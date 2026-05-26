//! Divergence-from-baseline novelty surface.
//!
//! Complement to [`crate::activity`] (which answers "where is the focus
//! right now") and [`crate::emergent`] (which clusters by embedding
//! density). Novelty answers "what's a new direction" — chunks whose
//! embedding diverges most from the per-project baseline mean.
//!
//! Algorithm:
//!
//! 1. Per-project baseline: `baseline[p] = mean({embedding | project = p
//!    AND ts >= now - baseline_days})`. If fewer than 10 chunks for a
//!    project, fall back to a global baseline (`project = None`).
//! 2. Per-chunk novelty: `score = 1 - cos(embedding, baseline[chunk.project])`.
//!    Range `[0, 2]`; high = unlike baseline = novel.
//! 3. Take the top-K most novel chunks from the recency window.
//! 4. **Re-cluster guard**: feed the top-K back through
//!    [`crate::cluster::find_clusters_with`] at `recluster_threshold`
//!    (defaults to [`crate::cluster::EMERGENT_THRESHOLD`]). Only
//!    clusters with `>= min_cluster` members surface — drops isolated
//!    rare strings (typos, hashes) that score high but don't cohere.
//!
//! Empty `Vec` is the expected null state when no qualifying clusters
//! surface (insufficient baseline corpus, no recent chunks, or the
//! re-cluster guard rejected everything). Callers should treat it as
//! "nothing novel enough to cluster" rather than a bug, and reach for
//! [`crate::activity`] (the burst surface) as a complementary view.
//! This contract is intentional: a softened guard that surfaces noise
//! is worse than an honest empty result.

use std::collections::HashMap;
use std::sync::Arc;

use chrono::{DateTime, Utc};
use ostk_recall_store::CorpusStore;
use ostk_recall_store::corpus::StoreError as CorpusError;
use thiserror::Error;

use crate::cluster::{EMERGENT_THRESHOLD, MIN_CLUSTER_SIZE, find_clusters_with};
use crate::cosine_similarity;

/// Default per-project baseline window in days.
pub const DEFAULT_BASELINE_DAYS: i64 = 7;

/// Default look-back window for novel chunks.
pub const DEFAULT_SINCE_HOURS: i64 = 24;

/// Default upper bound on surfaced clusters.
pub const DEFAULT_LIMIT: usize = 10;

/// Default minimum members per surfaced cluster. Re-exports the
/// emergent surface's [`crate::cluster::MIN_CLUSTER_SIZE`] so the
/// two density bars cannot quietly drift apart.
pub const DEFAULT_MIN_CLUSTER: usize = MIN_CLUSTER_SIZE;

/// Default re-cluster cosine threshold. Matches
/// [`crate::cluster::EMERGENT_THRESHOLD`] so a novelty cluster passes
/// the same density bar as an emergent cluster.
pub const DEFAULT_RECLUSTER_THRESHOLD: f32 = EMERGENT_THRESHOLD;

/// Upper bound on chunks scored per call. `find_clusters_with` is
/// `O(N²)` over the top-K subset; 2000 keeps the recency window
/// realistic while bounding clustering cost.
pub const RECENT_SAMPLE_LIMIT: usize = 2000;

/// After scoring, retain this many * `limit` of the most novel chunks
/// as input to `find_clusters_with`. Five gives the clusterer enough
/// candidates to find ≥ `min_cluster` coherent members per surfaced
/// cluster while bounding the `O(N²)` pairwise cost.
pub const TOPK_MULTIPLIER: usize = 5;

/// Maximum characters retained per sample snippet.
pub const SAMPLE_CHAR_LIMIT: usize = 200;

/// Samples surfaced per cluster.
pub const SAMPLES_PER_CLUSTER: usize = 3;

/// Minimum mean novelty for a surfaced cluster. Drops the "everything
/// is novel relative to itself" failure mode where a corpus dominated
/// by one topic produces a cluster whose mean novelty is near zero —
/// technically a coherent cluster, but not what a caller asking
/// "what's novel" wants to see. `0.3` puts the threshold solidly above
/// the cosine-of-very-similar-things range (typically < 0.05) without
/// excluding genuine "somewhat divergent but still on-topic" clusters.
pub const MIN_MEAN_NOVELTY: f32 = 0.3;

#[derive(Debug, Error)]
pub enum NoveltyError {
    #[error("corpus error: {0}")]
    Corpus(#[from] CorpusError),
}

/// Single surfaced novelty cluster.
#[derive(Debug, Clone)]
pub struct NoveltyReport {
    /// Dominant project of the cluster's members (mode). Empty string
    /// when every member has `project = None`.
    pub project: String,
    /// Number of chunks in the cluster.
    pub members: usize,
    /// Mean novelty score over the cluster's members. Range `[0, 2]`.
    pub mean_novelty: f32,
    /// Maximum novelty score within the cluster. Range `[0, 2]`.
    pub max_novelty: f32,
    /// Full cluster membership, sorted lexicographically. **Internal
    /// to the attention crate** — fed to `thread_query`'s v0.4.1+
    /// cross-axis backfill (which needs exact membership). Unbounded
    /// in size; MCP handlers must not echo it to the wire.
    pub chunk_ids: Vec<String>,
    /// Sample text snippets pulled from the cluster's members.
    pub samples: Vec<String>,
}

/// Run the novelty surface against the corpus.
///
/// `min_mean_novelty` is the floor for the post-cluster filter that drops
/// "coherent but uninteresting" clusters whose members happen to resemble
/// each other near the baseline. `0.0` disables the filter entirely;
/// [`MIN_MEAN_NOVELTY`] is the historically-baked default and the value
/// [`surface_default`] continues to pass for back-compat. Callers wanting
/// permissive output (the v0.3.1 discipline default) should pass `0.0`
/// explicitly — see `crates/attention-mcp/src/handlers.rs::thread_novelty`.
#[allow(clippy::too_many_lines)] // baseline computation + per-chunk scoring + recluster
                                 // + filter is one logical pipeline; splitting hurts more
                                 // than it helps. v0.4.0 `thread_query` consolidates this.
pub async fn surface_novelty(
    corpus: &Arc<CorpusStore>,
    since: DateTime<Utc>,
    baseline_days: i64,
    limit: usize,
    min_cluster: usize,
    recluster_threshold: f32,
    min_mean_novelty: f32,
) -> Result<Vec<NoveltyReport>, NoveltyError> {
    if limit == 0 {
        return Ok(Vec::new());
    }

    let recent = corpus
        .sample_recent_chunks_with_project(since, RECENT_SAMPLE_LIMIT)
        .await?;
    if recent.is_empty() {
        return Ok(Vec::new());
    }

    let mut baselines: HashMap<Option<String>, Option<Vec<f32>>> = HashMap::new();
    let global = corpus.project_baseline_mean(None, baseline_days).await?;
    baselines.insert(None, global.clone());
    let mut distinct_projects: Vec<Option<String>> = recent
        .iter()
        .map(|(_, p, _)| p.clone())
        .collect::<std::collections::HashSet<_>>()
        .into_iter()
        .collect();
    distinct_projects.sort();
    for proj in distinct_projects {
        let Some(p) = proj.as_deref() else { continue };
        let b = corpus.project_baseline_mean(Some(p), baseline_days).await?;
        baselines.insert(Some(p.to_string()), b);
    }

    let mut scored: Vec<ScoredChunk> = Vec::with_capacity(recent.len());
    for (chunk_id, project, embedding) in recent {
        let Some(baseline) = lookup_baseline(&baselines, project.as_deref()) else {
            continue;
        };
        let score = 1.0 - cosine_similarity(&embedding, baseline);
        scored.push(ScoredChunk {
            chunk_id,
            project,
            embedding,
            score,
        });
    }
    if scored.is_empty() {
        return Ok(Vec::new());
    }

    scored.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let topk = limit.saturating_mul(TOPK_MULTIPLIER).max(min_cluster);
    scored.truncate(topk);

    let cluster_input: Vec<(String, Vec<f32>)> = scored
        .iter()
        .map(|c| (c.chunk_id.clone(), c.embedding.clone()))
        .collect();
    let by_id: HashMap<&str, &ScoredChunk> = scored
        .iter()
        .map(|c| (c.chunk_id.as_str(), c))
        .collect();

    let clusters = find_clusters_with(&cluster_input, recluster_threshold, min_cluster, 2);
    if clusters.is_empty() {
        return Ok(Vec::new());
    }

    let all_ids: Vec<String> = clusters
        .iter()
        .flat_map(|c| c.chunk_ids.iter().take(SAMPLES_PER_CLUSTER).cloned())
        .collect();
    let texts = corpus.fetch_texts(&all_ids).await.unwrap_or_default();

    let mut reports: Vec<NoveltyReport> = clusters
        .into_iter()
        .map(|c| {
            let scores: Vec<f32> = c
                .chunk_ids
                .iter()
                .filter_map(|id| by_id.get(id.as_str()).map(|s| s.score))
                .collect();
            let max_novelty = scores.iter().copied().fold(0.0_f32, f32::max);
            let mean_novelty = if scores.is_empty() {
                0.0
            } else {
                #[allow(clippy::cast_precision_loss)]
                let n = scores.len() as f32;
                scores.iter().sum::<f32>() / n
            };
            let project = dominant_project(&c.chunk_ids, &by_id);
            let samples: Vec<String> = c
                .chunk_ids
                .iter()
                .take(SAMPLES_PER_CLUSTER)
                .filter_map(|id| texts.get(id))
                .map(|t| snippet(t, SAMPLE_CHAR_LIMIT))
                .collect();
            NoveltyReport {
                project,
                members: c.chunk_ids.len(),
                mean_novelty,
                max_novelty,
                chunk_ids: c.chunk_ids.clone(),
                samples,
            }
        })
        .collect();

    // Drop clusters whose mean novelty is below `min_mean_novelty` —
    // these are coherent-but-uninteresting clusters that survive
    // `find_clusters_with` simply because the chunks resemble each
    // other (e.g. baseline-aligned chunks form a cluster with mean
    // novelty near zero). Caller-tunable per v0.3.1 discipline rule
    // (no baked filters); `0.0` disables the filter entirely.
    if min_mean_novelty > 0.0 {
        reports.retain(|r| r.mean_novelty >= min_mean_novelty);
    }
    reports.sort_by(|a, b| {
        b.mean_novelty
            .partial_cmp(&a.mean_novelty)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    reports.truncate(limit);
    Ok(reports)
}

struct ScoredChunk {
    chunk_id: String,
    project: Option<String>,
    embedding: Vec<f32>,
    score: f32,
}

fn lookup_baseline<'a>(
    cache: &'a HashMap<Option<String>, Option<Vec<f32>>>,
    project: Option<&str>,
) -> Option<&'a Vec<f32>> {
    if let Some(p) = project {
        if let Some(Some(v)) = cache.get(&Some(p.to_string())) {
            return Some(v);
        }
    }
    if let Some(Some(v)) = cache.get(&None) {
        return Some(v);
    }
    None
}

fn dominant_project(chunk_ids: &[String], by_id: &HashMap<&str, &ScoredChunk>) -> String {
    let mut counts: HashMap<String, usize> = HashMap::new();
    for id in chunk_ids {
        if let Some(c) = by_id.get(id.as_str()) {
            if let Some(p) = &c.project {
                *counts.entry(p.clone()).or_insert(0) += 1;
            }
        }
    }
    counts
        .into_iter()
        .max_by(|a, b| a.1.cmp(&b.1).then_with(|| b.0.cmp(&a.0)))
        .map(|(p, _)| p)
        .unwrap_or_default()
}

fn snippet(text: &str, max_chars: usize) -> String {
    let collapsed: String = text.split_whitespace().collect::<Vec<_>>().join(" ");
    if collapsed.chars().count() <= max_chars {
        return collapsed;
    }
    let cut: String = collapsed.chars().take(max_chars).collect();
    format!("{cut}…")
}

/// Convenience wrapper using all defaults. Preserves the pre-v0.3.1
/// post-cluster `MIN_MEAN_NOVELTY` filter for back-compat with existing
/// library callers; MCP callers go through `thread_novelty` which
/// defaults to `0.0` (permissive) per the no-baked-filters discipline.
pub async fn surface_default(
    corpus: &Arc<CorpusStore>,
    since: DateTime<Utc>,
) -> Result<Vec<NoveltyReport>, NoveltyError> {
    surface_novelty(
        corpus,
        since,
        DEFAULT_BASELINE_DAYS,
        DEFAULT_LIMIT,
        DEFAULT_MIN_CLUSTER,
        DEFAULT_RECLUSTER_THRESHOLD,
        MIN_MEAN_NOVELTY,
    )
    .await
}

#[cfg(test)]
#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::float_cmp
)]
mod tests {
    use super::*;
    use ostk_recall_core::{Chunk, Links, Source};
    use tempfile::TempDir;

    fn near_axis(axis: usize, dim: usize, jitter: f32) -> Vec<f32> {
        let mut v = vec![0.0_f32; dim];
        v[axis] = 1.0;
        let other = dim - 1;
        if other > 0 {
            let share = jitter / other as f32;
            for (i, slot) in v.iter_mut().enumerate() {
                if i != axis {
                    *slot = share;
                }
            }
        }
        v
    }

    fn chunk_at(id: &str, ts: chrono::DateTime<chrono::Utc>) -> Chunk {
        Chunk {
            chunk_id: id.to_string(),
            source: Source::Markdown,
            project: Some("p".into()),
            source_id: format!("{id}.md"),
            chunk_index: 0,
            ts: Some(ts),
            role: None,
            text: format!("text-{id}"),
            sha256: Chunk::content_hash(id),
            links: Links::default(),
            extra: serde_json::Value::Null,
        }
    }

    async fn seed_baseline(
        store: &CorpusStore,
        n: usize,
        dim: usize,
        axis: usize,
        now: chrono::DateTime<chrono::Utc>,
    ) {
        let mut chunks = Vec::new();
        let mut embs = Vec::new();
        for i in 0..n {
            chunks.push(chunk_at(&format!("base-{i}"), now));
            embs.push(near_axis(axis, dim, 0.001));
        }
        store.upsert(&chunks, &embs).await.unwrap();
    }

    #[tokio::test]
    async fn surface_novelty_scores_orthogonal_higher_than_aligned() {
        let tmp = TempDir::new().unwrap();
        let dim = 16;
        let store = Arc::new(CorpusStore::open_or_create(tmp.path(), dim).await.unwrap());
        let now = chrono::Utc::now();

        seed_baseline(&store, 20, dim, 0, now).await;

        let mut chunks = Vec::new();
        let mut embs = Vec::new();
        for i in 0..3 {
            chunks.push(chunk_at(&format!("novel-{i}"), now));
            embs.push(near_axis(7, dim, 0.001));
        }
        chunks.push(chunk_at("aligned", now));
        embs.push(near_axis(0, dim, 0.001));
        store.upsert(&chunks, &embs).await.unwrap();

        let since = now - chrono::Duration::hours(1);
        let reports = surface_novelty(&store, since, 7, 10, 3, DEFAULT_RECLUSTER_THRESHOLD, MIN_MEAN_NOVELTY)
            .await
            .unwrap();
        assert_eq!(reports.len(), 1, "got {reports:#?}");
        let r = &reports[0];
        assert_eq!(r.members, 3);
        // Note: baseline mean is pulled by the novel chunks themselves
        // (the baseline window includes all chunks in the corpus), so
        // axis-7 chunks score `1 - cos(axis7, ~0.87*axis0 + ~0.13*axis7)`
        // ≈ 0.85 rather than the naive 1.0. The recluster guard still
        // correctly separates them from the baseline-aligned chunks.
        assert!(r.mean_novelty > 0.8, "got {}", r.mean_novelty);
        assert_eq!(r.project, "p");
    }

    #[tokio::test]
    async fn recluster_guard_drops_isolated_outliers() {
        let tmp = TempDir::new().unwrap();
        let dim = 16;
        let store = Arc::new(CorpusStore::open_or_create(tmp.path(), dim).await.unwrap());
        let now = chrono::Utc::now();

        seed_baseline(&store, 20, dim, 0, now).await;

        let mut chunks = Vec::new();
        let mut embs = Vec::new();
        for (i, axis) in [5_usize, 7, 9].iter().enumerate() {
            chunks.push(chunk_at(&format!("outlier-{i}"), now));
            embs.push(near_axis(*axis, dim, 0.001));
        }
        store.upsert(&chunks, &embs).await.unwrap();

        let since = now - chrono::Duration::hours(1);
        let reports = surface_novelty(&store, since, 7, 10, 3, DEFAULT_RECLUSTER_THRESHOLD, MIN_MEAN_NOVELTY)
            .await
            .unwrap();
        assert!(reports.is_empty(), "got {reports:#?}");
    }

    #[tokio::test]
    async fn recluster_guard_keeps_coherent_novel_cluster() {
        let tmp = TempDir::new().unwrap();
        let dim = 16;
        let store = Arc::new(CorpusStore::open_or_create(tmp.path(), dim).await.unwrap());
        let now = chrono::Utc::now();

        seed_baseline(&store, 20, dim, 0, now).await;

        let mut chunks = Vec::new();
        let mut embs = Vec::new();
        for (i, jitter) in [0.001_f32, 0.002, 0.003, 0.004].iter().enumerate() {
            chunks.push(chunk_at(&format!("cluster-{i}"), now));
            embs.push(near_axis(7, dim, *jitter));
        }
        store.upsert(&chunks, &embs).await.unwrap();

        let since = now - chrono::Duration::hours(1);
        let reports = surface_novelty(&store, since, 7, 10, 3, DEFAULT_RECLUSTER_THRESHOLD, MIN_MEAN_NOVELTY)
            .await
            .unwrap();
        assert_eq!(reports.len(), 1);
        assert_eq!(reports[0].members, 4);
        // Same caveat as `surface_novelty_scores_orthogonal_higher_than_aligned`:
        // baseline gets pulled by the novel chunks themselves.
        assert!(reports[0].mean_novelty > 0.8);
    }

    #[tokio::test]
    async fn min_mean_novelty_zero_keeps_clusters_filter_drops() {
        // v0.3.1 discipline: `min_mean_novelty = 0.0` disables the
        // post-cluster filter (permissive default for MCP callers).
        // A high floor (above the achievable score) drops everything.
        // Uses the same setup as `surface_novelty_scores_orthogonal_higher_than_aligned`.
        let tmp = TempDir::new().unwrap();
        let dim = 16;
        let store = Arc::new(CorpusStore::open_or_create(tmp.path(), dim).await.unwrap());
        let now = chrono::Utc::now();

        seed_baseline(&store, 20, dim, 0, now).await;

        let mut chunks = Vec::new();
        let mut embs = Vec::new();
        for i in 0..3 {
            chunks.push(chunk_at(&format!("novel-{i}"), now));
            embs.push(near_axis(7, dim, 0.001));
        }
        store.upsert(&chunks, &embs).await.unwrap();

        let since = now - chrono::Duration::hours(1);

        // Permissive: 0.0 floor → at least the novel cluster surfaces.
        // (May also surface the baseline-aligned cluster — proving the
        // historical 0.3 default was a baked filter, which is exactly
        // the discipline rule's point.)
        let permissive = surface_novelty(&store, since, 7, 10, 3, DEFAULT_RECLUSTER_THRESHOLD, 0.0)
            .await
            .unwrap();
        assert!(
            !permissive.is_empty(),
            "0.0 floor should surface ≥1 cluster"
        );

        // Strict: 1.5 floor (above the ~0.85 achievable here) → empty.
        let strict = surface_novelty(&store, since, 7, 10, 3, DEFAULT_RECLUSTER_THRESHOLD, 1.5)
            .await
            .unwrap();
        assert!(
            strict.is_empty(),
            "1.5 floor should drop all clusters; got {strict:#?}"
        );

        // Sanity: strict floor returns strictly fewer clusters than permissive.
        assert!(strict.len() < permissive.len());
    }

    #[tokio::test]
    async fn surface_novelty_returns_empty_when_no_baseline() {
        let tmp = TempDir::new().unwrap();
        let dim = 16;
        let store = Arc::new(CorpusStore::open_or_create(tmp.path(), dim).await.unwrap());
        let now = chrono::Utc::now();

        let mut chunks = Vec::new();
        let mut embs = Vec::new();
        for i in 0..5 {
            chunks.push(chunk_at(&format!("c{i}"), now));
            embs.push(near_axis(0, dim, 0.001));
        }
        store.upsert(&chunks, &embs).await.unwrap();

        let since = now - chrono::Duration::hours(1);
        let reports = surface_novelty(&store, since, 7, 10, 3, DEFAULT_RECLUSTER_THRESHOLD, MIN_MEAN_NOVELTY)
            .await
            .unwrap();
        assert!(reports.is_empty());
    }
}
