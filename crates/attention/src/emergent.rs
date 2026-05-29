//! On-demand emergent-thread discovery over the existing corpus.
//!
//! Unlike the in-stream pass driven by [`crate::weaver::AutoWeaver`]
//! (which only sees chunks coming through a fresh `IngestEvent`), this
//! module surfaces clusters from chunks **already in the corpus**. It
//! samples a recency window, runs the same [`crate::cluster`] primitive,
//! optionally writes the survivors to `threads_proposed`, and returns
//! human-readable reports with sample text per cluster.
//!
//! The deterministic-handle scheme from `weaver::generate_proposed_handle`
//! means re-running on the same day collapses to the same proposal rows
//! via the UNIQUE constraint — the function is idempotent.

use std::sync::Arc;

use chrono::{DateTime, Utc};
use ostk_recall_store::corpus::StoreError as CorpusError;
use ostk_recall_store::{CorpusStore, ProposedThreadRecord, ThreadsDb};
use thiserror::Error;

use crate::cluster::{
    EMERGENT_THRESHOLD, EmergentCluster, MIN_NEIGHBOURS_IN_CLUSTER, find_clusters_with,
};
use crate::weaver::generate_proposed_handle;

/// Returned for each surfaced cluster — enough for an operator to decide
/// whether to promote it to a real thread.
#[derive(Debug, Clone)]
pub struct EmergentReport {
    /// `proposed-<8-hex>` handle (deterministic from `chunk_ids` + date).
    pub handle: String,
    /// Number of chunks in the cluster.
    pub members: usize,
    /// Mean pairwise cosine similarity within the cluster (`0.0–1.0`).
    pub cohesion: f32,
    /// Full cluster membership, sorted lexicographically. **Internal
    /// to the attention crate** — fed to `thread_query`'s v0.4.1+
    /// cross-axis backfill (which needs exact membership). Unbounded
    /// in size; MCP handlers must not echo it to the wire.
    pub chunk_ids: Vec<String>,
    /// Short snippets from the cluster members — first ~120 chars of
    /// each, up to a configurable cap.
    pub samples: Vec<String>,
}

#[derive(Debug, Error)]
pub enum EmergentError {
    #[error("corpus error: {0}")]
    Corpus(#[from] CorpusError),
}

/// Default look-back window for emergent surfacing. Twelve hours is
/// "what's been going on recently" without lookback so deep that
/// yesterday's noise drowns out today's signal.
pub const DEFAULT_SINCE_HOURS: i64 = 12;

/// Default upper bound on chunks fed to `find_clusters`. `O(N²)` keeps
/// 500 well under 250k pairs — sub-second on a typical dev machine.
pub const DEFAULT_LIMIT: usize = 500;

/// Default minimum cluster size. Higher than the `AutoWeaver` default
/// (3) because backfill operates on a wider, noisier window — 5+ chunks
/// resonating means a real thread, not coincidence.
pub const DEFAULT_MIN_CLUSTER_SIZE: usize = 5;

/// Sample of text snippets surfaced per report.
pub const SAMPLE_LIMIT: usize = 5;

/// Run a one-shot emergent-thread discovery pass.
///
/// 1. Pulls up to `limit` non-stale chunks ingested since `since` from
///    the corpus (lance query).
/// 2. Clusters them via [`find_clusters`] at the configured cosine
///    `threshold`, keeping only clusters with at least
///    `min_cluster_size` members.
/// 3. If `persist` is true, writes each cluster as a
///    [`ProposedThreadRecord`] in `threads`. UNIQUE-constraint
///    collisions are treated as idempotent (already known) and don't
///    surface as errors.
/// 4. Fetches sample text for the cluster members and returns reports.
///
/// Returns an empty `Vec` when the recency window produced no chunks
/// or no qualifying clusters — those are valid "nothing to surface"
/// outcomes, not errors.
#[allow(clippy::too_many_arguments)] // v0.3.1: discipline rule says expose constants
// as args; the v0.4.0 `thread_query` work absorbs
// this surface into a single multi-axis verb.
pub async fn discover_and_surface(
    corpus: &Arc<CorpusStore>,
    threads: &Arc<ThreadsDb>,
    since: DateTime<Utc>,
    limit: usize,
    min_cluster_size: usize,
    threshold: f32,
    min_in_cluster_neighbours: usize,
    persist: bool,
) -> Result<Vec<EmergentReport>, EmergentError> {
    let sample = corpus.sample_recent_chunks(since, limit).await?;
    if sample.is_empty() {
        return Ok(Vec::new());
    }
    let clusters: Vec<EmergentCluster> = find_clusters_with(
        &sample,
        threshold,
        min_cluster_size,
        min_in_cluster_neighbours,
    );
    if clusters.is_empty() {
        return Ok(Vec::new());
    }

    let now = Utc::now();
    let mut reports: Vec<EmergentReport> = Vec::with_capacity(clusters.len());
    for cluster in clusters {
        let handle = generate_proposed_handle(&cluster.chunk_ids);

        if persist {
            let record = ProposedThreadRecord {
                id: 0,
                proposed_handle: handle.clone(),
                chunk_ids: cluster.chunk_ids.clone(),
                centroid_vec: cluster.centroid.clone(),
                cohesion: cluster.cohesion,
                created_at: now,
                promoted_to: None,
            };
            // Idempotent: the UNIQUE constraint on `proposed_handle`
            // collapses repeat runs on the same day. Other errors are
            // logged but don't abort the surfacing pass.
            if let Err(e) = threads.insert_proposed_thread(&record) {
                tracing::warn!(
                    error = %e,
                    handle = %handle,
                    "emergent: proposed-thread insert failed (likely UNIQUE collision)"
                );
            }
        }

        // Pull text for up to SAMPLE_LIMIT members so the operator has
        // something to read. The full chunk_ids set stays in the
        // ProposedThreadRecord; samples here are just for the report.
        let sample_ids: Vec<String> = cluster
            .chunk_ids
            .iter()
            .take(SAMPLE_LIMIT)
            .cloned()
            .collect();
        let texts = corpus.fetch_texts(&sample_ids).await.unwrap_or_default();
        let samples: Vec<String> = sample_ids
            .iter()
            .filter_map(|id| texts.get(id))
            .map(|t| snippet(t, 120))
            .collect();

        reports.push(EmergentReport {
            handle,
            members: cluster.chunk_ids.len(),
            cohesion: cluster.cohesion,
            chunk_ids: cluster.chunk_ids.clone(),
            samples,
        });
    }

    // Sort by cohesion desc — strongest signals first.
    reports.sort_by(|a, b| {
        b.cohesion
            .partial_cmp(&a.cohesion)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    Ok(reports)
}

/// Trim text to the first `max_chars` chars at a UTF-8 boundary,
/// collapsing inner whitespace. Used for the report `samples` field.
fn snippet(text: &str, max_chars: usize) -> String {
    let collapsed: String = text.split_whitespace().collect::<Vec<_>>().join(" ");
    if collapsed.chars().count() <= max_chars {
        return collapsed;
    }
    let cut: String = collapsed.chars().take(max_chars).collect();
    format!("{cut}…")
}

/// Convenience wrapper using all defaults. Equivalent to
/// `discover_and_surface` with `DEFAULT_LIMIT`, `DEFAULT_MIN_CLUSTER_SIZE`,
/// `EMERGENT_THRESHOLD`, `MIN_NEIGHBOURS_IN_CLUSTER`, persist = true.
pub async fn discover_default(
    corpus: &Arc<CorpusStore>,
    threads: &Arc<ThreadsDb>,
    since: DateTime<Utc>,
) -> Result<Vec<EmergentReport>, EmergentError> {
    discover_and_surface(
        corpus,
        threads,
        since,
        DEFAULT_LIMIT,
        DEFAULT_MIN_CLUSTER_SIZE,
        EMERGENT_THRESHOLD,
        MIN_NEIGHBOURS_IN_CLUSTER,
        true,
    )
    .await
}
