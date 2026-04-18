//! Hybrid dense + BM25 retrieval over the corpus table.

use std::collections::HashMap;
use std::sync::Arc;

use chrono::{DateTime, Utc};
use futures::TryStreamExt;
use lancedb::Connection;
use lancedb::index::scalar::FullTextSearchQuery;
use lancedb::query::{ExecutableQuery, QueryBase};
use lancedb::rerankers::rrf::RRFReranker;
use ostk_recall_pipeline::ChunkEmbedder;
use ostk_recall_store::CORPUS_TABLE;

use crate::error::Result;
use crate::rerank::RerankerLike;
use crate::row::{batch_to_hits, sql_escape};
use crate::types::{RecallHit, RecallParams};

/// Default cap on hits per `source_id` after RRF rerank. One chatty session
/// could otherwise dominate top-K via shared tokens.
pub const DEFAULT_MAX_PER_SOURCE_ID: usize = 3;

/// How many candidates to fetch from `LanceDB` relative to the requested
/// `limit`.
///
/// Leaves room after the per-`source_id` post-filter and the cross-encoder
/// rerank pass. Bumped from 4× → 6× because the cross-encoder needs a
/// wider candidate pool to find precise matches that BM25/dense scored
/// mid-pack.
pub const PREFETCH_MULTIPLIER: usize = 6;

/// Execute a hybrid recall against the corpus table.
///
/// Pipeline:
/// 1. Dense + BM25 retrieval, fused by RRF in `LanceDB` → ~`limit *
///    PREFETCH_MULTIPLIER` candidates.
/// 2. Optional cross-encoder rerank: if `reranker` is `Some`, score each
///    candidate's full text against `query` and reorder by the new score.
///    Without a reranker, the RRF-fused order is preserved.
/// 3. Per-`source_id` diversity filter, truncated to `limit`.
pub async fn recall(
    conn: &Connection,
    embedder: &dyn ChunkEmbedder,
    reranker: Option<&dyn RerankerLike>,
    params: &RecallParams,
) -> Result<Vec<RecallHit>> {
    let query_text = params.query.trim();
    if query_text.is_empty() {
        return Ok(Vec::new());
    }

    let limit = params.limit.unwrap_or(10).max(1);
    let max_per_source_id = params
        .max_per_source_id
        .unwrap_or(DEFAULT_MAX_PER_SOURCE_ID);
    // Pre-fetch more candidates than `limit` so the diversity filter and
    // the cross-encoder rerank both have room to operate.
    let fetch_limit = limit.saturating_mul(PREFETCH_MULTIPLIER).max(limit);

    let vec = embedder
        .encode_batch(&[query_text])
        .into_iter()
        .next()
        .unwrap_or_default();

    let table = conn.open_table(CORPUS_TABLE).execute().await?;

    let filter = build_filter(
        params.project.as_deref(),
        params.source.as_deref(),
        params.since,
    );

    let mut q = table
        .query()
        .nearest_to(vec)?
        .full_text_search(FullTextSearchQuery::new(query_text.to_string()))
        .rerank(Arc::new(RRFReranker::default()))
        .limit(fetch_limit);

    if let Some(f) = &filter {
        q = q.only_if(f);
    }

    let stream = q.execute().await?;
    let batches: Vec<_> = stream.try_collect().await?;

    let mut candidates = Vec::new();
    for b in &batches {
        candidates.extend(batch_to_hits(b)?);
    }

    // Cross-encoder pass. The candidate text we score is the snippet that
    // already shipped through `batch_to_hits` (≤400 chars) — short, dense,
    // and what the user would see in the answer anyway. Doing this in a
    // `spawn_blocking` keeps the async runtime healthy even though the
    // ONNX call is CPU-bound.
    let candidates = if let Some(reranker) = reranker {
        rerank_candidates(reranker, query_text, candidates)?
    } else {
        candidates
    };

    Ok(diversify_by_source_id(candidates, limit, max_per_source_id))
}

/// Apply the cross-encoder reranker to a candidate pool. Drops candidates
/// whose snippet is empty (the reranker would score them at noise floor
/// anyway, and they're not useful answers).
fn rerank_candidates(
    reranker: &dyn RerankerLike,
    query: &str,
    candidates: Vec<RecallHit>,
) -> Result<Vec<RecallHit>> {
    if candidates.is_empty() {
        return Ok(candidates);
    }
    // Build the doc list parallel to candidates. Empty-snippet rows would
    // still be re-attached by index but produce noisy scores; we keep them
    // in place but feed the reranker a single space so indices stay aligned.
    let docs: Vec<String> = candidates
        .iter()
        .map(|h| {
            if h.snippet.trim().is_empty() {
                " ".to_string()
            } else {
                h.snippet.clone()
            }
        })
        .collect();
    let take = candidates.len();
    let ranked = reranker
        .rerank(query, &docs, take)
        .map_err(|e| crate::error::QueryError::Decode(format!("rerank: {e}")))?;

    // Reassemble candidates in the new order; replace the score with the
    // cross-encoder score so downstream callers see the post-rerank rank.
    let mut by_idx: Vec<Option<RecallHit>> = candidates.into_iter().map(Some).collect();
    let mut out = Vec::with_capacity(ranked.len());
    for r in ranked {
        if let Some(slot) = by_idx.get_mut(r.idx).and_then(Option::take) {
            let mut hit = slot;
            hit.score = r.score;
            out.push(hit);
        }
    }
    // Append any candidates the reranker dropped (shouldn't happen with
    // top_k = take, but defensive).
    for slot in by_idx.into_iter().flatten() {
        out.push(slot);
    }
    Ok(out)
}

/// Post-filter ranked hits so no single `source_id` exceeds
/// `max_per_source_id`. `max_per_source_id == 0` disables the filter
/// (returns the first `limit` hits unchanged). Stops short once `limit`
/// hits are collected or candidates are exhausted.
fn diversify_by_source_id(
    candidates: Vec<RecallHit>,
    limit: usize,
    max_per_source_id: usize,
) -> Vec<RecallHit> {
    if max_per_source_id == 0 {
        let mut out = candidates;
        if out.len() > limit {
            out.truncate(limit);
        }
        return out;
    }
    let mut counts: HashMap<String, usize> = HashMap::new();
    let mut out = Vec::with_capacity(limit);
    for hit in candidates {
        if out.len() >= limit {
            break;
        }
        let count = counts.entry(hit.source_id.clone()).or_insert(0);
        if *count >= max_per_source_id {
            continue;
        }
        *count += 1;
        out.push(hit);
    }
    out
}

fn build_filter(
    project: Option<&str>,
    source: Option<&str>,
    since: Option<DateTime<Utc>>,
) -> Option<String> {
    let mut clauses: Vec<String> = Vec::new();
    if let Some(p) = project {
        clauses.push(format!("project = '{}'", sql_escape(p)));
    }
    if let Some(s) = source {
        clauses.push(format!("source = '{}'", sql_escape(s)));
    }
    if let Some(t) = since {
        clauses.push(format!("ts >= TIMESTAMP '{}'", t.to_rfc3339()));
    }
    if clauses.is_empty() {
        None
    } else {
        Some(clauses.join(" AND "))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rerank::{RerankedHit, RerankerLike};
    use ostk_recall_core::Links;

    #[test]
    fn build_filter_empty() {
        assert!(build_filter(None, None, None).is_none());
    }

    #[test]
    fn build_filter_project_and_source() {
        let f = build_filter(Some("foo"), Some("markdown"), None).unwrap();
        assert!(f.contains("project = 'foo'"));
        assert!(f.contains("source = 'markdown'"));
        assert!(f.contains("AND"));
    }

    #[test]
    fn build_filter_escapes_quotes() {
        let f = build_filter(Some("a'b"), None, None).unwrap();
        assert_eq!(f, "project = 'a''b'");
    }

    #[test]
    fn build_filter_since_is_iso() {
        let t = DateTime::parse_from_rfc3339("2026-04-17T10:00:00Z")
            .unwrap()
            .with_timezone(&Utc);
        let f = build_filter(None, None, Some(t)).unwrap();
        assert!(f.contains("TIMESTAMP '2026-04-17T10:00:00+00:00'"));
    }

    fn fake_hit(chunk_id: &str, source_id: &str) -> RecallHit {
        RecallHit {
            chunk_id: chunk_id.to_string(),
            project: None,
            source: "markdown".to_string(),
            source_id: source_id.to_string(),
            ts: None,
            snippet: String::new(),
            score: 1.0,
            links: Links::default(),
        }
    }

    #[test]
    fn diversify_caps_per_source_id() {
        // 5 from "X" interleaved at the top; 5 from unique sources
        // following — emulates a chatty session monopolizing top-K.
        let mut candidates = Vec::new();
        for i in 0..5 {
            candidates.push(fake_hit(&format!("x{i}"), "X"));
        }
        for i in 0..5 {
            candidates.push(fake_hit(&format!("u{i}"), &format!("U{i}")));
        }
        let out = diversify_by_source_id(candidates, 5, 2);
        assert_eq!(out.len(), 5);
        let x_count = out.iter().filter(|h| h.source_id == "X").count();
        assert!(
            x_count <= 2,
            "expected ≤2 hits from source_id X, got {x_count}"
        );
    }

    #[test]
    fn diversify_zero_means_unlimited() {
        let candidates: Vec<RecallHit> = (0..7).map(|i| fake_hit(&format!("x{i}"), "X")).collect();
        let out = diversify_by_source_id(candidates, 5, 0);
        assert_eq!(out.len(), 5);
        assert!(out.iter().all(|h| h.source_id == "X"));
    }

    #[test]
    fn diversify_returns_short_when_pool_exhausted() {
        // Only one source_id and a strict cap — fewer hits than limit
        // is acceptable.
        let candidates: Vec<RecallHit> = (0..10).map(|i| fake_hit(&format!("x{i}"), "X")).collect();
        let out = diversify_by_source_id(candidates, 5, 2);
        assert_eq!(out.len(), 2);
    }

    /// Fake reranker that scores documents by how many query terms they
    /// contain (case-insensitive whitespace tokenization). Lets us prove
    /// the rerank wire-up changes order without pulling ONNX.
    struct TokenOverlapReranker;

    impl RerankerLike for TokenOverlapReranker {
        fn rerank(
            &self,
            query: &str,
            docs: &[String],
            top_k: usize,
        ) -> crate::rerank::Result<Vec<RerankedHit>> {
            let q_terms: Vec<String> = query.split_whitespace().map(str::to_lowercase).collect();
            let mut scored: Vec<(usize, f32)> = docs
                .iter()
                .enumerate()
                .map(|(i, d)| {
                    let lower = d.to_lowercase();
                    #[allow(clippy::cast_precision_loss)]
                    let s = q_terms
                        .iter()
                        .filter(|t| lower.contains(t.as_str()))
                        .count() as f32;
                    (i, s)
                })
                .collect();
            scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            Ok(scored
                .into_iter()
                .take(top_k)
                .map(|(idx, score)| RerankedHit { idx, score })
                .collect())
        }
        fn model_id(&self) -> &'static str {
            "token-overlap-fake"
        }
    }

    fn hit_with_text(chunk_id: &str, source_id: &str, text: &str) -> RecallHit {
        RecallHit {
            chunk_id: chunk_id.to_string(),
            project: None,
            source: "markdown".to_string(),
            source_id: source_id.to_string(),
            ts: None,
            snippet: text.to_string(),
            score: 0.0,
            links: Links::default(),
        }
    }

    #[test]
    fn rerank_candidates_promotes_more_relevant() {
        // RRF orders the noisy doc first; reranker should push the
        // term-rich doc to the top.
        let candidates = vec![
            hit_with_text("noise", "n", "lorem ipsum dolor sit amet"),
            hit_with_text("good", "g", "the answer is rust async runtime"),
        ];
        let out =
            rerank_candidates(&TokenOverlapReranker, "rust async runtime", candidates).unwrap();
        assert_eq!(out[0].chunk_id, "good", "expected term-rich doc on top");
    }

    #[test]
    fn rerank_candidates_handles_empty_input() {
        let out = rerank_candidates(&TokenOverlapReranker, "anything", Vec::new()).unwrap();
        assert!(out.is_empty());
    }

    #[test]
    fn rerank_candidates_attaches_score() {
        let candidates = vec![
            hit_with_text("a", "a", "alpha alpha alpha"),
            hit_with_text("b", "b", "beta"),
        ];
        let out = rerank_candidates(&TokenOverlapReranker, "alpha", candidates).unwrap();
        assert!(out[0].score >= out[1].score);
        assert_eq!(out[0].chunk_id, "a");
    }
}
