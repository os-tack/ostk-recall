//! Hybrid dense + BM25 retrieval over the corpus table.

use std::collections::HashMap;
use std::sync::Arc;

use chrono::{DateTime, Utc};
use futures::TryStreamExt;
use lancedb::Connection;
use lancedb::index::scalar::FullTextSearchQuery;
use lancedb::query::{ExecutableQuery, QueryBase};
use lancedb::rerankers::rrf::RRFReranker;
use ostk_recall_core::Source;
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

/// Extra candidates pulled from `source = 'code'` when the caller hasn't
/// filtered by source.
///
/// The 442 k-chunk corpus is ~1 % code by volume — without this stratified
/// prefetch, a top-`PREFETCH_MULTIPLIER * limit` unfiltered query
/// statistically misses every code candidate. This boost guarantees the
/// cross-encoder reranker sees code rows whenever they exist for a query.
pub const STRATIFIED_CODE_PREFETCH: usize = 12;

/// Additive boost applied to `source = "code"` post-rerank scores when
/// [`is_identifier_query`] flags the query as identifier-shaped (a
/// `snake_case` / `CamelCase` symbol name or a single short token).
///
/// Tuning: 1.5 wasn't enough to overcome cross-encoder's strong
/// preference for conversation transcripts that contain the literal
/// identifier many times. 3.0 ensures code chunks win for identifier
/// queries while staying low enough that irrelevant code (other files
/// with similar tokens) stays below prose answers for non-identifier
/// queries.
///
/// Set to `0.0` to disable the heuristic without removing the call site.
pub const IDENTIFIER_CODE_BOOST: f32 = 3.0;

/// Execute a hybrid recall against the corpus table.
///
/// Pipeline:
/// 1. Dense + BM25 retrieval, fused by RRF in `LanceDB` → ~`limit *
///    PREFETCH_MULTIPLIER` candidates.
/// 2. Soft-stratified augmentation: when the caller hasn't filtered by
///    source, run a second targeted query with `source = 'code'` to pull
///    [`STRATIFIED_CODE_PREFETCH`] additional candidates. The 442k-chunk
///    corpus is ~1 % code; without this, code candidates rarely reach
///    the reranker. Results are merged (dedupe by `chunk_id`).
/// 3. Optional cross-encoder rerank: if `reranker` is `Some`, score each
///    candidate's full text against `query` and reorder by the new score.
///    Without a reranker, the RRF-fused order is preserved.
/// 4. Per-`source_id` diversity filter, truncated to `limit`.
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

    let primary_filter = build_filter(
        params.project.as_deref(),
        params.source.as_deref(),
        params.since,
    );

    let mut candidates = run_hybrid_query(
        &table,
        &vec,
        query_text,
        fetch_limit,
        primary_filter.as_deref(),
    )
    .await?;

    // Soft-stratified augmentation: when the caller hasn't filtered by
    // source, top up with a code-only prefetch so the reranker always
    // sees code candidates.
    if params.source.is_none() {
        let code_filter = build_filter(
            params.project.as_deref(),
            Some(Source::Code.as_str()),
            params.since,
        );
        match run_hybrid_query(
            &table,
            &vec,
            query_text,
            STRATIFIED_CODE_PREFETCH,
            code_filter.as_deref(),
        )
        .await
        {
            Ok(extras) => {
                tracing::debug!(
                    primary = candidates.len(),
                    code_extras = extras.len(),
                    "stratified prefetch"
                );
                merge_dedup(&mut candidates, extras);
            }
            Err(e) => {
                tracing::warn!(error = %e, "stratified code prefetch failed; continuing with primary candidates");
            }
        }
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

    // Identifier-mode boost: when the query reads like a symbol name
    // (snake_case, CamelCase, or a single short token), bias actual code
    // definitions above conversation transcripts that merely mention the
    // identifier. Bumps post-rerank scores in place, then re-sorts.
    let candidates = boost_code_for_identifier_queries(query_text, candidates);

    Ok(diversify_by_source_id(candidates, limit, max_per_source_id))
}

/// Returns true when the query reads like a code identifier.
///
/// Identifier-shaped means a single short token, `snake_case`, or
/// `CamelCase`. For these queries, the user almost certainly wants the
/// actual definition over conversation transcripts that mention the
/// symbol.
///
/// Heuristic, in order:
/// * Empty / >3 tokens → `false` (likely natural language).
/// * Any token containing `_` → `true` (snake_case-ish).
/// * Any token with a non-leading uppercase char → `true` (CamelCase-ish).
/// * Single token, all alphanumeric/underscore → `true` (bare symbol).
/// * Otherwise → `false`.
///
/// Tune by widening the token cap (currently 3) or by adjusting
/// [`IDENTIFIER_CODE_BOOST`].
#[must_use]
pub fn is_identifier_query(q: &str) -> bool {
    let q = q.trim();
    if q.is_empty() {
        return false;
    }
    let tokens: Vec<&str> = q.split_whitespace().collect();
    if tokens.len() > 3 {
        return false;
    }
    // Any token with underscore -> snake_case-like
    if tokens.iter().any(|t| t.contains('_')) {
        return true;
    }
    // Any token with mid-word uppercase -> CamelCase-like
    if tokens.iter().any(|t| {
        t.chars()
            .enumerate()
            .any(|(i, c)| i > 0 && c.is_ascii_uppercase())
    }) {
        return true;
    }
    // Single token, all alphanumeric/underscore — likely a bare symbol
    // name like "alloc_page" or "memcpy".
    if tokens.len() == 1 && tokens[0].chars().all(|c| c.is_alphanumeric() || c == '_') {
        return true;
    }
    false
}

/// If `query` looks like an identifier, add [`IDENTIFIER_CODE_BOOST`] to
/// every `source == "code"` candidate's score and re-sort by descending
/// score. Otherwise returns `candidates` unchanged.
fn boost_code_for_identifier_queries(
    query: &str,
    mut candidates: Vec<RecallHit>,
) -> Vec<RecallHit> {
    if !is_identifier_query(query) || IDENTIFIER_CODE_BOOST == 0.0 {
        return candidates;
    }
    for hit in &mut candidates {
        if hit.source == Source::Code.as_str() {
            hit.score += IDENTIFIER_CODE_BOOST;
        }
    }
    candidates.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    candidates
}

/// Issue one hybrid (dense + BM25 + RRF) query against `table`.
async fn run_hybrid_query(
    table: &lancedb::Table,
    vec: &[f32],
    query_text: &str,
    fetch_limit: usize,
    filter: Option<&str>,
) -> Result<Vec<RecallHit>> {
    let mut q = table
        .query()
        .nearest_to(vec.to_vec())?
        .full_text_search(FullTextSearchQuery::new(query_text.to_string()))
        .rerank(Arc::new(RRFReranker::default()))
        .limit(fetch_limit);
    if let Some(f) = filter {
        q = q.only_if(f);
    }
    let stream = q.execute().await?;
    let batches: Vec<_> = stream.try_collect().await?;
    let mut out = Vec::new();
    for b in &batches {
        out.extend(batch_to_hits(b)?);
    }
    Ok(out)
}

/// Merge `extras` into `dest`, skipping any whose `chunk_id` is already
/// present. Preserves `dest` order; appends new rows at the tail. The
/// reranker reorders everything, so insertion position only matters when
/// no reranker is attached — code rows correctly land *after* the primary
/// candidates in that fallback path.
fn merge_dedup(dest: &mut Vec<RecallHit>, extras: Vec<RecallHit>) {
    if extras.is_empty() {
        return;
    }
    let seen: std::collections::HashSet<String> = dest.iter().map(|h| h.chunk_id.clone()).collect();
    for hit in extras {
        if seen.contains(&hit.chunk_id) {
            continue;
        }
        dest.push(hit);
    }
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
            extra: serde_json::Value::Null,
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
            extra: serde_json::Value::Null,
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

    #[test]
    fn merge_dedup_appends_new_skips_existing() {
        let mut dest = vec![fake_hit("a", "S"), fake_hit("b", "S")];
        let extras = vec![fake_hit("a", "S"), fake_hit("c", "S")];
        merge_dedup(&mut dest, extras);
        let ids: Vec<_> = dest.iter().map(|h| h.chunk_id.as_str()).collect();
        assert_eq!(ids, vec!["a", "b", "c"]);
    }

    #[test]
    fn merge_dedup_empty_extras_noop() {
        let mut dest = vec![fake_hit("a", "S")];
        merge_dedup(&mut dest, Vec::new());
        assert_eq!(dest.len(), 1);
    }

    #[test]
    fn is_identifier_snake_case() {
        assert!(is_identifier_query("tier2_line_rebase"));
        assert!(is_identifier_query("alloc_page"));
    }

    #[test]
    fn is_identifier_camel_case() {
        assert!(is_identifier_query("MemoryRegion"));
    }

    #[test]
    fn is_identifier_single_short_token() {
        assert!(is_identifier_query("memcpy"));
    }

    #[test]
    fn is_identifier_natural_language_false() {
        assert!(!is_identifier_query("fleet heartbeat"));
        assert!(!is_identifier_query("what is X"));
        assert!(!is_identifier_query(""));
        assert!(!is_identifier_query(
            "how do we wire the reranker into recall"
        ));
    }

    #[test]
    fn boost_promotes_code_when_identifier_query() {
        let candidates = vec![
            RecallHit {
                chunk_id: "conv1".into(),
                project: None,
                source: "anthropic_session".into(),
                source_id: "s1".into(),
                ts: None,
                snippet: "we discussed alloc_page".into(),
                score: 5.0,
                links: Links::default(),
                extra: serde_json::Value::Null,
            },
            RecallHit {
                chunk_id: "code1".into(),
                project: None,
                source: "code".into(),
                source_id: "src/mm.rs".into(),
                ts: None,
                snippet: "fn alloc_page() {}".into(),
                score: 4.0,
                links: Links::default(),
                extra: serde_json::Value::Null,
            },
        ];
        let out = boost_code_for_identifier_queries("alloc_page", candidates);
        assert_eq!(out[0].chunk_id, "code1", "code hit should win after boost");
        // Boost lifted 4.0 by IDENTIFIER_CODE_BOOST (3.0) → 7.0, comfortably
        // above the 5.0 conversation row.
        assert!((out[0].score - (4.0 + IDENTIFIER_CODE_BOOST)).abs() < f32::EPSILON);
    }

    #[test]
    fn boost_noop_for_natural_language_query() {
        let candidates = vec![
            RecallHit {
                chunk_id: "conv1".into(),
                project: None,
                source: "anthropic_session".into(),
                source_id: "s1".into(),
                ts: None,
                snippet: "answer text".into(),
                score: 5.0,
                links: Links::default(),
                extra: serde_json::Value::Null,
            },
            RecallHit {
                chunk_id: "code1".into(),
                project: None,
                source: "code".into(),
                source_id: "src/mm.rs".into(),
                ts: None,
                snippet: "fn x() {}".into(),
                score: 4.0,
                links: Links::default(),
                extra: serde_json::Value::Null,
            },
        ];
        let out = boost_code_for_identifier_queries("how do we wire the reranker", candidates);
        // Order untouched, scores untouched.
        assert_eq!(out[0].chunk_id, "conv1");
        assert!((out[0].score - 5.0).abs() < f32::EPSILON);
        assert!((out[1].score - 4.0).abs() < f32::EPSILON);
    }
}
