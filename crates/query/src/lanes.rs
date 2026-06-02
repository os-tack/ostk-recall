//! Per-lane candidate generation + in-code RRF fusion.
//!
//! P3A replaces the opaque "Lance does dense + FTS + RRF in one query"
//! path in `hybrid.rs` with explicit per-lane queries whose evidence
//! (rank + score) is visible on each `Candidate`. Fusion is computed
//! in-code via reciprocal-rank fusion.
//!
//! Two retrieval modes per `architecture.md` § "Retrieval invariants":
//!
//! - [`ambient_candidates`]: dense only. **BM25 is OFF by invariant**
//!   — no user text query exists in ambient mode.
//! - [`explicit_candidates`]: BM25 + dense (the current `recall(text)`
//!   shape).
//!
//! MaxSim (P4) is NEVER a candidate lane. It's a rerank-only feature
//! that scores the union produced by the lanes here.

use std::collections::HashMap;

use futures::TryStreamExt;
use lancedb::Table;
use lancedb::index::scalar::FullTextSearchQuery;
use lancedb::query::{ExecutableQuery, QueryBase, Select};
use ostk_recall_core::Chunk;
use ostk_recall_store::{CORPUS_TABLE, CorpusStore};

use crate::candidate::Candidate;
use crate::context::{AttentionContext, QueryContext};
use crate::error::{QueryError, Result};

/// Reciprocal-rank fusion constant. 60.0 is the canonical default and
/// also what Lance's internal `RRFReranker::default()` uses; matching
/// it keeps the P3A golden test (top-3 identical, top-10 within 1
/// swap vs v0.5) tight.
pub const K_RRF: f32 = 60.0;

/// One lane's hit for a chunk: `(chunk_id, raw_score, rank)`. Rank is
/// 0-indexed (most-relevant first). Score semantics vary by lane —
/// BM25 returns `_score` (higher better), dense returns `_distance`
/// (lower better) — but the rank is what fuses lanes.
pub type LaneEntry = (String, f32, u32);

/// BM25 lane. Explicit-mode only.
///
/// Returns up to `k` chunks ranked by the FTS `_score` column. The
/// projection is intentionally narrow — only `chunk_id` and `_score`
/// are needed for lane evidence; full chunk data is fetched once for
/// the union (see [`build_candidates`]).
pub async fn lane_bm25(
    table: &Table,
    query_text: &str,
    filter: Option<&str>,
    k: usize,
) -> Result<Vec<LaneEntry>> {
    let mut q = table
        .query()
        .full_text_search(FullTextSearchQuery::new(query_text.to_string()))
        .select(Select::Columns(vec!["chunk_id".into()]))
        .limit(k);
    if let Some(f) = filter {
        q = q.only_if(f);
    }
    let stream = q.execute().await?;
    let batches = stream.try_collect::<Vec<_>>().await?;
    decode_lane(&batches, "_score")
}

/// Dense (vector) lane.
///
/// Returns up to `k` chunks ranked by `_distance` (ascending; closer
/// vectors first).
pub async fn lane_dense(
    table: &Table,
    query_vec: &[f32],
    filter: Option<&str>,
    k: usize,
) -> Result<Vec<LaneEntry>> {
    let mut q = table
        .query()
        .nearest_to(query_vec.to_vec())?
        .select(Select::Columns(vec!["chunk_id".into()]))
        .limit(k);
    if let Some(f) = filter {
        q = q.only_if(f);
    }
    let stream = q.execute().await?;
    let batches = stream.try_collect::<Vec<_>>().await?;
    decode_lane(&batches, "_distance")
}

/// Decode a stream of batches into `(chunk_id, score, rank)` entries.
/// Rank is assigned by row position in the returned stream — Lance
/// guarantees ordering matches the lane's ranking function.
///
/// **Hard requirement**: the score column (`_score` for BM25,
/// `_distance` for dense) MUST be present. If it isn't (e.g. Lance
/// changes the convention, or the query was misconfigured), this
/// errors rather than silently substituting `0.0`. The lane evidence
/// is the whole point of the P3A refactor — a silent zero would hide
/// regressions in retrieval quality.
fn decode_lane(batches: &[arrow::array::RecordBatch], score_col: &str) -> Result<Vec<LaneEntry>> {
    use arrow::array::{Array, Float32Array, StringArray};

    let mut out: Vec<LaneEntry> = Vec::new();
    for batch in batches {
        let n = batch.num_rows();
        if n == 0 {
            continue;
        }
        let id_col = batch
            .column_by_name("chunk_id")
            .ok_or_else(|| QueryError::Decode("chunk_id column missing".into()))?;
        let ids = id_col
            .as_any()
            .downcast_ref::<StringArray>()
            .ok_or_else(|| QueryError::Decode("chunk_id not Utf8".into()))?;
        let score_col_dyn = batch.column_by_name(score_col).ok_or_else(|| {
            QueryError::Decode(format!(
                "lane score column `{score_col}` missing — Lance did not surface lane evidence. \
                 Schema-changed upstream? Lane queries depend on `_score` (BM25) / `_distance` (dense)."
            ))
        })?;
        let score_arr = score_col_dyn
            .as_any()
            .downcast_ref::<Float32Array>()
            .ok_or_else(|| {
                QueryError::Decode(format!("lane score column `{score_col}` not Float32"))
            })?;
        for i in 0..n {
            let rank = u32::try_from(out.len()).unwrap_or(u32::MAX);
            // Null score is treated as 0.0 explicitly (not because the
            // column was absent — that errored above — but because
            // Lance may emit null when a row has no contribution to
            // the ranking function).
            let score = if score_arr.is_null(i) {
                0.0
            } else {
                score_arr.value(i)
            };
            out.push((ids.value(i).to_string(), score, rank));
        }
    }
    Ok(out)
}

/// Reciprocal-rank fusion over per-lane outputs.
///
/// For each chunk that appears in any lane:
///   `rrf(chunk) = Σ over lanes l : 1 / (K_RRF + rank_l(chunk))`
///
/// Returns a map keyed by chunk_id with the fused score. Chunks not
/// in any lane are simply absent.
#[must_use]
pub fn fuse_rrf(lanes: &[&[LaneEntry]]) -> HashMap<String, f32> {
    let mut out: HashMap<String, f32> = HashMap::new();
    for lane in lanes {
        for (id, _score, rank) in lane.iter() {
            let contrib = 1.0_f32 / (K_RRF + *rank as f32);
            *out.entry(id.clone()).or_insert(0.0) += contrib;
        }
    }
    out
}

/// Build the union of candidates from per-lane outputs.
///
/// `chunks` is a pre-fetched map of `chunk_id -> Chunk` covering all
/// ids that appear in any lane. The caller is responsible for the
/// fetch; doing it here would force the lanes module to know about
/// `CorpusStore` directly. (See `hybrid.rs` post-refactor for the
/// call site.)
///
/// Sets per-lane evidence on each candidate (bm25_score/rank,
/// dense_distance/rank), computes RRF, and stamps `rrf_score`. Does
/// NOT run any rank features — that's `RankEngine::rank`'s job.
#[must_use]
pub fn build_candidates(
    bm25: &[LaneEntry],
    dense: &[LaneEntry],
    mut chunks: HashMap<String, Chunk>,
) -> Vec<Candidate> {
    let mut by_id: HashMap<String, Candidate> = HashMap::new();

    for (id, score, rank) in bm25 {
        if let Some(chunk) = chunks.remove(id) {
            let c = by_id
                .entry(id.clone())
                .or_insert_with(|| Candidate::for_chunk(chunk));
            c.bm25_score = Some(*score);
            c.bm25_rank = Some(*rank);
        }
    }
    for (id, dist, rank) in dense {
        if let Some(c) = by_id.get_mut(id) {
            c.dense_distance = Some(*dist);
            c.dense_rank = Some(*rank);
        } else if let Some(chunk) = chunks.remove(id) {
            let mut c = Candidate::for_chunk(chunk);
            c.dense_distance = Some(*dist);
            c.dense_rank = Some(*rank);
            by_id.insert(id.clone(), c);
        }
    }

    let rrf = fuse_rrf(&[bm25, dense]);
    for (id, c) in by_id.iter_mut() {
        if let Some(score) = rrf.get(id) {
            c.rrf_score = Some(*score);
        }
    }

    // Deterministic order out: `HashMap::into_values` randomizes, and
    // downstream RankEngine sort breaks ties on `chunk_id`, so we may
    // as well present a stable sequence here too. Defensive: keeps
    // anything that consumes pre-rank candidates (tests, debug dumps)
    // reproducible.
    let mut out: Vec<Candidate> = by_id.into_values().collect();
    out.sort_by(|a, b| a.chunk.chunk_id.cmp(&b.chunk.chunk_id));
    out
}

/// Normalized `Rrf` feature score.
///
/// Maximum possible RRF for `n` lanes all ranked 0 is `n / K_RRF`.
/// In the P3A default config (dense + BM25), that's `2 / K_RRF`.
/// Dividing by `2 / K_RRF` saturates a perfect-rank-0-in-both chunk
/// at exactly 1.0. Chunks present in only one lane score around 0.5.
///
/// Default alpha.1 weight: `1.0`. This is the only feature with
/// non-zero default weight; the explicit-recall golden test verifies
/// numerical equivalence with v0.5.
#[must_use]
pub fn rrf_score_normalized(rrf_raw: f32) -> f32 {
    let max = 2.0_f32 / K_RRF;
    (rrf_raw / max).clamp(0.0, 1.0)
}

/// Ambient candidate generation — dense lane only.
///
/// **Invariant** (`architecture.md` § "Retrieval invariants"): there
/// is no user text query in ambient mode, so the BM25 lane is OFF.
/// `lens_ambient_skips_bm25` proves every returned candidate has
/// `bm25_rank == None`.
///
/// `attn_ctx.scope_vector` is the dense query vector — typically the
/// effective attention vector (`pinned → rolling → transient`). When
/// `scope_vector` is `None` (empty-mind boot), this returns an empty
/// `Vec`; the lens loop's empty-mind-skip catches that upstream.
///
/// Sized via `k_per_lane` rather than a single `limit` because P3B
/// will add multiple lanes (entity-anchored, concept-anchored); each
/// fetches `k_per_lane` and the union de-dupes.
///
/// The current P3A scope is dense only; entity/concept lanes are
/// scaffolded later (P7/P8). Filter is forwarded to Lance verbatim
/// (Lance SQL).
pub async fn ambient_candidates(
    store: &CorpusStore,
    attn_ctx: &AttentionContext,
    filter: Option<&str>,
    k_per_lane: usize,
) -> Result<Vec<Candidate>> {
    let Some(scope_vec) = attn_ctx.scope_vector.as_deref() else {
        return Ok(Vec::new());
    };
    let table = store
        .connection()
        .open_table(CORPUS_TABLE)
        .execute()
        .await?;
    let dense = lane_dense(&table, scope_vec, filter, k_per_lane).await?;
    finalize_candidates(store, &[], &dense).await
}

/// Relational candidate lane (relational-substrate slice 2): inject the chunks
/// of diffusion-reached neighbour concepts so they can **surface** in the lens,
/// not merely be re-ranked. Takes a precomputed [`RelationalSupport`] (the
/// caller runs the diffusion once and shares it with the `relational_lift`
/// feature). Empty when no neighbour evidence resolved to a chunk. Built
/// candidates carry their dense embedding (so `attention_affinity` can still
/// score them) but no dense/bm25 lane rank — they did not come from a
/// similarity query; `relational_lift` scores them by coordinate.
pub async fn relational_candidates(
    store: &CorpusStore,
    support: &ostk_recall_store::RelationalSupport,
) -> Result<Vec<Candidate>> {
    if support.inject_chunk_ids.is_empty() {
        return Ok(Vec::new());
    }
    let fetched = store.fetch_chunks_by_ids(&support.inject_chunk_ids).await?;
    let mut out = Vec::with_capacity(fetched.len());
    for (_id, (chunk, emb)) in fetched {
        let mut c = Candidate::for_chunk(chunk);
        c.dense_embedding = emb;
        out.push(c);
    }
    Ok(out)
}

/// Explicit candidate generation — BM25 + dense lanes.
///
/// Only `QueryContext::Explicit { text, embedding }` is valid here;
/// calling with `Ambient` returns an empty `Vec`. (The lens-loop call
/// site is `ambient_candidates` above.) `attn_ctx` is accepted for
/// signature symmetry with `ambient_candidates` and for future P3B
/// features that read it during candidate generation.
///
/// Stratified code prefetch is handled by the caller via
/// `merge_dense_with_extras` so this function stays focused on the
/// canonical two-lane shape and remains usable by tests that don't
/// want the prefetch heuristic.
pub async fn explicit_candidates(
    store: &CorpusStore,
    query: &QueryContext,
    _attn_ctx: &AttentionContext,
    filter: Option<&str>,
    k_per_lane: usize,
) -> Result<Vec<Candidate>> {
    let (text, vec) = match query {
        QueryContext::Explicit { text, embedding } => (text.as_str(), embedding.as_slice()),
        QueryContext::Ambient => return Ok(Vec::new()),
    };
    let table = store
        .connection()
        .open_table(CORPUS_TABLE)
        .execute()
        .await?;
    let bm25 = lane_bm25(&table, text, filter, k_per_lane).await?;
    let dense = lane_dense(&table, vec, filter, k_per_lane).await?;
    finalize_candidates(store, &bm25, &dense).await
}

/// Shared tail: union ids → batch-fetch chunks + embeddings → build
/// `Candidate`s with lane evidence + RRF stamped. Used by both
/// ambient and explicit modes.
async fn finalize_candidates(
    store: &CorpusStore,
    bm25: &[LaneEntry],
    dense: &[LaneEntry],
) -> Result<Vec<Candidate>> {
    let mut union_ids: Vec<String> = Vec::with_capacity(bm25.len() + dense.len());
    union_ids.extend(bm25.iter().map(|(id, _, _)| id.clone()));
    union_ids.extend(dense.iter().map(|(id, _, _)| id.clone()));
    union_ids.sort();
    union_ids.dedup();
    if union_ids.is_empty() {
        return Ok(Vec::new());
    }
    let fetched = store.fetch_chunks_by_ids(&union_ids).await?;
    let mut chunks: HashMap<String, Chunk> = HashMap::with_capacity(fetched.len());
    let mut embeddings: HashMap<String, Vec<f32>> = HashMap::new();
    for (id, (chunk, emb)) in fetched {
        if let Some(e) = emb {
            embeddings.insert(id.clone(), e);
        }
        chunks.insert(id, chunk);
    }
    let mut candidates = build_candidates(bm25, dense, chunks);
    for c in &mut candidates {
        if let Some(e) = embeddings.remove(&c.chunk.chunk_id) {
            c.dense_embedding = Some(e);
        }
    }
    Ok(candidates)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ostk_recall_core::{FacetSet, Links, Source};

    fn ce(id: &str, score: f32, rank: u32) -> LaneEntry {
        (id.to_string(), score, rank)
    }

    fn chunk(id: &str) -> Chunk {
        Chunk {
            chunk_id: id.to_string(),
            source: Source::Markdown,
            project: None,
            source_id: format!("src/{id}"),
            source_config_id: "test:cfg".into(),
            chunk_index: 0,
            ts: None,
            role: None,
            text: format!("text {id}"),
            sha256: format!("sha-{id}"),
            links: Links::default(),
            facets: FacetSet::default(),
            embedding_input_sha256: format!("emb-{id}"),
            extra: serde_json::Value::Null,
        }
    }

    #[test]
    fn fuse_rrf_in_both_lanes_sums_contributions() {
        let bm25 = vec![ce("a", 1.0, 0), ce("b", 0.8, 1)];
        let dense = vec![ce("a", 0.1, 0), ce("c", 0.2, 1)];
        let rrf = fuse_rrf(&[&bm25, &dense]);
        let a = rrf.get("a").copied().unwrap();
        // a is rank 0 in both lanes → 2 * (1 / (60 + 0)) = 2/60 = 0.0333...
        assert!((a - (2.0 / K_RRF)).abs() < 1e-6);
        let b = rrf.get("b").copied().unwrap();
        // b only in bm25 at rank 1 → 1 / (60 + 1) = 1/61
        assert!((b - (1.0 / (K_RRF + 1.0))).abs() < 1e-6);
    }

    #[test]
    fn rrf_score_normalized_saturates_at_one() {
        // both lanes rank 0 → exactly 1.0
        let raw = 2.0 / K_RRF;
        assert!((rrf_score_normalized(raw) - 1.0).abs() < 1e-6);
        // overshooting still clamps
        assert_eq!(rrf_score_normalized(10.0), 1.0);
        // zero stays zero
        assert_eq!(rrf_score_normalized(0.0), 0.0);
        // single lane rank 0 ≈ 0.5
        let single = 1.0 / K_RRF;
        assert!((rrf_score_normalized(single) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn build_candidates_unions_lanes_and_stamps_rrf() {
        let bm25 = vec![ce("a", 1.0, 0), ce("b", 0.8, 1)];
        let dense = vec![ce("a", 0.1, 0), ce("c", 0.2, 1)];
        let mut chunks = HashMap::new();
        for id in ["a", "b", "c"] {
            chunks.insert(id.to_string(), chunk(id));
        }

        let candidates = build_candidates(&bm25, &dense, chunks);
        assert_eq!(candidates.len(), 3);

        let a = candidates.iter().find(|c| c.chunk.chunk_id == "a").unwrap();
        assert!(a.has_bm25_evidence());
        assert!(a.has_dense_evidence());
        assert_eq!(a.bm25_rank, Some(0));
        assert_eq!(a.dense_rank, Some(0));
        assert!(a.rrf_score.unwrap() > 0.0);

        let b = candidates.iter().find(|c| c.chunk.chunk_id == "b").unwrap();
        assert!(b.has_bm25_evidence());
        assert!(!b.has_dense_evidence());

        let c = candidates.iter().find(|c| c.chunk.chunk_id == "c").unwrap();
        assert!(!c.has_bm25_evidence());
        assert!(c.has_dense_evidence());
    }

    #[test]
    fn ambient_build_skips_bm25_evidence() {
        // Critical P9b-min invariant: ambient candidates never carry
        // bm25 evidence (because the ambient candidate path never
        // calls lane_bm25). This test proves `build_candidates`
        // respects an empty BM25 slice — every candidate ends with
        // bm25_score == None.
        let bm25: Vec<LaneEntry> = Vec::new();
        let dense = vec![ce("a", 0.1, 0), ce("b", 0.2, 1)];
        let mut chunks = HashMap::new();
        for id in ["a", "b"] {
            chunks.insert(id.to_string(), chunk(id));
        }

        let candidates = build_candidates(&bm25, &dense, chunks);
        for c in &candidates {
            assert!(
                c.bm25_score.is_none() && c.bm25_rank.is_none(),
                "ambient candidate carried BM25 evidence: {:?}",
                c.chunk.chunk_id
            );
            assert!(c.has_dense_evidence());
        }
    }
}
