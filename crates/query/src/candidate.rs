//! Per-recall candidate type — carries lane evidence + the columns
//! that rank features need (dense embedding, entities).
//!
//! Designed as a layered envelope over `Chunk`: the base observation
//! stays pure, and lane/feature inputs accrete on the `Candidate` as
//! retrieval runs. Extending `Chunk` itself with `dense_embedding` /
//! `entities` would force every Lance row decoder (corpus stats,
//! audit, chunk display) to project columns it doesn't need.

use ostk_recall_core::Chunk;

/// A retrieval candidate carrying lane evidence and rank-feature inputs.
///
/// Constructed during candidate union (see `lanes::build_candidates`).
/// Each `Option<*>` field is `Some` exactly when the corresponding lane
/// produced this chunk in the current recall call.
///
/// `dense_embedding` and `entities` are projected from Lance during
/// candidate construction so features (DenseSimilarity, AttentionAffinity,
/// EntitySalience) can score without N+1 round-trips. They live here,
/// not on `Chunk`, by design — `Chunk` is the durable observation;
/// `Candidate` is the per-query view.
#[derive(Clone, Debug)]
pub struct Candidate {
    pub chunk: Chunk,
    /// Dense embedding for this chunk. Projected during candidate
    /// construction; `None` only when the corpus row had no embedding
    /// (legacy) or projection was skipped.
    pub dense_embedding: Option<Vec<f32>>,
    /// Entities extracted at ingest. Empty until P7 wires the entity
    /// ledger; P3A populates only when the entity column is present.
    pub entities: Vec<String>,

    // ─── Per-lane evidence ────────────────────────────────────────────
    /// BM25 raw score from the FTS lane. `None` in ambient mode (no
    /// text query exists) and when the chunk surfaced only via dense.
    pub bm25_score: Option<f32>,
    pub bm25_rank: Option<u32>,
    /// Vector distance from the dense lane (Lance returns L2 distance
    /// on normalized vectors; convert to similarity at the feature).
    pub dense_distance: Option<f32>,
    pub dense_rank: Option<u32>,

    // ─── Fused / reranker outputs ─────────────────────────────────────
    /// Reciprocal-rank fusion score, computed in-code from per-lane
    /// ranks. The `Rrf` feature reads this and is the only feature
    /// with non-zero default weight in alpha.1 — preserves the current
    /// hybrid retrieval behavior. (See p3-rank-evidence.md.)
    pub rrf_score: Option<f32>,
    /// Cross-encoder rerank score, populated by the post-rank stage
    /// when a reranker is configured.
    pub rerank_score: Option<f32>,
    // P4 lands `maxsim_score: Option<f32>` here, eval-gated and
    // default-off behind the `colbert` feature.
}

impl Candidate {
    /// Construct a bare candidate around `chunk`. Lane and feature
    /// fields are filled in by the lane functions during union.
    #[must_use]
    pub fn for_chunk(chunk: Chunk) -> Self {
        Self {
            chunk,
            dense_embedding: None,
            entities: Vec::new(),
            bm25_score: None,
            bm25_rank: None,
            dense_distance: None,
            dense_rank: None,
            rrf_score: None,
            rerank_score: None,
        }
    }

    /// Did this candidate surface via the BM25 lane?
    #[must_use]
    pub const fn has_bm25_evidence(&self) -> bool {
        self.bm25_rank.is_some()
    }

    /// Did this candidate surface via the dense lane?
    #[must_use]
    pub const fn has_dense_evidence(&self) -> bool {
        self.dense_rank.is_some()
    }
}
