//! Schema-only result/parameter types for the recall surface.
//!
//! These cross process boundaries: haystack (kernel) and ostk-recall-serve
//! (peer-process daemon) both depend on this module to serialize/deserialize
//! the wire shapes. Keep this file pure-rust + serde — no lance, no lancedb,
//! no datafusion, no arrow.
//!
//! Moved from `ostk_recall_query::types` and `ostk_recall_query::synthesis`
//! in v0.1.5 (cut #3 prep, →1848). The query crate keeps `pub use` re-exports
//! for backward compatibility with existing consumers.

use crate::attention::AttentionScope;
use crate::{ContextRole, Links, RecallIntent};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// Optional attention-bias rider for [`RecallParams`].
///
/// When present, the recall surface re-scores each hit by blending the
/// base hybrid score with two independent attention axes:
///
/// - **Thread-mediated** (`thread_weight`): the max
///   `score_thread(h)` over every thread `h` that
///   `find_threads_for_chunk(chunk_id)` returns. Lifts hits whose
///   chunk is already cited by a thread the operator is paying
///   attention to. This is the v0.4.x behaviour and is the default.
/// - **Embedding-mediated** (`embedding_weight`): the cosine
///   similarity between the hit's chunk embedding and the scope's
///   current attention vector
///   (`InMemoryAttention::scope_vector`). Lifts hits whose content
///   matches the operator's stated focus directly — no thread
///   required. Requires the attention runtime to share an embedder
///   with the corpus (production: `cli::commands::serve` asserts
///   this at startup). Opt-in (default 0.0) until callers opt in.
///
/// Composition: `final_score = base_score
///                            + thread_weight * thread_score
///                            + embedding_weight * embedding_score`,
/// with both axis scores clamped to `[0, 1]`. Setting both weights
/// to 0 is identity on the base score; mixing both lets the
/// substrate honour graph-derived attention and direct content
/// resonance at the same time. Same discipline as
/// `ThreadQueryAttribution`: every aggregate decomposes into named,
/// weighted contributions and shows up on the `RecallHit`.
///
/// **Back-compat:** the legacy `weight` field is accepted on the wire
/// as an alias for `thread_weight` so v0.4.x callers keep working
/// without changes. New callers should write `thread_weight` and
/// `embedding_weight`; `weight` will be removed at v1.0.0.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttentionBiasParams {
    pub scope: AttentionScope,
    /// Weight on the thread-mediated score. Default 1.0 — preserves
    /// pre-v0.5 behaviour for callers that omit this field.
    #[serde(default = "default_thread_weight", alias = "weight")]
    pub thread_weight: f32,
    /// Weight on the embedding-mediated score. Default 0.0 —
    /// opt-in for v0.5.x while clients migrate.
    #[serde(default)]
    pub embedding_weight: f32,
}

fn default_thread_weight() -> f32 {
    1.0
}

/// Parameters for the `recall` tool.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RecallParams {
    pub query: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub project: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub since: Option<DateTime<Utc>>,
    /// Half-open upper bound (`ts < before`). Combined with `since` this
    /// yields a `[since, before)` interval. P1 addition.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub before: Option<DateTime<Utc>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub limit: Option<usize>,
    /// Cap on hits sharing the same `source_id` after RRF reranking.
    /// Stops one chatty session from monopolizing top-K. `None` falls back
    /// to the default (3); `Some(0)` disables the filter entirely.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_per_source_id: Option<usize>,
    /// Intent-driven weighting for retrieval.
    #[serde(default)]
    pub intent: RecallIntent,
    /// Optional attention-bias rider. See [`AttentionBiasParams`].
    /// `None` is the historical behaviour: hits are ranked by the
    /// hybrid score alone.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub attention_bias: Option<AttentionBiasParams>,
}

/// One retrieval row, shaped for MCP consumers.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecallHit {
    pub chunk_id: String,
    pub project: Option<String>,
    pub source: String,
    pub source_id: String,
    pub ts: Option<DateTime<Utc>>,
    pub snippet: String,
    pub score: f32,
    pub links: Links,
    /// Scanner-supplied side-channel metadata (e.g. `symbols`, `kind`,
    /// `chunker`). Round-trips through the corpus `extra_json` column so
    /// MCP clients can show symbol-aware UX (file path + identifier
    /// plus its rust-analyzer kind). Defaults to `Value::Null` when the
    /// scanner didn't populate anything; serializes as JSON `null`.
    #[serde(default)]
    pub extra: serde_json::Value,
    /// Indicates if this chunk is from an orphan source that was marked stale
    /// rather than being physically deleted.
    #[serde(default)]
    pub stale: bool,
    /// The role this hit plays in a synthesized page.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub role: Option<ContextRole>,
    /// Hybrid-retrieval score before attention bias was applied.
    /// `None` when no bias was applied (then `score` is the base).
    /// When set, `score == base_score
    ///                    + thread_weight  * thread_score
    ///                    + embedding_weight * embedding_score`
    /// (within float tolerance) — argue with the math, not the vibe.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub base_score: Option<f32>,
    /// Thread-mediated bias contribution, clamped to `[0, 1]`. The
    /// max `score_thread(h)` over threads anchoring this chunk. Only
    /// present when `RecallParams.attention_bias` was supplied.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thread_score: Option<f32>,
    /// Embedding-mediated bias contribution, clamped to `[0, 1]`.
    /// Cosine of the chunk's embedding against the scope's current
    /// attention vector. Only present when bias was supplied and
    /// `embedding_weight > 0` (and the scope has been attended to
    /// and the chunk is still in the corpus).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub embedding_score: Option<f32>,
    /// Echo of `AttentionBiasParams.thread_weight`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thread_weight: Option<f32>,
    /// Echo of `AttentionBiasParams.embedding_weight`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub embedding_weight: Option<f32>,
    /// Deprecated alias for [`Self::thread_score`] (v0.4.x wire
    /// shape). Populated identically; remove at v1.0.0.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub attention_score: Option<f32>,
    /// Deprecated alias for [`Self::thread_weight`] (v0.4.x wire
    /// shape). Populated identically; remove at v1.0.0.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub attention_weight: Option<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecallLinkResult {
    pub chunk: RecallHit,
    pub parents: Vec<RecallHit>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SourceCount {
    pub source: String,
    pub count: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecallStats {
    pub total: usize,
    pub by_source: Vec<SourceCount>,
    pub model: String,
    pub dim: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub last_scan_at: Option<String>,
    /// Cross-encoder reranker info, if one is attached. Field is omitted
    /// from the JSON when `None` so old MCP clients keep parsing.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reranker: Option<RerankerStats>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RerankerStats {
    pub model: String,
    pub enabled: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditResult {
    pub columns: Vec<String>,
    pub rows: Vec<Vec<serde_json::Value>>,
}

/// A structured memory object compatible with ostk L1.5 page tables.
///
/// Data shape only — the synthesis logic that produces these (the
/// `Synthesizer` type) lives in the query crate because it needs the
/// runtime substrate (lance/lancedb). Haystack consumes `SynthesizedPage`
/// via deserialization from cached pages and the recall driver socket.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SynthesizedPage {
    /// e.g., "Symbol: `alloc_page`" or "Page: src/main.rs"
    pub title: String,
    /// The Primary hit
    pub head: RecallHit,
    /// Evolution (stale chunks)
    pub lineage: Vec<RecallHit>,
    /// Usage (transcripts/probes)
    pub evidence: Vec<RecallHit>,
    /// Count of total lineage hits found (for lazy loading)
    pub total_lineage: usize,
    /// Count of total evidence hits found (for lazy loading)
    pub total_evidence: usize,
    /// A 1-sentence synthesis bridging the roles
    pub summary: String,
}
