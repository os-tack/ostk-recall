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

use crate::{ContextRole, Links, RecallIntent};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

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
