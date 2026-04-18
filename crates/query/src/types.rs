use chrono::{DateTime, Utc};
use ostk_recall_core::Links;
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
    /// to [`hybrid::DEFAULT_MAX_PER_SOURCE_ID`] (3); `Some(0)` disables the
    /// filter entirely.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_per_source_id: Option<usize>,
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
