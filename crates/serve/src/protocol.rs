//! JSON-RPC 2.0 message types for the ostk-recall-serve driver.
//!
//! Wire shape per `docs/spec/driver-protocol.md`. Three methods:
//! `initialize`, `recall.fault`, `ping`. Newline-framed.

use ostk_recall_core::{RecallIntent, SynthesizedPage};
use serde::{Deserialize, Serialize};
use serde_json::Value;

// ── Envelope ──────────────────────────────────────────────────────────────

/// Inbound JSON-RPC request. `id` is `null` for notifications (which we
/// don't support on the request side — the daemon never gets pushed
/// notifications from the kernel, only requests).
#[derive(Debug, Deserialize)]
pub struct Request {
    pub jsonrpc: String,
    #[serde(default)]
    pub id: Value,
    pub method: String,
    #[serde(default)]
    pub params: Value,
}

/// Outbound response. Either `result` or `error` is set, never both.
#[derive(Debug, Serialize)]
pub struct Response {
    pub jsonrpc: &'static str,
    pub id: Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<RpcError>,
}

impl Response {
    pub fn ok(id: Value, result: Value) -> Self {
        Self {
            jsonrpc: "2.0",
            id,
            result: Some(result),
            error: None,
        }
    }

    pub fn err(id: Value, error: RpcError) -> Self {
        Self {
            jsonrpc: "2.0",
            id,
            result: None,
            error: Some(error),
        }
    }
}

#[derive(Debug, Serialize)]
pub struct RpcError {
    pub code: i32,
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<Value>,
}

impl RpcError {
    pub fn new(code: ErrorCode, message: impl Into<String>) -> Self {
        Self {
            code: code as i32,
            message: message.into(),
            data: None,
        }
    }

    pub fn with_data(mut self, data: Value) -> Self {
        self.data = Some(data);
        self
    }
}

/// Error codes per `docs/spec/driver-protocol.md`.
#[derive(Debug, Clone, Copy)]
#[repr(i32)]
pub enum ErrorCode {
    ParseError = -32700,
    InvalidRequest = -32600,
    MethodNotFound = -32601,
    InvalidParams = -32602,
    InternalError = -32603,
    CorpusNotFound = -32000,
    EmbedderLoadFailed = -32001,
    RecallFailed = -32002,
    NotInitialized = -32003,
}

// ── Method-specific payloads ──────────────────────────────────────────────

#[derive(Debug, Deserialize)]
pub struct InitializeParams {
    pub ostk_dir: String,
    #[serde(default = "default_embed_dim")]
    pub embed_dim: usize,
}

fn default_embed_dim() -> usize {
    256
}

#[derive(Debug, Serialize)]
pub struct InitializeResult {
    pub name: &'static str,
    pub version: &'static str,
    pub embedder: EmbedderInfo,
    pub corpus_root: String,
}

#[derive(Debug, Serialize)]
pub struct EmbedderInfo {
    pub model: String,
    pub dim: usize,
}

#[derive(Debug, Deserialize)]
pub struct FaultParams {
    pub query: String,
    #[serde(default)]
    pub intent: RecallIntent,
    #[serde(default)]
    pub limit: Option<usize>,
    #[serde(default)]
    pub max_per_source_id: Option<usize>,
}

#[derive(Debug, Serialize)]
pub struct FaultResult {
    pub pages: Vec<NamedPage>,
}

/// `(name, content)` pair returned by `recall.fault`. The daemon
/// synthesizes pages but does NOT write them to disk; the kernel calls
/// `store_page_owned()` for each.
#[derive(Debug, Serialize)]
pub struct NamedPage {
    pub name: String,
    /// Pre-serialized JSON of the SynthesizedPage. The kernel writes
    /// this string verbatim to the page-table.
    pub content: String,
}

impl NamedPage {
    pub fn from_page(page: &SynthesizedPage) -> Result<Self, serde_json::Error> {
        let slug = page
            .head
            .source_id
            .replace('/', ":")
            .replace('\\', ":")
            .replace(' ', "_");
        let name = format!("recall:{slug}");
        let content = serde_json::to_string(page)?;
        Ok(Self { name, content })
    }
}

#[derive(Debug, Serialize)]
pub struct PingResult {
    pub ok: bool,
}
