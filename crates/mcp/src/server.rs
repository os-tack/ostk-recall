//! MCP server: JSON-RPC over stdio.

use std::sync::Arc;

use ostk_recall_attention_mcp::{
    AttentionDispatch, AttentionHandlersError, DefaultAttentionHandlers, attention_tools,
    thread_tools,
};
use ostk_recall_query::{QueryEngine, QueryError, RecallParams, SynthesizedPage, Synthesizer};
use serde_json::{Value, json};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tracing::{error, info, warn};

use crate::protocol::{JsonRpcError, JsonRpcRequest, JsonRpcResponse};
use crate::tools::tool_list;

pub const PROTOCOL_VERSION: &str = "2025-06-18";

/// MCP stdio server. Holds an `Arc<QueryEngine>` so it can be cloned cheaply
/// across tasks if the caller ever wants to run multiple transports.
///
/// The optional `attention` dispatch unlocks the
/// `attention_*` / `thread_*` tool families when the caller (typically
/// `cli::commands::serve`) constructs a long-lived `AttentionDispatch`
/// and threads it through.
pub struct Server {
    engine: Arc<QueryEngine>,
    attention: Option<Arc<AttentionDispatch>>,
}

impl Server {
    #[must_use]
    pub fn new(engine: QueryEngine) -> Self {
        Self {
            engine: Arc::new(engine),
            attention: None,
        }
    }

    #[must_use]
    pub const fn from_arc(engine: Arc<QueryEngine>) -> Self {
        Self {
            engine,
            attention: None,
        }
    }

    /// Attach an `AttentionDispatch` so the attention/thread MCP tools
    /// become callable. Without one, those tools are not advertised in
    /// `tools/list` and `tools/call` returns method-not-found.
    #[must_use]
    pub fn with_attention(mut self, dispatch: Arc<AttentionDispatch>) -> Self {
        self.attention = Some(dispatch);
        self
    }

    pub const fn engine(&self) -> &Arc<QueryEngine> {
        &self.engine
    }

    /// Read newline-delimited JSON requests from stdin, dispatch, write
    /// responses to stdout. Returns when EOF is reached on stdin.
    pub async fn run_stdio(&self) -> std::io::Result<()> {
        let stdin = tokio::io::stdin();
        let mut stdout = tokio::io::stdout();
        let mut reader = BufReader::new(stdin);
        let mut line = String::new();

        info!("mcp server ready on stdio");
        loop {
            line.clear();
            let n = reader.read_line(&mut line).await?;
            if n == 0 {
                break;
            }
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }
            let value: Value = match serde_json::from_str(trimmed) {
                Ok(v) => v,
                Err(e) => {
                    let resp = JsonRpcResponse::err(
                        Value::Null,
                        JsonRpcError::parse(format!("parse error: {e}")),
                    );
                    write_response(&mut stdout, &resp).await?;
                    continue;
                }
            };
            let reply = self.handle_request(value).await;
            if let Some(r) = reply {
                write_response(&mut stdout, &r).await?;
            }
        }
        info!("mcp server shutting down");
        Ok(())
    }

    /// Dispatch one JSON-RPC message. Returns None for notifications (no id).
    pub async fn handle_request(&self, raw: Value) -> Option<JsonRpcResponse> {
        let req: JsonRpcRequest = match serde_json::from_value(raw.clone()) {
            Ok(r) => r,
            Err(e) => {
                let id = raw.get("id").cloned().unwrap_or(Value::Null);
                return Some(JsonRpcResponse::err(
                    id,
                    JsonRpcError::invalid_request(format!("malformed request: {e}")),
                ));
            }
        };
        let id = req.id.clone().unwrap_or(Value::Null);
        let is_notification = req.id.is_none();

        match req.method.as_str() {
            "initialize" => Some(JsonRpcResponse::ok(id, Self::handle_initialize())),
            "initialized" | "notifications/initialized" | "notifications/cancelled" => None,
            "ping" => Some(JsonRpcResponse::ok(id, json!({}))),
            "tools/list" => Some(JsonRpcResponse::ok(id, self.handle_tools_list())),
            "tools/call" => match self.handle_tools_call(req.params).await {
                Ok(v) => Some(JsonRpcResponse::ok(id, v)),
                Err(err) => Some(JsonRpcResponse::err(id, err)),
            },
            _ => {
                if is_notification {
                    warn!(method = %req.method, "unknown notification — ignoring");
                    None
                } else {
                    Some(JsonRpcResponse::err(
                        id,
                        JsonRpcError::method_not_found(&req.method),
                    ))
                }
            }
        }
    }

    fn handle_initialize() -> Value {
        json!({
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": { "tools": { "listChanged": false } },
            "serverInfo": {
                "name": "ostk-recall",
                "version": env!("CARGO_PKG_VERSION"),
            }
        })
    }

    fn handle_tools_list(&self) -> Value {
        let mut tools = tool_list(self.engine.has_audit());
        if self.attention.is_some() {
            tools.extend(attention_tools());
            tools.extend(thread_tools());
        }
        json!({ "tools": tools })
    }

    async fn handle_tools_call(&self, params: Value) -> std::result::Result<Value, JsonRpcError> {
        let name = params
            .get("name")
            .and_then(Value::as_str)
            .ok_or_else(|| JsonRpcError::invalid_params("missing tool name"))?
            .to_string();
        let args = params.get("arguments").cloned().unwrap_or(Value::Null);

        let result_json = match name.as_str() {
            "recall" => {
                let p: RecallParams = serde_json::from_value(args.clone())
                    .map_err(|e| JsonRpcError::invalid_params(format!("recall args: {e}")))?;
                let hits = self.engine.recall(p).await.map_err(query_error_to_rpc)?;
                json!({ "hits": hits })
            }
            "recall_link" => {
                let chunk_id = args
                    .get("chunk_id")
                    .and_then(Value::as_str)
                    .ok_or_else(|| JsonRpcError::invalid_params("missing chunk_id"))?;
                let out = self
                    .engine
                    .recall_link(chunk_id)
                    .await
                    .map_err(query_error_to_rpc)?;
                serde_json::to_value(out)
                    .map_err(|e| JsonRpcError::internal(format!("serialize: {e}")))?
            }
            "recall_stats" => {
                let out = self
                    .engine
                    .recall_stats()
                    .await
                    .map_err(query_error_to_rpc)?;
                serde_json::to_value(out)
                    .map_err(|e| JsonRpcError::internal(format!("serialize: {e}")))?
            }
            "recall_audit" => {
                let sql = args
                    .get("sql")
                    .and_then(Value::as_str)
                    .ok_or_else(|| JsonRpcError::invalid_params("missing sql"))?;
                let out = self.engine.recall_audit(sql).map_err(query_error_to_rpc)?;
                serde_json::to_value(out)
                    .map_err(|e| JsonRpcError::internal(format!("serialize: {e}")))?
            }
            "recall_fault" => {
                // →1848 cut #3: synthesize-style recall for haystack
                // `mem.fault_recall`. Embed + recall + Synthesizer::collapse;
                // return (name, content) pairs for the caller to write into
                // its page table. The daemon does NOT touch any page table.
                let p: RecallParams = serde_json::from_value(args.clone())
                    .map_err(|e| JsonRpcError::invalid_params(format!("recall_fault args: {e}")))?;
                if p.query.is_empty() {
                    return Err(JsonRpcError::invalid_params("query must be non-empty"));
                }
                let hits = self.engine.recall(p).await.map_err(query_error_to_rpc)?;
                let pages = Synthesizer::collapse(hits);
                let named: Vec<Value> = pages
                    .iter()
                    .map(named_page_value)
                    .collect::<std::result::Result<Vec<_>, _>>()
                    .map_err(|e| JsonRpcError::internal(format!("serialize page: {e}")))?;
                json!({ "pages": named })
            }
            other => {
                if let Some(d) = self.attention.as_ref() {
                    if is_attention_tool(other) {
                        let out = d
                            .dispatch(other, args)
                            .await
                            .map_err(attention_error_to_rpc)?;
                        let text = serde_json::to_string(&out).map_err(|e| {
                            JsonRpcError::internal(format!("serialize: {e}"))
                        })?;
                        return Ok(json!({
                            "content": [{ "type": "text", "text": text }],
                            "isError": false,
                        }));
                    }
                }
                return Err(JsonRpcError::method_not_found(&format!(
                    "tools/call/{other}"
                )));
            }
        };

        // Wrap in MCP content block.
        let text = serde_json::to_string(&result_json)
            .map_err(|e| JsonRpcError::internal(format!("serialize: {e}")))?;
        Ok(json!({
            "content": [
                { "type": "text", "text": text }
            ],
            "isError": false
        }))
    }
}

fn is_attention_tool(name: &str) -> bool {
    name.starts_with("attention_") || name.starts_with("thread_")
}

fn attention_error_to_rpc(err: AttentionHandlersError) -> JsonRpcError {
    match err {
        AttentionHandlersError::InvalidParams(m) => JsonRpcError::invalid_params(m),
        AttentionHandlersError::InvalidHandle(e) => {
            JsonRpcError::invalid_params(format!("invalid handle: {e}"))
        }
        AttentionHandlersError::PrivacyForbidden(tier) => {
            JsonRpcError::invalid_params(format!("privacy tier {tier:?} not permitted"))
        }
        other => JsonRpcError::internal(other.to_string()),
    }
}

async fn write_response<W: AsyncWriteExt + Unpin>(
    w: &mut W,
    resp: &JsonRpcResponse,
) -> std::io::Result<()> {
    let mut line = serde_json::to_vec(resp).map_err(|e| {
        error!(error = %e, "serialize response");
        std::io::Error::other(e)
    })?;
    line.push(b'\n');
    w.write_all(&line).await?;
    w.flush().await?;
    Ok(())
}

fn query_error_to_rpc(err: QueryError) -> JsonRpcError {
    match err {
        QueryError::Forbidden(msg) => JsonRpcError::invalid_params(msg),
        QueryError::NoEventsStore => JsonRpcError::invalid_request("events store not configured"),
        QueryError::NotFound(id) => JsonRpcError::invalid_params(format!("not found: {id}")),
        QueryError::Decode(m) => JsonRpcError::internal(format!("decode: {m}")),
        other => JsonRpcError::internal(other.to_string()),
    }
}

/// Build the `{name, content}` pair for one synthesized page. The name
/// is the kernel page-table key; content is the JSON-encoded
/// `SynthesizedPage` the kernel writes via `store_page_owned`.
fn named_page_value(page: &SynthesizedPage) -> std::result::Result<Value, serde_json::Error> {
    let slug = page
        .head
        .source_id
        .replace(['/', '\\'], ":")
        .replace(' ', "_");
    let name = format!("recall:{slug}");
    let content = serde_json::to_string(page)?;
    Ok(json!({ "name": name, "content": content }))
}
