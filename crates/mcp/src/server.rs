//! MCP server: JSON-RPC over stdio.

use std::sync::Arc;

use ostk_recall_query::{QueryEngine, QueryError, RecallParams};
use serde_json::{Value, json};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tracing::{error, info, warn};

use crate::protocol::{JsonRpcError, JsonRpcRequest, JsonRpcResponse};
use crate::tools::tool_list;

pub const PROTOCOL_VERSION: &str = "2025-06-18";

/// MCP stdio server. Holds an `Arc<QueryEngine>` so it can be cloned cheaply
/// across tasks if the caller ever wants to run multiple transports.
pub struct Server {
    engine: Arc<QueryEngine>,
}

impl Server {
    #[must_use]
    pub fn new(engine: QueryEngine) -> Self {
        Self {
            engine: Arc::new(engine),
        }
    }

    #[must_use]
    pub const fn from_arc(engine: Arc<QueryEngine>) -> Self {
        Self { engine }
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
        json!({ "tools": tool_list(self.engine.has_audit()) })
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
            other => {
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
