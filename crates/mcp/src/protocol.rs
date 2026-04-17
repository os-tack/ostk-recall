//! JSON-RPC 2.0 types used by the MCP stdio server.

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// A JSON-RPC request. The `id` field is `Value` because JSON-RPC allows
/// strings, numbers, or null; if absent the request is a notification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonRpcRequest {
    pub jsonrpc: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<Value>,
    pub method: String,
    #[serde(default, skip_serializing_if = "Value::is_null")]
    pub params: Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonRpcResponse {
    pub jsonrpc: String,
    pub id: Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<JsonRpcError>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonRpcError {
    pub code: i64,
    pub message: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub data: Option<Value>,
}

impl JsonRpcResponse {
    #[must_use]
    pub fn ok(id: Value, result: Value) -> Self {
        Self {
            jsonrpc: "2.0".to_string(),
            id,
            result: Some(result),
            error: None,
        }
    }

    #[must_use]
    pub fn err(id: Value, err: JsonRpcError) -> Self {
        Self {
            jsonrpc: "2.0".to_string(),
            id,
            result: None,
            error: Some(err),
        }
    }
}

pub mod codes {
    pub const PARSE_ERROR: i64 = -32700;
    pub const INVALID_REQUEST: i64 = -32600;
    pub const METHOD_NOT_FOUND: i64 = -32601;
    pub const INVALID_PARAMS: i64 = -32602;
    pub const INTERNAL_ERROR: i64 = -32603;
}

impl JsonRpcError {
    #[must_use]
    pub const fn new(code: i64, message: String) -> Self {
        Self {
            code,
            message,
            data: None,
        }
    }

    #[must_use]
    pub fn parse(msg: impl Into<String>) -> Self {
        Self::new(codes::PARSE_ERROR, msg.into())
    }

    #[must_use]
    pub fn invalid_request(msg: impl Into<String>) -> Self {
        Self::new(codes::INVALID_REQUEST, msg.into())
    }

    #[must_use]
    pub fn method_not_found(method: &str) -> Self {
        Self::new(
            codes::METHOD_NOT_FOUND,
            format!("method not found: {method}"),
        )
    }

    #[must_use]
    pub fn invalid_params(msg: impl Into<String>) -> Self {
        Self::new(codes::INVALID_PARAMS, msg.into())
    }

    #[must_use]
    pub fn internal(msg: impl Into<String>) -> Self {
        Self::new(codes::INTERNAL_ERROR, msg.into())
    }
}
