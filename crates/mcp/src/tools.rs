//! MCP tool definitions exposed by this server.

use serde_json::{Value, json};

#[must_use]
pub fn tool_recall() -> Value {
    json!({
        "name": "recall",
        "description": "Hybrid dense + BM25 retrieval across the local corpus.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": { "type": "string" },
                "project": { "type": "string" },
                "source": { "type": "string" },
                "since": { "type": "string", "format": "date-time" },
                "limit": { "type": "integer", "default": 10, "minimum": 1, "maximum": 100 }
            },
            "required": ["query"]
        }
    })
}

#[must_use]
pub fn tool_recall_link() -> Value {
    json!({
        "name": "recall_link",
        "description": "Fetch a chunk by id plus its parent chain.",
        "inputSchema": {
            "type": "object",
            "properties": { "chunk_id": { "type": "string" } },
            "required": ["chunk_id"]
        }
    })
}

#[must_use]
pub fn tool_recall_stats() -> Value {
    json!({
        "name": "recall_stats",
        "description": "Corpus stats: total count, by source, model info, last scan.",
        "inputSchema": { "type": "object", "properties": {} }
    })
}

#[must_use]
pub fn tool_recall_audit() -> Value {
    json!({
        "name": "recall_audit",
        "description": "Raw DuckDB SQL over audit_events (ostk_project sources only). SELECT only, one statement.",
        "inputSchema": {
            "type": "object",
            "properties": { "sql": { "type": "string" } },
            "required": ["sql"]
        }
    })
}

/// Return the tool list; `audit_available` drops `recall_audit` when false.
#[must_use]
pub fn tool_list(audit_available: bool) -> Vec<Value> {
    let mut out = vec![tool_recall(), tool_recall_link(), tool_recall_stats()];
    if audit_available {
        out.push(tool_recall_audit());
    }
    out
}
