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
                "before": { "type": "string", "format": "date-time" },
                "limit": { "type": "integer", "default": 10, "minimum": 1, "maximum": 100 },
                "max_per_source_id": {
                    "type": "integer",
                    "default": 3,
                    "minimum": 0,
                    "description": "Cap on hits sharing the same source_id after RRF rerank. 0 disables the diversity filter."
                }
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

/// →1848 cut #3: synthesize-style recall used by haystack `mem.fault_recall`.
///
/// Embeds the query, runs hybrid recall, runs `Synthesizer::collapse`, and
/// returns the resulting pages as `(name, content)` pairs. The caller
/// (haystack kernel) writes them to its page table via `store_page_owned`;
/// this tool does NOT touch the kernel page table. `page_handles` is the
/// provider-neutral dereference surface; legacy callers may continue to
/// consume `pages` unchanged.
///
/// Result shape (inside the MCP `content[0].text` JSON):
/// ```json
/// {
///   "pages": [{
///     "name": "recall:src:kernel:memory.rs",
///     "content": "<JSON-encoded SynthesizedPage>"
///   }],
///   "page_handles": [{
///     "type": "context_page",
///     "version": 1,
///     "logical_name": "recall:src:kernel:memory.rs",
///     "resolver": {
///       "tool": "context_load",
///       "arguments": { "name": "recall:src:kernel:memory.rs" }
///     }
///   }]
/// }
/// ```
#[must_use]
pub fn tool_recall_fault() -> Value {
    json!({
        "name": "recall_fault",
        "description": "Synthesize recall hits into virtual memory pages. Returns legacy pages as (name, content) pairs plus typed page_handles whose resolver explicitly calls context_load with the logical page name.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": { "type": "string" },
                "intent": {
                    "type": "string",
                    "enum": ["symbol", "narrative", "trace", "general"],
                    "default": "general"
                },
                "limit": { "type": "integer", "minimum": 1, "maximum": 100 },
                "max_per_source_id": {
                    "type": "integer",
                    "minimum": 0,
                    "description": "Cap on hits per source-id after RRF rerank. 0 disables the diversity filter."
                }
            },
            "required": ["query"]
        }
    })
}

/// Return the tool list; `audit_available` drops `recall_audit` when false.
#[must_use]
pub fn tool_list(audit_available: bool) -> Vec<Value> {
    let mut out = vec![
        tool_recall(),
        tool_recall_link(),
        tool_recall_stats(),
        tool_recall_fault(),
    ];
    if audit_available {
        out.push(tool_recall_audit());
    }
    out
}
