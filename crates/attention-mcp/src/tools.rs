//! MCP tool schemas for the attention substrate.
//!
//! Mirrors the shape used by `ostk-recall-mcp` in `tools.rs` — each
//! `tool_*` function returns a `serde_json::Value` describing the tool
//! to MCP clients via `tools/list`. The dispatch lives in
//! [`crate::handlers`]; this module is the wire description.
//!
//! Every tool that takes an `AttentionScope` carries it as an optional
//! input field; absence means "default scope" (project=None,
//! `privacy_tier=T1Project`). See `AttentionScope::default`.

use serde_json::{Value, json};

/// JSON schema fragment describing `AttentionScope`. Used inline by
/// every tool that takes a scope so the wire shape is consistent.
fn scope_schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "project": { "type": "string" },
            "session_id": { "type": "string" },
            "agent": { "type": "string" },
            "privacy_tier": {
                "type": "string",
                "enum": ["t0_private", "t1_project", "t2_trusted", "t3_public"],
                "default": "t1_project",
                "description": "Privacy tier. Defaults to t1_project when omitted."
            }
        }
    })
}

#[must_use]
pub fn tool_attention_attend() -> Value {
    json!({
        "name": "attention_attend",
        "description": "Ingest current conversational/tool context into the attention vector for the given scope.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "scope": scope_schema(),
                "context": { "type": "string" }
            },
            "required": ["context"]
        }
    })
}

#[must_use]
pub fn tool_attention_surface() -> Value {
    json!({
        "name": "attention_surface",
        "description": "Surface attention pages above the archive threshold for the given scope; honours PrivacyTier rules. Each page carries depth + ScoreAttribution.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "scope": scope_schema(),
                "limit": { "type": "integer", "minimum": 1, "maximum": 200, "default": 20 }
            }
        }
    })
}

#[must_use]
pub fn tool_attention_fold() -> Value {
    json!({
        "name": "attention_fold",
        "description": "Set fold depth (folded|half|full) for a thread handle within the scope.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "scope": scope_schema(),
                "handle": { "type": "string" },
                "depth": { "type": "string", "enum": ["folded", "half", "full"] }
            },
            "required": ["handle", "depth"]
        }
    })
}

#[must_use]
pub fn tool_attention_familiarize() -> Value {
    json!({
        "name": "attention_familiarize",
        "description": "Increment familiarity counter for a handle within the scope (called per turn-end). Returns the post-increment value.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "scope": scope_schema(),
                "handle": { "type": "string" }
            },
            "required": ["handle"]
        }
    })
}

#[must_use]
pub fn tool_attention_decay() -> Value {
    json!({
        "name": "attention_decay",
        "description": "Apply a multiplicative fade factor (0.0..=1.0 typical) to the floor for a handle across all scopes that hold it. Note: AttentionForwardStore::decay is scope-less in V1; this tool inherits that contract.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "handle": { "type": "string" },
                "factor": { "type": "number" }
            },
            "required": ["handle", "factor"]
        }
    })
}

#[must_use]
pub fn tool_thread_create() -> Value {
    json!({
        "name": "thread_create",
        "description": "Insert-or-replace a thread row in the durable ledger. Emits a ChainEvent::ThreadCreate when the handle is new.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "scope": scope_schema(),
                "handle": { "type": "string" },
                "body": { "type": "string", "description": "Markdown body (currently informational — body persistence is the threads scanner's responsibility)." },
                "tension": { "type": "string", "enum": ["active", "slack", "dormant"], "default": "active" }
            },
            "required": ["handle"]
        }
    })
}

#[must_use]
pub fn tool_thread_link() -> Value {
    json!({
        "name": "thread_link",
        "description": "Add a curated evidence link from a thread to a target path. Returns the new evidence_id.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "scope": scope_schema(),
                "handle": { "type": "string" },
                "target_path": { "type": "string" },
                "category": { "type": "string", "description": "Free-form (e.g. \"code\", \"doc\", \"transcript\")." }
            },
            "required": ["handle", "target_path", "category"]
        }
    })
}

#[must_use]
pub fn tool_thread_unlink() -> Value {
    json!({
        "name": "thread_unlink",
        "description": "Drop an evidence row by id. Emits ChainEvent::EvidenceRemove.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "evidence_id": { "type": "integer" }
            },
            "required": ["evidence_id"]
        }
    })
}

#[must_use]
pub fn tool_thread_promote() -> Value {
    json!({
        "name": "thread_promote",
        "description": "Promote a proposed thread to an active tension state. Updates the row's tension column; the scanner is responsible for moving the on-disk .proposed/ file. target_tier accepts active|slack.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "handle_from_proposed": { "type": "string" },
                "target_tier": { "type": "string", "enum": ["active", "slack"] }
            },
            "required": ["handle_from_proposed", "target_tier"]
        }
    })
}

#[must_use]
pub fn tool_thread_list() -> Value {
    json!({
        "name": "thread_list",
        "description": "List threads, optionally filtered by tension. PrivacyTier is honoured: scope at T1Project never sees T0Private rows from a different created_scope_key.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "scope": scope_schema(),
                "tension": { "type": "string", "enum": ["active", "slack", "dormant"] }
            }
        }
    })
}

/// Names of all attention-namespace tools (used by tests + introspection).
pub const ATTENTION_TOOL_NAMES: &[&str] = &[
    "attention_attend",
    "attention_surface",
    "attention_fold",
    "attention_familiarize",
    "attention_decay",
];

/// Names of all thread-namespace tools.
pub const THREAD_TOOL_NAMES: &[&str] = &[
    "thread_create",
    "thread_link",
    "thread_unlink",
    "thread_promote",
    "thread_list",
];

#[must_use]
pub fn attention_tools() -> Vec<Value> {
    vec![
        tool_attention_attend(),
        tool_attention_surface(),
        tool_attention_fold(),
        tool_attention_familiarize(),
        tool_attention_decay(),
    ]
}

#[must_use]
pub fn thread_tools() -> Vec<Value> {
    vec![
        tool_thread_create(),
        tool_thread_link(),
        tool_thread_unlink(),
        tool_thread_promote(),
        tool_thread_list(),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_attention_tools_have_required_shape() {
        for t in attention_tools() {
            assert!(t["name"].is_string());
            assert!(t["description"].is_string());
            assert_eq!(t["inputSchema"]["type"], "object");
        }
    }

    #[test]
    fn all_thread_tools_have_required_shape() {
        for t in thread_tools() {
            assert!(t["name"].is_string());
            assert!(t["description"].is_string());
            assert_eq!(t["inputSchema"]["type"], "object");
        }
    }

    #[test]
    fn tool_lists_match_published_names() {
        let attn: Vec<String> = attention_tools()
            .iter()
            .map(|t| t["name"].as_str().unwrap().to_string())
            .collect();
        assert_eq!(attn, ATTENTION_TOOL_NAMES);
        let thr: Vec<String> = thread_tools()
            .iter()
            .map(|t| t["name"].as_str().unwrap().to_string())
            .collect();
        assert_eq!(thr, THREAD_TOOL_NAMES);
    }
}
