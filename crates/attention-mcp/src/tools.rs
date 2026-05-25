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

#[must_use]
pub fn tool_thread_emergent() -> Value {
    json!({
        "name": "thread_emergent",
        "description": "Discover thread candidates from the existing corpus by clustering recent chunks. Returns clusters sorted by cohesion. Writes idempotent `proposed-<hash>` rows to the threads_proposed table unless `persist: false`.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "since_hours": { "type": "integer", "minimum": 1,
                                  "description": "Look-back window in hours. Default 12." },
                "limit": { "type": "integer", "minimum": 1, "maximum": 5000,
                            "description": "Max chunks fed to the clusterer. Default 500." },
                "min_cluster_size": { "type": "integer", "minimum": 2,
                                       "description": "Minimum members per surfaced cluster. Default 5." },
                "cohesion_threshold": { "type": "number", "minimum": 0.0, "maximum": 1.0,
                                         "description": "Pairwise cosine cut-off for cluster membership. Default matches cluster::EMERGENT_THRESHOLD (0.82). Lower = more permissive." },
                "min_neighbours": { "type": "integer", "minimum": 1,
                                     "description": "Minimum in-cluster neighbours each member must have. Default matches cluster::MIN_NEIGHBOURS_IN_CLUSTER (2). Density floor." },
                "persist": { "type": "boolean",
                              "description": "Write proposed rows to threads_proposed. Default true." }
            }
        }
    })
}

pub fn tool_thread_attention() -> Value {
    json!({
        "name": "thread_attention",
        "description": "Activity-burst attention surface. Groups recent non-stale chunks by (project, source_id) and ranks each group by `count * exp(-(now - max_ts) / decay_hours)`. Returns the per-source focus areas of the recency window with sample snippets. Robust to the 'thoughts are unique' problem because it doesn't touch embeddings.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "since_hours": { "type": "integer", "minimum": 1,
                                  "description": "Look-back window in hours. Default 24." },
                "limit": { "type": "integer", "minimum": 1, "maximum": 100,
                            "description": "Max bursts returned. Default 10." },
                "samples_per_burst": { "type": "integer", "minimum": 0, "maximum": 20,
                                        "description": "Sample snippets per burst. Default 3." },
                "decay_hours": { "type": "number", "minimum": 0.1,
                                  "description": "Half-life-style recency decay constant in hours. Default 6.0." }
            }
        }
    })
}

pub fn tool_thread_novelty() -> Value {
    json!({
        "name": "thread_novelty",
        "description": "Divergence-from-baseline novelty surface. Scores recent chunks by `1 - cos(embedding, project_baseline)` (high = unlike baseline = novel), then re-clusters the top-K through the same density guard the emergent surface uses. Returns clusters of coherent novel chunks, sorted by mean novelty. Empty result is the expected null state when nothing novel enough surfaces — pair with thread_attention for the activity-burst complement.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "since_hours": { "type": "integer", "minimum": 1,
                                  "description": "Look-back window in hours. Default 24." },
                "baseline_days": { "type": "integer", "minimum": 1,
                                    "description": "Per-project baseline window in days. Default 7." },
                "limit": { "type": "integer", "minimum": 1, "maximum": 100,
                            "description": "Max clusters returned. Default 10." },
                "min_cluster_size": { "type": "integer", "minimum": 2,
                                       "description": "Minimum members per surfaced cluster. Default matches cluster::MIN_CLUSTER_SIZE (3). (Aliased: also accepts `min_cluster` for v0.3.0 back-compat.)" },
                "min_cluster": { "type": "integer", "minimum": 2,
                                  "description": "Deprecated alias for `min_cluster_size`. Kept for v0.3.0 back-compat; new callers should use `min_cluster_size`." },
                "recluster_threshold": { "type": "number", "minimum": 0.0, "maximum": 1.0,
                                          "description": "Re-cluster cosine threshold. Default matches cluster::EMERGENT_THRESHOLD (0.82)." },
                "min_mean_novelty": { "type": "number", "minimum": 0.0, "maximum": 2.0,
                                       "description": "Drop clusters whose mean novelty is below this floor. Default 0.0 (filter off, permissive). The historical baked default was 0.3; pass that explicitly if you want pre-v0.3.1 behavior." }
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
    "thread_emergent",
    "thread_attention",
    "thread_novelty",
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
        tool_thread_emergent(),
        tool_thread_attention(),
        tool_thread_novelty(),
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
