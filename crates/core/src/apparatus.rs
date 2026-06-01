//! Structural apparatus — the one definition of "this chunk is harness
//! scaffolding, not cognition."
//!
//! High-volume templated envelopes (Claude Code tool-call blocks,
//! `<task-notification>` monitor events) are byte-repetitive, so they cluster
//! into degenerate high-cohesion units and must never *define* a thread or a
//! concept. This is the **structural** counterpart to the facet denylist
//! (`record_kind:harness_orchestration`): `block_kind` lives in `Chunk::extra`
//! and task-notifications are `block_kind=user`, so neither is expressible as a
//! `[weaver] exclude_facets` entry.
//!
//! The policy is **cognition-only, not recall-only**: callers exclude these
//! chunks from the weaver's anchor/proposal pass and from concept clustering,
//! but the chunks stay in the corpus and remain recall-able. (To drop them from
//! recall entirely, see `CorpusStore::mark_tool_blocks_stale`.)
//!
//! Both the Rust predicate ([`is_structural_apparatus`]) and any SQL-side
//! exclusion (e.g. the concept-clustering fetch) derive from the SAME
//! [`APPARATUS_BLOCK_KINDS`] / [`APPARATUS_TEXT_PREFIXES`] consts, so the two
//! encodings can never drift.

/// `block_kind` values (in `Chunk::extra`) that mark a chunk as apparatus.
pub const APPARATUS_BLOCK_KINDS: &[&str] = &["tool_use", "tool_result"];

/// Text prefixes (after `trim_start`) that mark a chunk as apparatus regardless
/// of `block_kind` (task-notifications arrive as `block_kind=user`).
pub const APPARATUS_TEXT_PREFIXES: &[&str] = &["<task-notification>"];

/// True if `(text, extra)` is structural apparatus per the shared consts.
///
/// `extra` is the chunk's `extra` JSON (`Chunk::extra`); `text` is its body.
#[must_use]
pub fn is_structural_apparatus(text: &str, extra: &serde_json::Value) -> bool {
    let block_kind = extra.get("block_kind").and_then(serde_json::Value::as_str);
    if block_kind.is_some_and(|k| APPARATUS_BLOCK_KINDS.contains(&k)) {
        return true;
    }
    let trimmed = text.trim_start();
    APPARATUS_TEXT_PREFIXES
        .iter()
        .any(|p| trimmed.starts_with(p))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn structural_apparatus_is_excluded_but_real_content_is_not() {
        // Tool-call envelopes and task-notifications are structural apparatus,
        // gated regardless of facets so they never seed degenerate units.
        assert!(is_structural_apparatus(
            "[tool_use Read: {\"file_path\":\"src/main.rs\"}]",
            &json!({ "block_kind": "tool_use" })
        ));
        assert!(is_structural_apparatus(
            "[tool_result: 1\t2026-05-12 claim id=os-builds-os]",
            &json!({ "block_kind": "tool_result" })
        ));
        // task-notifications are block_kind=user, caught by the text prefix
        // (with leading whitespace tolerated).
        assert!(is_structural_apparatus(
            "\n<task-notification>\n<task-id>bwl7bzhhn</task-id>",
            &json!({ "block_kind": "user" })
        ));
        // Real cognition must pass: assistant prose, genuine user turns, and
        // chunks with no `extra` are all woven.
        assert!(!is_structural_apparatus(
            "The kernel survives.",
            &json!({ "block_kind": "assistant_text" })
        ));
        assert!(!is_structural_apparatus(
            "commit that please, the os builds the os",
            &json!({ "block_kind": "user" })
        ));
        assert!(!is_structural_apparatus(
            "plain text, no extra",
            &json!(null)
        ));
    }
}
