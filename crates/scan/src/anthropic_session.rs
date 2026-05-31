//! Shared helper for parsing Anthropic-style session logs into per-block
//! chunks.
//!
//! Two callers share this logic:
//!
//! 1. [`crate::claude_code`] — `~/.claude/projects/*/*.jsonl` session files.
//!    Each line wraps an Anthropic message: `{ type, message: { role,
//!    content }, timestamp }`. `type in {user, assistant}` are kept;
//!    `file-history-snapshot`, `summary`, etc. are skipped.
//! 2. [`crate::ostk_project`] `.ostk/sessions/*.jsonl` — raw Anthropic
//!    Messages shape (`{ role, content, timestamp }` at the top level).
//!
//! The `content` field can be a plain string or an array of blocks
//! (`{type: "text"|"tool_use"|"tool_result"}`).
//!
//! # Chunking (Phase H)
//!
//! One chunk per **message block**:
//!
//! * user `text` (or whole-string content) → one chunk with `role = "user"`
//! * each assistant `text` block → one chunk with `role = "assistant"`
//! * each `tool_use` block → one chunk with `role = "tool"`
//! * each `tool_result` block → one chunk with `role = "tool_result"`
//!
//! Tool blocks (`tool_use`, `tool_result`) are still capped at
//! [`TOOL_RESULT_PREVIEW_CHARS`] characters to keep raw file dumps from
//! dominating BM25.
//!
//! Each chunk's `links.parent_ids` points to the previous chunk in the
//! session, so `recall_link` can walk backward and reconstruct the
//! surrounding context. `chunk_index` is monotonic over the entire session.
//!
//! Per-block chunking gives the cross-encoder reranker clean, single-
//! purpose units to score, instead of a hodgepodge "exchange" containing
//! prose, tool calls, and file listings smeared together.

use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

use chrono::{DateTime, Utc};
use ostk_recall_core::{Chunk, Links, Result, Source};
use serde::Deserialize;

/// Cap on rendered tool-block characters retained in a chunk.
///
/// Tool-result and tool-use blocks are auxiliary signal — full file dumps
/// dominate BM25 hits without adding query-relevant prose. Truncating to
/// 200 chars lets an error-snippet still leak through while a 5000-char
/// file listing doesn't.
pub const TOOL_RESULT_PREVIEW_CHARS: usize = 200;

/// Shape recognised by the parser. One line yields one `Record`.
#[derive(Debug, Deserialize)]
struct LogLine {
    /// Claude Code wrapper: `"user"` / `"assistant"` / `"file-history-snapshot"` / `"summary"` / ...
    #[serde(default)]
    r#type: Option<String>,
    /// Claude Code wraps an inner Anthropic message here.
    #[serde(default)]
    message: Option<InnerMessage>,
    /// Raw Anthropic shape uses these at the top level.
    #[serde(default)]
    role: Option<String>,
    #[serde(default)]
    content: Option<serde_json::Value>,
    #[serde(default)]
    timestamp: Option<DateTime<Utc>>,
}

#[derive(Debug, Deserialize)]
struct InnerMessage {
    #[serde(default)]
    role: Option<String>,
    #[serde(default)]
    content: Option<serde_json::Value>,
}

/// Kind of block surfaced as one chunk. Drives the `role` field on the
/// chunk and the `extra.block_kind` annotation downstream filters key on.
#[derive(Debug, Clone, Copy)]
enum BlockKind {
    UserText,
    AssistantText,
    ToolUse,
    ToolResult,
}

impl BlockKind {
    const fn as_role(self) -> &'static str {
        match self {
            Self::UserText => "user",
            Self::AssistantText => "assistant",
            Self::ToolUse => "tool",
            Self::ToolResult => "tool_result",
        }
    }

    const fn as_block_kind(self) -> &'static str {
        match self {
            Self::UserText => "user",
            Self::AssistantText => "assistant_text",
            Self::ToolUse => "tool_use",
            Self::ToolResult => "tool_result",
        }
    }
}

/// One pre-chunk record drawn from a single message block. Held in a Vec
/// while we walk the session, then materialized into [`Chunk`]s with the
/// session's parent-chain wiring and chunk-index numbering.
#[derive(Debug, Clone)]
struct Block {
    kind: BlockKind,
    text: String,
    ts: Option<DateTime<Utc>>,
}

/// Parse a session file into chunks.
///
/// * `source` — concrete `Source` to stamp each chunk with.
/// * `source_id_base` — base identifier (e.g. session filename); chunk
///   index is appended internally.
/// * `project` — carried onto chunks.
/// * `path` — absolute path stored in `Links.file_path`.
/// * `fallback_ts` — used when the first kept line has no `timestamp`.
pub fn parse_session_file(
    path: &Path,
    source: Source,
    source_id_base: &str,
    project: Option<&str>,
    source_config_id: &str,
    fallback_ts: Option<DateTime<Utc>>,
) -> Result<Vec<Chunk>> {
    let f = File::open(path)?;
    let reader = BufReader::new(f);

    let abs_path = path
        .canonicalize()
        .unwrap_or_else(|_| path.to_path_buf())
        .to_string_lossy()
        .into_owned();

    let mut blocks: Vec<Block> = Vec::new();
    for line in reader.lines() {
        let line = match line {
            Ok(l) => l,
            Err(e) => {
                tracing::warn!(path = %path.display(), error = %e, "session: read line failed");
                continue;
            }
        };
        if line.trim().is_empty() {
            continue;
        }
        let parsed: LogLine = match serde_json::from_str(&line) {
            Ok(p) => p,
            Err(e) => {
                tracing::warn!(path = %path.display(), error = %e, "session: bad jsonl line");
                continue;
            }
        };
        extract_blocks(&parsed, &mut blocks);
    }

    Ok(build_chunks(
        &blocks,
        source,
        source_id_base,
        project,
        source_config_id,
        &abs_path,
        fallback_ts,
    ))
}

/// Walk one log line into zero or more [`Block`]s, appending to `out`.
fn extract_blocks(line: &LogLine, out: &mut Vec<Block>) {
    // Type-level gate: Claude Code files have `type`. Skip anything not
    // user/assistant. Raw Anthropic files have no `type` field.
    if let Some(ty) = line.r#type.as_deref() {
        if ty != "user" && ty != "assistant" {
            return;
        }
    }
    let (role, content) = line.message.as_ref().map_or_else(
        || (line.role.clone(), line.content.clone()),
        |inner| (inner.role.clone(), inner.content.clone()),
    );
    let role = role.unwrap_or_else(|| line.r#type.clone().unwrap_or_default());
    if role != "user" && role != "assistant" {
        return;
    }
    let ts = line.timestamp;
    let Some(content) = content else { return };
    blocks_from_content(role.as_str(), content, ts, out);
}

/// Translate one message's `content` field into [`Block`]s. A bare string
/// becomes one block (user text or assistant text depending on role); an
/// array of blocks is split into one [`Block`] per content block of a
/// supported type. Empty/whitespace-only blocks are silently dropped.
fn blocks_from_content(
    role: &str,
    content: serde_json::Value,
    ts: Option<DateTime<Utc>>,
    out: &mut Vec<Block>,
) {
    match content {
        serde_json::Value::String(s) => {
            if s.trim().is_empty() {
                return;
            }
            let kind = if role == "assistant" {
                BlockKind::AssistantText
            } else {
                BlockKind::UserText
            };
            out.push(Block { kind, text: s, ts });
        }
        serde_json::Value::Array(items) => {
            for block in items {
                let Some(obj) = block.as_object() else {
                    continue;
                };
                let ty = obj.get("type").and_then(|v| v.as_str()).unwrap_or("");
                match ty {
                    "text" => {
                        let Some(text) = obj.get("text").and_then(|v| v.as_str()) else {
                            continue;
                        };
                        if text.trim().is_empty() {
                            continue;
                        }
                        let kind = if role == "assistant" {
                            BlockKind::AssistantText
                        } else {
                            BlockKind::UserText
                        };
                        out.push(Block {
                            kind,
                            text: text.to_string(),
                            ts,
                        });
                    }
                    "tool_use" => {
                        let name = obj.get("name").and_then(|v| v.as_str()).unwrap_or("tool");
                        let input = obj
                            .get("input")
                            .map(|v| serde_json::to_string(v).unwrap_or_default())
                            .unwrap_or_default();
                        let rendered = format!("[tool_use {name}: {input}]");
                        let capped = truncate_tool_block(&rendered);
                        if capped.trim().is_empty() {
                            continue;
                        }
                        out.push(Block {
                            kind: BlockKind::ToolUse,
                            text: capped,
                            ts,
                        });
                    }
                    "tool_result" => {
                        let tool_content = obj.get("content").map_or(String::new(), |v| match v {
                            serde_json::Value::String(s) => s.clone(),
                            other => serde_json::to_string(other).unwrap_or_default(),
                        });
                        let rendered = format!("[tool_result: {tool_content}]");
                        let capped = truncate_tool_block(&rendered);
                        if capped.trim().is_empty() {
                            continue;
                        }
                        out.push(Block {
                            kind: BlockKind::ToolResult,
                            text: capped,
                            ts,
                        });
                    }
                    _ => {
                        // unknown block; skip
                    }
                }
            }
        }
        _ => {}
    }
}

/// Truncate a rendered `tool_use`/`tool_result` block to
/// [`TOOL_RESULT_PREVIEW_CHARS`] *characters* (not bytes), appending a
/// truncation marker. Pass-through if already short enough. Char-counting
/// keeps multi-byte codepoints intact.
fn truncate_tool_block(text: &str) -> String {
    let char_count = text.chars().count();
    if char_count <= TOOL_RESULT_PREVIEW_CHARS {
        return text.to_string();
    }
    let head: String = text.chars().take(TOOL_RESULT_PREVIEW_CHARS).collect();
    let dropped = char_count - TOOL_RESULT_PREVIEW_CHARS;
    format!("{head}…[truncated {dropped} chars]")
}

/// Drop chunks whose text is Claude Code's slash-command surface scaffolding
/// — `<local-command-caveat>...`, `<command-name>...</command-name>`,
/// `<local-command-stdout>...`, etc. These chunks have `block_kind=user`
/// (Claude Code emits them as user messages wrapping the slash invocation)
/// so [`drop_tool_blocks`] does not catch them, but they are procedurally
/// identical: high-volume meta scaffolding with no thinking content. Left
/// in, they dominate emergent surfacing once tool blocks are removed.
#[must_use]
pub fn drop_local_command_wrappers(chunks: Vec<Chunk>) -> Vec<Chunk> {
    chunks
        .into_iter()
        .filter(|c| {
            let head = c.text.trim_start();
            !(head.starts_with("<local-command-")
                || head.starts_with("<command-name>")
                || head.starts_with("</command-name>")
                || head.starts_with("<command-message>")
                || head.starts_with("<command-args>"))
        })
        .collect()
}

/// Drop chunks whose `extra.block_kind` is `tool_use` or `tool_result`.
///
/// The parser itself preserves every block as a separate chunk so a future
/// tool-usage indexer can consume the full stream from the same source.
/// Callers feeding the attention corpus should run their parse output
/// through this filter — those blocks dominate the corpus by volume but
/// carry no thinking-substrate signal, and they were burying substantive
/// clusters in [`crate::attention::emergent`] surfacing.
#[must_use]
pub fn drop_tool_blocks(chunks: Vec<Chunk>) -> Vec<Chunk> {
    chunks
        .into_iter()
        .filter(|c| {
            c.extra
                .get("block_kind")
                .and_then(|v| v.as_str())
                .is_none_or(|kind| kind != "tool_use" && kind != "tool_result")
        })
        .collect()
}

/// Drop chunks whose text is a harness `<system-reminder>` block.
///
/// These are instructions the runtime injects into the transcript stream
/// (e.g. "Respond with just the action or changes…") — the *recording
/// apparatus*, not experienced content. Being byte-identical across hundreds
/// of turns, they form the largest degenerate clusters in
/// [`crate::attention::emergent`] surfacing (a 509-member, cohesion≈1.0 blob
/// observed in a live corpus). Like [`drop_tool_blocks`] and
/// [`drop_local_command_wrappers`], callers feeding the attention corpus
/// should run parse output through this filter.
///
/// Conservative by design: only chunks whose trimmed text *starts with* the
/// tag are dropped, so a reminder embedded inside a real message is retained
/// rather than discarding genuine content.
#[must_use]
pub fn drop_system_reminders(chunks: Vec<Chunk>) -> Vec<Chunk> {
    chunks
        .into_iter()
        .filter(|c| !c.text.trim_start().starts_with("<system-reminder>"))
        .collect()
}

/// Tag (don't drop) chunks whose text is a Claude Code multi-agent
/// `<teammate-message ...>` orchestration envelope — the inter-agent routing
/// wrapper the harness injects when agents coordinate (RT-7). Same apparatus
/// class as `<system-reminder>`: it is the recording/coordination apparatus,
/// not experienced thinking, and being templated it forms degenerate
/// high-familiarity threads (`team-lead`, `teammate-message`) and surfaces in
/// the ambient lens. *Unlike* pure reminders these envelopes carry task
/// descriptions with implementation-history value, so we **attenuate rather
/// than delete**: stamp `record_kind=harness_orchestration` so the lens
/// denylist (`config::default_lens_exclude_facets`) keeps them out of ambient
/// surfacing and the weaver skips them as anchor/proposal candidates, while
/// they stay fully recall-able. Conservative: only chunks whose trimmed text
/// *starts with* the tag are tagged, so an envelope quoted inside a real
/// message keeps its content facets.
#[must_use]
pub fn tag_harness_orchestration(chunks: Vec<Chunk>) -> Vec<Chunk> {
    chunks
        .into_iter()
        .map(|mut c| {
            if c.text.trim_start().starts_with("<teammate-message") {
                ostk_recall_core::merge_override(
                    &mut c.facets,
                    "record_kind",
                    vec!["harness_orchestration".to_string()],
                );
            }
            c
        })
        .collect()
}

/// Materialize blocks into chunks. `chunk_index` is monotonic across the
/// session; `parent_ids` holds the previous chunk's id so `recall_link`
/// can chain backward through the session.
fn build_chunks(
    blocks: &[Block],
    source: Source,
    source_id_base: &str,
    project: Option<&str>,
    source_config_id: &str,
    abs_path: &str,
    fallback_ts: Option<DateTime<Utc>>,
) -> Vec<Chunk> {
    let mut out: Vec<Chunk> = Vec::with_capacity(blocks.len());
    let mut chunk_index: u32 = 0;
    let mut prev_chunk_id: Option<String> = None;
    let first_ts = blocks.iter().find_map(|b| b.ts).or(fallback_ts);

    for block in blocks {
        let chunk_id = Chunk::make_id(source, source_id_base, chunk_index, source_config_id);
        let sha256 = Chunk::content_hash(&block.text);
        let links = Links {
            file_path: Some(abs_path.to_string()),
            parent_ids: prev_chunk_id.clone().into_iter().collect(),
            ..Links::default()
        };
        let extra = serde_json::json!({ "block_kind": block.kind.as_block_kind() });
        out.push(Chunk {
            chunk_id: chunk_id.clone(),
            source,
            project: project.map(str::to_string),
            source_id: source_id_base.to_string(),
            facets: Default::default(),
            embedding_input_sha256: String::new(),
            source_config_id: source_config_id.to_string(),
            chunk_index,
            ts: block.ts.or(first_ts),
            role: Some(block.kind.as_role().to_string()),
            text: block.text.clone(),
            sha256,
            links,
            extra,
        });
        prev_chunk_id = Some(chunk_id);
        chunk_index = chunk_index.saturating_add(1);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::TempDir;

    /// 1 user msg + assistant prose + 1 `tool_use` + 1 `tool_result` +
    /// assistant follow-up text → 5 chunks with the right roles and
    /// parent chain.
    #[test]
    fn per_message_chunks_roles_and_parent_chain() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("s.jsonl");
        let mut f = std::fs::File::create(&path).unwrap();
        // user
        writeln!(
            f,
            r#"{{"type":"user","message":{{"role":"user","content":"hello there"}},"timestamp":"2026-04-17T10:00:00Z"}}"#
        )
        .unwrap();
        // assistant: text + tool_use
        writeln!(
            f,
            r#"{{"type":"assistant","message":{{"role":"assistant","content":[{{"type":"text","text":"let me check"}},{{"type":"tool_use","name":"read","input":{{"path":"f.rs"}}}}]}},"timestamp":"2026-04-17T10:00:01Z"}}"#
        )
        .unwrap();
        // user: tool_result
        writeln!(
            f,
            r#"{{"type":"user","message":{{"role":"user","content":[{{"type":"tool_result","content":"file contents"}}]}},"timestamp":"2026-04-17T10:00:02Z"}}"#
        )
        .unwrap();
        // assistant: follow-up text
        writeln!(
            f,
            r#"{{"type":"assistant","message":{{"role":"assistant","content":[{{"type":"text","text":"done, here's the answer"}}]}},"timestamp":"2026-04-17T10:00:03Z"}}"#
        )
        .unwrap();

        let chunks = parse_session_file(
            &path,
            Source::ClaudeCode,
            "s.jsonl",
            Some("proj"),
            "test-cfg",
            None,
        )
        .unwrap();
        assert_eq!(chunks.len(), 5, "expected 5 chunks, got {}", chunks.len());

        // Roles in order: user, assistant, tool, tool_result, assistant
        let roles: Vec<&str> = chunks.iter().map(|c| c.role.as_deref().unwrap()).collect();
        assert_eq!(
            roles,
            vec!["user", "assistant", "tool", "tool_result", "assistant"]
        );

        // Block-kind annotations on extra
        let kinds: Vec<&str> = chunks
            .iter()
            .map(|c| c.extra.get("block_kind").unwrap().as_str().unwrap())
            .collect();
        assert_eq!(
            kinds,
            vec![
                "user",
                "assistant_text",
                "tool_use",
                "tool_result",
                "assistant_text"
            ]
        );

        // Parent chain: each chunk parents to the previous one's id.
        assert!(chunks[0].links.parent_ids.is_empty());
        for i in 1..chunks.len() {
            assert_eq!(
                chunks[i].links.parent_ids,
                vec![chunks[i - 1].chunk_id.clone()],
                "chunk {i} should parent to chunk {}",
                i - 1
            );
        }

        // chunk_index is monotonic.
        for (i, c) in chunks.iter().enumerate() {
            #[allow(clippy::cast_possible_truncation)]
            let expected = i as u32;
            assert_eq!(c.chunk_index, expected);
        }

        // Project carried through.
        assert!(chunks.iter().all(|c| c.project.as_deref() == Some("proj")));
    }

    /// Consecutive assistant text blocks (text → `tool_use` → text) →
    /// each text block is its own chunk.
    #[test]
    fn consecutive_assistant_text_blocks_split() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("s.jsonl");
        let mut f = std::fs::File::create(&path).unwrap();
        writeln!(
            f,
            r#"{{"type":"assistant","message":{{"role":"assistant","content":[{{"type":"text","text":"first"}},{{"type":"tool_use","name":"x","input":{{}}}},{{"type":"text","text":"second"}}]}},"timestamp":"2026-04-17T10:00:00Z"}}"#
        )
        .unwrap();
        let chunks =
            parse_session_file(&path, Source::ClaudeCode, "s.jsonl", None, "test-cfg", None)
                .unwrap();
        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks[0].text, "first");
        assert_eq!(chunks[0].role.as_deref(), Some("assistant"));
        assert!(chunks[1].text.contains("tool_use x"));
        assert_eq!(chunks[1].role.as_deref(), Some("tool"));
        assert_eq!(chunks[2].text, "second");
        assert_eq!(chunks[2].role.as_deref(), Some("assistant"));
    }

    /// Empty content blocks are skipped — no zero-text chunks.
    #[test]
    fn empty_blocks_skipped() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("s.jsonl");
        let mut f = std::fs::File::create(&path).unwrap();
        // Empty text block, then a real one
        writeln!(
            f,
            r#"{{"type":"assistant","message":{{"role":"assistant","content":[{{"type":"text","text":""}},{{"type":"text","text":"   "}},{{"type":"text","text":"real"}}]}},"timestamp":"2026-04-17T10:00:00Z"}}"#
        )
        .unwrap();
        // Whole-string user content that's just whitespace
        writeln!(
            f,
            r#"{{"type":"user","message":{{"role":"user","content":"   "}},"timestamp":"2026-04-17T10:00:01Z"}}"#
        )
        .unwrap();
        let chunks =
            parse_session_file(&path, Source::ClaudeCode, "s.jsonl", None, "test-cfg", None)
                .unwrap();
        assert_eq!(chunks.len(), 1, "expected only the non-empty text chunk");
        assert_eq!(chunks[0].text, "real");
    }

    #[test]
    fn parses_raw_anthropic_shape() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("s.jsonl");
        let mut f = std::fs::File::create(&path).unwrap();
        writeln!(
            f,
            r#"{{"role":"user","content":"q1","timestamp":"2026-04-17T10:00:00Z"}}"#
        )
        .unwrap();
        writeln!(
            f,
            r#"{{"role":"assistant","content":"a1","timestamp":"2026-04-17T10:00:01Z"}}"#
        )
        .unwrap();
        writeln!(
            f,
            r#"{{"role":"user","content":"q2","timestamp":"2026-04-17T10:00:02Z"}}"#
        )
        .unwrap();
        writeln!(
            f,
            r#"{{"role":"assistant","content":"a2","timestamp":"2026-04-17T10:00:03Z"}}"#
        )
        .unwrap();

        let chunks = parse_session_file(
            &path,
            Source::OstkSession,
            "s.jsonl",
            None,
            "test-cfg",
            None,
        )
        .unwrap();
        // 4 messages, each yields one block → 4 chunks.
        assert_eq!(chunks.len(), 4);
        assert_eq!(chunks[0].text, "q1");
        assert_eq!(chunks[0].role.as_deref(), Some("user"));
        assert_eq!(chunks[1].text, "a1");
        assert_eq!(chunks[1].role.as_deref(), Some("assistant"));
        assert_eq!(chunks[2].text, "q2");
        assert_eq!(chunks[3].text, "a2");
        // Parent chain reaches across the session.
        assert_eq!(chunks[3].links.parent_ids, vec![chunks[2].chunk_id.clone()]);
        assert_eq!(chunks[1].links.parent_ids, vec![chunks[0].chunk_id.clone()]);
    }

    #[test]
    fn tool_result_over_cap_is_truncated() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("s.jsonl");
        let mut f = std::fs::File::create(&path).unwrap();
        // Build a 5000-char tool_result blob — this is the exact pattern
        // that was dominating BM25 hits with raw file listings.
        let big_content: String = "abcdefghij".repeat(500); // 5000 chars
        let line = serde_json::json!({
            "type": "user",
            "message": {
                "role": "user",
                "content": [
                    { "type": "tool_result", "content": big_content }
                ]
            },
            "timestamp": "2026-04-17T10:00:00Z"
        });
        writeln!(f, "{line}").unwrap();
        let chunks =
            parse_session_file(&path, Source::ClaudeCode, "s.jsonl", None, "test-cfg", None)
                .unwrap();
        assert_eq!(chunks.len(), 1);
        let body = &chunks[0].text;
        assert_eq!(chunks[0].role.as_deref(), Some("tool_result"));
        assert!(
            body.contains("…[truncated"),
            "expected truncation marker, got: {body}"
        );
        // Full 5000-char string must not survive.
        assert!(
            !body.contains(&"abcdefghij".repeat(500)),
            "5000-char dump leaked through truncation"
        );
    }

    #[test]
    fn small_tool_result_is_preserved() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("s.jsonl");
        let mut f = std::fs::File::create(&path).unwrap();
        let small = "ok"; // 2 chars
        let line = serde_json::json!({
            "type": "user",
            "message": {
                "role": "user",
                "content": [
                    { "type": "tool_result", "content": small }
                ]
            },
            "timestamp": "2026-04-17T10:00:00Z"
        });
        writeln!(f, "{line}").unwrap();
        let chunks =
            parse_session_file(&path, Source::ClaudeCode, "s.jsonl", None, "test-cfg", None)
                .unwrap();
        assert_eq!(chunks.len(), 1);
        let body = &chunks[0].text;
        assert!(body.contains("[tool_result: ok]"));
        assert!(!body.contains("…[truncated"));
    }

    #[test]
    fn long_text_block_is_one_chunk() {
        // Per-message chunking does NOT slice prose into siblings; the
        // long text is preserved as a single chunk so the cross-encoder
        // can score the whole semantic unit at once.
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("s.jsonl");
        let mut f = std::fs::File::create(&path).unwrap();
        let long_text: String = "alpha-bravo ".repeat(1000); // 12000 chars
        let line = serde_json::json!({
            "type": "assistant",
            "message": {
                "role": "assistant",
                "content": [ { "type": "text", "text": long_text } ]
            },
            "timestamp": "2026-04-17T10:00:00Z"
        });
        writeln!(f, "{line}").unwrap();
        let chunks =
            parse_session_file(&path, Source::ClaudeCode, "s.jsonl", None, "test-cfg", None)
                .unwrap();
        assert_eq!(chunks.len(), 1);
        let body = &chunks[0].text;
        assert!(!body.contains("…[truncated"));
        // Every alpha-bravo occurrence is preserved.
        let count = body.matches("alpha-bravo").count();
        assert_eq!(count, 1000);
    }

    /// `drop_local_command_wrappers` filters Claude Code's slash-command
    /// surface scaffolding (which lands as `block_kind=user`) but leaves
    /// substantive user prose untouched.
    #[test]
    fn drop_local_command_wrappers_filters_scaffolding() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("s.jsonl");
        let mut f = std::fs::File::create(&path).unwrap();
        // Substantive user prose
        writeln!(
            f,
            r#"{{"type":"user","message":{{"role":"user","content":"think about this design"}},"timestamp":"2026-04-17T10:00:00Z"}}"#
        )
        .unwrap();
        // Slash command scaffolding
        writeln!(
            f,
            r#"{{"type":"user","message":{{"role":"user","content":"<local-command-caveat>Caveat: ...</local-command-caveat>"}},"timestamp":"2026-04-17T10:00:01Z"}}"#
        )
        .unwrap();
        writeln!(
            f,
            r#"{{"type":"user","message":{{"role":"user","content":"<command-name>/compact</command-name>"}},"timestamp":"2026-04-17T10:00:02Z"}}"#
        )
        .unwrap();
        writeln!(
            f,
            r#"{{"type":"user","message":{{"role":"user","content":"<local-command-stdout>Reconnected.</local-command-stdout>"}},"timestamp":"2026-04-17T10:00:03Z"}}"#
        )
        .unwrap();
        // Another substantive line
        writeln!(
            f,
            r#"{{"type":"assistant","message":{{"role":"assistant","content":"here's an idea"}},"timestamp":"2026-04-17T10:00:04Z"}}"#
        )
        .unwrap();
        let raw = parse_session_file(&path, Source::ClaudeCode, "s.jsonl", None, "test-cfg", None)
            .unwrap();
        assert_eq!(raw.len(), 5);
        let kept = drop_local_command_wrappers(raw);
        assert_eq!(kept.len(), 2);
        let texts: Vec<&str> = kept.iter().map(|c| c.text.as_str()).collect();
        assert_eq!(texts, vec!["think about this design", "here's an idea"]);
    }

    /// `drop_tool_blocks` filters tool_use/tool_result chunks but keeps
    /// user/assistant_text — and preserves their order + content.
    #[test]
    fn drop_tool_blocks_keeps_prose_only() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("s.jsonl");
        let mut f = std::fs::File::create(&path).unwrap();
        // user + assistant text + tool_use + tool_result + assistant text
        writeln!(
            f,
            r#"{{"type":"user","message":{{"role":"user","content":"q"}},"timestamp":"2026-04-17T10:00:00Z"}}"#
        )
        .unwrap();
        writeln!(
            f,
            r#"{{"type":"assistant","message":{{"role":"assistant","content":[{{"type":"text","text":"a1"}},{{"type":"tool_use","name":"r","input":{{}}}}]}},"timestamp":"2026-04-17T10:00:01Z"}}"#
        )
        .unwrap();
        writeln!(
            f,
            r#"{{"type":"user","message":{{"role":"user","content":[{{"type":"tool_result","content":"x"}}]}},"timestamp":"2026-04-17T10:00:02Z"}}"#
        )
        .unwrap();
        writeln!(
            f,
            r#"{{"type":"assistant","message":{{"role":"assistant","content":[{{"type":"text","text":"a2"}}]}},"timestamp":"2026-04-17T10:00:03Z"}}"#
        )
        .unwrap();
        let raw = parse_session_file(&path, Source::ClaudeCode, "s.jsonl", None, "test-cfg", None)
            .unwrap();
        assert_eq!(raw.len(), 5);
        let kept = drop_tool_blocks(raw);
        assert_eq!(kept.len(), 3);
        let kinds: Vec<&str> = kept
            .iter()
            .map(|c| c.extra.get("block_kind").unwrap().as_str().unwrap())
            .collect();
        assert_eq!(kinds, vec!["user", "assistant_text", "assistant_text"]);
        let texts: Vec<&str> = kept.iter().map(|c| c.text.as_str()).collect();
        assert_eq!(texts, vec!["q", "a1", "a2"]);
    }

    /// `drop_system_reminders` removes pure harness `<system-reminder>`
    /// chunks (the apparatus) but keeps real content — including a message
    /// that merely *mentions* a reminder mid-text.
    #[test]
    fn drop_system_reminders_filters_apparatus() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("s.jsonl");
        let mut f = std::fs::File::create(&path).unwrap();
        // Real user prose
        writeln!(
            f,
            r#"{{"type":"user","message":{{"role":"user","content":"design the weaver"}},"timestamp":"2026-04-17T10:00:00Z"}}"#
        )
        .unwrap();
        // Pure harness reminder (the apparatus)
        writeln!(
            f,
            r#"{{"type":"user","message":{{"role":"user","content":"<system-reminder>Respond with just the action or changes and without a thinking block.</system-reminder>"}},"timestamp":"2026-04-17T10:00:01Z"}}"#
        )
        .unwrap();
        // Real content that merely references a reminder mid-text — must survive
        writeln!(
            f,
            r#"{{"type":"assistant","message":{{"role":"assistant","content":"the <system-reminder> tag is harness instrumentation"}},"timestamp":"2026-04-17T10:00:02Z"}}"#
        )
        .unwrap();
        let raw = parse_session_file(&path, Source::ClaudeCode, "s.jsonl", None, "test-cfg", None)
            .unwrap();
        assert_eq!(raw.len(), 3);
        let kept = drop_system_reminders(raw);
        assert_eq!(kept.len(), 2, "only the pure reminder chunk is dropped");
        let texts: Vec<&str> = kept.iter().map(|c| c.text.as_str()).collect();
        assert_eq!(
            texts,
            vec![
                "design the weaver",
                "the <system-reminder> tag is harness instrumentation"
            ]
        );
    }

    /// `tag_harness_orchestration` stamps `record_kind=harness_orchestration`
    /// on `<teammate-message>` envelopes (kept, not dropped) and leaves real
    /// content untouched — including a message that merely mentions the term.
    #[test]
    fn tag_harness_orchestration_marks_only_envelopes() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("s.jsonl");
        let mut f = std::fs::File::create(&path).unwrap();
        writeln!(
            f,
            r#"{{"type":"assistant","message":{{"role":"assistant","content":"<teammate-message teammate_id=\"team-lead\">{{\"type\":\"task_assignment\",\"taskId\":\"5\"}}</teammate-message>"}},"timestamp":"2026-04-17T10:00:00Z"}}"#
        )
        .unwrap();
        writeln!(
            f,
            r#"{{"type":"assistant","message":{{"role":"assistant","content":"I'll message the team-lead about the design"}},"timestamp":"2026-04-17T10:00:01Z"}}"#
        )
        .unwrap();
        let raw = parse_session_file(&path, Source::ClaudeCode, "s.jsonl", None, "test-cfg", None)
            .unwrap();
        let tagged = tag_harness_orchestration(raw);
        assert_eq!(tagged.len(), 2, "nothing is dropped — tag, don't delete");
        let facets0 = ostk_recall_core::to_list(&tagged[0].facets);
        assert!(
            facets0
                .iter()
                .any(|f| f == "record_kind:harness_orchestration"),
            "the <teammate-message> envelope is tagged: {facets0:?}"
        );
        let facets1 = ostk_recall_core::to_list(&tagged[1].facets);
        assert!(
            !facets1
                .iter()
                .any(|f| f == "record_kind:harness_orchestration"),
            "prose that merely mentions team-lead is untouched: {facets1:?}"
        );
    }

    /// Skip wrapper types that aren't user/assistant.
    #[test]
    fn skips_non_user_assistant_types() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("s.jsonl");
        let mut f = std::fs::File::create(&path).unwrap();
        writeln!(f, r#"{{"type":"file-history-snapshot","messageId":"x"}}"#).unwrap();
        writeln!(f, r#"{{"type":"summary","content":"session summary"}}"#).unwrap();
        writeln!(
            f,
            r#"{{"type":"assistant","message":{{"role":"assistant","content":"hi"}},"timestamp":"2026-04-17T10:00:00Z"}}"#
        )
        .unwrap();
        let chunks =
            parse_session_file(&path, Source::ClaudeCode, "s.jsonl", None, "test-cfg", None)
                .unwrap();
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].text, "hi");
    }
}
