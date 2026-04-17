//! Shared helper for parsing Anthropic-style session logs into exchange
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
//! (`{type: "text"|"tool_use"|"tool_result"}`); all textual blocks are
//! concatenated.
//!
//! # Chunking
//!
//! One chunk per **exchange** = one assistant response (possibly with
//! interleaved `tool_use`/`tool_result` blocks) plus the preceding user turn.
//! Role on the chunk is `"assistant"` (it's semantically an answer). Each
//! chunk links to the previous exchange's `chunk_id` via
//! `Links.parent_ids`.
//!
//! Oversize assistant bodies (> `MAX_EXCHANGE_CHARS`) are paragraph-split
//! into multiple sibling chunks; the user turn stays with the first slice.

use std::fmt::Write as _;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

use chrono::{DateTime, Utc};
use ostk_recall_core::{Chunk, Links, Result, Source};
use serde::Deserialize;

/// Hard ceiling on a single chunk's text length. Assistant responses that
/// exceed this are paragraph-split into siblings.
pub const MAX_EXCHANGE_CHARS: usize = 4000;

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

#[derive(Debug, Clone)]
struct Turn {
    role: String,
    text: String,
    ts: Option<DateTime<Utc>>,
}

/// Parse a session file into chunks.
///
/// * `source` — concrete `Source` to stamp each chunk with.
/// * `source_id_base` — base identifier (e.g. session filename); chunk
///   index is appended by the caller pattern.
/// * `project` — carried onto chunks.
/// * `path` — absolute path stored in `Links.file_path`.
/// * `fallback_ts` — used when the first kept line has no `timestamp`.
pub fn parse_session_file(
    path: &Path,
    source: Source,
    source_id_base: &str,
    project: Option<&str>,
    fallback_ts: Option<DateTime<Utc>>,
) -> Result<Vec<Chunk>> {
    let f = File::open(path)?;
    let reader = BufReader::new(f);

    let abs_path = path
        .canonicalize()
        .unwrap_or_else(|_| path.to_path_buf())
        .to_string_lossy()
        .into_owned();

    let mut turns: Vec<Turn> = Vec::new();
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
        if let Some(turn) = turn_from_line(&parsed) {
            turns.push(turn);
        }
    }

    Ok(build_exchange_chunks(
        &turns,
        source,
        source_id_base,
        project,
        &abs_path,
        fallback_ts,
    ))
}

fn turn_from_line(line: &LogLine) -> Option<Turn> {
    // Type-level gate: Claude Code files have `type`. Skip anything not
    // user/assistant. Raw Anthropic files have no `type` field.
    if let Some(ty) = line.r#type.as_deref() {
        if ty != "user" && ty != "assistant" {
            return None;
        }
    }
    let (role, content) = line.message.as_ref().map_or_else(
        || (line.role.clone(), line.content.clone()),
        |inner| (inner.role.clone(), inner.content.clone()),
    );
    let role = role.unwrap_or_else(|| line.r#type.clone().unwrap_or_default());
    if role != "user" && role != "assistant" {
        return None;
    }
    let text = content.map(extract_text).unwrap_or_default();
    if text.trim().is_empty() {
        return None;
    }
    Some(Turn {
        role,
        text,
        ts: line.timestamp,
    })
}

/// Extract displayable text from an Anthropic `content` field, which may
/// be a string or an array of content blocks.
fn extract_text(content: serde_json::Value) -> String {
    match content {
        serde_json::Value::String(s) => s,
        serde_json::Value::Array(blocks) => {
            let mut out = String::new();
            for block in blocks {
                let Some(obj) = block.as_object() else {
                    continue;
                };
                let ty = obj.get("type").and_then(|v| v.as_str()).unwrap_or("");
                match ty {
                    "text" => {
                        if let Some(s) = obj.get("text").and_then(|v| v.as_str()) {
                            if !out.is_empty() {
                                out.push_str("\n\n");
                            }
                            out.push_str(s);
                        }
                    }
                    "tool_use" => {
                        let name = obj.get("name").and_then(|v| v.as_str()).unwrap_or("tool");
                        let input = obj
                            .get("input")
                            .map(|v| serde_json::to_string(v).unwrap_or_default())
                            .unwrap_or_default();
                        if !out.is_empty() {
                            out.push_str("\n\n");
                        }
                        let _ = write!(out, "[tool_use {name}: {input}]");
                    }
                    "tool_result" => {
                        let tool_content = obj.get("content").map_or(String::new(), |v| match v {
                            serde_json::Value::String(s) => s.clone(),
                            other => serde_json::to_string(other).unwrap_or_default(),
                        });
                        if !out.is_empty() {
                            out.push_str("\n\n");
                        }
                        let _ = write!(out, "[tool_result: {tool_content}]");
                    }
                    _ => {
                        // unknown block; keep a marker for debuggability
                    }
                }
            }
            out
        }
        _ => String::new(),
    }
}

fn build_exchange_chunks(
    turns: &[Turn],
    source: Source,
    source_id_base: &str,
    project: Option<&str>,
    abs_path: &str,
    fallback_ts: Option<DateTime<Utc>>,
) -> Vec<Chunk> {
    // Walk turns left-to-right: every `assistant` turn + the preceding
    // user turn forms one exchange.
    let mut out: Vec<Chunk> = Vec::new();
    let mut chunk_index: u32 = 0;
    let mut last_user: Option<&Turn> = None;
    let mut prev_chunk_id: Option<String> = None;
    let first_ts = turns.iter().find_map(|t| t.ts).or(fallback_ts);

    for turn in turns {
        match turn.role.as_str() {
            "user" => {
                last_user = Some(turn);
            }
            "assistant" => {
                let mut body = String::new();
                if let Some(u) = last_user {
                    body.push_str("USER: ");
                    body.push_str(u.text.trim());
                    body.push_str("\n\nASSISTANT: ");
                } else {
                    body.push_str("ASSISTANT: ");
                }
                body.push_str(turn.text.trim());
                last_user = None;

                let pieces = if body.len() <= MAX_EXCHANGE_CHARS {
                    vec![body]
                } else {
                    paragraph_split(&body, MAX_EXCHANGE_CHARS)
                };

                for piece in pieces {
                    let chunk_id = Chunk::make_id(source, source_id_base, chunk_index);
                    let sha256 = Chunk::content_hash(&piece);
                    let links = Links {
                        file_path: Some(abs_path.to_string()),
                        parent_ids: prev_chunk_id.clone().into_iter().collect(),
                        ..Links::default()
                    };
                    out.push(Chunk {
                        chunk_id: chunk_id.clone(),
                        source,
                        project: project.map(str::to_string),
                        source_id: source_id_base.to_string(),
                        chunk_index,
                        ts: turn.ts.or(first_ts),
                        role: Some("assistant".into()),
                        text: piece,
                        sha256,
                        links,
                        extra: serde_json::Value::Null,
                    });
                    prev_chunk_id = Some(chunk_id);
                    chunk_index = chunk_index.saturating_add(1);
                }
            }
            _ => {}
        }
    }
    out
}

/// Split on blank-line gaps, greedily packing into ≤`max_chars` pieces.
fn paragraph_split(text: &str, max_chars: usize) -> Vec<String> {
    let paras: Vec<&str> = text
        .split("\n\n")
        .map(str::trim_end)
        .filter(|p| !p.trim().is_empty())
        .collect();
    if paras.is_empty() {
        return vec![text.to_string()];
    }
    let mut out = Vec::new();
    let mut cur = String::new();
    for p in paras {
        let candidate = cur.len() + 2 + p.len();
        if cur.is_empty() {
            // first paragraph of a fresh chunk
        } else if candidate <= max_chars {
            cur.push_str("\n\n");
        } else {
            out.push(std::mem::take(&mut cur));
        }
        cur.push_str(p);
    }
    if !cur.trim().is_empty() {
        out.push(cur);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::TempDir;

    #[test]
    fn parses_claude_code_file_with_tool_use() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("s.jsonl");
        let mut f = std::fs::File::create(&path).unwrap();
        // Skip: file-history-snapshot
        writeln!(f, r#"{{"type":"file-history-snapshot","messageId":"x"}}"#).unwrap();
        // user turn
        writeln!(
            f,
            r#"{{"type":"user","message":{{"role":"user","content":"hello there"}},"timestamp":"2026-04-17T10:00:00Z"}}"#
        )
        .unwrap();
        // assistant turn with interleaved tool_use
        writeln!(
            f,
            r#"{{"type":"assistant","message":{{"role":"assistant","content":[{{"type":"text","text":"let me check"}},{{"type":"tool_use","name":"read","input":{{"path":"f.rs"}}}},{{"type":"text","text":"done, here's the answer"}}]}},"timestamp":"2026-04-17T10:00:01Z"}}"#
        )
        .unwrap();
        // summary (skip)
        writeln!(f, r#"{{"type":"summary","content":"session summary"}}"#).unwrap();

        let chunks =
            parse_session_file(&path, Source::ClaudeCode, "s.jsonl", Some("proj"), None).unwrap();
        assert_eq!(chunks.len(), 1);
        let c = &chunks[0];
        assert!(c.text.contains("hello there"));
        assert!(c.text.contains("let me check"));
        assert!(c.text.contains("done, here's the answer"));
        assert!(c.text.contains("tool_use read"));
        assert_eq!(c.role.as_deref(), Some("assistant"));
        assert_eq!(c.project.as_deref(), Some("proj"));
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

        let chunks = parse_session_file(&path, Source::OstkSession, "s.jsonl", None, None).unwrap();
        assert_eq!(chunks.len(), 2);
        assert!(chunks[0].text.contains("q1"));
        assert!(chunks[0].text.contains("a1"));
        assert!(chunks[1].text.contains("q2"));
        assert!(chunks[1].text.contains("a2"));
        // Second chunk parents to first.
        assert_eq!(chunks[1].links.parent_ids, vec![chunks[0].chunk_id.clone()]);
    }

    #[test]
    fn oversize_assistant_splits_into_siblings() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("s.jsonl");
        let mut f = std::fs::File::create(&path).unwrap();
        // Build ~12000 chars of paragraphs.
        let mut big = String::new();
        for i in 0..60 {
            let _ = write!(big, "paragraph {i} with some words. ");
            big.push_str(&"lorem ipsum ".repeat(10));
            big.push_str("\\n\\n");
        }
        writeln!(
            f,
            r#"{{"type":"user","message":{{"role":"user","content":"big one"}},"timestamp":"2026-04-17T10:00:00Z"}}"#
        )
        .unwrap();
        let line = serde_json::json!({
            "type": "assistant",
            "message": {
                "role": "assistant",
                "content": [ { "type": "text", "text": big.replace("\\n", "\n") } ]
            },
            "timestamp": "2026-04-17T10:00:01Z"
        });
        writeln!(f, "{line}").unwrap();

        let chunks = parse_session_file(&path, Source::ClaudeCode, "s.jsonl", None, None).unwrap();
        assert!(
            chunks.len() >= 2,
            "expected multi-chunk split, got {}",
            chunks.len()
        );
    }
}
