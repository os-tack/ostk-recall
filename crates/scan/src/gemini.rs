use std::fs;

use chrono::{DateTime, Utc};
use ostk_recall_core::{
    Chunk, Error, Result, Scanner, Source, SourceConfig, SourceItem, SourceKind, Links,
};
use serde_json::Value;

/// Scanner for Gemini CLI session logs.
///
/// The Gemini CLI writes two distinct on-disk shapes in `~/.gemini/tmp/`:
///
/// * **Shape A — `<project>/logs.json`**: a flat JSON array of user
///   prompts only. Each record is `{message, messageId, sessionId,
///   timestamp, type:"user"}`. Gemini responses are not logged here.
///
/// * **Shape B — `<project>/chats/session-*.json`**: a session envelope
///   `{sessionId, projectHash, startTime, lastUpdated, messages:[...]}`
///   where each message is `{id, timestamp, type:"user"|"gemini",
///   content, toolCalls?, thoughts?, tokens?, model?}`. The `content`
///   field is `[{text:"..."}]` for users and a plain string (often
///   empty when only `toolCalls` are present) for gemini.
///
/// Both shapes coexist in a real corpus (Scott's machine: 14 of the
/// first format, 151 of the second). The scanner detects shape from
/// the JSON top-level type and dispatches accordingly. A truly
/// unparseable file falls through to a single capped fallback chunk so
/// the body is still searchable.
#[derive(Debug, Default)]
pub struct GeminiScanner;

impl Scanner for GeminiScanner {
    fn kind(&self) -> SourceKind { SourceKind::Gemini }

    fn discover<'a>(
        &'a self,
        cfg: &'a SourceConfig,
    ) -> Box<dyn Iterator<Item = Result<SourceItem>> + 'a> {
        let roots = match cfg.expanded_paths() {
            Ok(v) => v,
            Err(e) => return Box::new(std::iter::once(Err(e))),
        };
        let project = cfg.project.clone();

        let iter = roots.into_iter().flat_map(move |root| {
            let project = project.clone();
            walkdir::WalkDir::new(&root)
                .into_iter()
                .filter_map(std::result::Result::ok)
                .filter(|e| e.file_type().is_file())
                .filter(|e| {
                    let name = e.file_name().to_string_lossy();
                    let is_session = name.starts_with("session-")
                        && e.path().extension().is_some_and(|x| x == "json");
                    let is_logs = name == "logs.json";
                    is_session || is_logs
                })
                .map(move |entry| {
                    let path = entry.path().to_path_buf();
                    // Use a path-relative source_id so logs.json files
                    // from different projects don't collide on the same
                    // ingest_chunks key.
                    let source_id = source_id_for(&path);
                    Ok(SourceItem {
                        source_id,
                        path: Some(path),
                        project: project.clone(),
                        bytes: None,
                        ignore: Vec::new(),
                    })
                })
        });
        Box::new(iter)
    }

    fn parse(&self, item: SourceItem) -> Result<Vec<Chunk>> {
        let path = item.path.as_ref().ok_or_else(|| Error::Parse("gemini: path missing".into()))?;
        let content = fs::read_to_string(path)?;

        let root: Value = match serde_json::from_str(&content) {
            Ok(v) => v,
            Err(e) => {
                tracing::warn!(
                    file = %path.display(),
                    error = %e,
                    "gemini: top-level JSON parse failed; emitting fallback chunk"
                );
                return Ok(fallback_chunk(&item, &content));
            }
        };

        let chunks = match &root {
            // Shape A: flat array of user prompts (logs.json).
            Value::Array(records) => parse_logs_array(&item, records),
            // Shape B: session envelope (chats/session-*.json).
            Value::Object(_) => parse_session_envelope(&item, &root),
            _ => {
                tracing::warn!(
                    file = %path.display(),
                    "gemini: top-level JSON was neither array nor object; fallback"
                );
                Vec::new()
            }
        };

        if chunks.is_empty() {
            return Ok(fallback_chunk(&item, &content));
        }
        Ok(chunks)
    }
}

/// Parse Shape A (`logs.json`): flat array of user prompts.
fn parse_logs_array(item: &SourceItem, records: &[Value]) -> Vec<Chunk> {
    let mut chunks = Vec::new();
    for rec in records {
        let Some(obj) = rec.as_object() else { continue };

        // logs.json records are user-only, but defensive against
        // future schema additions.
        let role_tag = obj.get("type").and_then(Value::as_str).unwrap_or("user");
        if role_tag != "user" && role_tag != "gemini" {
            continue;
        }

        // Real records use `message: String`; tolerate `content` shapes
        // for forward-compat with anything that grows out of session-*.
        let text = obj.get("message").and_then(Value::as_str).map(str::to_string)
            .or_else(|| obj.get("content").and_then(extract_text))
            .unwrap_or_default();
        let trimmed = text.trim();
        if trimmed.is_empty() {
            continue;
        }

        let body = format!("### {}\n{}", capitalize_role(role_tag), trimmed);
        let ts = parse_ts(obj.get("timestamp"));
        let session_id = obj.get("sessionId").and_then(Value::as_str).unwrap_or("").to_string();
        let extra = serde_json::json!({
            "session_id": session_id,
            "msg_type": role_tag,
            "message_id": obj.get("messageId").and_then(Value::as_str).unwrap_or(""),
            "shape": "logs.json",
        });

        let chunk_index = chunks.len() as u32;
        chunks.push(Chunk {
            chunk_id: Chunk::make_id(Source::Gemini, &item.source_id, chunk_index),
            source: Source::Gemini,
            project: item.project.clone(),
            source_id: item.source_id.clone(),
            chunk_index,
            ts,
            role: Some(role_tag.to_string()),
            text: body.clone(),
            sha256: Chunk::content_hash(&body),
            links: Links::default(),
            extra,
        });
    }
    chunks
}

/// Parse Shape B (`chats/session-*.json`): envelope with `messages` array.
fn parse_session_envelope(item: &SourceItem, root: &Value) -> Vec<Chunk> {
    let session_id = root.get("sessionId").and_then(Value::as_str).unwrap_or("").to_string();
    let project_hash = root.get("projectHash").and_then(Value::as_str).map(str::to_string);
    let messages = match root.get("messages") {
        Some(Value::Array(m)) => m.clone(),
        _ => return Vec::new(),
    };

    let mut chunks = Vec::new();
    for msg in &messages {
        let Some(obj) = msg.as_object() else { continue };
        let role_tag = match obj.get("type").and_then(Value::as_str) {
            Some(t @ ("user" | "gemini")) => t,
            _ => continue,
        };

        // Body: extract conversational text first; if empty, summarize
        // tool calls so a tool-only turn is still searchable.
        let text = obj.get("content").and_then(extract_text).unwrap_or_default();
        let mut body_text = text.trim().to_string();
        if body_text.is_empty() {
            if let Some(summary) = summarize_tool_calls(obj.get("toolCalls")) {
                body_text = summary;
            }
        }
        if body_text.is_empty() {
            continue;
        }

        let body = format!("### {}\n{}", capitalize_role(role_tag), body_text);
        let ts = parse_ts(obj.get("timestamp"));

        let mut extra = serde_json::json!({
            "session_id": session_id,
            "msg_type": role_tag,
            "shape": "session-envelope",
        });
        if let Some(ph) = &project_hash {
            extra["project_hash"] = Value::String(ph.clone());
        }
        if let Some(model) = obj.get("model").and_then(Value::as_str) {
            extra["model"] = Value::String(model.to_string());
        }
        if let Some(tokens) = obj.get("tokens").and_then(Value::as_object) {
            if let Some(thoughts) = tokens.get("thoughts").and_then(Value::as_u64) {
                extra["thought_tokens"] = thoughts.into();
            }
            if let Some(total) = tokens.get("total").and_then(Value::as_u64) {
                extra["total_tokens"] = total.into();
            }
        }

        let chunk_index = chunks.len() as u32;
        chunks.push(Chunk {
            chunk_id: Chunk::make_id(Source::Gemini, &item.source_id, chunk_index),
            source: Source::Gemini,
            project: item.project.clone(),
            source_id: item.source_id.clone(),
            chunk_index,
            ts,
            role: Some(role_tag.to_string()),
            text: body.clone(),
            sha256: Chunk::content_hash(&body),
            links: Links::default(),
            extra,
        });
    }
    chunks
}

/// `content` is one of: `String`, `[{text: String, ...}, ...]`,
/// `[String, ...]`, or `null`. Returns concatenated text or `None`.
fn extract_text(v: &Value) -> Option<String> {
    match v {
        Value::String(s) => Some(s.clone()),
        Value::Array(parts) => {
            let mut out = String::new();
            for p in parts {
                match p {
                    Value::String(s) => {
                        if !out.is_empty() { out.push('\n'); }
                        out.push_str(s);
                    }
                    Value::Object(obj) => {
                        if let Some(t) = obj.get("text").and_then(Value::as_str) {
                            if !out.is_empty() { out.push('\n'); }
                            out.push_str(t);
                        }
                    }
                    _ => {}
                }
            }
            if out.is_empty() { None } else { Some(out) }
        }
        _ => None,
    }
}

/// Summarize a `toolCalls` array as `[tool] name: description` lines so
/// gemini turns that consist purely of tool invocations are still
/// retrievable.
fn summarize_tool_calls(v: Option<&Value>) -> Option<String> {
    let arr = v?.as_array()?;
    if arr.is_empty() { return None; }
    let mut lines = Vec::new();
    for tc in arr {
        let Some(obj) = tc.as_object() else { continue };
        let name = obj.get("name").and_then(Value::as_str).unwrap_or("?");
        let desc = obj.get("args")
            .and_then(|a| a.get("description"))
            .and_then(Value::as_str)
            .or_else(|| obj.get("args")
                .and_then(|a| a.get("command"))
                .and_then(Value::as_str))
            .or_else(|| obj.get("args")
                .and_then(|a| a.get("file_path"))
                .and_then(Value::as_str))
            .unwrap_or("");
        if desc.is_empty() {
            lines.push(format!("[tool] {}", name));
        } else {
            lines.push(format!("[tool] {}: {}", name, desc));
        }
    }
    if lines.is_empty() { None } else { Some(lines.join("\n")) }
}

fn parse_ts(v: Option<&Value>) -> Option<DateTime<Utc>> {
    v.and_then(Value::as_str)
        .and_then(|s| DateTime::parse_from_rfc3339(s).ok())
        .map(|dt| dt.with_timezone(&Utc))
}

fn capitalize_role(role: &str) -> String {
    let mut chars = role.chars();
    match chars.next() {
        Some(c) => c.to_uppercase().chain(chars).collect(),
        None => String::new(),
    }
}

/// Build a stable, collision-resistant source_id from the file path.
/// `<project>/logs.json` becomes `<project>__logs.json`;
/// `<hash>/chats/session-X.json` becomes `<hash>__session-X.json`.
fn source_id_for(path: &std::path::Path) -> String {
    let file = path.file_name().map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "unknown.json".to_string());
    let parent_label = path.parent().and_then(|p| {
        let name = p.file_name().map(|s| s.to_string_lossy().into_owned())?;
        if name == "chats" {
            // Climb one more level to the project/hash dir.
            p.parent().and_then(|pp| pp.file_name().map(|s| s.to_string_lossy().into_owned()))
        } else {
            Some(name)
        }
    });
    match parent_label {
        Some(label) => format!("{}__{}", label, file),
        None => file,
    }
}

/// Emit one chunk holding the file's body. Caps the text at 8 KB so a
/// pathological session-N.json doesn't blow up the embedder.
fn fallback_chunk(item: &SourceItem, body: &str) -> Vec<Chunk> {
    const MAX_BODY: usize = 8 * 1024;
    let trimmed = body.trim();
    if trimmed.is_empty() {
        return Vec::new();
    }
    let text = truncate(trimmed, MAX_BODY).to_string();
    vec![Chunk {
        chunk_id: Chunk::make_id(Source::Gemini, &item.source_id, 0),
        source: Source::Gemini,
        project: item.project.clone(),
        source_id: item.source_id.clone(),
        chunk_index: 0,
        ts: None,
        role: None,
        text: text.clone(),
        sha256: Chunk::content_hash(&text),
        links: Links::default(),
        extra: serde_json::json!({"shape": "fallback"}),
    }]
}

fn truncate(s: &str, max: usize) -> &str {
    if s.len() <= max { return s; }
    let mut end = max;
    while !s.is_char_boundary(end) && end > 0 { end -= 1; }
    &s[..end]
}

#[cfg(test)]
mod tests {
    use super::*;
    use ostk_recall_core::SourceConfig;
    use std::path::PathBuf;
    use tempfile::TempDir;

    fn item(tmp: &std::path::Path, name: &str, body: &str) -> SourceItem {
        let path = tmp.join(name);
        std::fs::write(&path, body).unwrap();
        SourceItem {
            source_id: name.to_string(),
            path: Some(path),
            project: Some("test".to_string()),
            bytes: None,
            ignore: Vec::new(),
        }
    }

    #[test]
    fn parses_logs_json_flat_array() {
        // Real schema from ~/.gemini/tmp/<project>/logs.json
        let body = r#"[
            {"message":"first prompt","messageId":"a","sessionId":"sess1","timestamp":"2026-05-08T12:00:00Z","type":"user"},
            {"message":"second prompt","messageId":"b","sessionId":"sess1","timestamp":"2026-05-08T12:05:00Z","type":"user"}
        ]"#;
        let tmp = TempDir::new().unwrap();
        let it = item(tmp.path(), "logs.json", body);
        let chunks = GeminiScanner.parse(it).unwrap();
        assert_eq!(chunks.len(), 2, "two user prompts -> two chunks");
        assert!(chunks[0].text.contains("first prompt"));
        assert_eq!(chunks[0].role.as_deref(), Some("user"));
        assert_eq!(chunks[0].extra["shape"], "logs.json");
    }

    #[test]
    fn parses_session_envelope_with_mixed_content_shapes() {
        // Real schema from ~/.gemini/tmp/<project>/chats/session-*.json
        let body = r#"{
            "sessionId":"s1","projectHash":"h","startTime":"2026-05-08T12:00:00Z","lastUpdated":"2026-05-08T12:30:00Z",
            "messages":[
                {"id":"1","timestamp":"2026-05-08T12:00:00Z","type":"user","content":[{"text":"hello there"}]},
                {"id":"2","timestamp":"2026-05-08T12:00:30Z","type":"gemini","content":"hi back","tokens":{"input":10,"output":3,"total":13,"thoughts":2},"model":"gemini-3-pro"},
                {"id":"3","timestamp":"2026-05-08T12:01:00Z","type":"gemini","content":"","toolCalls":[{"name":"run_shell_command","args":{"description":"ls home dir","command":"ls"}}]}
            ]
        }"#;
        let tmp = TempDir::new().unwrap();
        let it = item(tmp.path(), "session-x.json", body);
        let chunks = GeminiScanner.parse(it).unwrap();
        assert_eq!(chunks.len(), 3, "user + gemini-text + gemini-tool -> three chunks");
        assert!(chunks[0].text.contains("hello there"));
        assert!(chunks[1].text.contains("hi back"));
        assert!(chunks[2].text.contains("[tool] run_shell_command"));
        assert!(chunks[2].text.contains("ls home dir"));
        assert_eq!(chunks[1].extra["model"], "gemini-3-pro");
        assert_eq!(chunks[1].extra["thought_tokens"], 2);
        assert_eq!(chunks[0].extra["shape"], "session-envelope");
    }

    #[test]
    fn unknown_shape_falls_back_to_single_chunk() {
        let tmp = TempDir::new().unwrap();
        let it = item(tmp.path(), "weird.json", "not json at all");
        let chunks = GeminiScanner.parse(it).unwrap();
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].extra["shape"], "fallback");
    }

    #[test]
    fn empty_messages_array_falls_back() {
        let tmp = TempDir::new().unwrap();
        let it = item(tmp.path(), "empty.json", r#"{"sessionId":"x","messages":[]}"#);
        let chunks = GeminiScanner.parse(it).unwrap();
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].extra["shape"], "fallback");
    }

    #[test]
    fn discovers_both_logs_and_session_files() {
        let tmp = TempDir::new().unwrap();
        let proj = tmp.path().join("proj1");
        let chats = proj.join("chats");
        std::fs::create_dir_all(&chats).unwrap();
        std::fs::write(proj.join("logs.json"), "[]").unwrap();
        std::fs::write(chats.join("session-abc.json"), "{}").unwrap();
        std::fs::write(chats.join("ignore.json"), "{}").unwrap();
        std::fs::write(proj.join("index.json"), "{}").unwrap();

        let cfg = SourceConfig {
            kind: SourceKind::Gemini,
            project: Some("test".to_string()),
            paths: vec![tmp.path().to_string_lossy().into_owned()],
            ignore: vec![],
            extensions: vec![],
        };
        let names: Vec<String> = GeminiScanner.discover(&cfg)
            .filter_map(std::result::Result::ok)
            .map(|i| i.path.unwrap().file_name().unwrap().to_string_lossy().into_owned())
            .collect();
        assert!(names.iter().any(|n| n == "logs.json"), "logs.json must be discovered: {names:?}");
        assert!(names.iter().any(|n| n == "session-abc.json"), "session-*.json must be discovered: {names:?}");
        assert!(!names.iter().any(|n| n == "ignore.json"), "non-session non-logs must be skipped: {names:?}");
        assert!(!names.iter().any(|n| n == "index.json"), "index.json must be skipped: {names:?}");
    }

    #[test]
    fn source_id_disambiguates_logs_across_projects() {
        let a = PathBuf::from("/tmp/.gemini/tmp/proj_a/logs.json");
        let b = PathBuf::from("/tmp/.gemini/tmp/proj_b/logs.json");
        assert_ne!(source_id_for(&a), source_id_for(&b));
        assert_eq!(source_id_for(&a), "proj_a__logs.json");
    }

    #[test]
    fn source_id_disambiguates_chat_sessions() {
        let p = PathBuf::from("/tmp/.gemini/tmp/abc123/chats/session-2026-05-08-x.json");
        assert_eq!(source_id_for(&p), "abc123__session-2026-05-08-x.json");
    }
}
