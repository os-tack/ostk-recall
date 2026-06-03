//! Codex CLI session log scanner.
//!
//! Walks `~/.codex/sessions/YYYY/MM/DD/rollout-*.jsonl` (or any configured
//! root) and emits one chunk per user turn. Codex writes JSONL while a session
//! is live, so malformed/truncated lines are skipped instead of failing the
//! whole file.
//!
//! The scanner treats `response_item` message records as the canonical
//! transcript stream and ignores event duplicates, function calls, tool output,
//! and reasoning records.

use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};

use chrono::{DateTime, Utc};
use ostk_recall_core::{
    Chunk, Error, Links, Result, Scanner, Source, SourceConfig, SourceItem, SourceKind,
};
use serde_json::Value;
use walkdir::WalkDir;

/// Scanner for Codex CLI rollout JSONL session logs.
#[derive(Debug, Default)]
pub struct CodexScanner;

#[derive(Debug, Clone)]
struct SessionMeta {
    session_id: Option<String>,
    cwd: Option<String>,
    originator: Option<String>,
    cli_version: Option<String>,
}

#[derive(Debug, Clone)]
struct PendingTurn {
    user_text: String,
    ts: Option<DateTime<Utc>>,
    assistant_parts: Vec<String>,
}

impl Scanner for CodexScanner {
    fn kind(&self) -> SourceKind {
        SourceKind::Codex
    }

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
            WalkDir::new(&root)
                .follow_links(false)
                .into_iter()
                .filter_map(std::result::Result::ok)
                .filter(|e| e.file_type().is_file())
                .filter(|e| is_codex_rollout(e.path()))
                .map(move |entry| {
                    let path = entry.path().to_path_buf();
                    let source_id = path
                        .file_name()
                        .map(|n| n.to_string_lossy().into_owned())
                        .unwrap_or_default();
                    Ok(SourceItem {
                        source_id,
                        path: Some(path),
                        project: project.clone(),
                        bytes: None,
                        ignore: Vec::new(),
                        source_config_id: "test-cfg".to_string(),
                    })
                })
        });
        Box::new(iter)
    }

    fn parse(&self, item: SourceItem) -> Result<Vec<Chunk>> {
        let path = item
            .path
            .as_ref()
            .ok_or_else(|| Error::Parse("codex: SourceItem.path missing".into()))?;
        parse_codex_session(path, &item)
    }
}

fn parse_codex_session(path: &Path, item: &SourceItem) -> Result<Vec<Chunk>> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);

    let mut meta = SessionMeta {
        session_id: None,
        cwd: None,
        originator: None,
        cli_version: None,
    };
    let mut current: Option<PendingTurn> = None;
    let mut chunks = Vec::new();

    for line in reader.lines() {
        let line = line?;
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }

        let Ok(v) = serde_json::from_str::<Value>(trimmed) else {
            continue;
        };

        if v.get("type").and_then(Value::as_str) == Some("session_meta") {
            update_session_meta(&mut meta, &v);
            continue;
        }

        let Some((role, text, ts)) = response_message(&v) else {
            continue;
        };
        if text.trim().is_empty() {
            continue;
        }

        match role {
            "user" => match current.as_mut() {
                Some(turn) if turn.assistant_parts.is_empty() => {
                    if !turn.user_text.is_empty() {
                        turn.user_text.push_str("\n\n");
                    }
                    turn.user_text.push_str(&text);
                }
                _ => {
                    flush_turn(&mut chunks, item, &meta, current.take());
                    current = Some(PendingTurn {
                        user_text: text,
                        ts,
                        assistant_parts: Vec::new(),
                    });
                }
            },
            "assistant" => {
                if let Some(turn) = current.as_mut() {
                    turn.assistant_parts.push(text);
                }
            }
            _ => {}
        }
    }

    flush_turn(&mut chunks, item, &meta, current);
    Ok(chunks)
}

fn flush_turn(
    chunks: &mut Vec<Chunk>,
    item: &SourceItem,
    meta: &SessionMeta,
    turn: Option<PendingTurn>,
) {
    let Some(turn) = turn else {
        return;
    };
    if turn.assistant_parts.is_empty() {
        return;
    }

    let assistant_text = turn.assistant_parts.join("\n\n");
    let combined_text = format!(
        "### User\n{}\n\n### Codex\n{}",
        turn.user_text, assistant_text
    );
    #[allow(clippy::cast_possible_truncation)]
    let chunk_index = chunks.len() as u32;
    let chunk_id = Chunk::make_id(
        Source::Codex,
        &item.source_id,
        chunk_index,
        &item.source_config_id,
    );
    let project = item
        .project
        .clone()
        .or_else(|| meta.cwd.as_deref().and_then(project_from_cwd));

    chunks.push(Chunk {
        chunk_id,
        source: Source::Codex,
        project,
        source_id: item.source_id.clone(),
        facets: Default::default(),
        embedding_input_sha256: String::new(),
        source_config_id: item.source_config_id.clone(),
        chunk_index,
        ts: turn.ts,
        role: Some("exchange".to_string()),
        text: combined_text.clone(),
        sha256: Chunk::content_hash(&combined_text),
        links: Links::default(),
        extra: serde_json::json!({
            "session_id": meta.session_id,
            "cwd": meta.cwd,
            "originator": meta.originator,
            "cli_version": meta.cli_version,
        }),
    });
}

fn is_codex_rollout(path: &Path) -> bool {
    path.extension()
        .and_then(|x| x.to_str())
        .is_some_and(|x| x.eq_ignore_ascii_case("jsonl"))
        && path
            .file_name()
            .and_then(|x| x.to_str())
            .is_some_and(|x| x.starts_with("rollout-"))
}

fn update_session_meta(meta: &mut SessionMeta, v: &Value) {
    let payload = v.get("payload").unwrap_or(&Value::Null);
    meta.session_id = payload
        .get("id")
        .and_then(Value::as_str)
        .map(str::to_string)
        .or_else(|| meta.session_id.clone());
    meta.cwd = payload
        .get("cwd")
        .and_then(Value::as_str)
        .map(str::to_string)
        .or_else(|| meta.cwd.clone());
    meta.originator = payload
        .get("originator")
        .and_then(Value::as_str)
        .map(str::to_string)
        .or_else(|| meta.originator.clone());
    meta.cli_version = payload
        .get("cli_version")
        .and_then(Value::as_str)
        .map(str::to_string)
        .or_else(|| meta.cli_version.clone());
}

fn response_message(v: &Value) -> Option<(&str, String, Option<DateTime<Utc>>)> {
    if v.get("type").and_then(Value::as_str) != Some("response_item") {
        return None;
    }

    let payload = v.get("payload")?;
    if payload.get("type").and_then(Value::as_str) != Some("message") {
        return None;
    }

    let role = payload.get("role").and_then(Value::as_str)?;
    let text = extract_content_text(payload.get("content")?);
    let ts = v
        .get("timestamp")
        .and_then(Value::as_str)
        .and_then(parse_rfc3339_utc);
    Some((role, text, ts))
}

fn extract_content_text(content: &Value) -> String {
    match content {
        Value::String(s) => s.clone(),
        Value::Array(parts) => parts
            .iter()
            .filter_map(|part| {
                part.as_str()
                    .map(str::to_string)
                    .or_else(|| part.get("text").and_then(Value::as_str).map(str::to_string))
            })
            .collect::<Vec<_>>()
            .join("\n"),
        _ => String::new(),
    }
}

fn parse_rfc3339_utc(s: &str) -> Option<DateTime<Utc>> {
    DateTime::parse_from_rfc3339(s)
        .ok()
        .map(|dt| dt.with_timezone(&Utc))
}

fn project_from_cwd(cwd: &str) -> Option<String> {
    PathBuf::from(cwd)
        .file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .filter(|name| !name.is_empty())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::TempDir;

    fn cfg_with(root: &Path, project: Option<&str>) -> SourceConfig {
        SourceConfig {
            kind: SourceKind::Codex,
            graph_only: false,
            project: project.map(str::to_string),
            paths: vec![root.to_string_lossy().into_owned()],
            ignore: vec![],
            extensions: vec![],
            entity_type: None,
            edges: Vec::new(),
            id: None,
            source_config_id: "test-cfg".to_string(),
            facets: Default::default(),
        }
    }

    fn item_for(path: &Path, project: Option<&str>) -> SourceItem {
        SourceItem {
            source_id: path.file_name().unwrap().to_string_lossy().into_owned(),
            path: Some(path.to_path_buf()),
            project: project.map(str::to_string),
            bytes: None,
            ignore: Vec::new(),
            source_config_id: "test-cfg".to_string(),
        }
    }

    #[test]
    fn discover_finds_rollout_jsonl_only() {
        let tmp = TempDir::new().unwrap();
        let dir = tmp.path().join("2026").join("06").join("03");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("rollout-2026-06-03T00-00-00-a.jsonl"), "").unwrap();
        std::fs::write(dir.join("history.jsonl"), "").unwrap();
        std::fs::write(dir.join("rollout-2026-06-03T00-00-00-a.txt"), "").unwrap();

        let scanner = CodexScanner;
        let cfg = cfg_with(tmp.path(), Some("configured-project"));
        let items: Vec<_> = scanner.discover(&cfg).filter_map(Result::ok).collect();
        assert_eq!(items.len(), 1);
        assert_eq!(items[0].project.as_deref(), Some("configured-project"));
    }

    #[test]
    fn parse_groups_assistant_parts_and_derives_project_from_cwd() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("rollout-session.jsonl");
        let mut f = std::fs::File::create(&path).unwrap();
        writeln!(
            f,
            r#"{{"timestamp":"2026-06-03T10:00:00Z","type":"session_meta","payload":{{"id":"s1","cwd":"/Users/x/projects/ostk-recall","originator":"codex-tui","cli_version":"0.136.0"}}}}"#
        )
        .unwrap();
        writeln!(
            f,
            r#"{{"timestamp":"2026-06-03T10:00:01Z","type":"response_item","payload":{{"type":"message","role":"user","content":[{{"type":"input_text","text":"implement codex watches"}}]}}}}"#
        )
        .unwrap();
        writeln!(
            f,
            r#"{{"timestamp":"2026-06-03T10:00:02Z","type":"response_item","payload":{{"type":"message","role":"user","content":[{{"type":"input_text","text":"make it live"}}]}}}}"#
        )
        .unwrap();
        writeln!(
            f,
            r#"{{"timestamp":"2026-06-03T10:00:02Z","type":"event_msg","payload":{{"type":"user_message","message":"duplicate event"}}}}"#
        )
        .unwrap();
        writeln!(
            f,
            r#"{{"timestamp":"2026-06-03T10:00:03Z","type":"response_item","payload":{{"type":"function_call","name":"shell"}}}}"#
        )
        .unwrap();
        writeln!(
            f,
            r#"{{"timestamp":"2026-06-03T10:00:04Z","type":"response_item","payload":{{"type":"message","role":"assistant","content":[{{"type":"output_text","text":"first note"}}]}}}}"#
        )
        .unwrap();
        writeln!(
            f,
            r#"{{"timestamp":"2026-06-03T10:00:05Z","type":"response_item","payload":{{"type":"message","role":"assistant","content":[{{"type":"output_text","text":"final answer"}}]}}}}"#
        )
        .unwrap();

        let scanner = CodexScanner;
        let chunks = scanner.parse(item_for(&path, None)).unwrap();
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].source, Source::Codex);
        assert_eq!(chunks[0].project.as_deref(), Some("ostk-recall"));
        assert_eq!(chunks[0].role.as_deref(), Some("exchange"));
        assert!(
            chunks[0]
                .text
                .contains("### User\nimplement codex watches\n\nmake it live")
        );
        assert!(
            chunks[0]
                .text
                .contains("### Codex\nfirst note\n\nfinal answer")
        );
        assert!(!chunks[0].text.contains("duplicate event"));
        assert_eq!(chunks[0].extra["session_id"], "s1");
    }

    #[test]
    fn parse_skips_truncated_live_line() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("rollout-live.jsonl");
        let mut f = std::fs::File::create(&path).unwrap();
        writeln!(
            f,
            r#"{{"timestamp":"2026-06-03T10:00:01Z","type":"response_item","payload":{{"type":"message","role":"user","content":[{{"text":"hello"}}]}}}}"#
        )
        .unwrap();
        writeln!(
            f,
            r#"{{"timestamp":"2026-06-03T10:00:02Z","type":"response_item","payload":{{"type":"message","role":"assistant","content":[{{"text":"hi"}}]}}}}"#
        )
        .unwrap();
        write!(
            f,
            r#"{{"timestamp":"2026-06-03T10:00:03Z","type":"response_item""#
        )
        .unwrap();

        let scanner = CodexScanner;
        let chunks = scanner.parse(item_for(&path, Some("cfg-project"))).unwrap();
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].project.as_deref(), Some("cfg-project"));
    }
}
