//! Claude Code session log scanner.
//!
//! Walks `~/.claude/projects/<slug>/*.jsonl` (or any directory configured
//! under `paths`). Each file is a session log of Anthropic Messages wrapped
//! in `{type, message, timestamp}` lines.
//!
//! One chunk per **exchange** = assistant response + preceding user turn.
//! Tool use and tool result blocks are concatenated into the assistant
//! body. See [`crate::anthropic_session`] for the shared parse helper.
//!
//! # `source_id` / project derivation
//!
//! * `source_id` = session filename (e.g. `"abc-1234.jsonl"`).
//! * `project`   = [`SourceConfig::project`] if set; otherwise derived
//!   from the immediate parent dir name. `~/.claude/projects` slugs look
//!   like `-Users-scottmeyer-projects-haystack` — we strip the leading
//!   `-` and keep the basename (`haystack`).

use std::path::{Path, PathBuf};

use chrono::{DateTime, Utc};
use ostk_recall_core::{
    Chunk, Error, Result, Scanner, Source, SourceConfig, SourceItem, SourceKind,
};
use walkdir::WalkDir;

use crate::anthropic_session::parse_session_file;

/// Scanner for Claude Code session logs.
#[derive(Debug, Default)]
pub struct ClaudeCodeScanner;

impl Scanner for ClaudeCodeScanner {
    fn kind(&self) -> SourceKind {
        SourceKind::ClaudeCode
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
                .filter(|e| {
                    e.path()
                        .extension()
                        .and_then(|x| x.to_str())
                        .is_some_and(|x| x.eq_ignore_ascii_case("jsonl"))
                })
                .map(move |entry| {
                    let path = entry.path().to_path_buf();
                    let source_id = path
                        .file_name()
                        .map(|n| n.to_string_lossy().into_owned())
                        .unwrap_or_default();
                    let project = project.clone().or_else(|| project_from_parent_dir(&path));
                    Ok(SourceItem {
                        source_id,
                        path: Some(path),
                        project,
                        bytes: None,
                        ignore: Vec::new(),
                    })
                })
        });
        Box::new(iter)
    }

    fn parse(&self, item: SourceItem) -> Result<Vec<Chunk>> {
        let path = item
            .path
            .as_ref()
            .ok_or_else(|| Error::Parse("claude_code: SourceItem.path missing".into()))?;
        let mtime = file_mtime_utc(path).ok();
        parse_session_file(
            path,
            Source::ClaudeCode,
            &item.source_id,
            item.project.as_deref(),
            mtime,
        )
    }
}

/// Given `/some/root/-Users-scottmeyer-projects-haystack/abc.jsonl`,
/// return `"haystack"`. For non-slug parents, return the dir name verbatim.
fn project_from_parent_dir(path: &Path) -> Option<String> {
    let parent = path.parent()?;
    let name = parent.file_name()?.to_string_lossy().into_owned();
    let trimmed = name.strip_prefix('-').unwrap_or(&name);
    // Replace remaining `-` with `/` and take the basename.
    let joined = trimmed.replace('-', "/");
    Some(
        PathBuf::from(joined)
            .file_name()
            .map_or_else(|| name.clone(), |n| n.to_string_lossy().into_owned()),
    )
}

fn file_mtime_utc(path: &Path) -> std::io::Result<DateTime<Utc>> {
    let meta = std::fs::metadata(path)?;
    let sys = meta.modified()?;
    Ok(DateTime::<Utc>::from(sys))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::TempDir;

    fn cfg_with(root: &Path, project: Option<&str>) -> SourceConfig {
        SourceConfig {
            kind: SourceKind::ClaudeCode,
            project: project.map(str::to_string),
            paths: vec![root.to_string_lossy().into_owned()],
            ignore: vec![],
            extensions: vec![],
        }
    }

    #[test]
    fn project_derivation() {
        let p = Path::new("/any/root/-Users-scottmeyer-projects-haystack/abc.jsonl");
        assert_eq!(project_from_parent_dir(p).as_deref(), Some("haystack"));
    }

    #[test]
    fn project_derivation_non_slug() {
        let p = Path::new("/tmp/sessions/abc.jsonl");
        assert_eq!(project_from_parent_dir(p).as_deref(), Some("sessions"));
    }

    #[test]
    fn discover_finds_jsonl() {
        let tmp = TempDir::new().unwrap();
        let sess = tmp.path().join("-Users-x-projects-foo");
        std::fs::create_dir_all(&sess).unwrap();
        std::fs::write(sess.join("abc.jsonl"), "").unwrap();
        std::fs::write(sess.join("def.jsonl"), "").unwrap();
        std::fs::write(sess.join("README.md"), "nope").unwrap();

        let scanner = ClaudeCodeScanner;
        let cfg = cfg_with(tmp.path(), None);
        let items: Vec<_> = scanner.discover(&cfg).filter_map(Result::ok).collect();
        assert_eq!(items.len(), 2);
        for it in &items {
            assert_eq!(it.project.as_deref(), Some("foo"));
        }
    }

    fn write_simple_session(path: &Path) {
        let mut f = std::fs::File::create(path).unwrap();
        writeln!(f, r#"{{"type":"file-history-snapshot","messageId":"x"}}"#).unwrap();
        writeln!(
            f,
            r#"{{"type":"user","message":{{"role":"user","content":"hello"}},"timestamp":"2026-04-17T10:00:00Z"}}"#
        )
        .unwrap();
        writeln!(
            f,
            r#"{{"type":"assistant","message":{{"role":"assistant","content":[{{"type":"text","text":"hi back"}}]}},"timestamp":"2026-04-17T10:00:01Z"}}"#
        )
        .unwrap();
    }

    fn write_session_with_tool_use(path: &Path) {
        let mut f = std::fs::File::create(path).unwrap();
        writeln!(
            f,
            r#"{{"type":"user","message":{{"role":"user","content":"do something"}},"timestamp":"2026-04-17T11:00:00Z"}}"#
        )
        .unwrap();
        writeln!(
            f,
            r#"{{"type":"assistant","message":{{"role":"assistant","content":[{{"type":"text","text":"start"}},{{"type":"tool_use","name":"read","input":{{"path":"x"}}}},{{"type":"text","text":"end"}}]}},"timestamp":"2026-04-17T11:00:01Z"}}"#
        )
        .unwrap();
    }

    #[test]
    fn parse_two_sessions_with_tool_use() {
        let tmp = TempDir::new().unwrap();
        let dir = tmp.path().join("-Users-x-projects-foo");
        std::fs::create_dir_all(&dir).unwrap();
        let s1 = dir.join("11111111.jsonl");
        let s2 = dir.join("22222222.jsonl");
        write_simple_session(&s1);
        write_session_with_tool_use(&s2);

        let scanner = ClaudeCodeScanner;
        let cfg = cfg_with(tmp.path(), None);
        let mut all_chunks = Vec::new();
        for item in scanner.discover(&cfg).filter_map(Result::ok) {
            all_chunks.extend(scanner.parse(item).unwrap());
        }
        assert_eq!(all_chunks.len(), 2);
        // Both sessions should yield one chunk each.
        let with_tool = all_chunks
            .iter()
            .find(|c| c.text.contains("tool_use read"))
            .expect("tool_use chunk present");
        assert!(with_tool.text.contains("start"));
        assert!(with_tool.text.contains("end"));
        assert_eq!(with_tool.project.as_deref(), Some("foo"));
    }
}
