//! Claude.ai conversation export scanner.
//!
//! Processes `claude-data-export-*.zip` files (or whatever `paths` globs
//! match). Opens each archive, extracts `conversations.json`, and emits one
//! chunk per message.
//!
//! Shape (Claude.ai, 2024/2025 export):
//! ```json
//! [
//!   {
//!     "uuid": "conv-uuid",
//!     "name": "conversation title",
//!     "chat_messages": [
//!       {"uuid": "msg-uuid", "text": "...", "sender": "human"|"assistant",
//!        "created_at": "2024-…"}
//!     ]
//!   }
//! ]
//! ```
//!
//! TODO(phase-d): Support `ChatGPT` `conversations.json` (different schema —
//! `mapping` tree of messages). Not in scope for Phase C.

use std::fs::File;
use std::io::{BufReader, Read};
use std::path::{Path, PathBuf};

use chrono::{DateTime, Utc};
use globset::{Glob, GlobSetBuilder};
use ostk_recall_core::{
    Chunk, Error, Links, Result, Scanner, Source, SourceConfig, SourceItem, SourceKind,
};
use serde::Deserialize;
use walkdir::WalkDir;

/// Scanner for Claude.ai export zips.
#[derive(Debug, Default)]
pub struct ZipExportScanner;

impl Scanner for ZipExportScanner {
    fn kind(&self) -> SourceKind {
        SourceKind::ZipExport
    }

    fn discover<'a>(
        &'a self,
        cfg: &'a SourceConfig,
    ) -> Box<dyn Iterator<Item = Result<SourceItem>> + 'a> {
        let plans: Vec<GlobPlan> = match plan_globs(&cfg.paths) {
            Ok(v) => v,
            Err(e) => return Box::new(std::iter::once(Err(e))),
        };
        let project = cfg.project.clone();
        let iter = plans.into_iter().flat_map(move |plan| {
            let project = project.clone();
            let matcher = plan.matcher.clone();
            WalkDir::new(&plan.root)
                .follow_links(false)
                .into_iter()
                .filter_map(std::result::Result::ok)
                .filter(|e| e.file_type().is_file())
                .filter(move |e| matcher.is_match(e.path()))
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
                    })
                })
        });
        Box::new(iter)
    }

    fn parse(&self, item: SourceItem) -> Result<Vec<Chunk>> {
        let path = item
            .path
            .as_ref()
            .ok_or_else(|| Error::Parse("zip_export: SourceItem.path missing".into()))?;
        let zip_name = path
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_default();
        let abs_path = path
            .canonicalize()
            .unwrap_or_else(|_| path.clone())
            .to_string_lossy()
            .into_owned();

        let body = read_conversations_json(path)?;
        let conversations: Vec<Conversation> = serde_json::from_str(&body).map_err(|e| {
            Error::Parse(format!(
                "zip_export: bad conversations.json in {}: {e}",
                path.display()
            ))
        })?;

        let mut chunks: Vec<Chunk> = Vec::new();
        let mut chunk_index: u32 = 0;
        for conv in conversations {
            let conv_uuid = conv.uuid.clone();
            let conv_name = conv.name.clone().unwrap_or_default();
            let mut prev_chunk_id: Option<String> = None;
            for msg in conv.chat_messages {
                let role = match msg.sender.as_deref() {
                    Some("human") => "user",
                    Some("assistant") => "assistant",
                    _ => continue,
                };
                let text = msg.text.clone().unwrap_or_default();
                if text.trim().is_empty() {
                    continue;
                }
                let source_id = format!("{zip_name}:{conv_uuid}:{}", msg.uuid);
                let chunk_id = Chunk::make_id(Source::ZipExport, &source_id, chunk_index);
                let sha256 = Chunk::content_hash(&text);
                let links = Links {
                    file_path: Some(abs_path.clone()),
                    parent_ids: prev_chunk_id.clone().into_iter().collect(),
                    ..Links::default()
                };
                let extra = serde_json::json!({
                    "conversation_name": conv_name,
                });
                chunks.push(Chunk {
                    chunk_id: chunk_id.clone(),
                    source: Source::ZipExport,
                    project: item.project.clone(),
                    source_id,
                    chunk_index,
                    ts: msg.created_at,
                    role: Some(role.into()),
                    text,
                    sha256,
                    links,
                    extra,
                });
                prev_chunk_id = Some(chunk_id);
                chunk_index = chunk_index.saturating_add(1);
            }
        }
        Ok(chunks)
    }
}

fn read_conversations_json(zip_path: &Path) -> Result<String> {
    let f = File::open(zip_path)?;
    let reader = BufReader::new(f);
    let mut archive = zip::ZipArchive::new(reader)
        .map_err(|e| Error::Parse(format!("zip open {}: {e}", zip_path.display())))?;
    // Find a file named `conversations.json` anywhere in the archive.
    for i in 0..archive.len() {
        let mut entry = archive
            .by_index(i)
            .map_err(|e| Error::Parse(format!("zip entry {i}: {e}")))?;
        if entry.name().ends_with("conversations.json") {
            let mut body = String::new();
            entry
                .read_to_string(&mut body)
                .map_err(|e| Error::Parse(format!("read conversations.json: {e}")))?;
            return Ok(body);
        }
    }
    Err(Error::Parse(format!(
        "zip_export: no conversations.json in {}",
        zip_path.display()
    )))
}

#[derive(Debug, Deserialize)]
struct Conversation {
    uuid: String,
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    chat_messages: Vec<Message>,
}

#[derive(Debug, Deserialize)]
struct Message {
    uuid: String,
    #[serde(default)]
    text: Option<String>,
    #[serde(default)]
    sender: Option<String>,
    #[serde(default)]
    created_at: Option<DateTime<Utc>>,
}

struct GlobPlan {
    root: PathBuf,
    matcher: globset::GlobSet,
}

fn plan_globs(raw: &[String]) -> Result<Vec<GlobPlan>> {
    let mut out = Vec::with_capacity(raw.len());
    for r in raw {
        let expanded = shellexpand::full(r)
            .map(std::borrow::Cow::into_owned)
            .map_err(|e| Error::PathExpand(e.to_string()))?;
        let (root, _) = split_glob(&expanded);
        let glob = Glob::new(&expanded).map_err(|e| Error::Scan(format!("bad glob {r}: {e}")))?;
        let mut b = GlobSetBuilder::new();
        b.add(glob);
        let set = b
            .build()
            .map_err(|e| Error::Scan(format!("build {r}: {e}")))?;
        out.push(GlobPlan { root, matcher: set });
    }
    Ok(out)
}

fn split_glob(raw: &str) -> (PathBuf, String) {
    let path = Path::new(raw);
    let mut root = PathBuf::new();
    let mut saw_meta = false;
    for comp in path.components() {
        let part = comp.as_os_str().to_string_lossy();
        if !saw_meta && part.chars().any(|c| matches!(c, '*' | '?' | '[' | '{')) {
            saw_meta = true;
        }
        if !saw_meta {
            root.push(comp);
        }
    }
    if root.as_os_str().is_empty() {
        root = PathBuf::from(".");
    }
    (root, raw.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::TempDir;
    use zip::write::SimpleFileOptions;

    fn build_fixture_zip(path: &Path) {
        let f = File::create(path).unwrap();
        let mut zw = zip::ZipWriter::new(f);
        let opts =
            SimpleFileOptions::default().compression_method(zip::CompressionMethod::Deflated);
        let body = serde_json::json!([
            {
                "uuid": "conv-1",
                "name": "First convo",
                "chat_messages": [
                    {"uuid": "m1", "text": "hello", "sender": "human",
                     "created_at": "2024-01-01T00:00:00Z"},
                    {"uuid": "m2", "text": "hi there", "sender": "assistant",
                     "created_at": "2024-01-01T00:00:01Z"},
                    {"uuid": "m3", "text": "thanks", "sender": "human",
                     "created_at": "2024-01-01T00:00:02Z"},
                ]
            },
            {
                "uuid": "conv-2",
                "name": "Second convo",
                "chat_messages": [
                    {"uuid": "n1", "text": "question", "sender": "human",
                     "created_at": "2024-02-01T00:00:00Z"},
                    {"uuid": "n2", "text": "answer", "sender": "assistant",
                     "created_at": "2024-02-01T00:00:01Z"},
                    {"uuid": "n3", "text": "ok", "sender": "human",
                     "created_at": "2024-02-01T00:00:02Z"},
                ]
            }
        ]);
        zw.start_file("conversations.json", opts).unwrap();
        zw.write_all(body.to_string().as_bytes()).unwrap();
        zw.finish().unwrap();
    }

    fn cfg_glob(pat: &str) -> SourceConfig {
        SourceConfig {
            kind: SourceKind::ZipExport,
            project: Some("export".into()),
            paths: vec![pat.into()],
            extensions: vec![],
        }
    }

    #[test]
    fn discover_matches_zip() {
        let tmp = TempDir::new().unwrap();
        let zip_path = tmp.path().join("claude-data-export-2024.zip");
        build_fixture_zip(&zip_path);
        let pat = format!("{}/claude-data-export-*.zip", tmp.path().display());
        let scanner = ZipExportScanner;
        let cfg = cfg_glob(&pat);
        let items: Vec<_> = scanner.discover(&cfg).filter_map(Result::ok).collect();
        assert_eq!(items.len(), 1);
    }

    #[test]
    fn parse_yields_one_chunk_per_message() {
        let tmp = TempDir::new().unwrap();
        let zip_path = tmp.path().join("claude-data-export-2024.zip");
        build_fixture_zip(&zip_path);
        let pat = format!("{}/claude-data-export-*.zip", tmp.path().display());
        let scanner = ZipExportScanner;
        let cfg = cfg_glob(&pat);
        let item = scanner.discover(&cfg).next().unwrap().unwrap();
        let chunks = scanner.parse(item).unwrap();
        assert_eq!(chunks.len(), 6, "2 conversations x 3 messages");
        // Role mapping.
        assert_eq!(chunks[0].role.as_deref(), Some("user"));
        assert_eq!(chunks[1].role.as_deref(), Some("assistant"));
        assert_eq!(chunks[2].role.as_deref(), Some("user"));
        // Conversation name carried in `extra`.
        let extra = &chunks[0].extra;
        assert_eq!(
            extra.get("conversation_name").and_then(|v| v.as_str()),
            Some("First convo")
        );
        // Parent linkage within conversation 1 (messages 0->1->2).
        assert!(chunks[0].links.parent_ids.is_empty());
        assert_eq!(chunks[1].links.parent_ids, vec![chunks[0].chunk_id.clone()]);
        // source_id = zipname:conv:msg
        assert!(
            chunks[0]
                .source_id
                .starts_with("claude-data-export-2024.zip:conv-1:")
        );
    }
}
