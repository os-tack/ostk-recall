//! Threads scanner.
//!
//! Reads `.ostk/threads/*.md` — the attention substrate's user-facing
//! surface. Each thread file is a markdown body with a YAML front-matter
//! header carrying the attention identity (`handle`, `tension`,
//! `familiarity`, `evidence`, timestamps).
//!
//! # Discovery shape
//!
//! Threads are **flat per project**: only `.md` files immediately under
//! each configured path. We deliberately do NOT recurse — the directory
//! layout is the contract. Hidden entries (`.proposed/`, `.INDEX.md`)
//! and the `INDEX.md` navigation aid are skipped.
//!
//! # Chunking
//!
//! Exactly **one chunk per thread file** — threads are the unit of
//! attention identity, so splitting them would fragment the handle that
//! downstream attention pages key off. The body is the chunk text
//! (front-matter stripped); the parsed front-matter rides in `extra`.
//!
//! # Degraded chunks
//!
//! Parsing is defensive. Missing or malformed front-matter does NOT
//! crash the scan — instead the scanner emits a chunk with
//! `extra.parse_error` set, `text` = the full file, and synthetic
//! defaults so the corpus row stays consistent. The warning is logged.

use std::path::{Path, PathBuf};

use chrono::{DateTime, Utc};
use ostk_recall_core::{
    Chunk, Error, Links, Result, Scanner, Source, SourceConfig, SourceItem, SourceKind,
};
use serde::Deserialize;
use tracing::warn;

const DEFAULT_TENSION: &str = "active";

/// Scanner for `.ostk/threads/*.md` attention threads.
#[derive(Debug, Default)]
pub struct ThreadScanner;

impl Scanner for ThreadScanner {
    fn kind(&self) -> SourceKind {
        SourceKind::Thread
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
            flat_thread_entries(&root).into_iter().map(move |path| {
                let source_id = thread_handle(&path);
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

    /// Path-filtered override: O(|paths|) — only the listed files are
    /// considered, with the same flat-file gating applied. Used by the
    /// watcher's incremental scan path.
    fn discover_paths<'a>(
        &'a self,
        cfg: &'a SourceConfig,
        paths: &'a [PathBuf],
    ) -> Box<dyn Iterator<Item = Result<SourceItem>> + 'a> {
        let roots = match cfg.expanded_paths() {
            Ok(v) => v,
            Err(e) => return Box::new(std::iter::once(Err(e))),
        };
        let project = cfg.project.clone();

        let owned: Vec<PathBuf> = paths.to_vec();
        let iter = owned.into_iter().filter_map(move |path| {
            // Must sit directly under a configured root.
            let parent = path.parent()?;
            if !roots.iter().any(|r| parent == r.as_path()) {
                return None;
            }
            if !is_thread_file(&path) {
                return None;
            }
            if !path.is_file() {
                return None;
            }
            Some(Ok(SourceItem {
                source_id: thread_handle(&path),
                path: Some(path),
                project: project.clone(),
                bytes: None,
                ignore: Vec::new(),
                source_config_id: "test-cfg".to_string(),
            }))
        });
        Box::new(iter)
    }

    fn parse(&self, item: SourceItem) -> Result<Vec<Chunk>> {
        let path = item
            .path
            .as_ref()
            .ok_or_else(|| Error::Parse("threads: SourceItem.path missing".into()))?;
        let text = std::fs::read_to_string(path)?;
        let mtime = file_mtime_utc(path).ok();

        let abs_path = path
            .canonicalize()
            .unwrap_or_else(|_| path.clone())
            .to_string_lossy()
            .into_owned();

        let mut parsed = parse_thread(&text, &item.source_id);
        if let Some(err) = parsed.parse_error.as_deref() {
            warn!(
                handle = %item.source_id,
                path = %abs_path,
                error = %err,
                "threads: degraded chunk — front-matter parse failed"
            );
        }

        let body = std::mem::take(&mut parsed.body);
        let chunk_id = Chunk::make_id(Source::Thread, &item.source_id, 0, &item.source_config_id);
        let sha256 = Chunk::content_hash(&body);
        let links = Links {
            file_path: Some(abs_path),
            ..Links::default()
        };
        let source_id = item.source_id.clone();

        Ok(vec![Chunk {
            chunk_id,
            source: Source::Thread,
            project: item.project,
            source_id,
            source_config_id: item.source_config_id.clone(),
            facets: Default::default(),
            embedding_input_sha256: String::new(),
            chunk_index: 0,
            ts: mtime,
            role: None,
            text: body,
            sha256,
            links,
            extra: parsed.into_extra(item.source_id),
        }])
    }
}

/// Flat-list discovery: every `.md` directly under `root`, excluding the
/// index file and any hidden entry. Non-existent roots yield nothing.
fn flat_thread_entries(root: &Path) -> Vec<PathBuf> {
    let Ok(rd) = std::fs::read_dir(root) else {
        return Vec::new();
    };
    let mut out: Vec<PathBuf> = rd
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.is_file() && is_thread_file(p))
        .collect();
    out.sort();
    out
}

/// A path qualifies as a thread when: extension is `md` (case-insensitive),
/// the file name does not start with `.`, and the file name is not
/// `INDEX.md` (navigation aid, not a thread).
fn is_thread_file(path: &Path) -> bool {
    let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
        return false;
    };
    if name.starts_with('.') {
        return false;
    }
    if name == "INDEX.md" {
        return false;
    }
    path.extension()
        .and_then(|x| x.to_str())
        .is_some_and(|x| x.eq_ignore_ascii_case("md"))
}

/// Derive the thread handle from a path: the file name with the `.md`
/// extension stripped. Falls back to the lossy file name if the
/// extension is missing.
fn thread_handle(path: &Path) -> String {
    path.file_stem().map_or_else(
        || {
            path.file_name()
                .unwrap_or_default()
                .to_string_lossy()
                .into_owned()
        },
        |s| s.to_string_lossy().into_owned(),
    )
}

fn file_mtime_utc(path: &Path) -> std::io::Result<DateTime<Utc>> {
    let meta = std::fs::metadata(path)?;
    let sys = meta.modified()?;
    Ok(DateTime::<Utc>::from(sys))
}

/// Raw front-matter — every field optional so a partial file still
/// parses. Defaults are applied in `ParsedThread::from_raw`.
#[derive(Debug, Default, Deserialize)]
struct RawFrontMatter {
    handle: Option<String>,
    tension: Option<String>,
    familiarity: Option<i64>,
    evidence: Option<Vec<serde_yaml::Value>>,
    last_touched: Option<String>,
    created_at: Option<String>,
}

/// Normalised view of a parsed thread file.
struct ParsedThread {
    body: String,
    tension: String,
    familiarity: i64,
    evidence_count: usize,
    last_touched: Option<String>,
    created_at: Option<String>,
    handle_in_yaml: Option<String>,
    parse_error: Option<String>,
}

impl ParsedThread {
    fn from_raw(raw: RawFrontMatter, body: String) -> Self {
        let evidence_count = raw.evidence.as_ref().map_or(0, Vec::len);
        Self {
            body,
            tension: raw.tension.unwrap_or_else(|| DEFAULT_TENSION.to_string()),
            familiarity: raw.familiarity.unwrap_or(0),
            evidence_count,
            last_touched: raw.last_touched,
            created_at: raw.created_at,
            handle_in_yaml: raw.handle,
            parse_error: None,
        }
    }

    /// Degraded form: front-matter missing or unparseable.
    fn degraded(body: String, error: String) -> Self {
        Self {
            body,
            tension: DEFAULT_TENSION.to_string(),
            familiarity: 0,
            evidence_count: 0,
            last_touched: None,
            created_at: None,
            handle_in_yaml: None,
            parse_error: Some(error),
        }
    }

    fn into_extra(self, source_id: String) -> serde_json::Value {
        let mut obj = serde_json::Map::new();
        obj.insert("handle".into(), serde_json::Value::String(source_id));
        obj.insert("tension".into(), serde_json::Value::String(self.tension));
        obj.insert(
            "familiarity".into(),
            serde_json::Value::Number(self.familiarity.into()),
        );
        obj.insert(
            "evidence_count".into(),
            serde_json::Value::Number((self.evidence_count as u64).into()),
        );
        obj.insert(
            "last_touched".into(),
            self.last_touched
                .map_or(serde_json::Value::Null, serde_json::Value::String),
        );
        obj.insert(
            "created_at".into(),
            self.created_at
                .map_or(serde_json::Value::Null, serde_json::Value::String),
        );
        if let Some(h) = self.handle_in_yaml {
            obj.insert("handle_in_yaml".into(), serde_json::Value::String(h));
        }
        if let Some(err) = self.parse_error {
            obj.insert("parse_error".into(), serde_json::Value::String(err));
        }
        serde_json::Value::Object(obj)
    }
}

/// Split a thread file into front-matter and body, then parse the
/// front-matter with `serde_yaml`. Tolerates: missing front-matter
/// (no leading `---`), unterminated front-matter (no closing `---`),
/// and malformed YAML — each path emits a degraded `ParsedThread`
/// whose body is the full file.
fn parse_thread(text: &str, _handle: &str) -> ParsedThread {
    let Some((yaml, body)) = split_front_matter(text) else {
        return ParsedThread::degraded(text.to_string(), "no front-matter delimiters".into());
    };

    match serde_yaml::from_str::<RawFrontMatter>(yaml) {
        Ok(raw) => ParsedThread::from_raw(raw, body.to_string()),
        Err(e) => ParsedThread::degraded(text.to_string(), format!("yaml: {e}")),
    }
}

/// Returns `(yaml_block, body)` if the text starts with a `---` line
/// and has a closing `---` line. Otherwise `None`.
///
/// The YAML block excludes the delimiter lines themselves. The body is
/// the remainder after the closing delimiter, with one leading newline
/// stripped if present.
fn split_front_matter(text: &str) -> Option<(&str, &str)> {
    let stripped = text.strip_prefix('\u{feff}').unwrap_or(text);
    let after_open = stripped
        .strip_prefix("---\n")
        .or_else(|| stripped.strip_prefix("---\r\n"))?;

    // Find a closing `---` line (start-of-line, possibly followed by \r\n).
    let bytes = after_open.as_bytes();
    let mut search_from = 0;
    loop {
        let rel = after_open[search_from..].find("\n---")?;
        let abs = search_from + rel + 1; // position of '-' of closing delim
        // The closing must be a full line: either end of input, or
        // followed by `\n` / `\r\n`.
        let end = abs + 3;
        let tail_ok = end == bytes.len()
            || bytes.get(end) == Some(&b'\n')
            || (bytes.get(end) == Some(&b'\r') && bytes.get(end + 1) == Some(&b'\n'));
        if tail_ok {
            let yaml = &after_open[..abs.saturating_sub(1)]; // drop trailing '\n'
            let after_close = if end == bytes.len() {
                ""
            } else if bytes.get(end) == Some(&b'\r') {
                &after_open[end + 2..]
            } else {
                &after_open[end + 1..]
            };
            return Some((yaml, after_close));
        }
        search_from = abs + 3;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::TempDir;

    fn cfg_with(root: &Path, project: &str) -> SourceConfig {
        SourceConfig {
            kind: SourceKind::Thread,
            project: Some(project.into()),
            paths: vec![root.to_string_lossy().into_owned()],
            ignore: vec![],
            extensions: vec![],
            id: None,
            source_config_id: "test-cfg".to_string(),
            facets: Default::default(),
        }
        }

    fn write_thread(dir: &Path, name: &str, body: &str) -> PathBuf {
        let path = dir.join(name);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).unwrap();
        }
        let mut f = std::fs::File::create(&path).unwrap();
        f.write_all(body.as_bytes()).unwrap();
        path
    }

    const COMPLETE: &str = "---\n\
handle: hoberman-thread-primitive\n\
tension: active\n\
created_at: 2026-05-22T22:00:00Z\n\
last_touched: 2026-05-23T15:00:00Z\n\
familiarity: 1\n\
evidence:\n  - \"needle:→1872 (epic)\"\n  - \"doc:foo.md#bar\"\n\
---\n\n# hoberman-thread-primitive\n\nBody text.\n";

    #[test]
    fn discovers_flat_md_files_only() {
        let tmp = TempDir::new().unwrap();
        write_thread(tmp.path(), "a.md", "---\nhandle: a\n---\nbody");
        write_thread(tmp.path(), "b.md", "---\nhandle: b\n---\nbody");
        write_thread(tmp.path(), "sub/c.md", "---\nhandle: c\n---\nbody");
        write_thread(tmp.path(), ".proposed/d.md", "---\nhandle: d\n---\nbody");
        write_thread(tmp.path(), "INDEX.md", "# index");
        // Also a hidden top-level file with a thread-like name.
        write_thread(tmp.path(), ".hidden.md", "---\nhandle: h\n---\nbody");

        let scanner = ThreadScanner;
        let cfg = cfg_with(tmp.path(), "p");
        let items: Vec<_> = scanner.discover(&cfg).filter_map(Result::ok).collect();
        let ids: Vec<&str> = items.iter().map(|i| i.source_id.as_str()).collect();
        assert_eq!(items.len(), 2, "got: {ids:?}");
        assert!(ids.contains(&"a"));
        assert!(ids.contains(&"b"));
    }

    #[test]
    fn parses_complete_front_matter() {
        let tmp = TempDir::new().unwrap();
        let path = write_thread(tmp.path(), "hoberman-thread-primitive.md", COMPLETE);

        let scanner = ThreadScanner;
        let cfg = cfg_with(tmp.path(), "p");
        let item = scanner.discover(&cfg).next().unwrap().unwrap();
        assert_eq!(item.source_id, "hoberman-thread-primitive");
        assert_eq!(item.path.as_deref(), Some(path.as_path()));

        let chunks = scanner.parse(item).unwrap();
        assert_eq!(chunks.len(), 1);
        let c = &chunks[0];
        assert_eq!(c.source, Source::Thread);
        assert_eq!(c.source_id, "hoberman-thread-primitive");
        assert_eq!(c.chunk_index, 0);

        let extra = c.extra.as_object().unwrap();
        assert_eq!(extra["handle"], "hoberman-thread-primitive");
        assert_eq!(extra["tension"], "active");
        assert_eq!(extra["familiarity"], 1);
        assert_eq!(extra["evidence_count"], 2);
        assert_eq!(extra["created_at"], "2026-05-22T22:00:00Z");
        assert_eq!(extra["last_touched"], "2026-05-23T15:00:00Z");
        assert!(extra.get("parse_error").is_none());
    }

    #[test]
    fn tolerates_missing_optional_fields() {
        let tmp = TempDir::new().unwrap();
        write_thread(
            tmp.path(),
            "minimal.md",
            "---\nhandle: minimal\n---\nbody only\n",
        );

        let scanner = ThreadScanner;
        let cfg = cfg_with(tmp.path(), "p");
        let item = scanner.discover(&cfg).next().unwrap().unwrap();
        let chunks = scanner.parse(item).unwrap();
        let extra = chunks[0].extra.as_object().unwrap();
        assert_eq!(extra["tension"], "active");
        assert_eq!(extra["familiarity"], 0);
        assert_eq!(extra["evidence_count"], 0);
        assert!(extra["last_touched"].is_null());
        assert!(extra["created_at"].is_null());
        assert!(extra.get("parse_error").is_none());
    }

    #[test]
    fn tolerates_no_front_matter() {
        let tmp = TempDir::new().unwrap();
        let body = "# just a heading\n\nNo front matter at all.\n";
        write_thread(tmp.path(), "plain.md", body);

        let scanner = ThreadScanner;
        let cfg = cfg_with(tmp.path(), "p");
        let item = scanner.discover(&cfg).next().unwrap().unwrap();
        let chunks = scanner.parse(item).unwrap();
        let c = &chunks[0];
        assert_eq!(c.text, body, "degraded chunk text = full file");
        let extra = c.extra.as_object().unwrap();
        assert!(extra.get("parse_error").is_some());
        assert_eq!(extra["tension"], "active");
        assert_eq!(extra["familiarity"], 0);
    }

    #[test]
    fn tolerates_malformed_yaml() {
        let tmp = TempDir::new().unwrap();
        // `evidence: [unterminated` is not valid YAML.
        let body = "---\nhandle: bad\nevidence: [unterminated\n---\nbody\n";
        write_thread(tmp.path(), "bad.md", body);

        let scanner = ThreadScanner;
        let cfg = cfg_with(tmp.path(), "p");
        let item = scanner.discover(&cfg).next().unwrap().unwrap();
        let chunks = scanner.parse(item).expect("must not crash");
        assert_eq!(chunks.len(), 1);
        let extra = chunks[0].extra.as_object().unwrap();
        assert!(extra.get("parse_error").is_some());
        // Degraded: text is the full file (front-matter not stripped).
        assert!(chunks[0].text.contains("---"));
    }

    #[test]
    fn parse_strips_front_matter_from_body() {
        let tmp = TempDir::new().unwrap();
        write_thread(tmp.path(), "thread.md", COMPLETE);
        let scanner = ThreadScanner;
        let cfg = cfg_with(tmp.path(), "p");
        let item = scanner.discover(&cfg).next().unwrap().unwrap();
        let chunks = scanner.parse(item).unwrap();
        assert!(
            !chunks[0].text.contains("---"),
            "body must not contain front-matter delimiter; got:\n{}",
            chunks[0].text
        );
        assert!(chunks[0].text.contains("# hoberman-thread-primitive"));
        assert!(chunks[0].text.contains("Body text."));
    }

    #[test]
    fn extra_contains_evidence_count() {
        let tmp = TempDir::new().unwrap();
        let body = "---\nhandle: e\nevidence:\n  - one\n  - two\n  - three\n---\nbody\n";
        write_thread(tmp.path(), "e.md", body);

        let scanner = ThreadScanner;
        let cfg = cfg_with(tmp.path(), "p");
        let item = scanner.discover(&cfg).next().unwrap().unwrap();
        let chunks = scanner.parse(item).unwrap();
        assert_eq!(chunks[0].extra["evidence_count"], 3);
    }

    #[test]
    fn discover_paths_short_circuits() {
        let tmp = TempDir::new().unwrap();
        write_thread(tmp.path(), "wanted.md", "---\nhandle: wanted\n---\nbody\n");
        // Decoy files that a full walk would yield; we ask for only
        // `wanted.md` and they must NOT appear in the result.
        write_thread(
            tmp.path(),
            "other-1.md",
            "---\nhandle: other-1\n---\nbody\n",
        );
        write_thread(
            tmp.path(),
            "other-2.md",
            "---\nhandle: other-2\n---\nbody\n",
        );

        let scanner = ThreadScanner;
        let cfg = cfg_with(tmp.path(), "p");
        let wanted = vec![tmp.path().join("wanted.md")];
        let items: Vec<_> = scanner
            .discover_paths(&cfg, &wanted)
            .filter_map(Result::ok)
            .collect();
        assert_eq!(items.len(), 1);
        assert_eq!(items[0].source_id, "wanted");
    }

    #[test]
    fn discover_paths_rejects_subdir_and_hidden() {
        let tmp = TempDir::new().unwrap();
        write_thread(tmp.path(), "sub/c.md", "---\nhandle: c\n---\nbody\n");
        write_thread(tmp.path(), ".hidden.md", "---\nhandle: h\n---\nbody\n");
        write_thread(tmp.path(), "INDEX.md", "# x");

        let scanner = ThreadScanner;
        let cfg = cfg_with(tmp.path(), "p");
        let paths = vec![
            tmp.path().join("sub/c.md"),
            tmp.path().join(".hidden.md"),
            tmp.path().join("INDEX.md"),
        ];
        let items: Vec<_> = scanner
            .discover_paths(&cfg, &paths)
            .filter_map(Result::ok)
            .collect();
        assert!(items.is_empty(), "all three should be filtered out");
    }

    #[test]
    fn idempotent_rescan() {
        let tmp = TempDir::new().unwrap();
        write_thread(tmp.path(), "a.md", COMPLETE);
        write_thread(tmp.path(), "b.md", "---\nhandle: b\n---\nbody\n");

        let scanner = ThreadScanner;
        let cfg = cfg_with(tmp.path(), "p");

        let run = || -> Vec<Chunk> {
            let mut items: Vec<_> = scanner.discover(&cfg).filter_map(Result::ok).collect();
            items.sort_by(|x, y| x.source_id.cmp(&y.source_id));
            items
                .into_iter()
                .flat_map(|it| scanner.parse(it).unwrap())
                .collect()
        };

        let first = run();
        let second = run();
        assert_eq!(first.len(), second.len());
        assert_eq!(first.len(), 2);
        for (a, b) in first.iter().zip(second.iter()) {
            assert_eq!(a.chunk_id, b.chunk_id, "chunk_id must be stable");
            assert_eq!(a.sha256, b.sha256);
            assert_eq!(a.text, b.text);
            assert_eq!(a.extra, b.extra);
        }
    }

    #[test]
    fn discover_walks_nothing_when_root_missing() {
        let tmp = TempDir::new().unwrap();
        let missing = tmp.path().join("does-not-exist");
        let scanner = ThreadScanner;
        let cfg = cfg_with(&missing, "p");
        let items: Vec<_> = scanner.discover(&cfg).filter_map(Result::ok).collect();
        assert!(items.is_empty());
    }

    #[test]
    fn split_front_matter_handles_crlf() {
        let text = "---\r\nhandle: x\r\n---\r\nbody\r\n";
        let (yaml, body) = split_front_matter(text).expect("must split");
        assert!(yaml.contains("handle: x"));
        assert!(body.starts_with("body"));
    }

    #[test]
    fn split_front_matter_returns_none_when_unterminated() {
        let text = "---\nhandle: x\nno closing delimiter\n";
        assert!(split_front_matter(text).is_none());
    }
}
