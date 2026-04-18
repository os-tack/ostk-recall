//! Composite scanner for haystack-style `.ostk/` project roots.
//!
//! One source config (`kind = "ostk_project"`, `paths = ["/path/to/proj"]`)
//! expands into chunks of seven distinct [`Source`] variants:
//!
//! | subsystem            | source                    | input path                    |
//! |----------------------|---------------------------|-------------------------------|
//! | decisions            | `OstkDecision`            | `.ostk/decisions.jsonl`       |
//! | needles              | `OstkNeedle`              | `.ostk/needles/issues.jsonl`  |
//! | audit (significant)  | `OstkAuditSignificant`    | `.ostk/audit.jsonl` (filter)  |
//! | conversations        | `OstkConversation`        | `.ostk/conversations/*.jsonl` |
//! | sessions             | `OstkSession`             | `.ostk/sessions/*.jsonl`      |
//! | memory pages         | `OstkMemory`              | `.ostk/memory/pages.jsonl`    |
//! | spec + draft docs    | `OstkSpec`                | `docs/{spec,draft}/**/*.md`   |
//! | source code          | `Code`                    | `src/**/*.rs` + friends       |
//!
//! The audit firehose (every row, significant or not) is streamed into an
//! [`EventsDb`] passed via [`OstkProjectScanner::with_events_db`]. If no
//! events sink is attached, the audit-file pass still emits significant
//! chunks but skips the `DuckDB` ingest.
//!
//! Chunking decisions reuse helpers where possible:
//! * spec/draft → [`crate::markdown::split_markdown`]
//! * code       → [`crate::code::walk_and_window`]
//! * sessions   → [`crate::anthropic_session::parse_session_file`]

use std::path::{Path, PathBuf};
use std::sync::Arc;

use chrono::{DateTime, Utc};
use ostk_recall_core::{
    Chunk, Error, Links, Result, Scanner, Source, SourceConfig, SourceItem, SourceKind,
};
use ostk_recall_store::{AuditEventRow, EventsDb};
use serde::Deserialize;

use crate::anthropic_session::parse_session_file;
use crate::code::walk_and_window;
use crate::markdown::split_markdown;
use crate::walk::walk_filtered;

/// Code extensions the composite scanner sweeps from each project's `src/`.
const CODE_EXTENSIONS: &[&str] = &["rs", "py", "ts", "tsx", "js", "go", "md"];

/// Batch size used when streaming audit rows into `DuckDB`.
const AUDIT_BATCH: usize = 500;

/// Composite `.ostk/` project scanner.
///
/// Optional `events` sink receives the full audit firehose (not just
/// significant events). When no sink is attached — e.g. during unit tests
/// that don't care about the duckdb side-table — the scanner silently
/// drops non-significant rows.
#[derive(Default)]
pub struct OstkProjectScanner {
    events: Option<Arc<EventsDb>>,
}

impl std::fmt::Debug for OstkProjectScanner {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OstkProjectScanner")
            .field("events", &self.events.is_some())
            .finish()
    }
}

impl OstkProjectScanner {
    #[must_use]
    pub const fn new() -> Self {
        Self { events: None }
    }

    #[must_use]
    pub fn with_events_db(mut self, events: Arc<EventsDb>) -> Self {
        self.events = Some(events);
        self
    }
}

impl Scanner for OstkProjectScanner {
    fn kind(&self) -> SourceKind {
        SourceKind::OstkProject
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
        let ignore_patterns = cfg.ignore.clone();
        let iter = roots.into_iter().map(move |root| {
            let source_id = root.to_string_lossy().into_owned();
            Ok(SourceItem {
                source_id,
                path: Some(root),
                project: project.clone(),
                bytes: None,
                ignore: ignore_patterns.clone(),
            })
        });
        Box::new(iter)
    }

    fn parse(&self, item: SourceItem) -> Result<Vec<Chunk>> {
        let ignore_patterns = item.ignore.clone();
        let root = item
            .path
            .ok_or_else(|| Error::Parse("ostk_project: SourceItem.path missing".into()))?;
        let project = item
            .project
            .or_else(|| root.file_name().map(|n| n.to_string_lossy().into_owned()));

        let mut out: Vec<Chunk> = Vec::new();

        // 1. decisions
        out.extend(scan_decisions(&root, project.as_deref())?);
        // 2. needles
        out.extend(scan_needles(&root, project.as_deref())?);
        // 3. audit (firehose → events.duckdb, significant → chunks)
        out.extend(scan_audit(
            &root,
            project.as_deref(),
            self.events.as_deref(),
        )?);
        // 4. conversations
        out.extend(scan_conversations(&root, project.as_deref())?);
        // 5. sessions
        out.extend(scan_sessions(&root, project.as_deref())?);
        // 6. memory pages
        out.extend(scan_memory(&root, project.as_deref())?);
        // 7. spec + draft markdown — honors per-source ignore patterns
        out.extend(scan_specs(&root, project.as_deref(), &ignore_patterns));
        // 8. source code — honors per-source ignore patterns
        out.extend(scan_code(&root, project.as_deref(), &ignore_patterns));

        Ok(out)
    }
}

// ────────────────────────────── decisions ──────────────────────────────

#[derive(Debug, Deserialize)]
struct DecisionRow {
    key: String,
    #[serde(default)]
    value: serde_json::Value,
    #[serde(default)]
    reason: Option<String>,
    #[serde(default)]
    timestamp: Option<DateTime<Utc>>,
}

fn scan_decisions(root: &Path, project: Option<&str>) -> Result<Vec<Chunk>> {
    let file = root.join(".ostk/decisions.jsonl");
    if !file.exists() {
        return Ok(Vec::new());
    }
    let abs = absolute(&file);
    let text = std::fs::read_to_string(&file)?;
    let mut chunks: Vec<Chunk> = Vec::new();
    let mut chunk_index: u32 = 0;
    for (lineno, line) in text.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let dec: DecisionRow = match serde_json::from_str(line) {
            Ok(v) => v,
            Err(e) => {
                tracing::warn!(line = lineno, error = %e, "decisions: bad jsonl");
                continue;
            }
        };
        let value_text = value_to_string(&dec.value);
        let body = if let Some(reason) = dec.reason.as_deref() {
            format!("{value_text}\n\n{reason}")
        } else {
            value_text
        };
        let chunk_id = Chunk::make_id(Source::OstkDecision, &dec.key, chunk_index);
        let sha256 = Chunk::content_hash(&body);
        let links = Links {
            file_path: Some(abs.clone()),
            ..Links::default()
        };
        chunks.push(Chunk {
            chunk_id,
            source: Source::OstkDecision,
            project: project.map(str::to_string),
            source_id: dec.key,
            chunk_index,
            ts: dec.timestamp,
            role: None,
            text: body,
            sha256,
            links,
            extra: serde_json::Value::Null,
        });
        chunk_index = chunk_index.saturating_add(1);
    }
    Ok(chunks)
}

fn value_to_string(v: &serde_json::Value) -> String {
    match v {
        serde_json::Value::String(s) => s.clone(),
        serde_json::Value::Null => String::new(),
        other => other.to_string(),
    }
}

// ─────────────────────────────── needles ────────────────────────────────

#[derive(Debug, Deserialize)]
struct NeedleRow {
    id: String,
    #[serde(default)]
    title: Option<String>,
    #[serde(default)]
    status: Option<String>,
    #[serde(default)]
    description: Option<String>,
    #[serde(default)]
    ac: Option<serde_json::Value>,
    #[serde(default)]
    created_at: Option<DateTime<Utc>>,
}

fn scan_needles(root: &Path, project: Option<&str>) -> Result<Vec<Chunk>> {
    let file = root.join(".ostk/needles/issues.jsonl");
    if !file.exists() {
        return Ok(Vec::new());
    }
    let abs = absolute(&file);
    let text = std::fs::read_to_string(&file)?;
    let mut chunks: Vec<Chunk> = Vec::new();
    let mut chunk_index: u32 = 0;
    for (lineno, line) in text.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let needle: NeedleRow = match serde_json::from_str(line) {
            Ok(v) => v,
            Err(e) => {
                tracing::warn!(line = lineno, error = %e, "needles: bad jsonl");
                continue;
            }
        };
        if needle.status.is_none() {
            continue;
        }
        if !needle.id.starts_with('→') {
            continue;
        }
        let title = needle.title.clone().unwrap_or_default();
        let desc = needle.description.clone().unwrap_or_default();
        let ac = needle.ac.as_ref().map(value_to_string).unwrap_or_default();
        let body = format!("{title}\n\n{desc}\n\n{ac}");
        let chunk_id = Chunk::make_id(Source::OstkNeedle, &needle.id, chunk_index);
        let sha256 = Chunk::content_hash(&body);
        let links = Links {
            file_path: Some(abs.clone()),
            ..Links::default()
        };
        chunks.push(Chunk {
            chunk_id,
            source: Source::OstkNeedle,
            project: project.map(str::to_string),
            source_id: needle.id,
            chunk_index,
            ts: needle.created_at,
            role: None,
            text: body,
            sha256,
            links,
            extra: serde_json::Value::Null,
        });
        chunk_index = chunk_index.saturating_add(1);
    }
    Ok(chunks)
}

// ───────────────────────────────── audit ────────────────────────────────

fn scan_audit(root: &Path, project: Option<&str>, events: Option<&EventsDb>) -> Result<Vec<Chunk>> {
    use std::fs::File;
    use std::io::{BufRead, BufReader};

    let file = root.join(".ostk/audit.jsonl");
    if !file.exists() {
        return Ok(Vec::new());
    }
    let abs = absolute(&file);

    let f = File::open(&file)?;
    let reader = BufReader::new(f);

    let mut chunks: Vec<Chunk> = Vec::new();
    let mut chunk_index: u32 = 0;

    let mut pending: Vec<AuditEventRow> = Vec::with_capacity(AUDIT_BATCH);
    let mut prev_hash: u64 = 0;

    for (lineno, line) in reader.lines().enumerate() {
        let line = match line {
            Ok(l) => l,
            Err(e) => {
                tracing::warn!(line = lineno, error = %e, "audit: read failed");
                continue;
            }
        };
        if line.trim().is_empty() {
            continue;
        }

        let value: serde_json::Value = match serde_json::from_str(&line) {
            Ok(v) => v,
            Err(e) => {
                tracing::warn!(line = lineno, error = %e, "audit: bad jsonl");
                continue;
            }
        };

        prev_hash = hash_update(prev_hash, &line);

        let row = build_audit_row(&value, project, prev_hash);

        if is_significant(&value) {
            let chunk =
                build_significant_chunk(&value, project, row.row_key.clone(), &abs, chunk_index);
            chunks.push(chunk);
            chunk_index = chunk_index.saturating_add(1);
        }

        pending.push(row);
        if pending.len() >= AUDIT_BATCH {
            flush_audit(events, &pending);
            pending.clear();
        }
    }
    if !pending.is_empty() {
        flush_audit(events, &pending);
    }

    Ok(chunks)
}

fn flush_audit(events: Option<&EventsDb>, rows: &[AuditEventRow]) {
    let Some(db) = events else { return };
    if let Err(e) = db.ingest_batch(rows) {
        tracing::warn!(error = %e, "audit: events ingest failed");
    }
}

fn build_audit_row(
    value: &serde_json::Value,
    project: Option<&str>,
    prev_hash: u64,
) -> AuditEventRow {
    let ts_str = value.get("ts").and_then(|v| v.as_str());
    let ts = ts_str
        .and_then(|s| DateTime::parse_from_rfc3339(s).ok())
        .map(|dt| dt.with_timezone(&Utc));
    let event = value
        .get("event")
        .and_then(|v| v.as_str())
        .map(str::to_string);
    let tool = value
        .get("tool")
        .and_then(|v| v.as_str())
        .map(str::to_string);
    let agent = value
        .get("agent")
        .and_then(|v| v.as_str())
        .map(str::to_string);
    let success = value.get("success").and_then(serde_json::Value::as_bool);
    let exit_code = value.get("exit_code").and_then(serde_json::Value::as_i64);
    let duration_ms = value.get("duration_ms").and_then(serde_json::Value::as_i64);

    let project_out = project.map(str::to_string).or_else(|| {
        value
            .get("project")
            .and_then(|v| v.as_str())
            .map(str::to_string)
    });
    let ts_part = ts_str.unwrap_or("<no-ts>").to_string();
    let project_part = project_out.clone().unwrap_or_default();
    let row_key = format!("{project_part}:{ts_part}:{prev_hash:016x}");

    AuditEventRow {
        row_key,
        project: project_out,
        ts,
        event,
        tool,
        agent,
        success,
        exit_code,
        duration_ms,
        raw: value.to_string(),
    }
}

fn is_significant(value: &serde_json::Value) -> bool {
    let event = value.get("event").and_then(|v| v.as_str()).unwrap_or("");
    let tool = value.get("tool").and_then(|v| v.as_str()).unwrap_or("");
    let success = value
        .get("success")
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(true);
    let duration_ms = value
        .get("duration_ms")
        .and_then(serde_json::Value::as_i64)
        .unwrap_or(0);

    if event == "decision" || event == "handoff" {
        return true;
    }
    if event.starts_with("escalation.") {
        return true;
    }
    if !success {
        return true;
    }
    if tool == "bash" && duration_ms > 30_000 {
        return true;
    }
    false
}

fn build_significant_chunk(
    value: &serde_json::Value,
    project: Option<&str>,
    row_key: String,
    abs_path: &str,
    chunk_index: u32,
) -> Chunk {
    let ts = value
        .get("ts")
        .and_then(|v| v.as_str())
        .and_then(|s| DateTime::parse_from_rfc3339(s).ok())
        .map(|dt| dt.with_timezone(&Utc));
    let event = value.get("event").and_then(|v| v.as_str()).unwrap_or("?");
    let tool = value.get("tool").and_then(|v| v.as_str()).unwrap_or("?");
    let agent = value.get("agent").and_then(|v| v.as_str()).unwrap_or("?");
    let success = value
        .get("success")
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(true);
    let outcome = if success { "succeeded" } else { "failed" };
    let ts_str = ts.map_or_else(|| "<no-ts>".to_string(), |t| t.to_rfc3339());
    let summary =
        format!("On {ts_str}, agent {agent} emitted {event} (tool={tool}) which {outcome}");

    let chunk_id = Chunk::make_id(Source::OstkAuditSignificant, &row_key, chunk_index);
    let sha256 = Chunk::content_hash(&summary);
    let links = Links {
        file_path: Some(abs_path.to_string()),
        duckdb_row_key: Some(row_key.clone()),
        ..Links::default()
    };
    Chunk {
        chunk_id,
        source: Source::OstkAuditSignificant,
        project: project.map(str::to_string),
        source_id: row_key,
        chunk_index,
        ts,
        role: None,
        text: summary,
        sha256,
        links,
        extra: serde_json::Value::Null,
    }
}

fn hash_update(prev: u64, line: &str) -> u64 {
    // Cheap rolling hash — stable across runs, not cryptographic.
    let mut h = prev;
    for b in line.bytes() {
        h = h.wrapping_mul(1_099_511_628_211).wrapping_add(u64::from(b));
    }
    h
}

// ─────────────────────────── conversations ──────────────────────────────

#[derive(Debug, Deserialize)]
struct ConversationRow {
    #[serde(default)]
    turn: Option<u64>,
    #[serde(default)]
    from: Option<String>,
    #[serde(default)]
    to: Option<String>,
    #[serde(default)]
    ts: Option<DateTime<Utc>>,
    #[serde(default)]
    msg: Option<String>,
}

fn scan_conversations(root: &Path, project: Option<&str>) -> Result<Vec<Chunk>> {
    let dir = root.join(".ostk/conversations");
    if !dir.exists() {
        return Ok(Vec::new());
    }
    let mut chunks: Vec<Chunk> = Vec::new();
    let mut chunk_index: u32 = 0;
    for entry in walk_filtered(&dir, &[]).filter(|e| {
        e.path()
            .extension()
            .and_then(|x| x.to_str())
            .is_some_and(|x| x.eq_ignore_ascii_case("jsonl"))
    }) {
        let path = entry.path();
        let stem = path
            .file_stem()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_default();
        let abs = absolute(path);
        let text = std::fs::read_to_string(path)?;
        for (lineno, line) in text.lines().enumerate() {
            if line.trim().is_empty() {
                continue;
            }
            let row: ConversationRow = match serde_json::from_str(line) {
                Ok(v) => v,
                Err(e) => {
                    tracing::warn!(line = lineno, error = %e, "conversations: bad jsonl");
                    continue;
                }
            };
            let from = row.from.clone().unwrap_or_default();
            let to = row.to.clone().unwrap_or_default();
            let msg = row.msg.clone().unwrap_or_default();
            if msg.trim().is_empty() {
                continue;
            }
            let body = format!("{from} → {to}: {msg}");
            let source_id = format!(
                "{stem}:{}",
                row.turn.unwrap_or_else(|| u64::from(chunk_index))
            );
            let chunk_id = Chunk::make_id(Source::OstkConversation, &source_id, chunk_index);
            let sha256 = Chunk::content_hash(&body);
            let links = Links {
                file_path: Some(abs.clone()),
                ..Links::default()
            };
            chunks.push(Chunk {
                chunk_id,
                source: Source::OstkConversation,
                project: project.map(str::to_string),
                source_id,
                chunk_index,
                ts: row.ts,
                role: None,
                text: body,
                sha256,
                links,
                extra: serde_json::Value::Null,
            });
            chunk_index = chunk_index.saturating_add(1);
        }
    }
    Ok(chunks)
}

// ─────────────────────────────── sessions ───────────────────────────────

fn scan_sessions(root: &Path, project: Option<&str>) -> Result<Vec<Chunk>> {
    let dir = root.join(".ostk/sessions");
    if !dir.exists() {
        return Ok(Vec::new());
    }
    let mut chunks: Vec<Chunk> = Vec::new();
    for entry in walk_filtered(&dir, &[]).filter(|e| {
        e.path()
            .extension()
            .and_then(|x| x.to_str())
            .is_some_and(|x| x.eq_ignore_ascii_case("jsonl"))
    }) {
        let path = entry.path();
        let source_id = path
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_default();
        let mtime = file_mtime_utc(path).ok();
        let session_chunks =
            parse_session_file(path, Source::OstkSession, &source_id, project, mtime)?;
        chunks.extend(session_chunks);
    }
    Ok(chunks)
}

// ──────────────────────────────── memory ────────────────────────────────

#[derive(Debug, Deserialize)]
struct MemoryRow {
    name: String,
    #[serde(default)]
    file_id: Option<String>,
    #[serde(default)]
    stored_at: Option<DateTime<Utc>>,
}

fn scan_memory(root: &Path, project: Option<&str>) -> Result<Vec<Chunk>> {
    let index = root.join(".ostk/memory/pages.jsonl");
    if !index.exists() {
        return Ok(Vec::new());
    }
    let memory_dir = root.join(".ostk/memory");
    let abs_index = absolute(&index);
    let text = std::fs::read_to_string(&index)?;
    let mut chunks: Vec<Chunk> = Vec::new();
    let mut chunk_index: u32 = 0;
    for (lineno, line) in text.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let page: MemoryRow = match serde_json::from_str(line) {
            Ok(v) => v,
            Err(e) => {
                tracing::warn!(line = lineno, error = %e, "memory: bad jsonl");
                continue;
            }
        };
        // Lookup the page blob.
        let mut body = String::new();
        if let Some(file_id) = page.file_id.as_deref() {
            let candidates = [
                memory_dir.join(format!("{file_id}.page")),
                memory_dir.join(file_id),
            ];
            for c in &candidates {
                if let Ok(contents) = std::fs::read_to_string(c) {
                    body = contents;
                    break;
                }
            }
        }
        if body.trim().is_empty() {
            // Try by name as a last resort.
            let candidate = memory_dir.join(format!("{}.page", page.name));
            if let Ok(c) = std::fs::read_to_string(&candidate) {
                body = c;
            }
        }
        if body.trim().is_empty() {
            continue;
        }
        let chunk_id = Chunk::make_id(Source::OstkMemory, &page.name, chunk_index);
        let sha256 = Chunk::content_hash(&body);
        let links = Links {
            file_path: Some(abs_index.clone()),
            ..Links::default()
        };
        chunks.push(Chunk {
            chunk_id,
            source: Source::OstkMemory,
            project: project.map(str::to_string),
            source_id: page.name,
            chunk_index,
            ts: page.stored_at,
            role: None,
            text: body,
            sha256,
            links,
            extra: serde_json::Value::Null,
        });
        chunk_index = chunk_index.saturating_add(1);
    }
    Ok(chunks)
}

// ─────────────────────────────── specs ──────────────────────────────────

fn scan_specs(root: &Path, project: Option<&str>, ignore_patterns: &[String]) -> Vec<Chunk> {
    let mut chunks: Vec<Chunk> = Vec::new();
    let mut chunk_index: u32 = 0;
    for subdir in ["docs/spec", "docs/draft"] {
        let dir = root.join(subdir);
        if !dir.exists() {
            continue;
        }
        for entry in walk_filtered(&dir, ignore_patterns).filter(|e| {
            e.path()
                .extension()
                .and_then(|x| x.to_str())
                .is_some_and(|x| x.eq_ignore_ascii_case("md"))
        }) {
            let path = entry.path();
            let source_id = path.strip_prefix(root).map_or_else(
                |_| path.to_string_lossy().into_owned(),
                |p| p.to_string_lossy().into_owned(),
            );
            let text = match std::fs::read_to_string(path) {
                Ok(t) => t,
                Err(e) => {
                    tracing::warn!(path = %path.display(), error = %e, "specs: read");
                    continue;
                }
            };
            let mtime = file_mtime_utc(path).ok();
            let abs = absolute(path);
            for seg in split_markdown(&text) {
                let chunk_id = Chunk::make_id(Source::OstkSpec, &source_id, chunk_index);
                let sha256 = Chunk::content_hash(&seg);
                let links = Links {
                    file_path: Some(abs.clone()),
                    ..Links::default()
                };
                chunks.push(Chunk {
                    chunk_id,
                    source: Source::OstkSpec,
                    project: project.map(str::to_string),
                    source_id: source_id.clone(),
                    chunk_index,
                    ts: mtime,
                    role: None,
                    text: seg,
                    sha256,
                    links,
                    extra: serde_json::Value::Null,
                });
                chunk_index = chunk_index.saturating_add(1);
            }
        }
    }
    chunks
}

// ──────────────────────────────── code ──────────────────────────────────

fn scan_code(root: &Path, project: Option<&str>, ignore_patterns: &[String]) -> Vec<Chunk> {
    let mut chunks: Vec<Chunk> = Vec::new();
    let mut chunk_index: u32 = 0;
    for subdir in ["src", "crates", "lib"] {
        let dir = root.join(subdir);
        if !dir.exists() {
            continue;
        }
        for entry in walk_filtered(&dir, ignore_patterns).filter(|e| {
            e.path()
                .extension()
                .and_then(|x| x.to_str())
                .is_some_and(|x| {
                    CODE_EXTENSIONS
                        .iter()
                        .any(|allowed| allowed.eq_ignore_ascii_case(x))
                })
        }) {
            let path = entry.path();
            let source_id = path.strip_prefix(root).map_or_else(
                |_| path.to_string_lossy().into_owned(),
                |p| p.to_string_lossy().into_owned(),
            );
            let text = match std::fs::read_to_string(path) {
                Ok(t) => t,
                Err(e) => {
                    tracing::warn!(path = %path.display(), error = %e, "code: read");
                    continue;
                }
            };
            let mtime = file_mtime_utc(path).ok();
            let abs = absolute(path);
            for window in
                walk_and_window(&text, crate::code::WINDOW_LINES, crate::code::OVERLAP_LINES)
            {
                let chunk_id = Chunk::make_id(Source::Code, &source_id, chunk_index);
                let sha256 = Chunk::content_hash(&window);
                let links = Links {
                    file_path: Some(abs.clone()),
                    ..Links::default()
                };
                chunks.push(Chunk {
                    chunk_id,
                    source: Source::Code,
                    project: project.map(str::to_string),
                    source_id: source_id.clone(),
                    chunk_index,
                    ts: mtime,
                    role: None,
                    text: window,
                    sha256,
                    links,
                    extra: serde_json::Value::Null,
                });
                chunk_index = chunk_index.saturating_add(1);
            }
        }
    }
    chunks
}

// ────────────────────────────── utility ────────────────────────────────

fn absolute(path: &Path) -> String {
    path.canonicalize()
        .unwrap_or_else(|_| PathBuf::from(path))
        .to_string_lossy()
        .into_owned()
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

    fn write_jsonl(path: &Path, lines: &[&str]) {
        let mut f = std::fs::File::create(path).unwrap();
        for l in lines {
            writeln!(f, "{l}").unwrap();
        }
    }

    fn write_fixture(root: &Path) {
        std::fs::create_dir_all(root.join(".ostk/needles")).unwrap();
        std::fs::create_dir_all(root.join(".ostk/conversations")).unwrap();
        std::fs::create_dir_all(root.join(".ostk/sessions")).unwrap();
        std::fs::create_dir_all(root.join(".ostk/memory")).unwrap();
        std::fs::create_dir_all(root.join("docs/spec")).unwrap();
        std::fs::create_dir_all(root.join("src")).unwrap();

        // 1 decision
        write_jsonl(
            &root.join(".ostk/decisions.jsonl"),
            &[
                r#"{"key":"K1","value":"adopt X","reason":"tradeoff study","timestamp":"2026-04-17T10:00:00Z"}"#,
            ],
        );
        // 1 needle (valid) + 1 needle (skipped: no status) + 1 (skipped: bad id)
        write_jsonl(
            &root.join(".ostk/needles/issues.jsonl"),
            &[
                r#"{"id":"→1000","title":"T1","status":"open","description":"d1","ac":["a","b"],"created_at":"2026-04-17T10:00:00Z"}"#,
                r#"{"id":"→1001","title":"T2","description":"no status"}"#,
                r#"{"id":"bad-id","title":"T3","status":"open"}"#,
            ],
        );
        // 2 audit events (1 failure → significant) + 1 decision event (significant)
        write_jsonl(
            &root.join(".ostk/audit.jsonl"),
            &[
                r#"{"ts":"2026-04-17T10:00:00Z","event":"tool.call","tool":"bash","agent":"a","success":true,"duration_ms":5}"#,
                r#"{"ts":"2026-04-17T10:00:01Z","event":"tool.call","tool":"bash","agent":"a","success":false,"exit_code":1,"duration_ms":100}"#,
                r#"{"ts":"2026-04-17T10:00:02Z","event":"decision","agent":"a","success":true}"#,
            ],
        );
        // 1 conversation message
        write_jsonl(
            &root.join(".ostk/conversations/alpha.jsonl"),
            &[
                r#"{"turn":1,"from":"alpha","to":"beta","ts":"2026-04-17T09:00:00Z","msg":"hi there"}"#,
            ],
        );
        // 1 session with 2 exchanges (anthropic shape)
        write_jsonl(
            &root.join(".ostk/sessions/s1.jsonl"),
            &[
                r#"{"role":"user","content":"q1","timestamp":"2026-04-17T08:00:00Z"}"#,
                r#"{"role":"assistant","content":"a1","timestamp":"2026-04-17T08:00:01Z"}"#,
                r#"{"role":"user","content":"q2","timestamp":"2026-04-17T08:00:02Z"}"#,
                r#"{"role":"assistant","content":"a2","timestamp":"2026-04-17T08:00:03Z"}"#,
            ],
        );
        // 1 memory page
        write_jsonl(
            &root.join(".ostk/memory/pages.jsonl"),
            &[r#"{"name":"p1","file_id":"p1","tokens":10,"stored_at":"2026-04-17T07:00:00Z"}"#],
        );
        std::fs::write(root.join(".ostk/memory/p1.page"), "memory blob body").unwrap();
        // 1 spec markdown
        std::fs::write(
            root.join("docs/spec/overview.md"),
            "# Overview\n\nIntro.\n\n## Details\n\nBody.\n",
        )
        .unwrap();
        // 1 source file
        std::fs::write(root.join("src/main.rs"), "fn main() {}\n").unwrap();
    }

    fn cfg_for(root: &Path, project: Option<&str>) -> SourceConfig {
        SourceConfig {
            kind: SourceKind::OstkProject,
            project: project.map(str::to_string),
            paths: vec![root.to_string_lossy().into_owned()],
            ignore: vec![],
            extensions: vec![],
        }
    }

    #[test]
    fn ostk_project_expands_into_all_subsystems() {
        let tmp = TempDir::new().unwrap();
        write_fixture(tmp.path());

        let events_dir = TempDir::new().unwrap();
        let events = Arc::new(EventsDb::open(events_dir.path()).unwrap());

        let scanner = OstkProjectScanner::new().with_events_db(Arc::clone(&events));
        let cfg = cfg_for(tmp.path(), Some("myproj"));
        let items: Vec<_> = scanner.discover(&cfg).filter_map(Result::ok).collect();
        assert_eq!(items.len(), 1);
        let chunks = scanner.parse(items.into_iter().next().unwrap()).unwrap();

        let count = |s: Source| chunks.iter().filter(|c| c.source == s).count();
        assert_eq!(count(Source::OstkDecision), 1, "decision count");
        assert_eq!(count(Source::OstkNeedle), 1, "needle count (1 valid)");
        assert_eq!(
            count(Source::OstkAuditSignificant),
            2,
            "failure + decision-event"
        );
        assert_eq!(count(Source::OstkConversation), 1, "conversation count");
        // Per-message chunking (Phase H): 4 anthropic messages → 4 chunks.
        assert_eq!(count(Source::OstkSession), 4, "per-message session chunks");
        assert_eq!(count(Source::OstkMemory), 1, "memory page");
        assert!(count(Source::OstkSpec) >= 1, "spec chunks");
        assert!(count(Source::Code) >= 1, "code chunks");

        // Audit firehose: all 3 rows in events.duckdb.
        assert_eq!(events.row_count().unwrap(), 3);

        // The failure event's row_key should resolve to a chunk via duckdb_row_key.
        let failure_chunk = chunks
            .iter()
            .find(|c| c.source == Source::OstkAuditSignificant && c.text.contains("failed"))
            .expect("failure chunk present");
        let row_key = failure_chunk
            .links
            .duckdb_row_key
            .clone()
            .expect("duckdb_row_key set");
        let row = events
            .get(&row_key)
            .unwrap()
            .expect("failure row present in events.duckdb");
        assert_eq!(row.success, Some(false));
    }

    #[test]
    fn ostk_project_without_events_sink_still_emits_chunks() {
        let tmp = TempDir::new().unwrap();
        write_fixture(tmp.path());
        let scanner = OstkProjectScanner::new();
        let cfg = cfg_for(tmp.path(), None);
        let item = scanner.discover(&cfg).next().unwrap().unwrap();
        let chunks = scanner.parse(item).unwrap();
        assert!(
            chunks
                .iter()
                .any(|c| c.source == Source::OstkAuditSignificant)
        );
    }

    #[test]
    fn needle_without_arrow_is_skipped() {
        let tmp = TempDir::new().unwrap();
        std::fs::create_dir_all(tmp.path().join(".ostk/needles")).unwrap();
        write_jsonl(
            &tmp.path().join(".ostk/needles/issues.jsonl"),
            &[r#"{"id":"no-arrow","title":"t","status":"open"}"#],
        );
        let chunks = scan_needles(tmp.path(), None).unwrap();
        assert!(chunks.is_empty());
    }
}
