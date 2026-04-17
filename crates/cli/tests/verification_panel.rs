//! Recall-QA verification panel.
//!
//! Drives `QueryEngine::recall` (plus `recall_link` and `recall_audit`)
//! against a single shared temp corpus built from every fixture under
//! `tests/fixtures/**`. Queries are declared in `tests/queries.yaml`;
//! each entry asserts that a distinctive keyword surfaces the expected
//! chunk within top-K.
//!
//! # Fake embedder + BM25-only ranking (known limitation)
//!
//! This test uses a deterministic `FakeEmbedder` so it runs offline.
//! Because the fake embedder's length-bucket vectors make the dense
//! side of the hybrid ranker effectively constant, retrieval is driven
//! by BM25 FTS alone. The fixtures and queries in `queries.yaml` are
//! engineered around this — every query targets a distinctive keyword
//! that appears in exactly one fixture.
//!
//! A second test, `verification_panel_semantic`, runs the same YAML
//! with the real embedder. It is gated behind `OSTK_RECALL_E2E=1` and
//! skipped by default (the real embedder needs model files on disk and
//! network access on first load). Enabling it exercises semantic
//! queries, where dense similarity matters.
//!
//! # Fixture coverage
//!
//! See `tests/fixtures/README.md` for the scanner → keyword → query
//! map. Each scanner (`markdown`, `code`, `claude_code`, `file_glob`,
//! `zip_export`, `ostk_project` with its six subtypes) gets at least
//! one query in the panel.

#![allow(
    clippy::option_if_let_else,
    clippy::redundant_closure_for_method_calls,
    clippy::too_many_arguments
)]

use std::fs::{self, File};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use ostk_recall_cli::commands::{self, InitOutcome};
use ostk_recall_pipeline::ChunkEmbedder;
use ostk_recall_query::{QueryEngine, RecallParams};
use tempfile::TempDir;
use zip::write::SimpleFileOptions;

const FAKE_DIM: usize = 16;

/// Deterministic length-bucket embedder. Makes every chunk land in one
/// of `FAKE_DIM` buckets based on `text.len() % FAKE_DIM`, which is
/// enough to keep the dense side of the hybrid ranker well-formed
/// without actually driving relevance. BM25 owns ranking.
struct FakeEmbedder;

impl ChunkEmbedder for FakeEmbedder {
    fn dim(&self) -> usize {
        FAKE_DIM
    }
    fn encode_batch(&self, texts: &[&str]) -> Vec<Vec<f32>> {
        texts
            .iter()
            .map(|t| {
                let mut v = vec![0.0; FAKE_DIM];
                let bucket = t.len() % FAKE_DIM;
                v[bucket] = 1.0;
                v
            })
            .collect()
    }
}

// ──────────────────────────── YAML-subset parser ────────────────────────────
//
// Intentionally small. Supports exactly the subset we use in
// `queries.yaml`:
//
// - List of mappings, each starting with `- key: value`.
// - Scalar values: strings (quoted or bare), integers.
// - Nested list values: `key:` followed by `  - item` lines.
// - Comments: `# ...` at any column.
//
// It is **not** a general YAML parser. If you need more features,
// swap in `serde_yml` as a dev-dep and parse structurally.

#[derive(Debug, Default, Clone)]
struct Entry {
    name: String,
    kind: String, // "recall" | "recall_link" | "recall_audit" | "negative"
    query: Option<String>,
    project: Option<String>,
    source: Option<String>,
    limit: Option<usize>,
    in_top_k: Option<usize>,
    must_contain: Vec<String>,
    chunk_id: Option<String>,
    parents_len: Option<usize>,
    sql: Option<String>,
    expected_count: Option<i64>,
    notes: Option<String>,
}

fn parse_yaml_panel(text: &str) -> Vec<Entry> {
    let mut out: Vec<Entry> = Vec::new();
    let mut current: Option<Entry> = None;
    let mut pending_list_key: Option<String> = None;

    for raw_line in text.lines() {
        // Strip trailing comments only when not inside a quoted string.
        // For simplicity we only strip `#` at column 0 or after a space
        // — good enough for our fixture.
        let line = strip_comment(raw_line);
        if line.trim().is_empty() {
            continue;
        }

        // New entry marker: "- key: value" at column 0.
        if let Some(rest) = line.strip_prefix("- ") {
            if let Some(entry) = current.take() {
                out.push(entry);
            }
            let mut e = Entry {
                kind: "recall".to_string(),
                ..Entry::default()
            };
            apply_kv(&mut e, rest.trim());
            current = Some(e);
            pending_list_key = None;
            continue;
        }

        // List-item continuation for the currently-pending list key.
        if let Some(bullet) = line.trim_start().strip_prefix("- ") {
            if let (Some(entry), Some(key)) = (current.as_mut(), pending_list_key.as_ref()) {
                push_list_item(entry, key, bullet.trim());
                continue;
            }
        }

        // "key: value" on an already-opened entry.
        if let Some(entry) = current.as_mut() {
            let trimmed = line.trim_start();
            if let Some((k, v)) = trimmed.split_once(':') {
                let k = k.trim();
                let v = v.trim();
                if v.is_empty() {
                    // Opens a list on the next lines.
                    pending_list_key = Some(k.to_string());
                } else {
                    pending_list_key = None;
                    apply_kv_pair(entry, k, v);
                }
            }
        }
    }

    if let Some(entry) = current.take() {
        out.push(entry);
    }
    out
}

fn strip_comment(line: &str) -> &str {
    // Strip `#` comments, but only outside quotes. Our fixtures don't
    // put `#` inside quoted strings, so a simple split works.
    if let Some(idx) = find_unquoted_hash(line) {
        &line[..idx]
    } else {
        line
    }
}

fn find_unquoted_hash(line: &str) -> Option<usize> {
    let mut in_single = false;
    let mut in_double = false;
    for (i, c) in line.char_indices() {
        match c {
            '\'' if !in_double => in_single = !in_single,
            '"' if !in_single => in_double = !in_double,
            '#' if !in_single && !in_double => return Some(i),
            _ => {}
        }
    }
    None
}

fn apply_kv(entry: &mut Entry, rest: &str) {
    if let Some((k, v)) = rest.split_once(':') {
        apply_kv_pair(entry, k.trim(), v.trim());
    }
}

fn apply_kv_pair(entry: &mut Entry, key: &str, raw_value: &str) {
    let value = unquote(raw_value);
    match key {
        "name" => entry.name = value,
        "kind" => entry.kind = value,
        "query" => entry.query = Some(value),
        "project" => entry.project = Some(value),
        "source" => entry.source = Some(value),
        "limit" => entry.limit = value.parse().ok(),
        "in_top_k" => entry.in_top_k = value.parse().ok(),
        "chunk_id" => entry.chunk_id = Some(value),
        "parents_len" => entry.parents_len = value.parse().ok(),
        "sql" => entry.sql = Some(value),
        "expected_count" => entry.expected_count = value.parse().ok(),
        "notes" => entry.notes = Some(value),
        _ => {}
    }
}

fn push_list_item(entry: &mut Entry, key: &str, item: &str) {
    let v = unquote(item);
    if key == "must_contain" {
        entry.must_contain.push(v);
    }
}

fn unquote(raw: &str) -> String {
    let s = raw.trim();
    if (s.starts_with('"') && s.ends_with('"') && s.len() >= 2)
        || (s.starts_with('\'') && s.ends_with('\'') && s.len() >= 2)
    {
        s[1..s.len() - 1].to_string()
    } else {
        s.to_string()
    }
}

// ─────────────────────────── fixture plumbing ───────────────────────────

fn fixtures_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("tests")
        .join("fixtures")
}

fn queries_yaml_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("tests")
        .join("queries.yaml")
}

/// Zip the checked-in `conversations.json` into a tempdir so the
/// `zip_export` scanner has an archive to open. Returns the tempdir
/// handle (caller keeps it alive) and the glob pattern.
fn build_zip_fixture() -> (TempDir, String) {
    let dir = TempDir::new().unwrap();
    let zip_path = dir.path().join("claude-data-export-panel.zip");
    let f = File::create(&zip_path).unwrap();
    let mut zw = zip::ZipWriter::new(f);
    let opts = SimpleFileOptions::default().compression_method(zip::CompressionMethod::Deflated);
    let src = fixtures_root()
        .join("zip_export")
        .join("conversations.json");
    let body = fs::read_to_string(&src).unwrap();
    zw.start_file("conversations.json", opts).unwrap();
    zw.write_all(body.as_bytes()).unwrap();
    zw.finish().unwrap();
    let pat = format!("{}/claude-data-export-*.zip", dir.path().display());
    (dir, pat)
}

fn write_config(
    path: &Path,
    corpus_root: &Path,
    markdown: &Path,
    code: &Path,
    claude_code: &Path,
    file_glob_pat: &str,
    zip_pat: &str,
    ostk_project: &Path,
) {
    let body = format!(
        r#"[corpus]
root = "{corpus}"

[embedder]
model = "unused-in-tests"

[[sources]]
kind = "markdown"
project = "notes"
paths = ["{markdown}"]

[[sources]]
kind = "code"
project = "code"
paths = ["{code}"]
extensions = ["rs"]

[[sources]]
kind = "claude_code"
project = "demo"
paths = ["{claude_code}"]

[[sources]]
kind = "file_glob"
project = "glob"
paths = ["{file_glob_pat}"]

[[sources]]
kind = "zip_export"
project = "claudeai"
paths = ["{zip_pat}"]

[[sources]]
kind = "ostk_project"
project = "panelproj"
paths = ["{ostk_project}"]
"#,
        corpus = corpus_root.display(),
        markdown = markdown.display(),
        code = code.display(),
        claude_code = claude_code.display(),
        file_glob_pat = file_glob_pat,
        zip_pat = zip_pat,
        ostk_project = ostk_project.display(),
    );
    fs::write(path, body).unwrap();
}

async fn build_corpus(embedder: Arc<dyn ChunkEmbedder>) -> (TempDir, TempDir, PathBuf, TempDir) {
    let corpus = TempDir::new().unwrap();
    let cfg_dir = TempDir::new().unwrap();
    let cfg_path = cfg_dir.path().join("config.toml");

    let fixtures = fixtures_root();
    let markdown = fixtures.join("markdown").join("notes");
    let code = fixtures.join("code");
    let claude_code = fixtures.join("claude_code");
    let file_glob = fixtures.join("file_glob");
    let file_glob_pat = format!("{}/**/*.txt", file_glob.display());
    let (zip_dir, zip_pat) = build_zip_fixture();
    let ostk_project = fixtures.join("ostk_project");

    assert!(markdown.exists(), "missing {}", markdown.display());
    assert!(code.exists(), "missing {}", code.display());
    assert!(claude_code.exists(), "missing {}", claude_code.display());
    assert!(file_glob.exists(), "missing {}", file_glob.display());
    assert!(ostk_project.exists(), "missing {}", ostk_project.display());

    write_config(
        &cfg_path,
        corpus.path(),
        &markdown,
        &code,
        &claude_code,
        &file_glob_pat,
        &zip_pat,
        &ostk_project,
    );

    match commands::init(&cfg_path, Arc::clone(&embedder))
        .await
        .unwrap()
    {
        InitOutcome::Initialized { .. } => {}
        InitOutcome::WroteStarter { .. } => panic!("expected Initialized"),
    }
    let scan = commands::scan(&cfg_path, Arc::clone(&embedder), None, false)
        .await
        .expect("scan");
    assert_eq!(scan.totals.errors, 0, "scan errors: {:?}", scan.totals);
    assert!(scan.totals.chunks_upserted > 0, "expected chunks");

    (corpus, cfg_dir, cfg_path, zip_dir)
}

// ────────────────────────────── assertion core ──────────────────────────────

#[derive(Debug, Clone)]
struct PanelResult {
    name: String,
    passed: bool,
    detail: String,
}

async fn evaluate_entry(engine: &QueryEngine, entry: &Entry) -> PanelResult {
    match entry.kind.as_str() {
        "recall" => evaluate_recall(engine, entry).await,
        "negative" => evaluate_negative(engine, entry).await,
        "recall_audit" => evaluate_audit(engine, entry),
        "recall_link" => evaluate_recall_link(engine, entry).await,
        other => PanelResult {
            name: entry.name.clone(),
            passed: false,
            detail: format!("unknown kind: {other}"),
        },
    }
}

async fn evaluate_recall(engine: &QueryEngine, entry: &Entry) -> PanelResult {
    let Some(query) = entry.query.clone() else {
        return PanelResult {
            name: entry.name.clone(),
            passed: false,
            detail: "query missing".into(),
        };
    };
    let top_k = entry.in_top_k.unwrap_or(10);
    let params = RecallParams {
        query,
        project: entry.project.clone(),
        source: entry.source.clone(),
        limit: Some(top_k.max(entry.limit.unwrap_or(top_k))),
        ..Default::default()
    };
    let hits = match engine.recall(params).await {
        Ok(h) => h,
        Err(e) => {
            return PanelResult {
                name: entry.name.clone(),
                passed: false,
                detail: format!("recall error: {e}"),
            };
        }
    };
    let window = hits.iter().take(top_k).collect::<Vec<_>>();

    // Project filter sanity — if the entry asked for a project filter,
    // every returned hit must match.
    if let Some(p) = &entry.project {
        for h in &window {
            if h.project.as_deref() != Some(p.as_str()) {
                return PanelResult {
                    name: entry.name.clone(),
                    passed: false,
                    detail: format!("project filter leak: got {:?}, want {}", h.project, p),
                };
            }
        }
    }
    // Source filter sanity.
    if let Some(s) = &entry.source {
        for h in &window {
            if h.source.as_str() != s.as_str() {
                return PanelResult {
                    name: entry.name.clone(),
                    passed: false,
                    detail: format!("source filter leak: got {}, want {}", h.source, s),
                };
            }
        }
    }

    if entry.must_contain.is_empty() {
        return PanelResult {
            name: entry.name.clone(),
            passed: !window.is_empty(),
            detail: format!("top={} hits (no substring assertions)", window.len()),
        };
    }

    for h in &window {
        for needle in &entry.must_contain {
            if h.snippet.contains(needle.as_str()) {
                return PanelResult {
                    name: entry.name.clone(),
                    passed: true,
                    detail: format!(
                        "hit {} (source={}, project={:?}) contains {:?}",
                        &h.chunk_id[..12.min(h.chunk_id.len())],
                        h.source,
                        h.project,
                        needle
                    ),
                };
            }
        }
    }
    let snippets: Vec<String> = window
        .iter()
        .map(|h| format!("[{}] {}…", h.source, &h.snippet[..80.min(h.snippet.len())]))
        .collect();
    PanelResult {
        name: entry.name.clone(),
        passed: false,
        detail: format!(
            "none of {:?} found in top-{} snippets: {:?}",
            entry.must_contain,
            window.len(),
            snippets
        ),
    }
}

async fn evaluate_negative(engine: &QueryEngine, entry: &Entry) -> PanelResult {
    // Negative assertion: with a query token that appears in NO
    // fixture, the probe substring (entry.must_contain or the query
    // itself) must not appear in any returned snippet. Hybrid recall
    // may still return chunks via its dense leg (every vector collapses
    // to FAKE_DIM buckets), but none of them should literally contain
    // the token.
    let Some(query) = entry.query.clone() else {
        return PanelResult {
            name: entry.name.clone(),
            passed: false,
            detail: "query missing".into(),
        };
    };
    let probe = entry
        .must_contain
        .first()
        .cloned()
        .unwrap_or_else(|| query.clone());
    let params = RecallParams {
        query,
        limit: Some(10),
        ..Default::default()
    };
    match engine.recall(params).await {
        Ok(hits) => {
            if let Some(leak) = hits.iter().find(|h| h.snippet.contains(probe.as_str())) {
                PanelResult {
                    name: entry.name.clone(),
                    passed: false,
                    detail: format!(
                        "probe {:?} leaked into snippet of chunk {}",
                        probe,
                        &leak.chunk_id[..12.min(leak.chunk_id.len())]
                    ),
                }
            } else {
                PanelResult {
                    name: entry.name.clone(),
                    passed: true,
                    detail: format!(
                        "probe {:?} absent from all {} returned snippets",
                        probe,
                        hits.len()
                    ),
                }
            }
        }
        Err(e) => PanelResult {
            name: entry.name.clone(),
            passed: false,
            detail: format!("recall error: {e}"),
        },
    }
}

fn evaluate_audit(engine: &QueryEngine, entry: &Entry) -> PanelResult {
    let Some(sql) = entry.sql.clone() else {
        return PanelResult {
            name: entry.name.clone(),
            passed: false,
            detail: "sql missing".into(),
        };
    };
    let expected = entry.expected_count.unwrap_or(0);
    match engine.recall_audit(&sql) {
        Ok(res) => {
            let actual = res
                .rows
                .first()
                .and_then(|r| r.first())
                .and_then(|v| v.as_i64())
                .unwrap_or(-1);
            PanelResult {
                name: entry.name.clone(),
                passed: actual == expected,
                detail: format!("count={actual} (expected {expected})"),
            }
        }
        Err(e) => PanelResult {
            name: entry.name.clone(),
            passed: false,
            detail: format!("recall_audit error: {e}"),
        },
    }
}

async fn evaluate_recall_link(engine: &QueryEngine, entry: &Entry) -> PanelResult {
    // Strategy: fire a recall for `entry.query`, take the top hit's
    // chunk_id, and chase recall_link. Assert parent count matches.
    let Some(query) = entry.query.clone() else {
        return PanelResult {
            name: entry.name.clone(),
            passed: false,
            detail: "query missing".into(),
        };
    };
    let hits = match engine
        .recall(RecallParams {
            query,
            limit: Some(10),
            ..Default::default()
        })
        .await
    {
        Ok(h) => h,
        Err(e) => {
            return PanelResult {
                name: entry.name.clone(),
                passed: false,
                detail: format!("recall error: {e}"),
            };
        }
    };
    // Find the first hit whose snippet still mentions the query text
    // (so we don't chase an unrelated chunk).
    let wanted = entry.must_contain.first().cloned();
    let Some(target) = hits.iter().find(|h| {
        wanted
            .as_ref()
            .is_none_or(|w| h.snippet.contains(w.as_str()))
    }) else {
        return PanelResult {
            name: entry.name.clone(),
            passed: false,
            detail: "no hit matched for recall_link seed".into(),
        };
    };
    match engine.recall_link(&target.chunk_id).await {
        Ok(link) => {
            let want = entry.parents_len.unwrap_or(0);
            let got = link.parents.len();
            PanelResult {
                name: entry.name.clone(),
                passed: got == want,
                detail: format!("parents={got} (expected {want})"),
            }
        }
        Err(e) => PanelResult {
            name: entry.name.clone(),
            passed: false,
            detail: format!("recall_link error: {e}"),
        },
    }
}

fn print_table(results: &[PanelResult]) {
    let max_name = results.iter().map(|r| r.name.len()).max().unwrap_or(4);
    eprintln!();
    eprintln!("{:<width$}   status   detail", "query", width = max_name);
    eprintln!("{}", "-".repeat(max_name + 60));
    for r in results {
        let status = if r.passed { "PASS" } else { "FAIL" };
        eprintln!(
            "{:<width$}   {}     {}",
            r.name,
            status,
            r.detail,
            width = max_name
        );
    }
    eprintln!();
}

async fn run_panel(embedder: Arc<dyn ChunkEmbedder>) {
    // Keep handles alive to preserve tempdirs until the test ends.
    let (_corpus, _cfg_dir, cfg_path, _zip_dir) = build_corpus(Arc::clone(&embedder)).await;
    let engine = commands::build_query_engine(&cfg_path, Arc::clone(&embedder))
        .await
        .expect("build_query_engine");

    let yaml_text = fs::read_to_string(queries_yaml_path()).expect("read tests/queries.yaml");
    let entries = parse_yaml_panel(&yaml_text);
    assert!(
        entries.len() >= 10,
        "expected >=10 panel entries, got {}",
        entries.len()
    );

    let mut results: Vec<PanelResult> = Vec::new();
    for entry in &entries {
        results.push(evaluate_entry(&engine, entry).await);
    }
    print_table(&results);
    let failed: Vec<&PanelResult> = results.iter().filter(|r| !r.passed).collect();
    assert!(
        failed.is_empty(),
        "{}/{} panel queries failed: {:#?}",
        failed.len(),
        results.len(),
        failed
    );
}

#[tokio::test]
async fn verification_panel() {
    let embedder: Arc<dyn ChunkEmbedder> = Arc::new(FakeEmbedder);
    run_panel(embedder).await;
}

/// Same panel, but with the real embedder. Skipped unless
/// `OSTK_RECALL_E2E=1` is set, because the real embedder needs model
/// files on disk and may hit the network on first load.
#[tokio::test]
async fn verification_panel_semantic() {
    if std::env::var("OSTK_RECALL_E2E").ok().as_deref() != Some("1") {
        eprintln!("skipping verification_panel_semantic (set OSTK_RECALL_E2E=1 to run)");
        return;
    }
    let embedder = match ostk_recall_embed::Embedder::load("minishlab/potion-retrieval-32M") {
        Ok(e) => Arc::new(e) as Arc<dyn ChunkEmbedder>,
        Err(e) => {
            eprintln!("skipping verification_panel_semantic: embedder load failed: {e}");
            return;
        }
    };
    run_panel(embedder).await;
}

// ──────────────────────────────── unit tests ────────────────────────────────

#[cfg(test)]
mod parser_tests {
    use super::*;

    #[test]
    fn parses_minimal_entry() {
        let src = r#"- name: "foo"
  query: "bar"
  in_top_k: 3
  must_contain:
    - "baz"
"#;
        let entries = parse_yaml_panel(src);
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].name, "foo");
        assert_eq!(entries[0].query.as_deref(), Some("bar"));
        assert_eq!(entries[0].in_top_k, Some(3));
        assert_eq!(entries[0].must_contain, vec!["baz".to_string()]);
        assert_eq!(entries[0].kind, "recall");
    }

    #[test]
    fn parses_multiple_entries_and_comments() {
        let src = r#"# header comment
- name: "a"
  query: "q"
- name: "b"
  kind: "negative"
  query: "qq"
"#;
        let entries = parse_yaml_panel(src);
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[1].kind, "negative");
    }
}
