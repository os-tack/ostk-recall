//! Source-code scanner.
//!
//! Walks every file matching configured `extensions` under configured
//! `paths`. Output strategy depends on the file extension and tooling:
//!
//! * `.rs` files in a Cargo workspace, when `fcp-rust` is on `$PATH`:
//!   delegated to [`crate::fcp_rust`] for symbol-bounded chunking. Each
//!   top-level item (fn, struct, enum, trait, impl, mod, const) becomes
//!   one chunk that includes the preceding ~5 lines (catches doc
//!   comments and attribute macros) and a synthetic header
//!   `// <kind> <name>` so BM25 surfaces symbol-name queries.
//! * Other extensions (and `.rs` fallback paths): 200-line windows with
//!   20-line overlap.
//!
//! TODO(phase-J): integrate `fcp-python`, `fcp-go`, `fcp-js` adapters
//! analogously. The line-window fallback stays in place for languages
//! without a semantic adapter, plus standalone `.rs` files outside any
//! Cargo workspace.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};

use chrono::{DateTime, Utc};
use ostk_recall_core::{
    Chunk, Error, Links, Result, Scanner, Source, SourceConfig, SourceItem, SourceKind,
};

use crate::fcp_rust::{self, FcpRustSession, RustSymbol, index_timeout_from_env};
use crate::walk::walk_filtered;

/// Lines per window (line-window fallback strategy).
pub const WINDOW_LINES: usize = 200;
/// Line overlap between adjacent windows.
pub const OVERLAP_LINES: usize = 20;
/// Lines of context to capture before each fcp-rust symbol. Catches doc
/// comments and `#[attribute]` macros immediately preceding the item.
pub const SYMBOL_LEADING_CONTEXT_LINES: u32 = 5;

/// Scanner for source-code trees. Holds a per-workspace cache of
/// `fcp-rust` sessions so successive `parse` calls amortize the
/// rust-analyzer cold-start cost.
#[derive(Default)]
pub struct CodeScanner {
    fcp_sessions: Mutex<HashMap<PathBuf, FcpRustSession>>,
    /// `Some(true)` once we've checked `$PATH` and found `fcp-rust`,
    /// `Some(false)` if absent. Lazy so unit tests don't pay for it.
    fcp_available: OnceLock<bool>,
}

impl std::fmt::Debug for CodeScanner {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let session_count = self.fcp_sessions.lock().map(|g| g.len()).unwrap_or(0);
        f.debug_struct("CodeScanner")
            .field("fcp_sessions", &session_count)
            .field("fcp_available", &self.fcp_available.get())
            .finish()
    }
}

impl CodeScanner {
    /// Returns true exactly once we've confirmed `fcp-rust` is on `$PATH`.
    /// Cached for the scanner's lifetime — installing fcp-rust mid-run
    /// won't take effect until the next ingest.
    fn fcp_rust_available(&self) -> bool {
        *self.fcp_available.get_or_init(|| {
            if which::which("fcp-rust").is_ok() {
                tracing::info!(
                    "code: fcp-rust detected on PATH; using semantic chunker for .rs files"
                );
                true
            } else {
                tracing::info!(
                    "code: fcp-rust not on PATH; falling back to line-window chunking for .rs files"
                );
                false
            }
        })
    }

    /// Borrow (or open) the fcp-rust session for `workspace_root`.
    /// Returns `None` if spawn or workspace-open fails — the caller
    /// should fall back to line-window chunking.
    fn session_for(&self, workspace_root: &Path) -> Option<()> {
        // Check under the lock, then drop it before any expensive
        // spawn/open work — keeps other parse calls unblocked.
        {
            let guard = self.fcp_sessions.lock().ok()?;
            if guard.contains_key(workspace_root) {
                return Some(());
            }
        }
        let mut session = match FcpRustSession::spawn() {
            Ok(s) => s,
            Err(e) => {
                tracing::warn!(error = %e, "code: failed to spawn fcp-rust");
                return None;
            }
        };
        if let Err(e) = session.open_workspace(workspace_root, index_timeout_from_env()) {
            tracing::warn!(
                error = %e,
                workspace = %workspace_root.display(),
                "code: fcp-rust open_workspace failed; falling back to line-window for this workspace"
            );
            return None;
        }
        // Re-acquire to insert. If a parallel parse won the race, drop
        // our duplicate session — the existing one is fine.
        {
            let mut guard = self.fcp_sessions.lock().ok()?;
            guard.entry(workspace_root.to_path_buf()).or_insert(session);
        }
        Some(())
    }

    /// Run `f` against the cached session for `workspace_root`. Returns
    /// `None` if no session is available. Holds the session lock for the
    /// duration of `f`; callers are expected to keep `f` short and
    /// non-recursive (no nested fcp-rust calls).
    fn with_session<F, R>(&self, workspace_root: &Path, f: F) -> Option<R>
    where
        F: FnOnce(&mut FcpRustSession) -> R,
    {
        let mut guard = self.fcp_sessions.lock().ok()?;
        let session = guard.get_mut(workspace_root)?;
        let out = f(session);
        drop(guard);
        Some(out)
    }

    /// Build chunks for a single Rust file using fcp-rust symbols.
    /// Returns `None` if fcp-rust isn't available, the workspace can't
    /// be opened, or the symbols query failed — caller falls back to
    /// line-window chunking.
    fn parse_rust_with_fcp(
        &self,
        item: &SourceItem,
        path: &Path,
        text: &str,
        mtime: Option<DateTime<Utc>>,
        abs_path: &str,
    ) -> Option<Vec<Chunk>> {
        if !self.fcp_rust_available() {
            return None;
        }
        let workspace = fcp_rust::find_cargo_workspace(path)?;
        self.session_for(&workspace)?;

        let total_lines = u32::try_from(text.lines().count()).unwrap_or(u32::MAX);
        let symbols = self
            .with_session(&workspace, |s| s.symbols(path, total_lines))?
            .ok()?;
        if symbols.is_empty() {
            // Empty symbols list: probably a tiny module declaration file.
            // Fall back to line-window so we still get a chunk.
            return None;
        }

        let file_lines: Vec<&str> = text.split_inclusive('\n').collect();
        let mut chunks = Vec::with_capacity(symbols.len());
        for (idx, sym) in symbols.iter().enumerate() {
            let chunk_text = slice_for_symbol(&file_lines, sym, SYMBOL_LEADING_CONTEXT_LINES);
            if chunk_text.trim().is_empty() {
                continue;
            }
            let body = format!("// {} {}\n{}", sym.kind, sym.name, chunk_text);
            let chunk_index = u32::try_from(idx).ok()?;
            let chunk_id = Chunk::make_id(Source::Code, &item.source_id, chunk_index);
            let sha256 = Chunk::content_hash(&body);
            let extra = serde_json::json!({
                "kind": sym.kind,
                "symbols": [sym.name.clone()],
                "line_start": sym.line_start,
                "line_end": sym.line_end,
                "chunker": "fcp-rust",
            });
            let links = Links {
                file_path: Some(abs_path.to_string()),
                ..Links::default()
            };
            chunks.push(Chunk {
                chunk_id,
                source: Source::Code,
                project: item.project.clone(),
                source_id: item.source_id.clone(),
                chunk_index,
                ts: mtime,
                role: None,
                text: body,
                sha256,
                links,
                extra,
            });
        }
        if chunks.is_empty() {
            None
        } else {
            Some(chunks)
        }
    }
}

impl Scanner for CodeScanner {
    fn kind(&self) -> SourceKind {
        SourceKind::Code
    }

    fn discover<'a>(
        &'a self,
        cfg: &'a SourceConfig,
    ) -> Box<dyn Iterator<Item = Result<SourceItem>> + 'a> {
        let roots = match cfg.expanded_paths() {
            Ok(v) => v,
            Err(e) => return Box::new(std::iter::once(Err(e))),
        };
        let extensions: Vec<String> = cfg
            .extensions
            .iter()
            .map(|e| e.trim_start_matches('.').to_ascii_lowercase())
            .collect();
        let project = cfg.project.clone();
        let ignore_patterns = cfg.ignore.clone();

        let iter = roots.into_iter().flat_map(move |root| {
            let project = project.clone();
            let root_for_rel = root.clone();
            let extensions = extensions.clone();
            walk_filtered(&root, &ignore_patterns)
                .filter(move |e| extension_matches(e.path(), &extensions))
                .map(move |entry| {
                    let path = entry.path().to_path_buf();
                    let source_id = relative_source_id(&root_for_rel, &path);
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
        let path = item
            .path
            .as_ref()
            .ok_or_else(|| Error::Parse("code: SourceItem.path missing".into()))?;
        let text = std::fs::read_to_string(path)?;
        let mtime = file_mtime_utc(path).ok();
        let abs_path = path
            .canonicalize()
            .unwrap_or_else(|_| path.clone())
            .to_string_lossy()
            .into_owned();

        // Rust path: try fcp-rust first; on any failure, fall through to
        // the line-window strategy below.
        let is_rust = path
            .extension()
            .and_then(|e| e.to_str())
            .is_some_and(|e| e.eq_ignore_ascii_case("rs"));
        if is_rust {
            if let Some(chunks) = self.parse_rust_with_fcp(&item, path, &text, mtime, &abs_path) {
                return Ok(chunks);
            }
        }

        // TODO(phase-J): wire fcp-python / fcp-go / fcp-js here. Until
        // then, every non-Rust extension uses the line-window chunker.
        let windows = walk_and_window(&text, WINDOW_LINES, OVERLAP_LINES);
        let mut chunks = Vec::with_capacity(windows.len());
        for (idx, window) in windows.into_iter().enumerate() {
            let chunk_index = u32::try_from(idx).map_err(|_| {
                Error::Parse(format!(
                    "code: chunk_index {idx} exceeds u32 for {}",
                    item.source_id
                ))
            })?;
            let chunk_id = Chunk::make_id(Source::Code, &item.source_id, chunk_index);
            let sha256 = Chunk::content_hash(&window);
            let links = Links {
                file_path: Some(abs_path.clone()),
                ..Links::default()
            };
            chunks.push(Chunk {
                chunk_id,
                source: Source::Code,
                project: item.project.clone(),
                source_id: item.source_id.clone(),
                chunk_index,
                ts: mtime,
                role: None,
                text: window,
                sha256,
                links,
                extra: serde_json::Value::Null,
            });
        }
        Ok(chunks)
    }
}

/// Build the chunk body for a single symbol: pull `[line_start - leading,
/// line_end]` from `lines`, clamping to file bounds.
#[must_use]
pub fn slice_for_symbol(lines: &[&str], sym: &RustSymbol, leading: u32) -> String {
    if lines.is_empty() {
        return String::new();
    }
    let total = u32::try_from(lines.len()).unwrap_or(u32::MAX);
    // Convert to 0-based indexing for slicing.
    let start_one = sym.line_start.saturating_sub(leading).max(1);
    let start = (start_one - 1) as usize;
    let end_one = sym.line_end.min(total);
    let end = end_one as usize;
    if start >= lines.len() || end <= start {
        return String::new();
    }
    lines[start..end].concat()
}

fn extension_matches(path: &Path, exts: &[String]) -> bool {
    if exts.is_empty() {
        return false;
    }
    path.extension().and_then(|e| e.to_str()).is_some_and(|e| {
        let lower = e.to_ascii_lowercase();
        exts.contains(&lower)
    })
}

fn relative_source_id(root: &Path, path: &Path) -> String {
    path.strip_prefix(root).map_or_else(
        |_| {
            path.file_name()
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_default()
        },
        |rel| {
            rel.components()
                .map(|c| c.as_os_str().to_string_lossy().into_owned())
                .collect::<Vec<_>>()
                .join("/")
        },
    )
}

fn file_mtime_utc(path: &Path) -> std::io::Result<DateTime<Utc>> {
    let meta = std::fs::metadata(path)?;
    let sys = meta.modified()?;
    Ok(DateTime::<Utc>::from(sys))
}

/// Slice `text` into line-based windows of `window` lines with `overlap`
/// lines carried over between adjacent windows.
///
/// * Files with fewer than `window + 1` lines return a single chunk
///   containing the whole file.
/// * Empty or whitespace-only content returns an empty `Vec`.
/// * `overlap >= window` is clamped to `window - 1` to guarantee forward
///   progress.
#[must_use]
pub fn walk_and_window(text: &str, window: usize, overlap: usize) -> Vec<String> {
    if text.trim().is_empty() {
        return Vec::new();
    }
    let lines: Vec<&str> = text.split_inclusive('\n').collect();
    if lines.len() <= window {
        return vec![lines.concat()];
    }

    let window = window.max(1);
    let overlap = overlap.min(window.saturating_sub(1));
    let stride = window - overlap;

    let mut out: Vec<String> = Vec::new();
    let mut start = 0usize;
    while start < lines.len() {
        let end = (start + window).min(lines.len());
        out.push(lines[start..end].concat());
        if end == lines.len() {
            break;
        }
        start += stride;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fmt::Write as _;
    use tempfile::TempDir;

    fn cfg_with(root: &Path, exts: &[&str]) -> SourceConfig {
        SourceConfig {
            kind: SourceKind::Code,
            project: Some("code".into()),
            paths: vec![root.to_string_lossy().into_owned()],
            ignore: vec![],
            extensions: exts.iter().map(|s| (*s).to_string()).collect(),
        }
    }

    #[test]
    fn window_short_file_one_chunk() {
        let body = "fn a() {}\nfn b() {}\n";
        let chunks = walk_and_window(body, 200, 20);
        assert_eq!(chunks.len(), 1);
        assert!(chunks[0].contains("fn a"));
    }

    #[test]
    fn window_long_file_overlaps() {
        let mut body = String::new();
        for i in 0..400 {
            let _ = writeln!(body, "line {i}");
        }
        let chunks = walk_and_window(&body, 200, 20);
        assert!(
            chunks.len() >= 2,
            "expected multiple windows, got {}",
            chunks.len()
        );
        // Overlap: last 20 lines of chunk[0] should appear at start of chunk[1].
        let tail_rev: Vec<&str> = chunks[0]
            .lines()
            .rev()
            .take(20)
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .collect();
        let head: Vec<&str> = chunks[1].lines().take(20).collect();
        assert_eq!(tail_rev, head, "expected 20-line overlap");
    }

    #[test]
    fn empty_body_no_chunks() {
        assert!(walk_and_window("", 200, 20).is_empty());
        assert!(walk_and_window("\n\n\n", 200, 20).is_empty());
    }

    #[test]
    fn discover_filters_by_extension() {
        let tmp = TempDir::new().unwrap();
        std::fs::create_dir_all(tmp.path().join("src")).unwrap();
        std::fs::write(tmp.path().join("src/a.rs"), "fn a() {}\n").unwrap();
        std::fs::write(tmp.path().join("src/b.rs"), "fn b() {}\n").unwrap();
        std::fs::write(tmp.path().join("src/c.txt"), "skipped\n").unwrap();

        let scanner = CodeScanner::default();
        let cfg = cfg_with(tmp.path(), &["rs"]);
        let items: Vec<_> = scanner.discover(&cfg).filter_map(Result::ok).collect();
        assert_eq!(items.len(), 2);
    }

    #[test]
    fn parse_short_rust_file_one_chunk() {
        // Standalone .rs file outside any Cargo workspace → fcp-rust path
        // declines, falls through to line-window strategy → one chunk.
        let tmp = TempDir::new().unwrap();
        let file = tmp.path().join("main.rs");
        std::fs::write(&file, "fn main() {}\nfn other() {}\n").unwrap();
        let scanner = CodeScanner::default();
        let cfg = cfg_with(tmp.path(), &["rs"]);
        let item = scanner.discover(&cfg).next().unwrap().unwrap();
        let chunks = scanner.parse(item).unwrap();
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].chunk_index, 0);
        assert!(
            chunks[0]
                .links
                .file_path
                .as_deref()
                .unwrap()
                .ends_with("main.rs")
        );
    }

    #[test]
    fn parse_400_line_rust_file_two_chunks_via_window() {
        // 400 line-noise file outside any Cargo workspace exercises the
        // line-window fallback path and proves the strategy still works.
        let tmp = TempDir::new().unwrap();
        let file = tmp.path().join("big.rs");
        let mut body = String::new();
        for i in 0..400 {
            let _ = writeln!(body, "// line {i}");
        }
        std::fs::write(&file, &body).unwrap();
        let scanner = CodeScanner::default();
        let cfg = cfg_with(tmp.path(), &["rs"]);
        let item = scanner.discover(&cfg).next().unwrap().unwrap();
        let chunks = scanner.parse(item).unwrap();
        assert!(
            chunks.len() >= 2,
            "expected at least 2 windows, got {}",
            chunks.len()
        );
        assert_eq!(chunks[0].chunk_index, 0);
        assert_eq!(chunks[1].chunk_index, 1);
    }

    #[test]
    fn slice_for_symbol_clamps_to_file() {
        let body = "a\nb\nc\nd\ne\nf\n";
        let lines: Vec<&str> = body.split_inclusive('\n').collect();
        // symbol on line 4 with leading=2 → start at line 2 (b)
        let sym = RustSymbol {
            name: "x".into(),
            kind: "fn".into(),
            line_start: 4,
            line_end: 5,
        };
        let s = slice_for_symbol(&lines, &sym, 2);
        assert_eq!(s, "b\nc\nd\ne\n");
    }

    #[test]
    fn slice_for_symbol_handles_oversized_end() {
        let body = "a\nb\nc\n";
        let lines: Vec<&str> = body.split_inclusive('\n').collect();
        let sym = RustSymbol {
            name: "x".into(),
            kind: "fn".into(),
            line_start: 2,
            line_end: u32::MAX,
        };
        let s = slice_for_symbol(&lines, &sym, 0);
        assert_eq!(s, "b\nc\n");
    }

    /// Live integration: when `fcp-rust` is on PATH, parsing a real
    /// Rust file inside the ostk-recall workspace should yield multiple
    /// symbol-bounded chunks (not one line-window chunk).
    #[test]
    fn parse_rust_file_uses_fcp_rust_when_available() {
        if which::which("fcp-rust").is_err() {
            eprintln!("skipping: fcp-rust not on PATH");
            return;
        }
        // Test from inside this workspace so find_cargo_workspace succeeds.
        let cwd = std::env::current_dir().unwrap();
        let ws = fcp_rust::find_cargo_workspace(&cwd).unwrap_or(cwd);
        let target = ws.join("crates/scan/src/code.rs");
        if !target.exists() {
            eprintln!("skipping: fixture file missing");
            return;
        }
        let scanner = CodeScanner::default();
        let item = SourceItem {
            source_id: "crates/scan/src/code.rs".into(),
            path: Some(target),
            project: Some("test".into()),
            bytes: None,
            ignore: Vec::new(),
        };
        let chunks = scanner.parse(item).expect("parse failed");
        assert!(
            chunks.len() >= 3,
            "expected multiple symbol chunks, got {}",
            chunks.len()
        );
        // At least one chunk should carry the synthetic header for CodeScanner.
        let has_header = chunks.iter().any(|c| {
            c.text.starts_with("// struct CodeScanner") || c.text.contains("// struct CodeScanner")
        });
        assert!(has_header, "expected CodeScanner symbol header");
        // Extra metadata should be populated.
        let with_extra = chunks
            .iter()
            .find(|c| c.extra.get("chunker").and_then(|v| v.as_str()) == Some("fcp-rust"))
            .expect("expected at least one chunk tagged chunker=fcp-rust");
        assert!(with_extra.extra.get("symbols").is_some());
    }
}
