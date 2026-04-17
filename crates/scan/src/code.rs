//! Source-code scanner.
//!
//! Walks every file matching configured `extensions` under configured
//! `paths` and chunks each file into 200-line windows with 20-line overlap.
//!
//! TODO(phase-d+): swap the line-window strategy for tree-sitter-driven
//! semantic chunks (functions, impls, top-level items). The [`Scanner`]
//! contract is stable either way — only [`walk_and_window`] changes.

use std::path::Path;

use chrono::{DateTime, Utc};
use ostk_recall_core::{
    Chunk, Error, Links, Result, Scanner, Source, SourceConfig, SourceItem, SourceKind,
};
use walkdir::WalkDir;

/// Lines per window.
pub const WINDOW_LINES: usize = 200;
/// Line overlap between adjacent windows.
pub const OVERLAP_LINES: usize = 20;

/// Scanner for source-code trees.
#[derive(Debug, Default)]
pub struct CodeScanner;

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

        let iter = roots.into_iter().flat_map(move |root| {
            let project = project.clone();
            let root_for_rel = root.clone();
            let extensions = extensions.clone();
            WalkDir::new(&root)
                .follow_links(false)
                .into_iter()
                .filter_map(std::result::Result::ok)
                .filter(|e| e.file_type().is_file())
                .filter(move |e| extension_matches(e.path(), &extensions))
                .map(move |entry| {
                    let path = entry.path().to_path_buf();
                    let source_id = relative_source_id(&root_for_rel, &path);
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
            .ok_or_else(|| Error::Parse("code: SourceItem.path missing".into()))?;
        let text = std::fs::read_to_string(path)?;
        let mtime = file_mtime_utc(path).ok();
        let abs_path = path
            .canonicalize()
            .unwrap_or_else(|_| path.clone())
            .to_string_lossy()
            .into_owned();

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

        let scanner = CodeScanner;
        let cfg = cfg_with(tmp.path(), &["rs"]);
        let items: Vec<_> = scanner.discover(&cfg).filter_map(Result::ok).collect();
        assert_eq!(items.len(), 2);
    }

    #[test]
    fn parse_short_rust_file_one_chunk() {
        let tmp = TempDir::new().unwrap();
        let file = tmp.path().join("main.rs");
        std::fs::write(&file, "fn main() {}\nfn other() {}\n").unwrap();
        let scanner = CodeScanner;
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
    fn parse_400_line_rust_file_two_chunks() {
        let tmp = TempDir::new().unwrap();
        let file = tmp.path().join("big.rs");
        let mut body = String::new();
        for i in 0..400 {
            let _ = writeln!(body, "// line {i}");
        }
        std::fs::write(&file, &body).unwrap();
        let scanner = CodeScanner;
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
}
