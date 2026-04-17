//! Generic text scanner driven by glob patterns.
//!
//! Config `paths` are `globset` patterns. Matched files are read as UTF-8;
//! invalid UTF-8 files are skipped with a `warn` log (no scan failure).
//!
//! Chunking strategy is the same paragraph-split-with-overlap used for
//! oversize markdown sections: paragraph boundaries (blank-line gaps), with
//! a 2000-token / 200-token-overlap budget.

use std::path::{Path, PathBuf};

use chrono::{DateTime, Utc};
use globset::{Glob, GlobSetBuilder};
use ostk_recall_core::{
    Chunk, Error, Links, Result, Scanner, Source, SourceConfig, SourceItem, SourceKind,
};
use walkdir::WalkDir;

/// Token budget per chunk (same proxy as markdown: bytes/4).
const MAX_CHUNK_TOKENS: usize = 2000;
/// Token overlap between adjacent chunks.
const OVERLAP_TOKENS: usize = 200;
const BYTES_PER_TOKEN: usize = 4;

/// Scanner for generic text files matched by glob patterns.
#[derive(Debug, Default)]
pub struct FileGlobScanner;

impl Scanner for FileGlobScanner {
    fn kind(&self) -> SourceKind {
        SourceKind::FileGlob
    }

    fn discover<'a>(
        &'a self,
        cfg: &'a SourceConfig,
    ) -> Box<dyn Iterator<Item = Result<SourceItem>> + 'a> {
        let raw_paths = cfg.paths.clone();
        let project = cfg.project.clone();

        // Resolve each glob to (walk_root, glob_pattern).
        let planned: Vec<GlobPlan> = match plan_globs(&raw_paths) {
            Ok(v) => v,
            Err(e) => return Box::new(std::iter::once(Err(e))),
        };

        let iter = planned.into_iter().flat_map(move |plan| {
            let project = project.clone();
            let walk_root = plan.root.clone();
            let matcher = plan.matcher.clone();
            WalkDir::new(&plan.root)
                .follow_links(false)
                .into_iter()
                .filter_map(std::result::Result::ok)
                .filter(|e| e.file_type().is_file())
                .filter(move |e| matcher.is_match(e.path()))
                .map(move |entry| {
                    let path = entry.path().to_path_buf();
                    let source_id = relative_source_id(&walk_root, &path);
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
            .ok_or_else(|| Error::Parse("file_glob: SourceItem.path missing".into()))?;
        let bytes = std::fs::read(path).map_err(Error::Io)?;
        let Ok(s) = std::str::from_utf8(&bytes) else {
            tracing::warn!(path = %path.display(), "skipping non-utf8 file");
            return Ok(Vec::new());
        };
        let text = s.to_string();
        if text.trim().is_empty() {
            return Ok(Vec::new());
        }

        let mtime = file_mtime_utc(path).ok();
        let abs_path = path
            .canonicalize()
            .unwrap_or_else(|_| path.clone())
            .to_string_lossy()
            .into_owned();

        let segments = split_paragraphs(&text);
        let mut chunks = Vec::with_capacity(segments.len());
        for (idx, seg) in segments.into_iter().enumerate() {
            let chunk_index = u32::try_from(idx).map_err(|_| {
                Error::Parse(format!(
                    "file_glob: chunk_index {idx} exceeds u32 for {}",
                    item.source_id
                ))
            })?;
            let chunk_id = Chunk::make_id(Source::FileGlob, &item.source_id, chunk_index);
            let sha256 = Chunk::content_hash(&seg);
            let links = Links {
                file_path: Some(abs_path.clone()),
                ..Links::default()
            };
            chunks.push(Chunk {
                chunk_id,
                source: Source::FileGlob,
                project: item.project.clone(),
                source_id: item.source_id.clone(),
                chunk_index,
                ts: mtime,
                role: None,
                text: seg,
                sha256,
                links,
                extra: serde_json::Value::Null,
            });
        }
        Ok(chunks)
    }
}

/// Plan derived from one user-supplied glob string.
struct GlobPlan {
    /// Non-glob prefix; where `WalkDir` starts.
    root: PathBuf,
    /// Compiled glob matcher (matches against the full candidate path).
    matcher: globset::GlobSet,
}

fn plan_globs(raw: &[String]) -> Result<Vec<GlobPlan>> {
    let mut out = Vec::with_capacity(raw.len());
    for r in raw {
        let expanded = shellexpand_path(r)?;
        let (root, pattern) = split_glob(&expanded);
        let glob = Glob::new(&pattern).map_err(|e| Error::Scan(format!("bad glob {r}: {e}")))?;
        let mut b = GlobSetBuilder::new();
        b.add(glob);
        let set = b
            .build()
            .map_err(|e| Error::Scan(format!("build globset {r}: {e}")))?;
        out.push(GlobPlan { root, matcher: set });
    }
    Ok(out)
}

fn shellexpand_path(raw: &str) -> Result<String> {
    shellexpand::full(raw)
        .map(std::borrow::Cow::into_owned)
        .map_err(|e| Error::PathExpand(e.to_string()))
}

/// Split a glob string at the first component that contains a glob
/// metacharacter. The portion before becomes the walk root; the entire
/// original string (with a leading-`/` tolerance) is compiled as the
/// full-path matcher.
fn split_glob(raw: &str) -> (PathBuf, String) {
    let path = Path::new(raw);
    let mut root = PathBuf::new();
    let mut saw_meta = false;
    for comp in path.components() {
        let part = comp.as_os_str().to_string_lossy();
        if !saw_meta && contains_glob_meta(&part) {
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

fn contains_glob_meta(s: &str) -> bool {
    s.chars().any(|c| matches!(c, '*' | '?' | '[' | '{'))
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

/// Greedily pack paragraphs (blank-line-separated runs) into chunks of
/// ≤`MAX_CHUNK_TOKENS`, with `OVERLAP_TOKENS` worth of tail carried forward
/// between chunks.
#[must_use]
pub fn split_paragraphs(text: &str) -> Vec<String> {
    let paragraphs: Vec<&str> = text
        .split("\n\n")
        .map(str::trim_end)
        .filter(|p| !p.trim().is_empty())
        .collect();
    if paragraphs.is_empty() {
        return Vec::new();
    }
    let max_bytes = MAX_CHUNK_TOKENS * BYTES_PER_TOKEN;
    let overlap_bytes = OVERLAP_TOKENS * BYTES_PER_TOKEN;

    let mut out: Vec<String> = Vec::new();
    let mut current = String::new();
    for para in paragraphs {
        let candidate_len = current.len() + para.len() + 2;
        if current.is_empty() {
            // first paragraph of a fresh chunk
        } else if candidate_len <= max_bytes {
            current.push_str("\n\n");
        } else {
            let finished = std::mem::take(&mut current);
            let tail = tail_bytes(&finished, overlap_bytes);
            out.push(finished);
            current.push_str(&tail);
            if !current.is_empty() {
                current.push_str("\n\n");
            }
        }
        current.push_str(para);
        if current.len() > max_bytes * 2 {
            let finished = std::mem::take(&mut current);
            let tail = tail_bytes(&finished, overlap_bytes);
            out.push(finished);
            current.push_str(&tail);
        }
    }
    if !current.trim().is_empty() {
        out.push(current);
    }
    out
}

fn tail_bytes(s: &str, n: usize) -> String {
    if s.len() <= n {
        return s.to_string();
    }
    let start = s.len() - n;
    let boundary = (start..=s.len())
        .find(|&i| s.is_char_boundary(i))
        .unwrap_or(s.len());
    s[boundary..].to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn cfg_glob(pat: &str, project: &str) -> SourceConfig {
        SourceConfig {
            kind: SourceKind::FileGlob,
            project: Some(project.into()),
            paths: vec![pat.into()],
            extensions: vec![],
        }
    }

    #[test]
    fn split_basic_paragraphs() {
        let body = "alpha\n\nbeta\n\ngamma\n";
        let chunks = split_paragraphs(body);
        assert_eq!(chunks.len(), 1);
        assert!(chunks[0].contains("alpha"));
        assert!(chunks[0].contains("gamma"));
    }

    #[test]
    fn paragraphs_only_empty_returns_none() {
        assert!(split_paragraphs("").is_empty());
        assert!(split_paragraphs("\n\n\n").is_empty());
    }

    #[test]
    fn split_glob_separates_root_and_pattern() {
        let (root, pat) = split_glob("/tmp/foo/*.txt");
        assert_eq!(root, PathBuf::from("/tmp/foo"));
        assert_eq!(pat, "/tmp/foo/*.txt");
    }

    #[test]
    fn split_glob_deep_pattern() {
        let (root, _pat) = split_glob("/a/b/**/c.txt");
        assert_eq!(root, PathBuf::from("/a/b"));
    }

    #[test]
    fn discover_and_parse_txt_files() {
        let tmp = TempDir::new().unwrap();
        std::fs::create_dir_all(tmp.path().join("docs")).unwrap();
        std::fs::write(
            tmp.path().join("docs/a.txt"),
            "alpha body.\n\nmore alpha.\n",
        )
        .unwrap();
        std::fs::write(tmp.path().join("docs/b.txt"), "beta body.\n").unwrap();
        std::fs::write(tmp.path().join("docs/skip.bin"), b"\x00\x01binary\x02\x03").unwrap();

        let pat = format!("{}/**/*.txt", tmp.path().display());
        let scanner = FileGlobScanner;
        let cfg = cfg_glob(&pat, "p");
        let items: Vec<_> = scanner.discover(&cfg).filter_map(Result::ok).collect();
        assert_eq!(items.len(), 2);

        let mut all_chunks = Vec::new();
        for item in items {
            all_chunks.extend(scanner.parse(item).unwrap());
        }
        assert_eq!(all_chunks.len(), 2);
        assert!(
            all_chunks
                .iter()
                .all(|c| matches!(c.source, Source::FileGlob))
        );
    }

    #[test]
    fn binary_file_yields_no_chunks() {
        let tmp = TempDir::new().unwrap();
        std::fs::write(tmp.path().join("blob.dat"), b"\xff\xfe\x00garbage\x00").unwrap();
        let pat = format!("{}/**/*.dat", tmp.path().display());
        let scanner = FileGlobScanner;
        let cfg = cfg_glob(&pat, "p");
        let items: Vec<_> = scanner.discover(&cfg).filter_map(Result::ok).collect();
        assert_eq!(items.len(), 1);
        let chunks = scanner.parse(items.into_iter().next().unwrap()).unwrap();
        assert!(chunks.is_empty(), "non-utf8 file should produce no chunks");
    }
}
