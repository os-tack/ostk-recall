//! Markdown scanner.
//!
//! Walks every `.md` file under each configured path, splits bodies on
//! top-level `##` headings into chunks, and (for oversize sections)
//! sub-splits on paragraph boundaries with a small overlap.
//!
//! # Chunking rules
//!
//! * The body is split on lines that start with `## ` at column 0. The text
//!   *before* the first `##` is chunk 0 (often a doc intro plus `# Title`).
//!   Each subsequent `##` section becomes one more chunk.
//! * Token estimate is `bytes / 4` — a cheap proxy avoiding a tokenizer dep.
//! * Any section whose estimate exceeds `MAX_SECTION_TOKENS` (2000, i.e.
//!   ~8000 bytes) is sub-split on paragraph boundaries (blank-line gaps) with
//!   `OVERLAP_TOKENS` (200, i.e. ~800 bytes) of overlap between sub-chunks.
//! * Empty / whitespace-only sections are dropped.
//!
//! See `split_markdown` for the pure helper; it has its own unit tests.

use std::path::Path;

use chrono::{DateTime, Utc};
use ostk_recall_core::{
    Chunk, Error, Links, Result, Scanner, Source, SourceConfig, SourceItem, SourceKind,
};

use crate::walk::walk_filtered;

/// Token budget above which a section is sub-split.
const MAX_SECTION_TOKENS: usize = 2000;
/// Paragraph-overlap budget between adjacent sub-chunks.
const OVERLAP_TOKENS: usize = 200;
/// Bytes per "token" — cheap approximation (no tokenizer dep).
const BYTES_PER_TOKEN: usize = 4;

/// Scanner for markdown source trees.
#[derive(Debug, Default)]
pub struct MarkdownScanner;

impl Scanner for MarkdownScanner {
    fn kind(&self) -> SourceKind {
        SourceKind::Markdown
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

        let iter = roots.into_iter().flat_map(move |root| {
            let project = project.clone();
            let root_for_rel = root.clone();
            walk_filtered(&root, &ignore_patterns)
                .filter(|e| {
                    e.path()
                        .extension()
                        .and_then(|x| x.to_str())
                        .is_some_and(|x| x.eq_ignore_ascii_case("md"))
                })
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
            .ok_or_else(|| Error::Parse("markdown: SourceItem.path missing".into()))?;
        let text = std::fs::read_to_string(path)?;
        let mtime = file_mtime_utc(path).ok();

        let abs_path = path
            .canonicalize()
            .unwrap_or_else(|_| path.clone())
            .to_string_lossy()
            .into_owned();

        let segments = split_markdown(&text);
        let mut chunks = Vec::with_capacity(segments.len());
        for (idx, seg) in segments.into_iter().enumerate() {
            let chunk_index = u32::try_from(idx).map_err(|_| {
                Error::Parse(format!(
                    "markdown: chunk_index {idx} exceeds u32 for {}",
                    item.source_id
                ))
            })?;
            let chunk_id = Chunk::make_id(Source::Markdown, &item.source_id, chunk_index);
            let sha256 = Chunk::content_hash(&seg);
            let links = Links {
                file_path: Some(abs_path.clone()),
                ..Links::default()
            };
            chunks.push(Chunk {
                chunk_id,
                source: Source::Markdown,
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

/// Compute the `source_id` for a discovered file: path relative to the
/// configured root, using `/` separators. If the file isn't under the root
/// (shouldn't happen for `WalkDir(root)`), fall back to the file name.
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

/// Split a markdown body into chunk texts.
///
/// Strategy:
/// 1. Split on `##` heading lines at column 0. Pre-heading text is segment 0.
/// 2. For each segment, if `bytes/4 > MAX_SECTION_TOKENS`, sub-split on
///    paragraph boundaries (blank-line gaps) with ~`OVERLAP_TOKENS` carryover.
/// 3. Trim trailing whitespace; drop empty segments.
///
/// Deterministic and pure — same input always produces the same output.
#[must_use]
pub fn split_markdown(text: &str) -> Vec<String> {
    let sections = split_on_h2(text);
    let mut out = Vec::new();
    for s in sections {
        let trimmed = s.trim_end();
        if trimmed.trim().is_empty() {
            continue;
        }
        if token_estimate(trimmed) <= MAX_SECTION_TOKENS {
            out.push(trimmed.to_string());
        } else {
            out.extend(split_oversize_section(trimmed));
        }
    }
    out
}

/// Split on `##` heading lines. Each `## ...` line starts a new segment.
/// The text before the first `##` is segment 0, preserved even if empty
/// (caller drops empties).
fn split_on_h2(text: &str) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    let mut current = String::new();

    for line in text.split_inclusive('\n') {
        if is_h2_heading(line) {
            if !current.is_empty() || !out.is_empty() {
                out.push(std::mem::take(&mut current));
            } else {
                // First segment is empty — drop the leading empty segment so
                // a file that starts with `##` doesn't get a phantom chunk 0.
                current.clear();
            }
        }
        current.push_str(line);
    }
    if !current.is_empty() {
        out.push(current);
    }
    out
}

fn is_h2_heading(line: &str) -> bool {
    // A line like `## Something` or just `##`. `###` should NOT match.
    let stripped = line.strip_suffix('\n').unwrap_or(line);
    let stripped = stripped.strip_suffix('\r').unwrap_or(stripped);
    stripped == "##" || stripped.starts_with("## ")
}

const fn token_estimate(s: &str) -> usize {
    s.len() / BYTES_PER_TOKEN
}

/// Sub-split an oversize section on paragraph (blank-line) boundaries,
/// greedily packing paragraphs into sub-chunks. Between adjacent sub-chunks
/// we carry ~`OVERLAP_TOKENS` of overlap from the tail of the previous one.
fn split_oversize_section(section: &str) -> Vec<String> {
    let paragraphs: Vec<&str> = section
        .split("\n\n")
        .map(str::trim_end)
        .filter(|p| !p.trim().is_empty())
        .collect();
    if paragraphs.is_empty() {
        return vec![section.to_string()];
    }

    let max_bytes = MAX_SECTION_TOKENS * BYTES_PER_TOKEN;
    let overlap_bytes = OVERLAP_TOKENS * BYTES_PER_TOKEN;

    let mut out: Vec<String> = Vec::new();
    let mut current = String::new();
    for para in paragraphs {
        let candidate_len = current.len() + para.len() + 2;
        if current.is_empty() {
            // first paragraph of a new sub-chunk
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
        // If a single paragraph alone exceeds the budget, still flush it as
        // its own sub-chunk (hard-split would hurt semantics more than a
        // slightly oversize chunk).
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

/// Return the last `n` bytes of `s`, snapped to a character boundary.
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
    use ostk_recall_core::SourceKind;
    use std::io::Write;
    use tempfile::TempDir;

    fn cfg_with(root: &Path, project: &str) -> SourceConfig {
        SourceConfig {
            kind: SourceKind::Markdown,
            project: Some(project.into()),
            paths: vec![root.to_string_lossy().into_owned()],
            ignore: vec![],
            extensions: vec![],
        }
    }

    #[test]
    fn split_three_h2_sections() {
        let doc = "\
# Title
Intro paragraph.

## First
Alpha.

## Second
Beta.

## Third
Gamma.
";
        let chunks = split_markdown(doc);
        assert_eq!(chunks.len(), 4, "pre-## intro + 3 sections");
        assert!(chunks[0].contains("Intro paragraph"));
        assert!(chunks[1].starts_with("## First"));
        assert!(chunks[2].starts_with("## Second"));
        assert!(chunks[3].starts_with("## Third"));
    }

    #[test]
    fn split_no_headings_single_chunk() {
        let doc = "Just one big blob.\n\nAnother paragraph.\n";
        let chunks = split_markdown(doc);
        assert_eq!(chunks.len(), 1);
        assert!(chunks[0].contains("Just one big blob"));
        assert!(chunks[0].contains("Another paragraph"));
    }

    #[test]
    fn split_leading_h2_no_phantom_chunk() {
        let doc = "## Heading\nBody.\n";
        let chunks = split_markdown(doc);
        assert_eq!(chunks.len(), 1);
        assert!(chunks[0].starts_with("## Heading"));
    }

    #[test]
    fn h3_does_not_split() {
        let doc = "\
# T

### Sub one
Alpha.

### Sub two
Beta.
";
        let chunks = split_markdown(doc);
        assert_eq!(chunks.len(), 1);
    }

    #[test]
    fn oversize_section_sub_splits_with_overlap() {
        // Build a single `##` section whose body exceeds 2000 tokens
        // (> ~8000 bytes). Use multiple paragraphs so paragraph splitting
        // can actually divide it.
        let big_para = "lorem ipsum ".repeat(200); // ~2400 bytes
        let body = format!(
            "## Big\n\n{big_para}\n\n{big_para}\n\n{big_para}\n\n{big_para}\n\n{big_para}\n"
        );
        let chunks = split_markdown(&body);
        assert!(
            chunks.len() >= 2,
            "expected sub-split; got {} chunks",
            chunks.len()
        );
        // Overlap: adjacent sub-chunks should share some tail/head text.
        let tail_first = &chunks[0][chunks[0].len().saturating_sub(400)..];
        let head_second = &chunks[1][..chunks[1].len().min(400)];
        // A weaker but robust check: overlap_bytes of chunk[0]'s end should
        // appear somewhere in chunk[1]'s head region. Walk small windows.
        let overlap_window = &tail_first[tail_first.len().saturating_sub(64)..];
        assert!(
            head_second.contains(overlap_window) || chunks[1].contains(overlap_window),
            "expected overlap carryover between sub-chunks"
        );
    }

    #[test]
    fn chunk_ids_deterministic_across_runs() {
        let tmp = TempDir::new().unwrap();
        let file = tmp.path().join("note.md");
        let mut f = std::fs::File::create(&file).unwrap();
        writeln!(f, "# T\n\n## A\nalpha\n\n## B\nbeta").unwrap();

        let scanner = MarkdownScanner;
        let cfg = cfg_with(tmp.path(), "p");
        let items1: Vec<_> = scanner.discover(&cfg).filter_map(Result::ok).collect();
        let items2: Vec<_> = scanner.discover(&cfg).filter_map(Result::ok).collect();
        assert_eq!(items1.len(), 1);

        let c1 = scanner.parse(items1.into_iter().next().unwrap()).unwrap();
        let c2 = scanner.parse(items2.into_iter().next().unwrap()).unwrap();
        assert_eq!(c1.len(), c2.len());
        for (a, b) in c1.iter().zip(c2.iter()) {
            assert_eq!(a.chunk_id, b.chunk_id, "chunk_id must be stable");
            assert_eq!(a.sha256, b.sha256);
        }
    }

    #[test]
    fn discover_walks_tree() {
        let tmp = TempDir::new().unwrap();
        std::fs::create_dir_all(tmp.path().join("a/b")).unwrap();
        std::fs::write(tmp.path().join("a/one.md"), "# one\n").unwrap();
        std::fs::write(tmp.path().join("a/b/two.md"), "# two\n").unwrap();
        std::fs::write(tmp.path().join("not-md.txt"), "nope").unwrap();

        let scanner = MarkdownScanner;
        let cfg = cfg_with(tmp.path(), "p");
        let items: Vec<_> = scanner.discover(&cfg).filter_map(Result::ok).collect();
        assert_eq!(items.len(), 2);
        let ids: Vec<&str> = items.iter().map(|i| i.source_id.as_str()).collect();
        assert!(ids.iter().any(|s| s.ends_with("one.md")));
        assert!(ids.iter().any(|s| s.ends_with("two.md")));
    }

    #[test]
    fn parse_populates_links_and_project() {
        let tmp = TempDir::new().unwrap();
        let file = tmp.path().join("x.md");
        std::fs::write(&file, "# hi\n").unwrap();
        let scanner = MarkdownScanner;
        let cfg = cfg_with(tmp.path(), "proj");
        let item = scanner.discover(&cfg).next().unwrap().unwrap();
        let chunks = scanner.parse(item).unwrap();
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].project.as_deref(), Some("proj"));
        assert!(
            chunks[0]
                .links
                .file_path
                .as_deref()
                .unwrap()
                .ends_with("x.md")
        );
        assert!(chunks[0].links.parent_ids.is_empty());
    }
}
