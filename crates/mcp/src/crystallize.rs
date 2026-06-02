//! Relational-substrate slice 5: crystallize a typed concept node into a stub
//! markdown file — the "propose, then confirm" reification step.
//!
//! A recurring prose name is materialized as a `Proposed` typed node by the
//! scanner (no file yet). When a human confirms (via the `memory_concept`
//! `crystallize` action or the `ostk-recall crystallize` CLI), this writes a
//! stub `.md` under the source that declares the node's `entity_type`, so the
//! next scan ingests it as a first-class, content-bearing node. Shared by both
//! surfaces (`cli → mcp`); lives in `mcp` because `mcp → core`/`store` but
//! `mcp ↛ cli`.
//!
//! Safety: the slug is validated as a canonical handle here (so a crafted
//! handle cannot escape the source directory), and the file is created
//! atomically with `create_new` — it **never overwrites** an existing file,
//! even under a concurrent call.

use std::path::PathBuf;

use ostk_recall_core::config::expand_path;
use ostk_recall_core::{Config, SourceKind};
use ostk_recall_store::slugify;
use thiserror::Error;

/// Why a crystallize attempt could not produce a path.
#[derive(Debug, Error)]
pub enum CrystallizeError {
    /// `slug` is not a canonical handle — refused before any path join so a
    /// crafted handle (e.g. `../x`) cannot escape the source directory.
    #[error("invalid slug {slug:?}: not a canonical handle")]
    InvalidSlug { slug: String },
    /// No markdown source declares this `entity_type` for this project.
    #[error("no markdown source declares entity_type={kind:?} in project {project:?}")]
    NoSource { project: String, kind: String },
    /// The matched source declares no paths to write under.
    #[error("the markdown source for entity_type={kind:?} declares no paths")]
    MissingPath { kind: String },
    /// Path expansion (`~`/env) failed.
    #[error("path expansion failed: {0}")]
    Expand(String),
    /// Filesystem write failed.
    #[error("write failed: {0}")]
    Io(String),
}

/// A request to crystallize one resolved typed node. Built from the **resolved**
/// `ConceptRecord` (its own `project`/`kind`/`handle`), never a bare lookup —
/// `resolve_concept` may fall back to a global concept.
#[derive(Debug, Clone)]
pub struct CrystallizeRequest<'a> {
    /// The resolved node's project (`""` = global).
    pub project: &'a str,
    /// The resolved node's `entity_type` (selects the source + dir).
    pub kind: &'a str,
    /// The resolved node's handle (the file stem).
    pub slug: &'a str,
    /// Optional summary, written under the heading.
    pub summary: Option<&'a str>,
}

/// The outcome of a crystallize: the resolved path and whether a new file was
/// written (`false` = the file already existed; never overwritten).
#[derive(Debug, Clone)]
pub struct CrystallizeOutcome {
    pub path: PathBuf,
    pub created: bool,
}

/// Write a stub markdown file for `req` under the matching markdown source, or
/// report why it cannot. Idempotent: an existing file is left untouched
/// (`created == false`).
pub fn crystallize(
    cfg: &Config,
    req: &CrystallizeRequest,
) -> Result<CrystallizeOutcome, CrystallizeError> {
    // Traversal guard: the store schema does not enforce handle shape and this
    // helper is public, so refuse anything that isn't already a canonical handle
    // BEFORE any path join (`slugify(x) == Some(x)` iff x is canonical). Blocks
    // `../x` and friends from escaping the source directory.
    if slugify(req.slug).as_deref() != Some(req.slug) {
        return Err(CrystallizeError::InvalidSlug {
            slug: req.slug.to_string(),
        });
    }
    // First markdown source declaring this entity_type for this project. The
    // markdown filter is required: config permits `entity_type` on any source
    // kind, but crystallize writes a `.md` stub a markdown source will re-ingest.
    let source = cfg
        .sources
        .iter()
        .find(|s| {
            matches!(s.kind, SourceKind::Markdown)
                && s.entity_type.as_deref() == Some(req.kind)
                && s.project.as_deref().unwrap_or("") == req.project
        })
        .ok_or_else(|| CrystallizeError::NoSource {
            project: req.project.to_string(),
            kind: req.kind.to_string(),
        })?;

    let raw = source
        .paths
        .first()
        .ok_or_else(|| CrystallizeError::MissingPath {
            kind: req.kind.to_string(),
        })?;
    let dir = expand_path(raw).map_err(|e| CrystallizeError::Expand(e.to_string()))?;
    let path = dir.join(format!("{}.md", req.slug));

    let body = render_stub(req, &source.edges);
    std::fs::create_dir_all(&dir).map_err(|e| CrystallizeError::Io(e.to_string()))?;
    // Atomic create — never overwrite, even under a concurrent crystallize. An
    // exists-then-write check is a TOCTOU race; `create_new` lets the OS
    // arbitrate, and an already-present file yields `created: false`.
    match std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&path)
    {
        Ok(mut f) => {
            use std::io::Write as _;
            f.write_all(body.as_bytes())
                .map_err(|e| CrystallizeError::Io(e.to_string()))?;
            Ok(CrystallizeOutcome {
                path,
                created: true,
            })
        }
        Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => Ok(CrystallizeOutcome {
            path,
            created: false,
        }),
        Err(e) => Err(CrystallizeError::Io(e.to_string())),
    }
}

/// Render the stub: frontmatter (`project`, `kind`, one empty key per the
/// source's `edges` vocabulary so the file is pre-wired for slice-3 authored
/// edges), a title-cased heading, and an optional summary.
fn render_stub(req: &CrystallizeRequest, edges: &[String]) -> String {
    let mut s = String::new();
    s.push_str("---\n");
    s.push_str(&format!("project: {}\n", req.project));
    s.push_str(&format!("kind: {}\n", req.kind));
    for e in edges {
        s.push_str(&format!("{e}: []\n"));
    }
    s.push_str("---\n\n");
    s.push_str(&format!("# {}\n", title_case(req.slug)));
    if let Some(summary) = req.summary.filter(|v| !v.is_empty()) {
        s.push('\n');
        s.push_str(summary);
        s.push('\n');
    }
    s
}

/// `sarah-connor` → `Sarah Connor`. Splits on `-`/`_`, capitalizes each word.
fn title_case(slug: &str) -> String {
    slug.split(['-', '_'])
        .filter(|w| !w.is_empty())
        .map(|w| {
            let mut chars = w.chars();
            chars.next().map_or_else(String::new, |first| {
                first.to_uppercase().collect::<String>() + chars.as_str()
            })
        })
        .collect::<Vec<_>>()
        .join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;
    use ostk_recall_core::Config;
    use tempfile::TempDir;

    /// Build a config (via the real TOML load path) with one `person` markdown
    /// source rooted at `<dir>/people`.
    fn cfg_with_person_dir(dir: &std::path::Path) -> Config {
        let people = dir.join("people");
        let toml = format!(
            r#"
[corpus]
root = "{root}/corpus"
[embedder]
model = "minishlab/potion-retrieval-32M"
[[sources]]
kind = "markdown"
project = "memories"
paths = ["{people}"]
extensions = ["md"]
entity_type = "person"
edges = ["families", "works_on"]
"#,
            root = dir.display(),
            people = people.display(),
        );
        let cfg_path = dir.join("config.toml");
        std::fs::write(&cfg_path, toml).unwrap();
        Config::load(&cfg_path).unwrap()
    }

    #[test]
    fn writes_stub_then_never_overwrites() {
        let tmp = TempDir::new().unwrap();
        let cfg = cfg_with_person_dir(tmp.path());
        let req = CrystallizeRequest {
            project: "memories",
            kind: "person",
            slug: "sarah",
            summary: None,
        };
        let out = crystallize(&cfg, &req).unwrap();
        assert!(out.created);
        assert!(out.path.ends_with("people/sarah.md"));
        let body = std::fs::read_to_string(&out.path).unwrap();
        assert!(body.contains("kind: person"));
        assert!(body.contains("project: memories"));
        assert!(body.contains("families: []"));
        assert!(body.contains("# Sarah"));

        // Second call must not overwrite.
        std::fs::write(&out.path, "EDITED").unwrap();
        let out2 = crystallize(&cfg, &req).unwrap();
        assert!(!out2.created);
        assert_eq!(std::fs::read_to_string(&out2.path).unwrap(), "EDITED");
    }

    #[test]
    fn no_source_for_unknown_kind() {
        let tmp = TempDir::new().unwrap();
        let cfg = cfg_with_person_dir(tmp.path());
        let req = CrystallizeRequest {
            project: "memories",
            kind: "meeting",
            slug: "standup",
            summary: None,
        };
        assert!(matches!(
            crystallize(&cfg, &req),
            Err(CrystallizeError::NoSource { .. })
        ));
    }

    #[test]
    fn title_case_multiword() {
        assert_eq!(title_case("sarah"), "Sarah");
        assert_eq!(title_case("sarah-connor"), "Sarah Connor");
    }

    #[test]
    fn rejects_non_canonical_slug_traversal() {
        let tmp = TempDir::new().unwrap();
        let cfg = cfg_with_person_dir(tmp.path());
        let req = CrystallizeRequest {
            project: "memories",
            kind: "person",
            slug: "../escape",
            summary: None,
        };
        assert!(matches!(
            crystallize(&cfg, &req),
            Err(CrystallizeError::InvalidSlug { .. })
        ));
        assert!(!tmp.path().join("escape.md").exists());
    }
}
