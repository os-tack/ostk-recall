//! Relational-substrate slice 3: seed typed concept *nodes* + authored
//! *edges* into the concept ledger from markdown files in configured
//! directories.
//!
//! "Dropping a file is authoring" — the filesystem is the ontology
//! (`.ostk/threads/cognitive-memory/relational-substrate.md`).
//!
//! A file `memories/people/tori.md` under a source declaring
//! `entity_type = "person"` seeds a `person` node `tori`; a frontmatter field
//! named in `edges` (e.g. `families: [sarah]`) seeds an authored edge
//! `tori --families--> sarah`. Authored edges enter at the low prior
//! ([`AUTHORED_EDGE_CONFIDENCE`]) and earn conductance through use.
//!
//! This runs in the cli scan commands (where `ThreadsDb` is in scope) — the
//! ingest `Pipeline` has no ledger handle. Idempotent: re-seeding ensures
//! (never duplicates) and re-touches edges (bumps recency = use).

use std::path::{Path, PathBuf};

use ostk_recall_core::SourceConfig;
use ostk_recall_scan::threads::split_front_matter;
use ostk_recall_scan::walk::walk_filtered;
use ostk_recall_store::{
    AUTHORED_EDGE_CONFIDENCE, ConceptRecord, ConceptStatus, EdgeSource, ThreadsDb, slugify,
};
use serde_json::json;

/// Per-run seeding counts (logged at info; returned for tests).
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct SeedStats {
    /// Concepts freshly created (file-nodes + auto-created edge targets).
    pub nodes_seeded: u64,
    /// Concepts that already existed (re-ensured / kind-backfilled).
    pub nodes_touched: u64,
    /// Authored edges written (or re-touched) successfully.
    pub edges_seeded: u64,
    /// Markdown files visited.
    pub files_scanned: u64,
    /// Stems / edge targets that `slugify` rejected (too short, all-digit…).
    pub slug_rejects: u64,
}

/// `true` for a `*.md` file (frontmatter is markdown; node-seeding is
/// markdown-specific, independent of which `SourceKind` ingests the file
/// for recall).
fn is_md(p: &Path) -> bool {
    p.extension()
        .and_then(|e| e.to_str())
        .is_some_and(|e| e.eq_ignore_ascii_case("md"))
}

fn canonical_path(p: &Path) -> PathBuf {
    std::fs::canonicalize(p).unwrap_or_else(|_| p.to_path_buf())
}

/// `true` if `path` (already under `root`) is not excluded by the root's
/// ignore files (`.gitignore`/`.ostk-recall-ignore`) or the per-source
/// `ignore` patterns.
///
/// A per-path check for the incremental driver — keeps ignore semantics
/// consistent with the full scan without rewalking the tree on every edit. A
/// matcher-build failure errs open (seed it) rather than silently dropping.
fn path_seedable(root: &Path, ignore_patterns: &[String], path: &Path) -> bool {
    let Ok(rel) = path.strip_prefix(root) else {
        return false;
    };
    let mut b = ignore::gitignore::GitignoreBuilder::new(root);
    let _ = b.add(root.join(".gitignore"));
    let _ = b.add(root.join(".ostk-recall-ignore"));
    for pat in ignore_patterns {
        let _ = b.add_line(None, pat);
    }
    b.build()
        .map_or(true, |gi| !gi.matched(rel, false).is_ignore())
}

/// Coerce a frontmatter value into a list of target names. Accepts a YAML
/// sequence of strings OR a single scalar string; anything else → empty.
fn yaml_to_names(v: &serde_yaml::Value) -> Vec<String> {
    match v {
        serde_yaml::Value::String(s) => vec![s.clone()],
        serde_yaml::Value::Sequence(seq) => seq
            .iter()
            .filter_map(|x| x.as_str().map(ToString::to_string))
            .collect(),
        _ => Vec::new(),
    }
}

/// Resolve `raw` to its canonical concept, creating the node only if no
/// existing concept (handle/alias/merge-forward AND project→global fallback)
/// matches — so a frontmatter target never *shadows* a canonical concept it
/// merely aliases or that lives in a wider scope. Returns the full
/// [`ConceptRecord`] (id + scope) so the caller can insert edges *by id* and
/// avoid any handle re-resolution ambiguity. On create, emits a
/// `ConceptPromoted("proposed")` chain trace; an existing node has its NULL
/// `kind` backfilled in its own scope (COALESCE). `None` only if the name
/// doesn't slugify (counted as a reject).
fn canonical_or_create(
    threads: &ThreadsDb,
    project: &str,
    raw: &str,
    kind: Option<&str>,
    stats: &mut SeedStats,
) -> Option<ConceptRecord> {
    let Some(slug) = slugify(raw) else {
        stats.slug_rejects += 1;
        return None;
    };
    match threads.resolve_concept(project, &slug) {
        Ok(Some(rec)) => {
            // Backfill `kind` onto the canonical IN ITS OWN scope; never create
            // a `(project, handle)` copy that would shadow a wider-scope node.
            if kind.is_some() {
                if let Err(e) = threads.ensure_typed_concept(
                    &rec.project,
                    &rec.handle,
                    ConceptStatus::Proposed,
                    kind,
                ) {
                    tracing::warn!(handle = %rec.handle, error = %e, "kind backfill failed");
                }
            }
            stats.nodes_touched += 1;
            Some(rec)
        }
        Ok(None) => {
            match threads.ensure_typed_concept(project, &slug, ConceptStatus::Proposed, kind) {
                Ok((rec, created)) => {
                    if created {
                        stats.nodes_seeded += 1;
                        if let Err(e) = threads.record_concept_promoted(project, &slug, "proposed")
                        {
                            tracing::warn!(handle = %slug, error = %e, "ConceptPromoted emit failed");
                        }
                    } else {
                        stats.nodes_touched += 1;
                    }
                    Some(rec)
                }
                Err(e) => {
                    tracing::warn!(handle = %slug, error = %e, "ensure_typed_concept failed; skipping");
                    None
                }
            }
        }
        Err(e) => {
            tracing::warn!(handle = %slug, error = %e, "resolve failed; skipping");
            None
        }
    }
}

/// Seed one markdown file: its node, then an authored edge per `edges` field.
fn seed_file(
    threads: &ThreadsDb,
    cfg: &SourceConfig,
    kind: &str,
    project: &str,
    path: &Path,
    stats: &mut SeedStats,
) {
    stats.files_scanned += 1;
    let stem = path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or_default();
    let Some(from) = canonical_or_create(threads, project, stem, Some(kind), stats) else {
        return;
    };
    if cfg.edges.is_empty() {
        return;
    }
    let text = match std::fs::read_to_string(path) {
        Ok(t) => t,
        Err(e) => {
            tracing::warn!(path = %path.display(), error = %e, "read failed; node kept, no edges");
            return;
        }
    };
    let Some((yaml, _body)) = split_front_matter(&text) else {
        return; // no frontmatter → node only
    };
    let fm: std::collections::BTreeMap<String, serde_yaml::Value> = match serde_yaml::from_str(yaml)
    {
        Ok(m) => m,
        Err(e) => {
            tracing::warn!(path = %path.display(), error = %e, "frontmatter parse failed; node kept");
            return;
        }
    };
    for field in &cfg.edges {
        let Some(val) = fm.get(field) else { continue };
        for name in yaml_to_names(val) {
            let Some(to) = canonical_or_create(threads, project, &name, None, stats) else {
                continue;
            };
            let evidence =
                json!({ "source_config_id": cfg.source_config_id, "field": field }).to_string();
            // Insert *by id* so the exact resolved endpoints are used — a
            // same-handle local concept can't shadow a global one the alias
            // resolved to. `by = "scanner"` is the provenance tag (the scan id
            // lives in `evidence`). Self-loops are rejected → skipped.
            match threads.add_concept_edge_by_id(
                from.id,
                field,
                to.id,
                AUTHORED_EDGE_CONFIDENCE,
                EdgeSource::Authored,
                Some("scanner"),
                Some(&evidence),
            ) {
                Ok((_, created)) => {
                    stats.edges_seeded += 1;
                    // Emit the cognition-stream event only on first authoring,
                    // never on idempotent re-scan (re-touch is silent use).
                    if created {
                        if let Err(e) = threads.record_concept_connected(
                            project,
                            &from.handle,
                            field,
                            &to.handle,
                            EdgeSource::Authored,
                            Some("scanner"),
                        ) {
                            tracing::warn!(error = %e, "ConceptConnected emit failed");
                        }
                    }
                }
                Err(e) => {
                    tracing::debug!(from = %from.handle, to = %to.handle, error = %e, "edge skipped");
                }
            }
        }
    }
}

/// Full-scan driver: seed every markdown file under the source's roots.
///
/// No-op unless `entity_type` is set. Honors `.gitignore`/`.ostk-recall-ignore`/
/// `cfg.ignore` via [`walk_filtered`] — seeds exactly what the corpus ingests.
pub fn seed_nodes_from_source(threads: &ThreadsDb, cfg: &SourceConfig) -> SeedStats {
    let mut stats = SeedStats::default();
    let Some(kind) = cfg.entity_type.as_deref() else {
        return stats;
    };
    let project = cfg.project.as_deref().unwrap_or("");
    let roots = cfg.expanded_paths().unwrap_or_default();
    for root in roots {
        for entry in walk_filtered(&root, &cfg.ignore) {
            let p = entry.path();
            if is_md(p) {
                seed_file(threads, cfg, kind, project, p, &mut stats);
            }
        }
    }
    stats
}

/// Incremental driver: seed only the changed `paths` that fall under this
/// source's roots.
///
/// Reuses [`walk_filtered`] so ignore semantics match the full scan exactly
/// (a changed-but-ignored file is never seeded), then intersects with the
/// changed set. No-op unless `entity_type` is set.
pub fn seed_nodes_for_paths(
    threads: &ThreadsDb,
    cfg: &SourceConfig,
    paths: &[PathBuf],
) -> SeedStats {
    let mut stats = SeedStats::default();
    let Some(kind) = cfg.entity_type.as_deref() else {
        return stats;
    };
    let project = cfg.project.as_deref().unwrap_or("");
    let roots: Vec<PathBuf> = cfg
        .expanded_paths()
        .unwrap_or_default()
        .iter()
        .map(|r| canonical_path(r))
        .collect();
    // O(changed paths), not O(tree): seed each changed `.md` file iff it sits
    // under one of this source's roots and isn't excluded. No per-edit rewalk.
    for p in paths {
        if !is_md(p) {
            continue;
        }
        let pc = canonical_path(p);
        // A delete/rename event names a path that no longer exists. Skip it —
        // otherwise seed_file would create/re-promote a node for a file that
        // was just removed (read fails *after* the node is written).
        if !pc.is_file() {
            continue;
        }
        if let Some(root) = roots.iter().find(|r| pc.starts_with(r)) {
            if path_seedable(root, &cfg.ignore, &pc) {
                seed_file(threads, cfg, kind, project, &pc, &mut stats);
            }
        }
    }
    stats
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    use ostk_recall_core::SourceKind;
    use ostk_recall_store::{AliasSource, ChainEvent, ChainSink, EdgeDirection, SqliteChainSink};
    use tempfile::TempDir;

    const PROJ: &str = "memories";

    fn src(dir: &Path, entity_type: Option<&str>, edges: &[&str], ignore: &[&str]) -> SourceConfig {
        SourceConfig {
            kind: SourceKind::Markdown,
            project: Some(PROJ.into()),
            paths: vec![dir.to_string_lossy().into_owned()],
            ignore: ignore.iter().map(ToString::to_string).collect(),
            extensions: vec![],
            facets: std::collections::BTreeMap::new(),
            id: None,
            entity_type: entity_type.map(ToString::to_string),
            edges: edges.iter().map(ToString::to_string).collect(),
            source_config_id: "seed-test".into(),
        }
    }

    fn write_md(dir: &Path, name: &str, body: &str) -> PathBuf {
        let p = dir.join(name);
        std::fs::write(&p, body).unwrap();
        p
    }

    #[test]
    fn seeds_proposed_node_with_kind() {
        let tmp = TempDir::new().unwrap();
        let db = ThreadsDb::open(tmp.path()).unwrap();
        let dir = tmp.path().join("people");
        std::fs::create_dir_all(&dir).unwrap();
        write_md(&dir, "tori.md", "no frontmatter here\n");
        let stats = seed_nodes_from_source(&db, &src(&dir, Some("person"), &[], &[]));
        assert_eq!(stats.nodes_seeded, 1);
        assert_eq!(stats.edges_seeded, 0);
        let rec = db.get_concept(PROJ, "tori").unwrap().unwrap();
        assert_eq!(rec.kind.as_deref(), Some("person"));
        assert_eq!(rec.status, ConceptStatus::Proposed);
    }

    #[test]
    fn frontmatter_creates_authored_edge_with_target() {
        let tmp = TempDir::new().unwrap();
        let db = ThreadsDb::open(tmp.path()).unwrap();
        let dir = tmp.path().join("people");
        std::fs::create_dir_all(&dir).unwrap();
        write_md(&dir, "tori.md", "---\nfamilies: [sarah]\n---\nbody\n");
        let stats = seed_nodes_from_source(&db, &src(&dir, Some("person"), &["families"], &[]));
        assert_eq!(stats.edges_seeded, 1);
        let edges = db
            .list_concept_edges(PROJ, "tori", EdgeDirection::From)
            .unwrap();
        assert_eq!(edges.len(), 1);
        assert_eq!(edges[0].to_handle, "sarah");
        assert_eq!(edges[0].relation, "families");
        assert_eq!(edges[0].source, EdgeSource::Authored);
        assert!((edges[0].confidence - AUTHORED_EDGE_CONFIDENCE).abs() < 1e-6);
        // Target auto-created, untyped (kind set only by its own file).
        let sarah = db.get_concept(PROJ, "sarah").unwrap().unwrap();
        assert_eq!(sarah.kind, None);
    }

    #[test]
    fn edge_field_accepts_scalar_string() {
        let tmp = TempDir::new().unwrap();
        let db = ThreadsDb::open(tmp.path()).unwrap();
        let dir = tmp.path().to_path_buf();
        write_md(&dir, "tori.md", "---\nfamilies: sarah\n---\n");
        seed_nodes_from_source(&db, &src(&dir, Some("person"), &["families"], &[]));
        let edges = db
            .list_concept_edges(PROJ, "tori", EdgeDirection::From)
            .unwrap();
        assert_eq!(edges.len(), 1);
        assert_eq!(edges[0].to_handle, "sarah");
    }

    #[test]
    fn reseed_is_idempotent() {
        let tmp = TempDir::new().unwrap();
        let db = ThreadsDb::open(tmp.path()).unwrap();
        let dir = tmp.path().to_path_buf();
        write_md(&dir, "tori.md", "---\nfamilies: [sarah]\n---\n");
        let cfg = src(&dir, Some("person"), &["families"], &[]);
        seed_nodes_from_source(&db, &cfg);
        seed_nodes_from_source(&db, &cfg);
        // Node not duplicated; edge re-touched (use), origin preserved.
        assert_eq!(db.list_concepts(Some(PROJ), None).unwrap().len(), 2); // tori + sarah
        let edges = db
            .list_concept_edges(PROJ, "tori", EdgeDirection::From)
            .unwrap();
        assert_eq!(edges.len(), 1);
        assert_eq!(edges[0].touch_count, 2);
        assert_eq!(edges[0].source, EdgeSource::Authored);
    }

    #[test]
    fn slug_reject_handled_gracefully() {
        let tmp = TempDir::new().unwrap();
        let db = ThreadsDb::open(tmp.path()).unwrap();
        let dir = tmp.path().to_path_buf();
        write_md(&dir, "ab.md", "x\n"); // 2-char stem → slugify rejects
        let stats = seed_nodes_from_source(&db, &src(&dir, Some("person"), &[], &[]));
        assert_eq!(stats.slug_rejects, 1);
        assert_eq!(stats.nodes_seeded, 0);
    }

    #[test]
    fn non_entity_source_is_noop() {
        let tmp = TempDir::new().unwrap();
        let db = ThreadsDb::open(tmp.path()).unwrap();
        let dir = tmp.path().to_path_buf();
        write_md(&dir, "tori.md", "x\n");
        let stats = seed_nodes_from_source(&db, &src(&dir, None, &[], &[]));
        assert_eq!(stats, SeedStats::default());
        assert!(db.get_concept(PROJ, "tori").unwrap().is_none());
    }

    #[test]
    fn project_scoping() {
        let tmp = TempDir::new().unwrap();
        let db = ThreadsDb::open(tmp.path()).unwrap();
        let dir = tmp.path().to_path_buf();
        write_md(&dir, "tori.md", "x\n");
        seed_nodes_from_source(&db, &src(&dir, Some("person"), &[], &[]));
        assert!(db.get_concept(PROJ, "tori").unwrap().is_some());
        assert!(db.get_concept("", "tori").unwrap().is_none(), "not global");
    }

    #[test]
    fn self_loop_target_skipped() {
        let tmp = TempDir::new().unwrap();
        let db = ThreadsDb::open(tmp.path()).unwrap();
        let dir = tmp.path().to_path_buf();
        write_md(&dir, "tori.md", "---\nfamilies: [tori]\n---\n");
        let stats = seed_nodes_from_source(&db, &src(&dir, Some("person"), &["families"], &[]));
        assert_eq!(stats.edges_seeded, 0, "self-loop rejected");
        assert!(
            db.list_concept_edges(PROJ, "tori", EdgeDirection::From)
                .unwrap()
                .is_empty()
        );
    }

    #[test]
    fn alias_shadow_guard() {
        let tmp = TempDir::new().unwrap();
        let db = ThreadsDb::open(tmp.path()).unwrap();
        // Pre-seed canonical `sarah-connor` with alias `sarah`.
        let (rec, _) = db
            .ensure_typed_concept(PROJ, "sarah-connor", ConceptStatus::Active, Some("person"))
            .unwrap();
        db.touch_alias(rec.id, "sarah", AliasSource::User).unwrap();
        let dir = tmp.path().to_path_buf();
        write_md(&dir, "tori.md", "---\nfamilies: [sarah]\n---\n");
        seed_nodes_from_source(&db, &src(&dir, Some("person"), &["families"], &[]));
        // Edge targets the canonical concept, NOT a fresh shadow `sarah`.
        let edges = db
            .list_concept_edges(PROJ, "tori", EdgeDirection::From)
            .unwrap();
        assert_eq!(edges.len(), 1);
        assert_eq!(edges[0].to_handle, "sarah-connor");
        assert!(
            db.get_concept(PROJ, "sarah").unwrap().is_none(),
            "no shadow node"
        );
    }

    #[test]
    fn global_canonical_not_shadowed() {
        let tmp = TempDir::new().unwrap();
        let db = ThreadsDb::open(tmp.path()).unwrap();
        // A GLOBAL canonical concept (project "") with alias `sarah`.
        let (rec, _) = db
            .ensure_typed_concept("", "sarah-connor", ConceptStatus::Active, Some("person"))
            .unwrap();
        db.touch_alias(rec.id, "sarah", AliasSource::User).unwrap();
        let dir = tmp.path().to_path_buf();
        write_md(&dir, "tori.md", "---\nfamilies: [sarah]\n---\n");
        seed_nodes_from_source(&db, &src(&dir, Some("person"), &["families"], &[]));
        // The edge resolves to the GLOBAL canonical — NOT a local copy in
        // `memories` (resolve_concept falls back to global; we must not shadow).
        let edges = db
            .list_concept_edges(PROJ, "tori", EdgeDirection::From)
            .unwrap();
        assert_eq!(edges.len(), 1);
        assert_eq!(edges[0].to_handle, "sarah-connor");
        assert!(
            db.get_concept(PROJ, "sarah-connor").unwrap().is_none(),
            "no local shadow of the global concept"
        );
        assert!(db.get_concept(PROJ, "sarah").unwrap().is_none());
    }

    #[test]
    fn incremental_skips_ignored_changed_path() {
        let tmp = TempDir::new().unwrap();
        let db = ThreadsDb::open(tmp.path()).unwrap();
        let dir = tmp.path().to_path_buf();
        let secret = write_md(&dir, "secret.md", "x\n");
        let stats = seed_nodes_for_paths(
            &db,
            &src(&dir, Some("person"), &[], &["secret.md"]),
            &[secret],
        );
        assert_eq!(stats.files_scanned, 0, "ignored changed path not seeded");
        assert!(db.get_concept(PROJ, "secret").unwrap().is_none());
    }

    #[test]
    fn incremental_skips_path_outside_root() {
        let tmp = TempDir::new().unwrap();
        let db = ThreadsDb::open(tmp.path()).unwrap();
        let root = tmp.path().join("people");
        std::fs::create_dir_all(&root).unwrap();
        let outside = write_md(tmp.path(), "stray.md", "x\n"); // sibling of root, not under it
        seed_nodes_for_paths(&db, &src(&root, Some("person"), &[], &[]), &[outside]);
        assert!(
            db.get_concept(PROJ, "stray").unwrap().is_none(),
            "path outside root skipped"
        );
    }

    #[test]
    fn deleted_changed_path_does_not_seed() {
        let tmp = TempDir::new().unwrap();
        let db = ThreadsDb::open(tmp.path()).unwrap();
        let dir = tmp.path().to_path_buf();
        let ghost = dir.join("ghost.md"); // never created (a delete event)
        let stats = seed_nodes_for_paths(&db, &src(&dir, Some("person"), &[], &[]), &[ghost]);
        assert_eq!(stats.files_scanned, 0, "missing path is not seeded");
        assert!(db.get_concept(PROJ, "ghost").unwrap().is_none());
    }

    #[test]
    fn target_resolves_to_global_not_local_same_handle() {
        let tmp = TempDir::new().unwrap();
        let db = ThreadsDb::open(tmp.path()).unwrap();
        // Global canonical with alias `sarah`...
        let (g, _) = db
            .ensure_typed_concept("", "sarah-connor", ConceptStatus::Active, Some("person"))
            .unwrap();
        db.touch_alias(g.id, "sarah", AliasSource::User).unwrap();
        // ...and a DIFFERENT local concept that happens to share the handle.
        db.ensure_typed_concept(PROJ, "sarah-connor", ConceptStatus::Active, Some("robot"))
            .unwrap();
        let dir = tmp.path().to_path_buf();
        write_md(&dir, "tori.md", "---\nfamilies: [sarah]\n---\n");
        seed_nodes_from_source(&db, &src(&dir, Some("person"), &["families"], &[]));
        // `sarah` resolves to the GLOBAL canonical; the by-id edge targets it,
        // NOT the local same-handle concept (which handle-resolution prefers).
        let to_global = db
            .list_concept_edges("", "sarah-connor", EdgeDirection::To)
            .unwrap();
        assert_eq!(to_global.len(), 1, "edge targets the global canonical");
        let to_local = db
            .list_concept_edges(PROJ, "sarah-connor", EdgeDirection::To)
            .unwrap();
        assert!(to_local.is_empty(), "local same-handle concept got no edge");
    }

    #[test]
    fn ignored_file_not_seeded() {
        let tmp = TempDir::new().unwrap();
        let db = ThreadsDb::open(tmp.path()).unwrap();
        let dir = tmp.path().to_path_buf();
        write_md(&dir, "tori.md", "x\n");
        write_md(&dir, "secret.md", "x\n");
        seed_nodes_from_source(&db, &src(&dir, Some("person"), &[], &["secret.md"]));
        assert!(db.get_concept(PROJ, "tori").unwrap().is_some());
        assert!(
            db.get_concept(PROJ, "secret").unwrap().is_none(),
            "ignored file skipped"
        );
    }

    #[test]
    fn incremental_seeds_only_changed_paths() {
        let tmp = TempDir::new().unwrap();
        let db = ThreadsDb::open(tmp.path()).unwrap();
        let dir = tmp.path().to_path_buf();
        let tori = write_md(&dir, "tori.md", "x\n");
        write_md(&dir, "mike.md", "x\n");
        let stats = seed_nodes_for_paths(&db, &src(&dir, Some("person"), &[], &[]), &[tori]);
        assert_eq!(stats.files_scanned, 1);
        assert!(db.get_concept(PROJ, "tori").unwrap().is_some());
        assert!(
            db.get_concept(PROJ, "mike").unwrap().is_none(),
            "unchanged file not seeded"
        );
    }

    #[test]
    fn emits_chain_events_on_creation_only() {
        let tmp = TempDir::new().unwrap();
        let sink: Arc<dyn ChainSink> = Arc::new(SqliteChainSink::open(tmp.path()).unwrap());
        let db = ThreadsDb::open_with_sink(tmp.path(), Arc::clone(&sink)).unwrap();
        let dir = tmp.path().join("people");
        std::fs::create_dir_all(&dir).unwrap();
        write_md(&dir, "tori.md", "---\nfamilies: [sarah]\n---\n");
        let cfg = src(&dir, Some("person"), &["families"], &[]);
        seed_nodes_from_source(&db, &cfg);
        let kinds: Vec<String> = db
            .iter_chain()
            .unwrap()
            .iter()
            .map(|e| ChainEvent::kind_str(e).to_string())
            .collect();
        assert_eq!(
            kinds.iter().filter(|k| *k == "concept_promoted").count(),
            2,
            "tori + sarah each promoted once"
        );
        assert_eq!(
            kinds.iter().filter(|k| *k == "concept_connected").count(),
            1
        );
        // Re-scan: idempotent → no new chain events.
        let before = db.iter_chain().unwrap().len();
        seed_nodes_from_source(&db, &cfg);
        assert_eq!(
            db.iter_chain().unwrap().len(),
            before,
            "rescan emits nothing"
        );
    }
}
