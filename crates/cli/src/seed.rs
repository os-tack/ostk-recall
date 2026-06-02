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

use std::collections::{BTreeSet, HashMap};
use std::path::{Path, PathBuf};

use aho_corasick::{AhoCorasick, MatchKind};
use ostk_recall_core::SourceConfig;
use ostk_recall_scan::threads::split_front_matter;
use ostk_recall_scan::walk::walk_filtered;
use ostk_recall_store::{
    AUTHORED_EDGE_CONFIDENCE, ConceptRecord, ConceptStatus, EdgeSource,
    OBSERVED_MENTION_CONFIDENCE, ThreadsDb, slugify,
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

// ---------------------------------------------------------------------------
// Relational-substrate slice 4: gazetteer prose mention-linking.
//
// After node-seeding (slice 3) has run for every source, plain prose in a
// file's BODY that *names* a known node lights an `observed` `mentions` edge
// from that file's node to the named node — no markup, no wiki-links (the
// anti-Obsidian win). Authored edges (slice 3) are the topology hypothesis;
// observed edges are evidence of use. Both are gated by the same conductance
// rule: use, not assertion, is what makes current flow.
// ---------------------------------------------------------------------------

/// Edge relation for a bare prose mention (generic; typed edges come from
/// frontmatter `edges` fields in slice 3).
const MENTION_RELATION: &str = "mentions";

/// Surface-form floor (in normalized chars). Shorter handles/aliases (`io`,
/// `id`) are too noisy to match safely against free prose.
const MIN_MENTION_CHARS: usize = 3;

/// Per-run mention-linking counts (logged at info; returned for tests).
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct MentionStats {
    /// Markdown files whose body was scanned for mentions.
    pub files_scanned: u64,
    /// Observed `mentions` edges written (or re-touched) successfully.
    pub mentions_linked: u64,
    /// Distinct surface forms dropped because they were ambiguous within a
    /// scope tier (named, not silent — the disambiguation seam).
    pub ambiguous_skipped: u64,
}

/// Surface-form claims, split into the four tiers `resolve_concept` consults
/// in order (`resolve_one`, concepts.rs): a **handle** beats an **alias**, and
/// within each the file's **project** beats **global**. The first non-empty
/// tier wins; only a collision *within* that winning tier is ambiguous. (A
/// handle tier usually holds one id since `(project, handle)` is unique, but
/// distinct handles can *normalize* to the same form — e.g. `foo-bar` and
/// `foo_bar` both fold to `foo bar` — so any tier, handle or alias, may be
/// ambiguous and is dropped.)
#[derive(Default)]
struct FormClaims {
    project_handle: BTreeSet<i64>,
    global_handle: BTreeSet<i64>,
    project_alias: BTreeSet<i64>,
    global_alias: BTreeSet<i64>,
}

/// Outcome of resolving a surface form to a single concept id.
enum Pick {
    One(i64),
    Ambiguous,
    None,
}

impl FormClaims {
    /// First non-empty tier, in `resolve_one` order; `Ambiguous` if that tier
    /// names more than one distinct concept.
    fn pick(&self) -> Pick {
        let tiers = [
            &self.project_handle,
            &self.global_handle,
            &self.project_alias,
            &self.global_alias,
        ];
        match tiers.into_iter().find(|t| !t.is_empty()) {
            None => Pick::None,
            Some(t) if t.len() == 1 => Pick::One(*t.iter().next().expect("len==1")),
            Some(_) => Pick::Ambiguous,
        }
    }
}

/// A built matcher over one project's known node surface forms (handles +
/// aliases of project∪global concepts), plus the id→handle map for chain
/// events. Empty `ac` (no forms) matches nothing.
struct Gazetteer {
    ac: Option<AhoCorasick>,
    /// Parallel to the automaton's pattern ids: `ids[pattern] = concept id`.
    ids: Vec<i64>,
    /// Canonical handle per concept id (for the `ConceptConnected` event).
    handles: HashMap<i64, String>,
}

impl Gazetteer {
    /// Distinct concept ids named in `body_norm` (already normalized),
    /// excluding `self_id`. Word boundaries are enforced against the
    /// space-delimited normalized text, so `ostk` never matches inside
    /// `ostkx`.
    fn matches(&self, body_norm: &str, self_id: i64) -> Vec<i64> {
        let Some(ac) = self.ac.as_ref() else {
            return Vec::new();
        };
        let bytes = body_norm.as_bytes();
        let mut found = BTreeSet::new();
        for m in ac.find_iter(body_norm) {
            let (s, e) = (m.start(), m.end());
            let before = s == 0 || bytes[s - 1] == b' ';
            let after = e == bytes.len() || bytes[e] == b' ';
            if before && after {
                let id = self.ids[m.pattern().as_usize()];
                if id != self_id {
                    found.insert(id);
                }
            }
        }
        found.into_iter().collect()
    }
}

/// Lowercase, replace every non-alphanumeric char with a space, and collapse
/// runs of whitespace. Applied identically to gazetteer forms and file bodies
/// so hyphens, markdown (`**ostk-recall**`), and punctuation fold away and
/// `ostk-recall` ⇒ `ostk recall` matches both hyphenated and spaced prose.
fn normalize(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut prev_space = true; // leading run is trimmed
    for ch in s.chars() {
        if ch.is_alphanumeric() {
            out.extend(ch.to_lowercase());
            prev_space = false;
        } else if !prev_space {
            out.push(' ');
            prev_space = true;
        }
    }
    if out.ends_with(' ') {
        out.pop();
    }
    out
}

/// Normalize a surface form and apply the length floor; `None` if too short.
fn gaz_form(raw: &str) -> Option<String> {
    let norm = normalize(raw);
    (norm.chars().count() >= MIN_MENTION_CHARS).then_some(norm)
}

/// Build the gazetteer for `project` (its own concepts + globals). Terminal
/// (`rejected`/`merged`) concepts are excluded as targets; a merged-away
/// handle still resolves because it now lives as an alias of the canonical.
/// Returns the matcher and the count of ambiguous surface forms dropped.
fn build_gazetteer(threads: &ThreadsDb, project: &str) -> (Gazetteer, u64) {
    let concepts = threads
        .list_concepts(Some(project), None)
        .unwrap_or_default();
    let mut claims: HashMap<String, FormClaims> = HashMap::new();
    let mut handles: HashMap<i64, String> = HashMap::new();
    for c in &concepts {
        if c.status.is_terminal() {
            continue;
        }
        handles.insert(c.id, c.handle.clone());
        let project_scoped = !c.project.is_empty();
        // Handle claim (the highest-precedence tier).
        if let Some(norm) = gaz_form(&c.handle) {
            let entry = claims.entry(norm).or_default();
            if project_scoped {
                entry.project_handle.insert(c.id);
            } else {
                entry.global_handle.insert(c.id);
            }
        }
        // Alias claims (lower precedence than any handle).
        if let Ok(aliases) = threads.list_aliases(c.id) {
            for a in aliases {
                if let Some(norm) = gaz_form(&a.alias) {
                    let entry = claims.entry(norm).or_default();
                    if project_scoped {
                        entry.project_alias.insert(c.id);
                    } else {
                        entry.global_alias.insert(c.id);
                    }
                }
            }
        }
    }

    let mut patterns: Vec<String> = Vec::new();
    let mut ids: Vec<i64> = Vec::new();
    let mut ambiguous = 0u64;
    for (form, fc) in claims {
        match fc.pick() {
            Pick::One(id) => {
                ids.push(id);
                patterns.push(form);
            }
            Pick::Ambiguous => ambiguous += 1,
            Pick::None => {}
        }
    }

    let ac = if patterns.is_empty() {
        None
    } else {
        AhoCorasick::builder()
            .match_kind(MatchKind::LeftmostLongest)
            .build(&patterns)
            .ok()
    };
    (Gazetteer { ac, ids, handles }, ambiguous)
}

/// Scan one markdown file's body for gazetteer mentions and write observed
/// edges from the file's node to each distinct named node (≠ self).
fn link_file_mentions(
    threads: &ThreadsDb,
    gaz: &Gazetteer,
    cfg: &SourceConfig,
    project: &str,
    path: &Path,
    stats: &mut MentionStats,
) {
    stats.files_scanned += 1;
    let stem = path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or_default();
    let Some(slug) = slugify(stem) else {
        return;
    };
    // Resolve the file's own node the SAME way slice 3 does (alias / merge /
    // project→global aware), so a file whose stem resolves to a canonical or
    // global concept links from that record — never a fresh local shadow.
    let self_rec = match threads.resolve_concept(project, &slug) {
        Ok(Some(rec)) => rec,
        Ok(None) => return, // no node for this file (node-seeding didn't run)
        Err(e) => {
            tracing::warn!(handle = %slug, error = %e, "mention: resolve self failed; skipping");
            return;
        }
    };
    let text = match std::fs::read_to_string(path) {
        Ok(t) => t,
        Err(e) => {
            tracing::warn!(path = %path.display(), error = %e, "mention: read failed; skipping");
            return;
        }
    };
    // Body only — frontmatter `edges` fields are slice 3's job.
    let body = split_front_matter(&text).map_or(text.as_str(), |(_, b)| b);
    let body_norm = normalize(body);
    let targets = gaz.matches(&body_norm, self_rec.id);
    if targets.is_empty() {
        return;
    }
    let evidence =
        json!({ "source_config_id": cfg.source_config_id, "via": "gazetteer" }).to_string();
    for tid in targets {
        match threads.add_concept_edge_by_id(
            self_rec.id,
            MENTION_RELATION,
            tid,
            OBSERVED_MENTION_CONFIDENCE,
            EdgeSource::Observed,
            Some("scanner"),
            Some(&evidence),
        ) {
            Ok((_, created)) => {
                stats.mentions_linked += 1;
                // The authoritative edge is the by-id row inserted above. This
                // chain event is AUDIT-ONLY: `ConceptConnected` carries a single
                // `project` + bare handles (no endpoint ids/scopes), so a
                // cross-scope mention (e.g. a global file-node) records the
                // file's source project for both endpoints. If replay ever
                // becomes authoritative (slice 2), the event needs endpoint
                // ids/scopes. Mirrors slice 3's `seed_file`.
                if created {
                    if let Some(to_handle) = gaz.handles.get(&tid) {
                        if let Err(e) = threads.record_concept_connected(
                            project,
                            &self_rec.handle,
                            MENTION_RELATION,
                            to_handle,
                            EdgeSource::Observed,
                            Some("scanner"),
                        ) {
                            tracing::warn!(error = %e, "mention: ConceptConnected emit failed");
                        }
                    }
                }
            }
            Err(e) => {
                tracing::debug!(from = %self_rec.handle, to_id = tid, error = %e, "mention edge skipped");
            }
        }
    }
}

/// Full-scan driver: link prose mentions for every markdown file under each
/// `entity_type` source.
///
/// Runs AFTER node-seeding for all sources so the gazetteer is complete (a
/// person can mention a meeting). The gazetteer is built once per project and
/// reused across sources sharing that project.
pub fn link_mentions_from_sources(threads: &ThreadsDb, sources: &[&SourceConfig]) -> MentionStats {
    let mut stats = MentionStats::default();
    let mut gaz_cache: HashMap<String, Gazetteer> = HashMap::new();
    for cfg in sources {
        if cfg.entity_type.is_none() {
            continue;
        }
        let project = cfg.project.as_deref().unwrap_or("");
        if !gaz_cache.contains_key(project) {
            let (gaz, ambiguous) = build_gazetteer(threads, project);
            stats.ambiguous_skipped += ambiguous;
            gaz_cache.insert(project.to_string(), gaz);
        }
        let gaz = &gaz_cache[project];
        for root in cfg.expanded_paths().unwrap_or_default() {
            for entry in walk_filtered(&root, &cfg.ignore) {
                let p = entry.path();
                if is_md(p) {
                    link_file_mentions(threads, gaz, cfg, project, p, &mut stats);
                }
            }
        }
    }
    stats
}

/// Incremental driver: link mentions only for the changed `paths` that fall
/// under an `entity_type` source.
///
/// Reuses the exact delete / ignore / root guards [`seed_nodes_for_paths`]
/// applies so a deleted or ignored changed path is treated identically by both
/// passes. The gazetteer is rebuilt from the DB (so it is complete after this
/// scan's node-seeding).
pub fn link_mentions_for_paths(
    threads: &ThreadsDb,
    sources: &[&SourceConfig],
    paths: &[PathBuf],
) -> MentionStats {
    let mut stats = MentionStats::default();
    let mut gaz_cache: HashMap<String, Gazetteer> = HashMap::new();
    for cfg in sources {
        if cfg.entity_type.is_none() {
            continue;
        }
        let project = cfg.project.as_deref().unwrap_or("");
        let roots: Vec<PathBuf> = cfg
            .expanded_paths()
            .unwrap_or_default()
            .iter()
            .map(|r| canonical_path(r))
            .collect();
        if !gaz_cache.contains_key(project) {
            let (gaz, ambiguous) = build_gazetteer(threads, project);
            stats.ambiguous_skipped += ambiguous;
            gaz_cache.insert(project.to_string(), gaz);
        }
        let gaz = &gaz_cache[project];
        for p in paths {
            if !is_md(p) {
                continue;
            }
            let pc = canonical_path(p);
            if !pc.is_file() {
                continue; // delete/rename event — path no longer exists
            }
            if let Some(root) = roots.iter().find(|r| pc.starts_with(r)) {
                if path_seedable(root, &cfg.ignore, &pc) {
                    link_file_mentions(threads, gaz, cfg, project, &pc, &mut stats);
                }
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

    // --- Slice 4: gazetteer prose mention-linking ---------------------------

    /// Collect the `mentions` edges out of `(project, handle)` as
    /// `(to_handle, source, confidence, touch_count)`.
    fn mentions_from(
        db: &ThreadsDb,
        project: &str,
        handle: &str,
    ) -> Vec<(String, EdgeSource, f32, u32)> {
        db.list_concept_edges(project, handle, EdgeDirection::From)
            .unwrap()
            .into_iter()
            .filter(|e| e.relation == "mentions")
            .map(|e| (e.to_handle, e.source, e.confidence, e.touch_count))
            .collect()
    }

    #[test]
    fn mention_links_observed_edge() {
        let tmp = TempDir::new().unwrap();
        let db = ThreadsDb::open(tmp.path()).unwrap();
        let dir = tmp.path().to_path_buf();
        write_md(&dir, "tori.md", "Tori works on ostk-recall with Mike.\n");
        write_md(&dir, "mike.md", "An engineer.\n");
        write_md(&dir, "ostk-recall.md", "The memory layer.\n");
        let cfg = src(&dir, Some("person"), &[], &[]);
        seed_nodes_from_source(&db, &cfg);
        let stats = link_mentions_from_sources(&db, &[&cfg]);
        let m = mentions_from(&db, PROJ, "tori");
        assert_eq!(m.len(), 2, "tori names mike + ostk-recall (not self)");
        assert!(
            m.iter()
                .all(|(_, src, conf, _)| *src == EdgeSource::Observed
                    && (*conf - OBSERVED_MENTION_CONFIDENCE).abs() < 1e-6)
        );
        assert!(m.iter().any(|(h, ..)| h == "mike"));
        assert!(m.iter().any(|(h, ..)| h == "ostk-recall"));
        assert_eq!(stats.mentions_linked, 2);
        assert_eq!(stats.files_scanned, 3);
    }

    #[test]
    fn mention_skips_self() {
        let tmp = TempDir::new().unwrap();
        let db = ThreadsDb::open(tmp.path()).unwrap();
        let dir = tmp.path().to_path_buf();
        write_md(&dir, "tori.md", "Tori thinks about Tori a lot.\n");
        let cfg = src(&dir, Some("person"), &[], &[]);
        seed_nodes_from_source(&db, &cfg);
        link_mentions_from_sources(&db, &[&cfg]);
        assert!(
            mentions_from(&db, PROJ, "tori").is_empty(),
            "no self-mention edge"
        );
    }

    #[test]
    fn mention_respects_word_boundary() {
        let tmp = TempDir::new().unwrap();
        let db = ThreadsDb::open(tmp.path()).unwrap();
        let dir = tmp.path().to_path_buf();
        write_md(&dir, "mike.md", "node\n");
        write_md(&dir, "bob.md", "mikexyz appears, never standalone\n"); // substring only
        write_md(&dir, "amy.md", "amy saw Mike today\n"); // standalone
        let cfg = src(&dir, Some("person"), &[], &[]);
        seed_nodes_from_source(&db, &cfg);
        link_mentions_from_sources(&db, &[&cfg]);
        assert!(
            mentions_from(&db, PROJ, "bob").is_empty(),
            "substring is not a mention"
        );
        let amy = mentions_from(&db, PROJ, "amy");
        assert_eq!(amy.len(), 1);
        assert_eq!(amy[0].0, "mike");
    }

    #[test]
    fn mention_skips_ambiguous_surface() {
        let tmp = TempDir::new().unwrap();
        let db = ThreadsDb::open(tmp.path()).unwrap();
        let dir = tmp.path().to_path_buf();
        // Two project concepts claim the same alias `acme` → ambiguous.
        let (a, _) = db
            .ensure_typed_concept(PROJ, "alpha", ConceptStatus::Active, Some("person"))
            .unwrap();
        db.touch_alias(a.id, "acme", AliasSource::User).unwrap();
        let (b, _) = db
            .ensure_typed_concept(PROJ, "beta", ConceptStatus::Active, Some("person"))
            .unwrap();
        db.touch_alias(b.id, "acme", AliasSource::User).unwrap();
        write_md(&dir, "doc.md", "we discussed acme today\n");
        let cfg = src(&dir, Some("person"), &[], &[]);
        seed_nodes_from_source(&db, &cfg);
        let stats = link_mentions_from_sources(&db, &[&cfg]);
        assert!(
            mentions_from(&db, PROJ, "doc").is_empty(),
            "ambiguous surface not linked"
        );
        assert!(stats.ambiguous_skipped >= 1);
    }

    #[test]
    fn mention_is_idempotent() {
        let tmp = TempDir::new().unwrap();
        let db = ThreadsDb::open(tmp.path()).unwrap();
        let dir = tmp.path().to_path_buf();
        write_md(&dir, "tori.md", "Tori knows Mike.\n");
        write_md(&dir, "mike.md", "x\n");
        let cfg = src(&dir, Some("person"), &[], &[]);
        seed_nodes_from_source(&db, &cfg);
        link_mentions_from_sources(&db, &[&cfg]);
        link_mentions_from_sources(&db, &[&cfg]);
        let m = mentions_from(&db, PROJ, "tori");
        assert_eq!(m.len(), 1);
        assert_eq!(m[0].0, "mike");
        assert_eq!(m[0].3, 2, "re-scan re-touches (touch_count bumps)");
        assert_eq!(m[0].1, EdgeSource::Observed, "origin preserved");
    }

    #[test]
    fn mention_chain_event_on_creation_only() {
        let tmp = TempDir::new().unwrap();
        let sink: std::sync::Arc<dyn ChainSink> =
            std::sync::Arc::new(SqliteChainSink::open(tmp.path()).unwrap());
        let db = ThreadsDb::open_with_sink(tmp.path(), std::sync::Arc::clone(&sink)).unwrap();
        let dir = tmp.path().to_path_buf();
        write_md(&dir, "tori.md", "Tori knows Mike.\n");
        write_md(&dir, "mike.md", "x\n");
        let cfg = src(&dir, Some("person"), &[], &[]);
        seed_nodes_from_source(&db, &cfg); // promotes tori + mike, no edges
        let connected = |db: &ThreadsDb| {
            db.iter_chain()
                .unwrap()
                .iter()
                .filter(|e| ChainEvent::kind_str(e) == "concept_connected")
                .count()
        };
        assert_eq!(connected(&db), 0);
        link_mentions_from_sources(&db, &[&cfg]);
        assert_eq!(connected(&db), 1, "one mention edge created → one event");
        link_mentions_from_sources(&db, &[&cfg]);
        assert_eq!(connected(&db), 1, "re-scan re-touch emits nothing");
    }

    #[test]
    fn mention_scans_body_not_frontmatter() {
        let tmp = TempDir::new().unwrap();
        let db = ThreadsDb::open(tmp.path()).unwrap();
        let dir = tmp.path().to_path_buf();
        // sarah is named only in frontmatter, never in the body.
        write_md(
            &dir,
            "tori.md",
            "---\nfamilies: [sarah]\n---\nTori writes prose here.\n",
        );
        let cfg = src(&dir, Some("person"), &["families"], &[]);
        seed_nodes_from_source(&db, &cfg); // authored tori --families--> sarah
        link_mentions_from_sources(&db, &[&cfg]);
        let all = db
            .list_concept_edges(PROJ, "tori", EdgeDirection::From)
            .unwrap();
        assert!(
            all.iter()
                .any(|e| e.relation == "families" && e.to_handle == "sarah")
        );
        assert!(
            all.iter().all(|e| e.relation != "mentions"),
            "frontmatter-only name is not a body mention"
        );
    }

    #[test]
    fn mention_links_via_alias_to_canonical() {
        let tmp = TempDir::new().unwrap();
        let db = ThreadsDb::open(tmp.path()).unwrap();
        let dir = tmp.path().to_path_buf();
        db.ensure_typed_concept(PROJ, "real-name", ConceptStatus::Active, Some("person"))
            .unwrap();
        db.ensure_typed_concept(PROJ, "oldname", ConceptStatus::Active, Some("person"))
            .unwrap();
        // oldname → alias of canonical, tombstoned merged row.
        db.merge_concept(PROJ, "oldname", "real-name").unwrap();
        write_md(&dir, "doc.md", "we saw oldname yesterday\n");
        let cfg = src(&dir, Some("person"), &[], &[]);
        seed_nodes_from_source(&db, &cfg);
        link_mentions_from_sources(&db, &[&cfg]);
        let m = mentions_from(&db, PROJ, "doc");
        assert_eq!(m.len(), 1, "merged handle links via the canonical's alias");
        assert_eq!(
            m[0].0, "real-name",
            "links to the live canonical, not the tombstone"
        );
    }

    #[test]
    fn mention_excludes_terminal_status_target() {
        let tmp = TempDir::new().unwrap();
        let db = ThreadsDb::open(tmp.path()).unwrap();
        let dir = tmp.path().to_path_buf();
        db.ensure_typed_concept(PROJ, "ghost", ConceptStatus::Active, Some("person"))
            .unwrap();
        db.set_concept_status(PROJ, "ghost", ConceptStatus::Rejected, None)
            .unwrap();
        write_md(&dir, "doc.md", "ghost ghost ghost\n");
        let cfg = src(&dir, Some("person"), &[], &[]);
        seed_nodes_from_source(&db, &cfg);
        link_mentions_from_sources(&db, &[&cfg]);
        assert!(
            mentions_from(&db, PROJ, "doc").is_empty(),
            "a rejected concept is not a gazetteer target"
        );
    }

    #[test]
    fn mention_from_node_resolves_to_global_canonical() {
        let tmp = TempDir::new().unwrap();
        let db = ThreadsDb::open(tmp.path()).unwrap();
        let dir = tmp.path().to_path_buf();
        // A GLOBAL `tori` and a project-local target `mike`.
        db.ensure_typed_concept("", "tori", ConceptStatus::Active, Some("person"))
            .unwrap();
        db.ensure_typed_concept(PROJ, "mike", ConceptStatus::Active, Some("person"))
            .unwrap();
        write_md(&dir, "tori.md", "Tori knows Mike.\n");
        let cfg = src(&dir, Some("person"), &[], &[]);
        seed_nodes_from_source(&db, &cfg); // resolves tori → global, no local shadow
        link_mentions_from_sources(&db, &[&cfg]);
        // The mention edge hangs off the GLOBAL tori, not a local memories/tori.
        let global = mentions_from(&db, "", "tori");
        assert_eq!(global.len(), 1);
        assert_eq!(global[0].0, "mike");
        assert!(
            db.get_concept(PROJ, "tori").unwrap().is_none(),
            "no local shadow of the global file-node"
        );
    }

    #[test]
    fn mention_cross_scope_event_is_audit_only() {
        // Locks the documented audit-only behavior: the authoritative edge is
        // the by-id DB row (hangs off GLOBAL tori), but the ConceptConnected
        // chain event — lacking endpoint scopes — records the file's SOURCE
        // project. If this shape changes, this test should fail loudly.
        let tmp = TempDir::new().unwrap();
        let sink: std::sync::Arc<dyn ChainSink> =
            std::sync::Arc::new(SqliteChainSink::open(tmp.path()).unwrap());
        let db = ThreadsDb::open_with_sink(tmp.path(), std::sync::Arc::clone(&sink)).unwrap();
        let dir = tmp.path().to_path_buf();
        db.ensure_typed_concept("", "tori", ConceptStatus::Active, Some("person"))
            .unwrap();
        db.ensure_typed_concept(PROJ, "mike", ConceptStatus::Active, Some("person"))
            .unwrap();
        write_md(&dir, "tori.md", "Tori knows Mike.\n");
        let cfg = src(&dir, Some("person"), &[], &[]);
        seed_nodes_from_source(&db, &cfg);
        link_mentions_from_sources(&db, &[&cfg]);
        // Authoritative edge: from the GLOBAL tori.
        assert_eq!(mentions_from(&db, "", "tori")[0].0, "mike");
        // Audit event: from handle `tori`, but project = the source project.
        let evs = db.iter_chain().unwrap();
        let ev = evs
            .iter()
            .find_map(|e| match e {
                ChainEvent::ConceptConnected {
                    project,
                    from,
                    relation,
                    to,
                    ..
                } if relation == "mentions" => Some((project.clone(), from.clone(), to.clone())),
                _ => None,
            })
            .expect("a mention ConceptConnected event");
        assert_eq!(ev.1, "tori");
        assert_eq!(ev.2, "mike");
        assert_eq!(
            ev.0, PROJ,
            "audit-only: event records the source project, not the global from-scope"
        );
    }

    #[test]
    fn mention_handle_beats_alias_across_scope() {
        // resolve_one order: a handle (even global) beats an alias (even
        // project-local). A global handle `foo` must win over a project concept
        // merely aliased `foo`.
        let tmp = TempDir::new().unwrap();
        let db = ThreadsDb::open(tmp.path()).unwrap();
        let dir = tmp.path().to_path_buf();
        db.ensure_typed_concept("", "foo", ConceptStatus::Active, Some("person"))
            .unwrap();
        let (bar, _) = db
            .ensure_typed_concept(PROJ, "bar", ConceptStatus::Active, Some("person"))
            .unwrap();
        db.touch_alias(bar.id, "foo", AliasSource::User).unwrap();
        write_md(&dir, "doc.md", "doc cites foo here\n");
        let cfg = src(&dir, Some("person"), &[], &[]);
        seed_nodes_from_source(&db, &cfg);
        link_mentions_from_sources(&db, &[&cfg]);
        let m = mentions_from(&db, PROJ, "doc");
        assert_eq!(m.len(), 1);
        assert_eq!(
            m[0].0, "foo",
            "global handle beats project alias (resolve_one precedence)"
        );
    }

    #[test]
    fn mention_normalized_handle_collision_is_ambiguous() {
        // `foo-bar` and `foo_bar` are distinct handles that NORMALIZE to the
        // same form; with no precedence tiebreak the handle tier holds two ids
        // → ambiguous → dropped (not silently linked to one of them).
        let tmp = TempDir::new().unwrap();
        let db = ThreadsDb::open(tmp.path()).unwrap();
        let dir = tmp.path().to_path_buf();
        db.ensure_typed_concept(PROJ, "foo-bar", ConceptStatus::Active, Some("person"))
            .unwrap();
        db.ensure_typed_concept(PROJ, "foo_bar", ConceptStatus::Active, Some("person"))
            .unwrap();
        write_md(&dir, "doc.md", "doc names foo bar here\n");
        let cfg = src(&dir, Some("person"), &[], &[]);
        seed_nodes_from_source(&db, &cfg);
        let stats = link_mentions_from_sources(&db, &[&cfg]);
        assert!(
            mentions_from(&db, PROJ, "doc").is_empty(),
            "normalized handle collision is ambiguous, not linked"
        );
        assert!(stats.ambiguous_skipped >= 1);
    }

    #[test]
    fn mention_incremental_skips_deleted_and_ignored() {
        let tmp = TempDir::new().unwrap();
        let db = ThreadsDb::open(tmp.path()).unwrap();
        let dir = tmp.path().to_path_buf();
        write_md(&dir, "mike.md", "x\n");
        write_md(&dir, "tori.md", "Tori knows Mike.\n");
        write_md(&dir, "ignored.md", "Mentions Mike too\n");
        let cfg = src(&dir, Some("person"), &[], &["ignored.md"]);
        seed_nodes_from_source(&db, &cfg); // tori + mike nodes (ignored.md skipped)

        // Deleted / never-existing changed path → not scanned.
        let ghost = dir.join("ghost.md");
        let s_del = link_mentions_for_paths(&db, &[&cfg], &[ghost]);
        assert_eq!(s_del.files_scanned, 0, "missing path not scanned");

        // Ignored changed path → not scanned even though the file exists.
        let s_ign = link_mentions_for_paths(&db, &[&cfg], &[dir.join("ignored.md")]);
        assert_eq!(s_ign.files_scanned, 0, "ignored path not scanned");

        // A real changed path → scanned and linked.
        let s_ok = link_mentions_for_paths(&db, &[&cfg], &[dir.join("tori.md")]);
        assert_eq!(s_ok.files_scanned, 1);
        let m = mentions_from(&db, PROJ, "tori");
        assert_eq!(m.len(), 1);
        assert_eq!(m[0].0, "mike");
    }
}
