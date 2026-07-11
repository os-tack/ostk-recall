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

use std::collections::{BTreeSet, HashMap, HashSet};
use std::path::{Path, PathBuf};

use aho_corasick::{AhoCorasick, MatchKind};
use ostk_recall_core::{Source, SourceConfig};
use ostk_recall_scan::threads::split_front_matter;
use ostk_recall_scan::walk::walk_filtered;
use ostk_recall_store::{
    AUTHORED_EDGE_CONFIDENCE, AliasSource, ConceptRecord, ConceptStatus, EdgeSource,
    EvidenceAttach, IngestDb, OBSERVED_MENTION_CONFIDENCE, ThreadsDb, slugify,
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
    /// Slice 5: recurring unresolved prose names materialized as context-typed
    /// `Proposed` nodes (born this pass, gate-crossed).
    pub candidates_materialized: u64,
    /// Slice 5: recurring prose names dropped because their dominant containing
    /// kind was a tie at/above the recurrence gate (can't context-type — named,
    /// not silent).
    pub candidates_ambiguous: u64,
}

// ---------------------------------------------------------------------------
// Relational-substrate slice 5: automagic promotion.
//
// During the mention phase (above), prose tokens that the gazetteer does NOT
// resolve are conservative proper-name CANDIDATES. We aggregate them in memory
// across the whole full-scan, then materialize only those recurring across
// `PROSE_RECURRENCE_MIN_DOCS` distinct docs of one dominant `entity_type` — born
// `Candidate`, promoted in-pass to `Proposed` (conf 0.4, scoped — no global
// reflect sweep), and connected via observed `mentions` edges so the new node
// joins the diffusible graph. A human later confirms a `crystallize` to write a
// stub file. Propose, never auto-write.
// ---------------------------------------------------------------------------

/// Distinct docs of the dominant kind a prose name must recur across before it
/// is materialized as a context-typed node. `>= REFLECT_EVIDENCE_THRESHOLD` (2)
/// by construction, so a materialized node already clears the reflect gate.
const PROSE_RECURRENCE_MIN_DOCS: usize = 3;

/// Lowercased non-name words that survive the capitalized-mid-sentence rule too
/// often (sentence openers that recur mid-sentence, weekday/month names). A
/// candidate whose normalized form is here is dropped before aggregation.
const STOPWORDS: &[&str] = &[
    "the",
    "and",
    "but",
    "for",
    "nor",
    "yet",
    "with",
    "from",
    "into",
    "this",
    "that",
    "these",
    "those",
    "your",
    "their",
    "our",
    "his",
    "her",
    "its",
    "monday",
    "tuesday",
    "wednesday",
    "thursday",
    "friday",
    "saturday",
    "sunday",
    "january",
    "february",
    "march",
    "april",
    "may",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
    "today",
    "tomorrow",
    "yesterday",
    "okay",
];

/// A conservative proper-name candidate extracted from raw prose: `display` is
/// the cased form (for the eventual handle), `norm` its normalized key (for
/// dedup + known-form lookup).
#[derive(Debug, Clone, PartialEq, Eq)]
struct ProseCandidate {
    norm: String,
    display: String,
}

/// One doc that contributed a prose mention of a candidate: the doc's node id +
/// handle (for the audit `ConceptConnected`) and its source `entity_type` (for
/// the dominant-kind gate + context-typing).
#[derive(Debug, Clone)]
struct DocHit {
    doc_id: i64,
    doc_handle: String,
    kind: String,
}

/// In-memory aggregate for one normalized candidate within a project: a
/// representative display form + the distinct docs that named it. This map IS
/// the candidate stage — only gate-crossers get reified.
#[derive(Debug, Default)]
struct CandidateAgg {
    display: String,
    hits: Vec<DocHit>,
}

/// True for a name-token char: ASCII/unicode letter, or an internal `'`/`-`
/// (trimmed at the edges by [`refine_token`]).
fn is_name_char(c: char) -> bool {
    c.is_alphabetic() || c == '\'' || c == '-'
}

/// Strip a leading markdown line marker (`#`/`>`/`-`/`*`/`N.`) and surrounding
/// whitespace so the word that follows is treated as sentence-start.
fn strip_line_marker(line: &str) -> &str {
    let t = line.trim_start();
    // ATX heading / blockquote / bullet markers.
    let t = t.trim_start_matches(['#', '>', '-', '*', '+']).trim_start();
    // Ordered-list `N.` / `N)` marker.
    let bytes = t.as_bytes();
    let digits = bytes.iter().take_while(|b| b.is_ascii_digit()).count();
    if digits > 0 && matches!(bytes.get(digits), Some(b'.' | b')')) {
        t[digits + 1..].trim_start()
    } else {
        t
    }
}

/// Trim edge `'`/`-`, then strip a trailing possessive (`'s`/`'S`/`'`) so
/// `Sarah's` and `Sarah` fold to the same word. Returns `None` if nothing
/// alphabetic remains.
fn refine_token(token: &str) -> Option<String> {
    let core = token.trim_matches(|c| c == '\'' || c == '-');
    let core = core
        .strip_suffix("'s")
        .or_else(|| core.strip_suffix("'S"))
        .or_else(|| core.strip_suffix('\''))
        .unwrap_or(core);
    let core = core.trim_matches(|c| c == '\'' || c == '-');
    (!core.is_empty() && core.chars().any(char::is_alphabetic)).then(|| core.to_string())
}

/// Extract conservative capitalized-unigram proper-name candidates from raw
/// `body` (case-preserving). Pure: sentence-position + stoplist + length
/// heuristics only, no NER and no DB. A token qualifies iff its first char is
/// uppercase AND it is not at a sentence start (sentence-initial capitals are
/// ambiguous; a recurring real name appears mid-sentence in ≥1 doc). Deduped by
/// normalized form within the call.
fn extract_prose_name_candidates(body: &str) -> Vec<ProseCandidate> {
    let mut out: Vec<ProseCandidate> = Vec::new();
    let mut seen: HashSet<String> = HashSet::new();
    for raw_line in body.lines() {
        let line = strip_line_marker(raw_line);
        let chars: Vec<char> = line.chars().collect();
        let mut sentence_start = true;
        let mut i = 0;
        while i < chars.len() {
            let c = chars[i];
            if is_name_char(c) {
                let start = i;
                while i < chars.len() && is_name_char(chars[i]) {
                    i += 1;
                }
                let token: String = chars[start..i].iter().collect();
                let at_start = sentence_start;
                sentence_start = false;
                if at_start {
                    continue; // sentence-initial capital is ambiguous — drop
                }
                let Some(word) = refine_token(&token) else {
                    continue;
                };
                if !word.chars().next().is_some_and(char::is_uppercase) {
                    continue;
                }
                let norm = normalize(&word);
                if norm.chars().count() < MIN_MENTION_CHARS || STOPWORDS.contains(&norm.as_str()) {
                    continue;
                }
                if seen.insert(norm.clone()) {
                    out.push(ProseCandidate {
                        norm,
                        display: word,
                    });
                }
            } else {
                if c == '.' || c == '!' || c == '?' {
                    sentence_start = true;
                }
                i += 1;
            }
        }
    }
    out
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
    /// Slice 5: normalized handle + alias forms of EVERY concept in
    /// project∪global — including terminal (`rejected`/`merged`), ambiguous, and
    /// the file's own forms. The "already known" oracle for prose candidates:
    /// unlike [`Gazetteer::matches`] (which excludes self and drops ambiguous),
    /// a hit here means the name is already accounted for and must not become a
    /// duplicate latent node.
    known: HashSet<String>,
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
    let mut known: HashSet<String> = HashSet::new();
    for c in &concepts {
        let aliases = threads.list_aliases(c.id).unwrap_or_default();
        // Slice-5 known-form oracle: capture handle + alias forms of EVERY
        // concept (incl. terminal/ambiguous/self) BEFORE the terminal skip and
        // the ambiguity drop below, so a rejected name is not re-proposed and a
        // self/ambiguous alias does not re-materialize as a duplicate.
        if let Some(norm) = gaz_form(&c.handle) {
            known.insert(norm);
        }
        for a in &aliases {
            if let Some(norm) = gaz_form(&a.alias) {
                known.insert(norm);
            }
        }
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
        for a in &aliases {
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
    (
        Gazetteer {
            ac,
            ids,
            handles,
            known,
        },
        ambiguous,
    )
}

/// Scan one markdown file's body for gazetteer mentions and write observed
/// edges from the file's node to each distinct named node (≠ self).
fn link_file_mentions(
    threads: &ThreadsDb,
    gaz: &Gazetteer,
    cfg: &SourceConfig,
    project: &str,
    path: &Path,
    agg: Option<&mut HashMap<String, CandidateAgg>>,
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
    // Slice 5: fold unresolved prose-name candidates into the per-project
    // aggregate (the in-memory candidate stage; the driver materializes
    // gate-crossers after the file loop). Runs regardless of gazetteer hits.
    // `None` on incremental scans, which never materialize.
    if let Some(agg) = agg {
        collect_prose_candidates(gaz, cfg, &self_rec, body, agg);
    }
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

/// Slice 5: fold the file's unresolved prose-name candidates into the
/// per-project `agg`. A token survives only if it is not a known surface form
/// (`gaz.known` — handles/aliases of every concept, the right oracle vs.
/// `gaz.matches`) and not the file naming its own handle. Each surviving
/// candidate records one [`DocHit`] per distinct doc.
fn collect_prose_candidates(
    gaz: &Gazetteer,
    cfg: &SourceConfig,
    self_rec: &ConceptRecord,
    body: &str,
    agg: &mut HashMap<String, CandidateAgg>,
) {
    let kind = cfg.entity_type.as_deref().unwrap_or_default();
    for cand in extract_prose_name_candidates(body) {
        if gaz.known.contains(&cand.norm) {
            continue; // already a known handle/alias (any status) — not a candidate
        }
        let Some(slug) = slugify(&cand.display) else {
            continue;
        };
        if slug == self_rec.handle {
            continue; // the doc naming its own handle
        }
        let entry = agg.entry(cand.norm).or_insert_with(|| CandidateAgg {
            display: cand.display.clone(),
            hits: Vec::new(),
        });
        if !entry.hits.iter().any(|h| h.doc_id == self_rec.id) {
            entry.hits.push(DocHit {
                doc_id: self_rec.id,
                doc_handle: self_rec.handle.clone(),
                kind: kind.to_string(),
            });
        }
    }
}

/// Slice 5: reify the gate-crossing prose candidates for one project. A
/// candidate is materialized iff it recurs across `>= PROSE_RECURRENCE_MIN_DOCS`
/// distinct docs of a single **dominant** `entity_type` (a tie at/above the gate
/// is dropped as ambiguous — it cannot be context-typed). Each materialized name
/// is born `Candidate` then promoted in-pass to `Proposed` (conf 0.4, scoped —
/// no global reflect sweep) and connected via observed `mentions` edges from
/// every contributing doc so it joins the diffusible graph this scan.
fn materialize_prose_candidates(
    threads: &ThreadsDb,
    project: &str,
    agg: &HashMap<String, CandidateAgg>,
    stats: &mut MentionStats,
) {
    for cand in agg.values() {
        // Distinct docs per containing kind.
        let mut by_kind: HashMap<&str, BTreeSet<i64>> = HashMap::new();
        for hit in &cand.hits {
            by_kind
                .entry(hit.kind.as_str())
                .or_default()
                .insert(hit.doc_id);
        }
        let max = by_kind.values().map(BTreeSet::len).max().unwrap_or(0);
        if max < PROSE_RECURRENCE_MIN_DOCS {
            continue; // below the recurrence gate
        }
        let tops: Vec<&str> = by_kind
            .iter()
            .filter(|(_, docs)| docs.len() == max)
            .map(|(k, _)| *k)
            .collect();
        if tops.len() != 1 {
            stats.candidates_ambiguous += 1; // tie at/above gate — can't context-type
            continue;
        }
        let kind = tops[0];
        let Some(slug) = slugify(&cand.display) else {
            continue;
        };
        // Idempotency / race guard: skip if the name now resolves to a node.
        if matches!(threads.resolve_concept(project, &slug), Ok(Some(_))) {
            continue;
        }
        let (rec, created) = match threads.ensure_typed_concept(
            project,
            &slug,
            ConceptStatus::Candidate,
            Some(kind),
        ) {
            Ok(v) => v,
            Err(e) => {
                tracing::warn!(handle = %slug, error = %e, "prose-promotion: ensure candidate failed");
                continue;
            }
        };
        // Gate crossed → promote to Proposed, scoped to this handle (no global
        // reflect sweep). `set_concept_status` carries the 0.4 confidence that
        // `ensure_typed_concept` only sets for a Candidate birth.
        if let Err(e) =
            threads.set_concept_status(project, &slug, ConceptStatus::Proposed, Some(0.4))
        {
            tracing::warn!(handle = %slug, error = %e, "prose-promotion: promote to proposed failed");
        }
        if created {
            // Single chain event: the "proposed" advancement (candidate birth is
            // silent). Mirrors slice-3's creation event; audit-only.
            let _ = threads.record_concept_promoted(project, &slug, "proposed");
        }
        stats.candidates_materialized += 1;

        // Connect to every contributing doc (all kinds — a real mention is a
        // real mention, and the next gazetteer scan would write these anyway).
        let evidence = json!({ "via": "prose-promotion" }).to_string();
        let mut linked: BTreeSet<i64> = BTreeSet::new();
        for hit in &cand.hits {
            if !linked.insert(hit.doc_id) {
                continue;
            }
            match threads.add_concept_edge_by_id(
                hit.doc_id,
                MENTION_RELATION,
                rec.id,
                OBSERVED_MENTION_CONFIDENCE,
                EdgeSource::Observed,
                Some("scanner"),
                Some(&evidence),
            ) {
                Ok((_, edge_created)) => {
                    stats.mentions_linked += 1;
                    if edge_created {
                        if let Err(e) = threads.record_concept_connected(
                            project,
                            &hit.doc_handle,
                            MENTION_RELATION,
                            &rec.handle,
                            EdgeSource::Observed,
                            Some("scanner"),
                        ) {
                            tracing::warn!(error = %e, "prose-promotion: ConceptConnected emit failed");
                        }
                    }
                }
                Err(e) => {
                    tracing::debug!(to = %slug, from_id = hit.doc_id, error = %e, "prose-promotion edge skipped");
                }
            }
        }
    }
}

/// Full-scan driver: link prose mentions for every markdown file under each
/// `entity_type` source, then materialize recurring unresolved prose names.
///
/// Runs AFTER node-seeding for all sources so the gazetteer is complete (a
/// person can mention a meeting). The gazetteer is built once per project and
/// reused across sources sharing that project. Candidate aggregation spans the
/// whole scan, so cross-doc recurrence is exact; materialization runs per
/// project after the file loop.
pub fn link_mentions_from_sources(threads: &ThreadsDb, sources: &[&SourceConfig]) -> MentionStats {
    let mut stats = MentionStats::default();
    let mut gaz_cache: HashMap<String, Gazetteer> = HashMap::new();
    let mut agg_cache: HashMap<String, HashMap<String, CandidateAgg>> = HashMap::new();
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
        let agg = agg_cache.entry(project.to_string()).or_default();
        for root in cfg.expanded_paths().unwrap_or_default() {
            for entry in walk_filtered(&root, &cfg.ignore) {
                let p = entry.path();
                if is_md(p) {
                    link_file_mentions(threads, gaz, cfg, project, p, Some(&mut *agg), &mut stats);
                }
            }
        }
    }
    for (project, agg) in &agg_cache {
        materialize_prose_candidates(threads, project, agg, &mut stats);
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
    // Slice 5: an incremental scan sees one changed file at a time, so it can
    // never observe the cross-doc recurrence gate. It passes `None` for the
    // candidate aggregate (no materialization) — new-candidate detection rides
    // full scans; the slice-4 gazetteer path still maintains edges to known nodes.
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
                    link_file_mentions(threads, gaz, cfg, project, &pc, None, &mut stats);
                }
            }
        }
    }
    stats
}

// ---------------------------------------------------------------------
// Graph-only doc-topology harvest (relational-substrate; graph-growth plan)
// ---------------------------------------------------------------------
//
// A `graph_only` source seeds the *active docs' authorial link graph* as
// concept topology WITHOUT ingesting/embedding — the chunks are already owned
// by an ingesting source (e.g. `ostk_project` over the same tree). Each `.md`
// becomes an `Active` `doc` concept with a **path-stable** handle; inline
// markdown links → `references` edges; frontmatter `relates-to` → `relates_to`
// edges. Prose mentions are deliberately NOT harvested here (the gazetteer is
// skipped for graph-only sources — doc handles like `read`/`index` would match
// common prose and manufacture frequency-noise). Evidence is attached to the
// chunk an ingesting source already produced (by coordinate), never re-embedded.

/// Per-run doc-graph harvest counts.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct DocGraphStats {
    /// `doc` concept nodes freshly created.
    pub nodes_seeded: u64,
    /// `doc` concept nodes that already existed (re-ensured).
    pub nodes_touched: u64,
    /// `references` / `relates_to` edges written or re-touched.
    pub edges_seeded: u64,
    /// Evidence rows attached to already-ingested chunks (no re-embed).
    pub evidence_attached: u64,
    /// Markdown files visited.
    pub files_scanned: u64,
    /// Link targets dropped (external, anchor-only, out-of-root, non-doc).
    pub links_skipped: u64,
}

/// Relation minted for an inline `[..](other.md)` link.
const DOC_LINK_RELATION: &str = "references";
/// Relation minted for a frontmatter `relates-to:` reference.
const DOC_FRONTMATTER_RELATION: &str = "relates_to";

/// Slug-normalize without `slugify`'s length cap (lowercase, keep
/// `[a-z0-9_]`, collapse other runs to `-`, trim trailing `-`).
fn slug_norm(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut prev_dash = false;
    for ch in s.trim().chars() {
        let c = ch.to_ascii_lowercase();
        if c.is_ascii_alphanumeric() || c == '_' {
            out.push(c);
            prev_dash = false;
        } else if !prev_dash && !out.is_empty() {
            out.push('-');
            prev_dash = true;
        }
    }
    while out.ends_with('-') {
        out.pop();
    }
    out
}

/// Path-stable doc handle from a docs-root-relative path.
/// `spec/agent-lifecycle.md` → `spec-agent-lifecycle`. Documents are nodes,
/// not titles, so identity is the *path* (stems collide: many `INDEX.md`).
/// Results over `slugify`'s 40-char ceiling collapse to a readable prefix +
/// short blake3 hash, so distinct deep paths never alias on truncation.
fn doc_handle(rel: &Path) -> Option<String> {
    let mut raw = String::new();
    for comp in rel.with_extension("").components() {
        if let std::path::Component::Normal(s) = comp {
            if !raw.is_empty() {
                raw.push('-');
            }
            raw.push_str(&s.to_string_lossy());
        }
    }
    let norm = slug_norm(&raw);
    if norm.len() < 3 {
        return None;
    }
    if norm.len() <= 40 {
        return Some(norm);
    }
    let short = &blake3::hash(norm.as_bytes()).to_hex()[..8];
    let prefix = norm.chars().take(31).collect::<String>();
    let prefix = prefix.trim_end_matches('-');
    Some(format!("{prefix}-{short}"))
}

/// Resolve an inline-link target to a docs-root-relative path, applying the
/// skip/strip rules. `from_rel` is the source doc's docs-root-relative path.
/// Returns `None` for anything that isn't an in-root `.md` document.
fn resolve_link_rel(from_rel: &Path, raw: &str) -> Option<String> {
    // Markdown allows `](url "title")` — take the URL token only.
    let url = raw.split_whitespace().next().unwrap_or("");
    // Strip `#fragment` / `?query`.
    let url = url.split(['#', '?']).next().unwrap_or("");
    if url.is_empty()
        || url.contains("://")
        || url.starts_with("mailto:")
        || url.starts_with("//")
        || url.starts_with('/')
    {
        return None; // empty / pure-anchor / scheme / absolute → skip
    }
    if !url.to_ascii_lowercase().ends_with(".md") {
        return None; // only document links
    }
    // Base = the source doc's directory (docs-root-relative).
    let mut stack: Vec<String> = from_rel
        .parent()
        .map(|p| {
            p.components()
                .filter_map(|c| match c {
                    std::path::Component::Normal(s) => Some(s.to_string_lossy().into_owned()),
                    _ => None,
                })
                .collect()
        })
        .unwrap_or_default();
    for comp in Path::new(url).components() {
        match comp {
            std::path::Component::CurDir => {}
            std::path::Component::ParentDir => {
                stack.pop()?; // underflow ⇒ escapes the doc root ⇒ skip
            }
            std::path::Component::Normal(s) => stack.push(s.to_string_lossy().into_owned()),
            _ => return None, // RootDir / Prefix ⇒ absolute ⇒ skip
        }
    }
    if stack.is_empty() {
        return None;
    }
    Some(stack.join("/"))
}

/// Find inline markdown link targets (`](target)`) in a body.
fn extract_md_link_targets(body: &str) -> Vec<String> {
    let mut out = Vec::new();
    let bytes = body.as_bytes();
    let mut i = 0;
    while i + 1 < bytes.len() {
        if bytes[i] == b']' && bytes[i + 1] == b'(' {
            let start = i + 2;
            if let Some(close) = body[start..].find(')') {
                out.push(body[start..start + close].to_string());
                i = start + close + 1;
                continue;
            }
        }
        i += 1;
    }
    out
}

/// First non-empty `# Heading` line (the doc title), for an alias.
fn doc_title(body: &str) -> Option<String> {
    body.lines()
        .map(str::trim)
        .find_map(|l| {
            l.strip_prefix("# ")
                .map(str::trim)
                .filter(|t| !t.is_empty())
        })
        .map(ToString::to_string)
}

/// A seeded doc node, carried from the node pass to the edge pass.
struct DocNode {
    rel: PathBuf,
    handle: String,
    id: i64,
    body: String,
}

/// Seed one doc node (`Active`, kind from `entity_type`), its stem/title
/// aliases, and coordinate evidence to an already-ingested chunk (no
/// re-embed). Returns the concept id on success.
#[allow(clippy::too_many_arguments)] // cohesive per-doc seed inputs; a struct would just relocate them
fn seed_doc_node(
    threads: &ThreadsDb,
    ingest: &IngestDb,
    project: &str,
    kind: &str,
    root_name: &str,
    rel: &Path,
    handle: &str,
    body: &str,
    stats: &mut DocGraphStats,
) -> Option<i64> {
    let (rec, created) =
        match threads.ensure_typed_concept(project, handle, ConceptStatus::Active, Some(kind)) {
            Ok(v) => v,
            Err(e) => {
                tracing::debug!(handle, error = %e, "doc-graph: ensure node failed");
                return None;
            }
        };
    if created {
        stats.nodes_seeded += 1;
        if let Err(e) = threads.record_concept_promoted(project, handle, "active") {
            tracing::debug!(handle, error = %e, "doc-graph: ConceptPromoted emit failed");
        }
    } else {
        stats.nodes_touched += 1;
    }
    // Aliases: bare stem + `# Title` (documents are reachable by their name,
    // even though identity is the path).
    if let Some(stem) = rel.file_stem().and_then(|s| s.to_str()) {
        let _ = threads.touch_alias(rec.id, stem, AliasSource::Path);
    }
    if let Some(title) = doc_title(body) {
        let _ = threads.touch_alias(rec.id, &title, AliasSource::Path);
    }
    // Evidence: map docs-root-relative `rel` → the ingesting source's
    // project-root coordinate (`<root_name>/<rel>`, e.g. `docs/spec/foo.md`)
    // and anchor on the first chunk (lowest `chunk_index`). No re-embed.
    // Corpus coordinates are `/`-separated on every platform — join the
    // components rather than lossy-print the platform path, which would
    // key the lookup on `\` under Windows.
    let rel_slash = rel
        .components()
        .map(|c| c.as_os_str().to_string_lossy())
        .collect::<Vec<_>>()
        .join("/");
    let source_id = format!("{root_name}/{rel_slash}");
    if let Ok(chunks) = ingest.chunk_ids_for_source_id(Source::OstkSpec.as_str(), &source_id) {
        if let Some(anchor) = chunks.first() {
            let attached = threads.attach_concept_evidence(&EvidenceAttach {
                concept_id: rec.id,
                project,
                source: Source::OstkSpec.as_str(),
                source_id: &source_id,
                chunk_id: Some(anchor),
                content_sha256: None,
                anchor_vec: None,
                score: None,
                reason: Some("doc-graph harvest"),
            });
            match attached {
                Ok(()) => stats.evidence_attached += 1,
                Err(e) => tracing::debug!(handle, error = %e, "doc-graph: evidence attach failed"),
            }
        }
    }
    Some(rec.id)
}

/// Harvest edges out of one doc body, resolving targets via `resolve`
/// (handle → concept id). `references` from inline links, `relates_to` from
/// frontmatter `relates-to:` `.md` tokens.
#[allow(clippy::too_many_arguments)] // cohesive per-doc edge-harvest inputs
fn harvest_doc_edges(
    threads: &ThreadsDb,
    project: &str,
    from_handle: &str,
    from_id: i64,
    from_rel: &Path,
    body: &str,
    resolve: &dyn Fn(&str) -> Option<i64>,
    stats: &mut DocGraphStats,
) {
    let mut add = |to_handle: &str, to_id: i64, relation: &str| {
        if to_id == from_id {
            return;
        }
        match threads.add_concept_edge_by_id(
            from_id,
            relation,
            to_id,
            OBSERVED_MENTION_CONFIDENCE,
            EdgeSource::Observed,
            Some("scanner"),
            Some(&json!({ "via": "doc-graph" }).to_string()),
        ) {
            Ok((_, created)) => {
                stats.edges_seeded += 1;
                // Audit-only cognition event, on first creation only (the
                // by-id row above is authoritative; re-touch is silent use).
                if created {
                    if let Err(e) = threads.record_concept_connected(
                        project,
                        from_handle,
                        relation,
                        to_handle,
                        EdgeSource::Observed,
                        Some("scanner"),
                    ) {
                        tracing::debug!(error = %e, "doc-graph: ConceptConnected emit failed");
                    }
                }
            }
            Err(e) => tracing::debug!(error = %e, "doc-graph: edge skipped"),
        }
    };

    // Inline links (file-relative) → references.
    for raw in extract_md_link_targets(body) {
        match resolve_link_rel(from_rel, &raw) {
            Some(rel) => match doc_handle(Path::new(&rel)) {
                Some(h) => match resolve(&h) {
                    Some(to_id) => add(&h, to_id, DOC_LINK_RELATION),
                    None => stats.links_skipped += 1, // in-root path, not a seeded doc
                },
                None => stats.links_skipped += 1,
            },
            None => stats.links_skipped += 1, // external / anchor / out-of-root
        }
    }

    // Frontmatter `relates-to:` → relates_to. Best-effort over the free-text
    // value, handling both the inline form (`relates-to: a.md, b.md`) and the
    // YAML block-list form (`relates-to:` then indented `- a.md` lines). Each
    // `.md` token is resolved docs-root-relative after stripping a leading
    // `docs/` or `./` (frontmatter paths are often project-root-relative).
    if let Some((yaml, _)) = split_front_matter(body) {
        let mut resolve_tokens = |src: &str| {
            for tok in src.split([' ', ',', '(', ')', '\t']) {
                let tok = tok.trim_matches(|c: char| {
                    !c.is_ascii_alphanumeric() && c != '/' && c != '.' && c != '-' && c != '_'
                });
                if !tok.to_ascii_lowercase().ends_with(".md") {
                    continue;
                }
                let stripped = tok.trim_start_matches("./");
                let stripped = stripped.strip_prefix("docs/").unwrap_or(stripped);
                if let Some(h) = doc_handle(Path::new(stripped)) {
                    if let Some(to_id) = resolve(&h) {
                        add(&h, to_id, DOC_FRONTMATTER_RELATION);
                    }
                }
            }
        };
        let mut in_relates = false;
        for line in yaml.lines() {
            let trimmed = line.trim_start();
            if let Some(rest) = trimmed
                .strip_prefix("relates-to:")
                .or_else(|| trimmed.strip_prefix("relates_to:"))
            {
                in_relates = true;
                resolve_tokens(rest); // inline values on the key line
            } else if in_relates && trimmed.starts_with('-') {
                resolve_tokens(trimmed.trim_start_matches('-')); // block-list item
            } else if !trimmed.is_empty() {
                in_relates = false; // a new key ends the relates-to block
            }
        }
    }
}

/// Full-scan driver: harvest the doc-topology graph for one `graph_only` source.
///
/// Two passes — seed every node first (so link targets resolve), then harvest
/// edges. No-op unless `entity_type` is set (config-validated).
pub fn harvest_doc_graph_from_source(
    threads: &ThreadsDb,
    ingest: &IngestDb,
    cfg: &SourceConfig,
) -> DocGraphStats {
    let mut stats = DocGraphStats::default();
    let project = cfg.project.as_deref().unwrap_or("");
    let Some(kind) = cfg.entity_type.as_deref() else {
        return stats;
    };
    let roots = cfg.expanded_paths().unwrap_or_default();
    for root in &roots {
        let root_name = root
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or_default();
        // Pass 1 — seed nodes; build handle→id for in-root link resolution.
        let mut docs: Vec<DocNode> = Vec::new();
        let mut seeded: HashMap<String, i64> = HashMap::new();
        for entry in walk_filtered(root, &cfg.ignore).filter(|e| is_md(e.path())) {
            let path = entry.path();
            let Ok(rel) = path.strip_prefix(root) else {
                continue;
            };
            let Some(handle) = doc_handle(rel) else {
                continue;
            };
            stats.files_scanned += 1;
            let body = std::fs::read_to_string(path).unwrap_or_default();
            if let Some(id) = seed_doc_node(
                threads, ingest, project, kind, root_name, rel, &handle, &body, &mut stats,
            ) {
                seeded.insert(handle.clone(), id);
                docs.push(DocNode {
                    rel: rel.to_path_buf(),
                    handle,
                    id,
                    body,
                });
            }
        }
        // Pass 2 — edges, resolved against the now-complete node set.
        let resolve = |h: &str| seeded.get(h).copied();
        for doc in &docs {
            harvest_doc_edges(
                threads,
                project,
                &doc.handle,
                doc.id,
                &doc.rel,
                &doc.body,
                &resolve,
                &mut stats,
            );
        }
    }
    stats
}

/// Incremental driver: harvest only the changed `paths` for a `graph_only` source.
///
/// Built for a future watch wiring (not exposed via the project-scoped
/// `--source` in this phase). Targets resolve against the existing ledger (a
/// full scan seeds the complete node set).
pub fn harvest_doc_graph_for_paths(
    threads: &ThreadsDb,
    ingest: &IngestDb,
    cfg: &SourceConfig,
    paths: &[PathBuf],
) -> DocGraphStats {
    let mut stats = DocGraphStats::default();
    let project = cfg.project.as_deref().unwrap_or("");
    let Some(kind) = cfg.entity_type.as_deref() else {
        return stats;
    };
    let roots = cfg.expanded_paths().unwrap_or_default();
    let resolve = |h: &str| {
        threads
            .resolve_concept(project, h)
            .ok()
            .flatten()
            .map(|r| r.id)
    };
    for pc in paths {
        if !is_md(pc) {
            continue;
        }
        let Some(root) = roots.iter().find(|r| pc.starts_with(r)) else {
            continue;
        };
        let root_name = root
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or_default();
        let Ok(rel) = pc.strip_prefix(root) else {
            continue;
        };
        let Some(handle) = doc_handle(rel) else {
            continue;
        };
        stats.files_scanned += 1;
        let body = std::fs::read_to_string(pc).unwrap_or_default();
        if let Some(id) = seed_doc_node(
            threads, ingest, project, kind, root_name, rel, &handle, &body, &mut stats,
        ) {
            harvest_doc_edges(
                threads, project, &handle, id, rel, &body, &resolve, &mut stats,
            );
        }
    }
    stats
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    use ostk_recall_core::SourceKind;
    use ostk_recall_store::{
        AliasSource, ChainEvent, ChainSink, EdgeDirection, IngestChunkRow, SqliteChainSink,
    };
    use tempfile::TempDir;

    const PROJ: &str = "memories";

    fn src(dir: &Path, entity_type: Option<&str>, edges: &[&str], ignore: &[&str]) -> SourceConfig {
        SourceConfig {
            kind: SourceKind::Markdown,
            graph_only: false,
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

    // ── Slice 5: prose-name extraction + automagic promotion ──────────────

    fn mentions_to(db: &ThreadsDb, project: &str, handle: &str) -> Vec<(String, EdgeSource, f32)> {
        db.list_concept_edges(project, handle, EdgeDirection::To)
            .unwrap()
            .into_iter()
            .filter(|e| e.relation == "mentions")
            .map(|e| (e.from_handle, e.source, e.confidence))
            .collect()
    }

    #[test]
    fn extractor_keeps_midsentence_drops_start_and_lowercase() {
        let c = extract_prose_name_candidates("We met Sarah today. Bob waved.");
        let norms: Vec<&str> = c.iter().map(|p| p.norm.as_str()).collect();
        assert!(norms.contains(&"sarah"), "mid-sentence capital kept");
        assert!(!norms.contains(&"bob"), "sentence-start capital dropped");
        assert!(!norms.contains(&"today"), "lowercase ignored");
    }

    #[test]
    fn extractor_folds_possessive() {
        let c = extract_prose_name_candidates("I saw Sarah's car and Sarah again.");
        let n: Vec<&str> = c.iter().map(|p| p.norm.as_str()).collect();
        assert_eq!(
            n.iter().filter(|x| **x == "sarah").count(),
            1,
            "Sarah's and Sarah fold to one `sarah`"
        );
    }

    #[test]
    fn extractor_drops_stopwords_and_months() {
        let c = extract_prose_name_candidates("We met in December and on Monday.");
        let n: Vec<&str> = c.iter().map(|p| p.norm.as_str()).collect();
        assert!(!n.contains(&"december"));
        assert!(!n.contains(&"monday"));
    }

    #[test]
    fn prose_promotion_materializes_recurring_person() {
        let tmp = TempDir::new().unwrap();
        let sink: Arc<dyn ChainSink> = Arc::new(SqliteChainSink::open(tmp.path()).unwrap());
        let db = ThreadsDb::open_with_sink(tmp.path(), Arc::clone(&sink)).unwrap();
        let dir = tmp.path().join("people");
        std::fs::create_dir_all(&dir).unwrap();
        // Sarah recurs mid-sentence across 3 person docs; Quentin in 1; December
        // is a stopword/month. Sentence-initial doc names (Alice/Bob/Carol) are
        // dropped by the sentence-start rule.
        write_md(&dir, "alice.md", "We met. Alice spoke with Sarah today.\n");
        write_md(&dir, "bob.md", "Notes. Bob saw Sarah at lunch.\n");
        write_md(&dir, "carol.md", "Recap. Carol called Sarah in December.\n");
        write_md(&dir, "dave.md", "Aside. Dave mentioned Quentin once.\n");
        // An unrelated candidate in another project must stay untouched (no global sweep).
        db.ensure_concept("other", "ghost", ConceptStatus::Candidate)
            .unwrap();
        let cfg = src(&dir, Some("person"), &[], &[]);
        seed_nodes_from_source(&db, &cfg);
        let stats = link_mentions_from_sources(&db, &[&cfg]);

        let rec = db
            .resolve_concept(PROJ, "sarah")
            .unwrap()
            .expect("sarah materialized");
        assert_eq!(rec.kind.as_deref(), Some("person"));
        assert_eq!(rec.status, ConceptStatus::Proposed);
        assert!((rec.confidence - 0.4).abs() < 1e-3, "born proposed at 0.4");
        assert_eq!(stats.candidates_materialized, 1);

        let inbound = mentions_to(&db, PROJ, "sarah");
        assert_eq!(
            inbound.len(),
            3,
            "one observed mentions edge per contributing doc"
        );
        assert!(inbound.iter().all(|(_, s, c)| *s == EdgeSource::Observed
            && (*c - OBSERVED_MENTION_CONFIDENCE).abs() < 1e-6));

        // No one-off bloat.
        assert!(db.get_concept(PROJ, "quentin").unwrap().is_none());
        assert!(db.get_concept(PROJ, "december").unwrap().is_none());
        // Scoping: the unrelated candidate is untouched.
        assert_eq!(
            db.get_concept("other", "ghost").unwrap().unwrap().status,
            ConceptStatus::Candidate
        );

        // Chain: exactly one ConceptPromoted{sarah,"proposed"}, never "candidate".
        let promotions: Vec<String> = db
            .iter_chain()
            .unwrap()
            .into_iter()
            .filter_map(|e| match e {
                ChainEvent::ConceptPromoted {
                    handle, to_status, ..
                } if handle == "sarah" => Some(to_status),
                _ => None,
            })
            .collect();
        assert_eq!(promotions, vec!["proposed".to_string()]);

        // Idempotent re-scan: sarah is now known, edges re-touch (no dup), no new event.
        let stats2 = link_mentions_from_sources(&db, &[&cfg]);
        assert_eq!(stats2.candidates_materialized, 0);
        assert_eq!(mentions_to(&db, PROJ, "sarah").len(), 3);
    }

    #[test]
    fn prose_promotion_dominant_kind_tie_dropped() {
        // Robin recurs across 3 person + 3 meeting docs → tie at the gate → can't
        // context-type → dropped as ambiguous (not materialized).
        let tmp = TempDir::new().unwrap();
        let db = ThreadsDb::open(tmp.path()).unwrap();
        let pdir = tmp.path().join("people");
        let mdir = tmp.path().join("meetings");
        std::fs::create_dir_all(&pdir).unwrap();
        std::fs::create_dir_all(&mdir).unwrap();
        for n in ["alpha", "bravo", "gamma"] {
            write_md(&pdir, &format!("{n}.md"), "Notes. We saw Robin here.\n");
        }
        for n in ["delta", "sigma", "omega"] {
            write_md(&mdir, &format!("{n}.md"), "Recap. They met Robin again.\n");
        }
        let pcfg = src(&pdir, Some("person"), &[], &[]);
        let mcfg = src(&mdir, Some("meeting"), &[], &[]);
        seed_nodes_from_source(&db, &pcfg);
        seed_nodes_from_source(&db, &mcfg);
        let stats = link_mentions_from_sources(&db, &[&pcfg, &mcfg]);
        assert!(db.get_concept(PROJ, "robin").unwrap().is_none());
        assert_eq!(stats.candidates_ambiguous, 1);
        assert_eq!(stats.candidates_materialized, 0);
    }

    #[test]
    fn prose_promotion_skips_ambiguous_and_rejected_known_forms() {
        // Forms that `gaz.matches` would MISS — an ambiguous alias (dropped from
        // the matcher) and a rejected node's form (terminal, excluded from the
        // matcher) — must still be recognized as known via `known_forms`, so a
        // recurring prose mention never re-materializes them as duplicates.
        let tmp = TempDir::new().unwrap();
        let db = ThreadsDb::open(tmp.path()).unwrap();
        let dir = tmp.path().join("people");
        std::fs::create_dir_all(&dir).unwrap();
        let (r1, _) = db
            .ensure_typed_concept(PROJ, "robin-a", ConceptStatus::Active, Some("person"))
            .unwrap();
        let (r2, _) = db
            .ensure_typed_concept(PROJ, "robin-b", ConceptStatus::Active, Some("person"))
            .unwrap();
        db.touch_alias(r1.id, "Robin", AliasSource::User).unwrap(); // ambiguous: two
        db.touch_alias(r2.id, "Robin", AliasSource::User).unwrap(); // ids share "robin"
        db.ensure_typed_concept(PROJ, "trent", ConceptStatus::Rejected, Some("person"))
            .unwrap();
        for n in ["memo1", "memo2", "memo3"] {
            write_md(
                &dir,
                &format!("{n}.md"),
                "Notes. We saw Robin and Trent here.\n",
            );
        }
        let cfg = src(&dir, Some("person"), &[], &[]);
        seed_nodes_from_source(&db, &cfg);
        let stats = link_mentions_from_sources(&db, &[&cfg]);
        assert_eq!(
            stats.candidates_materialized, 0,
            "ambiguous + rejected known forms never re-materialize"
        );
        assert!(db.get_concept(PROJ, "robin").unwrap().is_none());
        assert_eq!(
            db.get_concept(PROJ, "trent").unwrap().unwrap().status,
            ConceptStatus::Rejected
        );
    }

    // ---- doc-graph harvest: pure helpers ----

    #[test]
    fn doc_handle_is_path_based() {
        assert_eq!(
            doc_handle(Path::new("spec/agent-lifecycle.md")).as_deref(),
            Some("spec-agent-lifecycle")
        );
        // Stems collide; paths don't.
        assert_eq!(
            doc_handle(Path::new("spec/index.md")).as_deref(),
            Some("spec-index")
        );
        assert_eq!(
            doc_handle(Path::new("draft/index.md")).as_deref(),
            Some("draft-index")
        );
        assert_ne!(
            doc_handle(Path::new("spec/index.md")),
            doc_handle(Path::new("draft/index.md")),
            "same stem in different dirs must not collide"
        );
    }

    #[test]
    fn doc_handle_long_path_uses_prefix_plus_hash() {
        let long = Path::new("spec/a-very-very-long-document-name-that-blows-past-forty-chars.md");
        let h = doc_handle(long).expect("long path still yields a handle");
        assert!(
            h.len() <= 40,
            "handle within slug ceiling: {} ({})",
            h,
            h.len()
        );
        assert!(h.contains('-'), "prefix-hash shape");
        // Deterministic + collision-resistant: a different long path → different handle.
        let other = Path::new("spec/a-very-very-long-document-name-that-blows-past-forty-OTHER.md");
        assert_ne!(doc_handle(long), doc_handle(other));
        assert_eq!(doc_handle(long), doc_handle(long), "deterministic");
    }

    #[test]
    fn resolve_link_rel_in_root_paths() {
        let from = Path::new("spec/foo.md");
        assert_eq!(
            resolve_link_rel(from, "bar.md").as_deref(),
            Some("spec/bar.md")
        );
        assert_eq!(
            resolve_link_rel(from, "./bar.md").as_deref(),
            Some("spec/bar.md")
        );
        assert_eq!(
            resolve_link_rel(from, "../draft/x.md").as_deref(),
            Some("draft/x.md")
        );
        // Fragment + query are stripped.
        assert_eq!(
            resolve_link_rel(from, "bar.md#sec").as_deref(),
            Some("spec/bar.md")
        );
        assert_eq!(
            resolve_link_rel(from, "bar.md?v=1").as_deref(),
            Some("spec/bar.md")
        );
        // `](url "title")` — only the URL token is used.
        assert_eq!(
            resolve_link_rel(from, "bar.md \"A Title\"").as_deref(),
            Some("spec/bar.md")
        );
    }

    #[test]
    fn resolve_link_rel_skips_non_doc_targets() {
        let from = Path::new("spec/foo.md");
        for skip in [
            "https://example.com/x.md", // absolute URL
            "http://example.com",       // absolute URL, non-md
            "mailto:a@b.md",            // mailto scheme
            "#section",                 // pure anchor
            "/abs/x.md",                // absolute fs path
            "image.png",                // non-doc extension
            "../../outside.md",         // escapes the doc root (from spec/foo.md)
            "",                         // empty
        ] {
            assert!(
                resolve_link_rel(from, skip).is_none(),
                "should skip {skip:?}"
            );
        }
        // From a top-level doc, a single `../` already escapes the root.
        assert!(resolve_link_rel(Path::new("foo.md"), "../bar.md").is_none());
    }

    #[test]
    fn extract_md_link_targets_finds_inline_links() {
        let body = "see [a](./a.md) and [b](b.md \"title\") plus [c](https://x).";
        let got = extract_md_link_targets(body);
        assert_eq!(got, vec!["./a.md", "b.md \"title\"", "https://x"]);
    }

    // ---- doc-graph harvest: integration ----

    fn write_doc(root: &Path, rel: &str, body: &str) {
        let p = root.join(rel);
        std::fs::create_dir_all(p.parent().unwrap()).unwrap();
        std::fs::write(p, body).unwrap();
    }

    #[test]
    fn harvest_seeds_active_doc_nodes_and_reference_edges() {
        let tmp = TempDir::new().unwrap();
        let db_dir = TempDir::new().unwrap();
        let ing_dir = TempDir::new().unwrap();
        let db = ThreadsDb::open(db_dir.path()).unwrap();
        let ingest = IngestDb::open(ing_dir.path()).unwrap();
        let root = tmp.path();

        // a → b (in-root), plus external/anchor/out-of-root links that must skip.
        write_doc(
            root,
            "spec/a.md",
            "# Doc A\nlinks [to b](b.md), [ext](https://x.com/y.md), [anc](#s), [out](../../o.md)",
        );
        write_doc(root, "spec/b.md", "# Doc B\nleaf");

        let cfg = src(root, Some("doc"), &[], &[]);
        let stats = harvest_doc_graph_from_source(&db, &ingest, &cfg);

        assert_eq!(stats.nodes_seeded, 2, "two doc nodes");
        // Both nodes are Active + kind=doc.
        for h in ["spec-a", "spec-b"] {
            let rec = db.get_concept(PROJ, h).unwrap().unwrap();
            assert_eq!(rec.status, ConceptStatus::Active, "{h} active");
            assert_eq!(rec.kind.as_deref(), Some("doc"), "{h} kind=doc");
        }
        // Exactly one references edge a→b; the external/anchor/out-of-root skipped.
        assert_eq!(
            stats.edges_seeded, 1,
            "only the in-root .md link becomes an edge"
        );
        assert!(
            stats.links_skipped >= 3,
            "external/anchor/out-of-root skipped"
        );
        let edges = db
            .list_concept_edges(PROJ, "spec-a", EdgeDirection::From)
            .unwrap();
        assert_eq!(edges.len(), 1);
        assert_eq!(edges[0].relation, "references");
        assert_eq!(edges[0].to_handle, "spec-b");
        assert!(matches!(edges[0].source, EdgeSource::Observed));

        // Re-harvest is idempotent (no new nodes/edges; re-touch only).
        let again = harvest_doc_graph_from_source(&db, &ingest, &cfg);
        assert_eq!(again.nodes_seeded, 0, "existing nodes not re-created");
        let edges2 = db
            .list_concept_edges(PROJ, "spec-a", EdgeDirection::From)
            .unwrap();
        assert_eq!(edges2.len(), 1, "edge re-touched, not duplicated");
    }

    #[test]
    fn harvest_attaches_evidence_to_already_ingested_chunk() {
        let tmp = TempDir::new().unwrap();
        let db_dir = TempDir::new().unwrap();
        let ing_dir = TempDir::new().unwrap();
        let db = ThreadsDb::open(db_dir.path()).unwrap();
        let ingest = IngestDb::open(ing_dir.path()).unwrap();
        let root = tmp.path();
        let root_name = root.file_name().unwrap().to_str().unwrap();

        write_doc(root, "spec/a.md", "# Doc A\nbody");
        // Pre-seed the ingesting source's chunk at the mapped coordinate
        // (project-root-relative `<root_name>/spec/a.md`), two chunks so the
        // anchor must be the first by chunk_index.
        for idx in [1_u32, 0] {
            ingest
                .record_chunk(
                    &IngestChunkRow {
                        chunk_id: format!("chunk-{idx}"),
                        source: Source::OstkSpec.as_str().to_string(),
                        source_id: format!("{root_name}/spec/a.md"),
                        source_config_id: "test".into(),
                        chunk_index: idx,
                        content_sha256: format!("sha-{idx}"),
                        embedding_input_sha256: String::new(),
                    },
                    None,
                )
                .unwrap();
        }

        let cfg = src(root, Some("doc"), &[], &[]);
        let stats = harvest_doc_graph_from_source(&db, &ingest, &cfg);
        assert_eq!(
            stats.evidence_attached, 1,
            "one evidence anchor for the ingested doc"
        );
    }

    #[test]
    fn harvest_emits_concept_connected_on_creation_only() {
        let tmp = TempDir::new().unwrap();
        let ing_dir = TempDir::new().unwrap();
        let sink: Arc<dyn ChainSink> = Arc::new(SqliteChainSink::open(tmp.path()).unwrap());
        let db = ThreadsDb::open_with_sink(tmp.path(), Arc::clone(&sink)).unwrap();
        let ingest = IngestDb::open(ing_dir.path()).unwrap();
        let root = tmp.path().join("docs");
        std::fs::create_dir_all(&root).unwrap();
        write_doc(&root, "spec/a.md", "# A\nsee [b](b.md)");
        write_doc(&root, "spec/b.md", "# B\nleaf");

        let cfg = src(&root, Some("doc"), &[], &[]);
        harvest_doc_graph_from_source(&db, &ingest, &cfg);
        let kinds: Vec<String> = db
            .iter_chain()
            .unwrap()
            .iter()
            .map(|e| ChainEvent::kind_str(e).to_string())
            .collect();
        assert_eq!(
            kinds.iter().filter(|k| *k == "concept_connected").count(),
            1,
            "the a→b references edge emits one ConceptConnected"
        );
        // Idempotent re-harvest: no new chain events (re-touch is silent use).
        let before = db.iter_chain().unwrap().len();
        harvest_doc_graph_from_source(&db, &ingest, &cfg);
        assert_eq!(
            db.iter_chain().unwrap().len(),
            before,
            "rescan emits nothing"
        );
    }

    #[test]
    fn harvest_relates_to_handles_yaml_block_list() {
        let tmp = TempDir::new().unwrap();
        let db_dir = TempDir::new().unwrap();
        let ing_dir = TempDir::new().unwrap();
        let db = ThreadsDb::open(db_dir.path()).unwrap();
        let ingest = IngestDb::open(ing_dir.path()).unwrap();
        let root = tmp.path();
        // Block-list frontmatter form (not inline) — the codex Low finding.
        write_doc(
            root,
            "spec/a.md",
            "---\nrelates-to:\n  - spec/b.md\n  - docs/spec/c.md\nstatus: live\n---\n# A\nbody",
        );
        write_doc(root, "spec/b.md", "# B");
        write_doc(root, "spec/c.md", "# C");

        let cfg = src(root, Some("doc"), &[], &[]);
        harvest_doc_graph_from_source(&db, &ingest, &cfg);
        let edges = db
            .list_concept_edges(PROJ, "spec-a", EdgeDirection::From)
            .unwrap();
        let relates: Vec<&str> = edges
            .iter()
            .filter(|e| e.relation == "relates_to")
            .map(|e| e.to_handle.as_str())
            .collect();
        // `spec/b.md` (docs-root-relative) and `docs/spec/c.md` (docs/-prefixed)
        // both resolve; `status: live` ends the block and is ignored.
        assert!(
            relates.contains(&"spec-b"),
            "block-list item resolved: {relates:?}"
        );
        assert!(
            relates.contains(&"spec-c"),
            "docs/-prefixed item resolved: {relates:?}"
        );
    }
}
