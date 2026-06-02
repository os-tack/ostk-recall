//! Concept ledger — durable, use-driven concept object permanence.
//!
//! The successor to the shelved emergent-clustering P8 (see
//! `.ostk/threads/cognitive-memory/p8-concepts.md` and
//! `memory-tool-surface.md`). P8 tried to *discover* concepts by
//! clustering raw chunks; empirically the corpus is topically monolithic
//! and HDBSCAN yields one macro-basin or dust. The reframing: the missing
//! layer is not retrieval (recall already finds the answer) but durable
//! concept *objects* a human/agent can promote, merge, correct.
//!
//! Concepts are **use-driven**: minted from recall events as low-trust
//! `candidate`s, promoted through `proposed` to `active` only on repeated
//! use, multi-source evidence, or operator confirmation. Provenance is
//! mandatory — every concept carries source-tagged aliases + evidence, so
//! a model belief with no support stays a visible, correctable candidate
//! rather than silently becoming memory.
//!
//! # Identity is `(project, handle)`
//!
//! `project = ""` means a **global** concept (cross-project — `mish`,
//! `ostk`). A non-empty project scopes the concept so fresh-project
//! handles (`auth`, `client`, `kernel`) don't collide across projects.
//! Lookups [`resolve`](ThreadsDb::resolve_concept) prefer the exact
//! `(project, handle)` then fall back to the global row.
//!
//! # Evidence keys on the durable coordinate, not `chunk_id`
//!
//! `chunk_id` is a coordinate hash that churns on re-chunk / move /
//! reingest (the same way thread anchors orphaned, fixed via `anchor_vec`).
//! So evidence keys on `(source, source_id)` — the stable coordinate —
//! and carries `last_resolved_chunk_id` (a re-resolvable cache),
//! `content_sha256` (change detection), `anchor_vec` (the chunk embedding,
//! for nearest-chunk re-resolution after a re-chunk), and `relation_state`
//! (`active|moved|orphaned`, gated rather than deleted). The
//! [`reconcile_concept_evidence`] sweep keeps the cache fresh as ids churn.
//!
//! Storage lives in `threads.sqlite` (tables in [`ThreadsDb::migrate`]).
//! These methods are an `impl` block on [`ThreadsDb`] split here for
//! readability; they share the same `Mutex<Connection>` via `lock()`.

use std::collections::HashMap;

use chrono::{DateTime, Utc};
use rusqlite::{OptionalExtension, params};

use crate::corpus::{CorpusStore, Result, StoreError};
use crate::ingest::IngestDb;
use crate::threads::{ThreadsDb, bytes_to_f32_vec, f32_vec_to_bytes};

/// Confidence stamped on a freshly observed candidate concept. Low by
/// design — candidates must not influence recall; only `active` concepts
/// are eligible (enforced by the future rank feature).
pub const CANDIDATE_CONFIDENCE: f32 = 0.1;

/// Confidence stamped on a freshly *authored* edge.
///
/// Low by design: an authored edge is a topology *hypothesis* — a low-weight
/// prior that must earn conductance through use (re-observation), not a fact
/// asserted at full strength. "Authoring sets the topology hypothesis; use
/// sets the conductance" (relational-substrate.md).
pub const AUTHORED_EDGE_CONFIDENCE: f32 = 0.1;

/// Confidence stamped on an `observed` edge minted from a gazetteer prose
/// mention (relational-substrate slice 4).
///
/// Flat low prior, like an authored edge: a bare prose mention is evidence of
/// *use*, but conductance is still earned through re-observation, not asserted
/// at full strength. (Lift-weighting of observed edges — "observed weighted by
/// lift" — is a later refinement; this is the flat placeholder.)
pub const OBSERVED_MENTION_CONFIDENCE: f32 = 0.1;

/// The global (cross-project) namespace. `project = ""` concepts are
/// visible from every project's resolution fallback.
pub const GLOBAL_PROJECT: &str = "";

/// Lifecycle status of a concept card.
///
/// `candidate → proposed → active`, with `rejected` (kept as an
/// anti-pattern so it is not re-proposed) and `merged` (folded into a
/// canonical concept; the row keeps a `merged_into` pointer) as terminal
/// branches. Wire form is `snake_case` to match the `status` column.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ConceptStatus {
    Candidate,
    Proposed,
    Active,
    Rejected,
    Merged,
}

impl ConceptStatus {
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Candidate => "candidate",
            Self::Proposed => "proposed",
            Self::Active => "active",
            Self::Rejected => "rejected",
            Self::Merged => "merged",
        }
    }

    /// Terminal states are not re-touched by observation — a `rejected`
    /// concept stays rejected (anti-pattern memory) and a `merged` one
    /// resolves forward to its canonical, so observation must not write to
    /// the tombstone.
    #[must_use]
    pub const fn is_terminal(self) -> bool {
        matches!(self, Self::Rejected | Self::Merged)
    }

    /// Parse a wire string; errors as `InvalidEnumValue` on anything else.
    pub fn parse(s: &str) -> Result<Self> {
        match s {
            "candidate" => Ok(Self::Candidate),
            "proposed" => Ok(Self::Proposed),
            "active" => Ok(Self::Active),
            "rejected" => Ok(Self::Rejected),
            "merged" => Ok(Self::Merged),
            other => Err(StoreError::InvalidEnumValue {
                field: "concept.status".into(),
                value: other.into(),
            }),
        }
    }
}

/// Where an alias was first observed. Drives provenance — a `user`-sourced
/// alias is operator-curated; a `model`-sourced one is unverified belief
/// and stays low-trust until corroborated.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AliasSource {
    Query,
    Path,
    Symbol,
    User,
    Model,
}

impl AliasSource {
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Query => "query",
            Self::Path => "path",
            Self::Symbol => "symbol",
            Self::User => "user",
            Self::Model => "model",
        }
    }

    pub fn parse(s: &str) -> Result<Self> {
        match s {
            "query" => Ok(Self::Query),
            "path" => Ok(Self::Path),
            "symbol" => Ok(Self::Symbol),
            "user" => Ok(Self::User),
            "model" => Ok(Self::Model),
            other => Err(StoreError::InvalidEnumValue {
                field: "concept_alias.source".into(),
                value: other.into(),
            }),
        }
    }
}

/// How an edge entered the substrate — its origin, not its strength.
///
/// `authored` = an operator drew it (a low-weight topology prior, see
/// [`AUTHORED_EDGE_CONFIDENCE`]); `observed` = co-occurrence the system
/// noticed; `promoted` = a latent (similarity) edge walked enough to be
/// reified (no writer yet — slice 2's diffusion produces it). Origin is
/// immutable: re-observation is *use*, not re-authoring, so this never
/// changes once set (relational-substrate.md).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EdgeSource {
    Authored,
    Observed,
    Promoted,
}

impl EdgeSource {
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Authored => "authored",
            Self::Observed => "observed",
            Self::Promoted => "promoted",
        }
    }

    pub fn parse(s: &str) -> Result<Self> {
        match s {
            "authored" => Ok(Self::Authored),
            "observed" => Ok(Self::Observed),
            "promoted" => Ok(Self::Promoted),
            other => Err(StoreError::InvalidEnumValue {
                field: "concept_edge.source".into(),
                value: other.into(),
            }),
        }
    }
}

/// Resolution state of an evidence row against the live corpus.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EvidenceState {
    /// `last_resolved_chunk_id` points at a live corpus chunk.
    Active,
    /// The source path moved; coordinate re-resolved to a new path.
    Moved,
    /// The underlying source/chunk is gone. Kept (not deleted) so the
    /// concept's provenance trail survives — gate, don't delete.
    Orphaned,
}

impl EvidenceState {
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Active => "active",
            Self::Moved => "moved",
            Self::Orphaned => "orphaned",
        }
    }

    pub fn parse(s: &str) -> Result<Self> {
        match s {
            "active" => Ok(Self::Active),
            "moved" => Ok(Self::Moved),
            "orphaned" => Ok(Self::Orphaned),
            other => Err(StoreError::InvalidEnumValue {
                field: "concept_evidence.relation_state".into(),
                value: other.into(),
            }),
        }
    }
}

/// A concept card row.
#[derive(Debug, Clone, PartialEq)]
pub struct ConceptRecord {
    pub id: i64,
    /// `""` = global (cross-project).
    pub project: String,
    pub handle: String,
    pub summary: Option<String>,
    pub status: ConceptStatus,
    pub confidence: f32,
    /// Canonical handle this concept was merged into (when `status == Merged`).
    pub merged_into: Option<String>,
    /// Typed-node kind (`person`, `meeting`, …) when file-seeded; `None` =
    /// untyped (observed/asserted concepts). See `relational-substrate.md`.
    pub kind: Option<String>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

/// A source-tagged alias for a concept.
#[derive(Debug, Clone, PartialEq)]
pub struct ConceptAlias {
    pub alias: String,
    pub source: AliasSource,
    pub confidence: f32,
    pub first_seen_at: DateTime<Utc>,
    pub last_seen_at: DateTime<Utc>,
    pub touch_count: u32,
}

/// A coordinate-anchored evidence link backing a concept.
#[derive(Debug, Clone, PartialEq)]
pub struct ConceptEvidence {
    pub source: String,
    pub source_id: String,
    pub last_resolved_chunk_id: Option<String>,
    pub content_sha256: Option<String>,
    pub score: Option<f32>,
    pub reason: Option<String>,
    pub relation_state: EvidenceState,
    pub first_seen_at: DateTime<Utc>,
    pub last_seen_at: DateTime<Utc>,
    pub touch_count: u32,
}

/// Parameters for [`ThreadsDb::attach_concept_evidence`]. Grouped into a
/// struct so the durable coordinate, the resolvable pointer, and the
/// optional embedding/hash travel together (and to keep the arg count sane).
#[derive(Debug, Clone)]
pub struct EvidenceAttach<'a> {
    pub concept_id: i64,
    pub project: &'a str,
    pub source: &'a str,
    pub source_id: &'a str,
    pub chunk_id: Option<&'a str>,
    pub content_sha256: Option<&'a str>,
    pub anchor_vec: Option<&'a [f32]>,
    pub score: Option<f32>,
    pub reason: Option<&'a str>,
}

/// A concept→concept relation edge (handles resolved for ergonomics).
#[derive(Debug, Clone, PartialEq)]
pub struct ConceptEdge {
    pub id: i64,
    pub from_handle: String,
    pub relation: String,
    pub to_handle: String,
    pub confidence: f32,
    pub evidence_json: Option<String>,
    pub first_seen_at: DateTime<Utc>,
    pub last_seen_at: DateTime<Utc>,
    pub touch_count: u32,
    /// How the edge entered (immutable origin). See [`EdgeSource`].
    pub source: EdgeSource,
    /// Operator that authored/observed the edge, if known. Operators are
    /// *users* of the substrate, not graph nodes — this is a provenance
    /// tag, not an endpoint.
    pub by: Option<String>,
}

/// A narrative memory event attached to a concept (`memory_remember` of
/// kind note/decision/fact/open_question) — a durable, timestamped
/// provenance row rather than summary text.
#[derive(Debug, Clone, PartialEq)]
pub struct ConceptNote {
    pub id: i64,
    pub kind: String,
    pub text: String,
    pub created_at: DateTime<Utc>,
}

/// Direction filter for [`ThreadsDb::list_concept_edges`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EdgeDirection {
    From,
    To,
}

// ---------------------------------------------------------------------
// Deterministic candidate extraction (pure — no I/O, no LLM)
// ---------------------------------------------------------------------

/// A minimal read-only view of a recall hit, decoupled from `RecallHit`
/// (which lives in `ostk-recall-core`) so this crate stays leaf-level and
/// the extractor is unit-testable without a corpus.
#[derive(Debug, Clone)]
pub struct HitView<'a> {
    /// The hit's `source_id` — often a path (`~/projects/mish`,
    /// `crates/store/src/threads.rs`).
    pub source_id: &'a str,
    /// Scanner-supplied symbol tokens (`extra.symbols`), if any.
    pub symbols: &'a [String],
}

/// A single extracted candidate term: its normalized concept `handle`,
/// the surface `alias` it was seen as, and the `source` that produced it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExtractedTerm {
    pub handle: String,
    pub alias: String,
    pub source: AliasSource,
}

/// Tiny stoplist for query-term extraction — common interrogatives /
/// fillers that would otherwise mint junk concepts.
const QUERY_STOPWORDS: &[&str] = &[
    "what", "whats", "where", "when", "which", "who", "whom", "whose", "why", "how", "the", "and",
    "are", "is", "was", "were", "does", "did", "do", "for", "from", "with", "this", "that",
    "these", "those", "about", "into", "between", "explain", "show", "tell", "find", "list",
];

/// Path basenames that are structural code-layout, not concepts. A hit in
/// `crates/store/src/lib.rs` must not mint a concept `lib`. These are the
/// universal Rust/JS/Py file-layout names; the slug filters do the rest.
const STRUCTURAL_BASENAMES: &[&str] = &[
    "lib",
    "mod",
    "main",
    "index",
    "mode",
    "test",
    "tests",
    "types",
    "type",
    "util",
    "utils",
    "common",
    "helper",
    "helpers",
    "error",
    "errors",
    "config",
    "constants",
    "const",
    "src",
    "target",
    "build",
    "readme",
    "license",
    "init",
    "setup",
    "core",
];

/// Normalize a surface term into a concept handle slug.
///
/// Lowercases, keeps `[a-z0-9_-]`, collapses other runs to `-`. Returns
/// `None` if the result is empty, shorter than 3 chars, longer than 40,
/// or all digits (a bare number is not a concept).
#[must_use]
pub fn slugify(term: &str) -> Option<String> {
    let mut out = String::with_capacity(term.len());
    let mut prev_dash = false;
    for ch in term.trim().chars() {
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
    if out.len() < 3 || out.len() > 40 {
        return None;
    }
    if out.chars().all(|c| c.is_ascii_digit() || c == '-') {
        return None;
    }
    Some(out)
}

/// Last path segment without extension, for path-shaped `source_id`s.
/// `~/projects/mish` → `mish`; `crates/store/src/threads.rs` → `threads`.
/// Returns `None` when the input is not path-shaped.
fn path_basename_stem(source_id: &str) -> Option<&str> {
    if !source_id.contains('/') {
        return None;
    }
    let last = source_id.rsplit('/').next()?;
    let stem = last.split('.').next().unwrap_or(last);
    if stem.is_empty() { None } else { Some(stem) }
}

/// Extract conservative, deterministic candidate terms from a query plus
/// its top recall hits. No embeddings, no clustering, no model belief —
/// just path basenames, scanner symbols, and proper query terms, each
/// filtered through [`slugify`] and the structural-basename stoplist. No
/// terms → empty vec is the correct null state. Deduplicated by `handle`
/// (first occurrence wins; path > symbol > query order).
#[must_use]
pub fn extract_concept_terms(query: &str, hits: &[HitView]) -> Vec<ExtractedTerm> {
    let mut seen = std::collections::HashSet::new();
    let mut out = Vec::new();
    let mut push = |handle: String, alias: String, source: AliasSource, out: &mut Vec<_>| {
        if seen.insert(handle.clone()) {
            out.push(ExtractedTerm {
                handle,
                alias,
                source,
            });
        }
    };

    // 1) Path basenames from hit source_ids (highest provenance), minus
    //    structural code-layout names (lib/mod/tests/…).
    for hit in hits {
        if let Some(stem) = path_basename_stem(hit.source_id) {
            if STRUCTURAL_BASENAMES.contains(&stem.to_ascii_lowercase().as_str()) {
                continue;
            }
            if let Some(handle) = slugify(stem) {
                push(handle, stem.to_string(), AliasSource::Path, &mut out);
            }
        }
    }

    // 2) Scanner-supplied symbols (CamelCase / snake_case / module::path).
    for hit in hits {
        for sym in hit.symbols {
            let tail = sym.rsplit("::").next().unwrap_or(sym);
            if let Some(handle) = slugify(tail) {
                push(handle, tail.to_string(), AliasSource::Symbol, &mut out);
            }
        }
    }

    // 3) Proper terms from the query: quoted spans first, then tokens that
    //    survive the stoplist + slugify.
    for span in quoted_spans(query) {
        if let Some(handle) = slugify(&span) {
            push(handle, span, AliasSource::Query, &mut out);
        }
    }
    for tok in query.split(|c: char| !c.is_ascii_alphanumeric() && c != '_') {
        if tok.is_empty() {
            continue;
        }
        let lower = tok.to_ascii_lowercase();
        if QUERY_STOPWORDS.contains(&lower.as_str()) {
            continue;
        }
        if let Some(handle) = slugify(tok) {
            push(handle, tok.to_string(), AliasSource::Query, &mut out);
        }
    }

    out
}

/// Extract double-quoted spans (keeps multi-word proper terms intact).
fn quoted_spans(s: &str) -> Vec<String> {
    let mut spans = Vec::new();
    let mut current: Option<String> = None;
    for ch in s.chars() {
        match (ch, current.as_mut()) {
            ('"', None) => current = Some(String::new()),
            ('"', Some(buf)) => {
                let done = std::mem::take(buf);
                if !done.trim().is_empty() {
                    spans.push(done);
                }
                current = None;
            }
            (_, Some(buf)) => buf.push(ch),
            (_, None) => {}
        }
    }
    spans
}

// ---------------------------------------------------------------------
// Concept ledger CRUD on ThreadsDb
// ---------------------------------------------------------------------

impl ThreadsDb {
    /// Get-or-create a concept by `(project, handle)`. If absent, inserts a
    /// row with `initial_status`; if present, the existing row is returned
    /// UNCHANGED (status is never downgraded by re-observation — promotion
    /// is one-way and explicit). Returns `(record, created)`.
    pub fn ensure_concept(
        &self,
        project: &str,
        handle: &str,
        initial_status: ConceptStatus,
    ) -> Result<(ConceptRecord, bool)> {
        let now = Utc::now();
        let conn = self.lock();
        let confidence = if matches!(initial_status, ConceptStatus::Candidate) {
            CANDIDATE_CONFIDENCE
        } else {
            0.0
        };
        let n = conn.execute(
            "INSERT INTO concepts (project, handle, status, confidence, created_at, updated_at)
             VALUES (?, ?, ?, ?, ?, ?)
             ON CONFLICT(project, handle) DO NOTHING",
            params![
                project,
                handle,
                initial_status.as_str(),
                confidence,
                now.to_rfc3339(),
                now.to_rfc3339(),
            ],
        )?;
        let record = get_concept_exact(&conn, project, handle)?.ok_or_else(|| {
            StoreError::InvalidEnumValue {
                field: "concept.handle".into(),
                value: format!("{project}/{handle} vanished after ensure"),
            }
        })?;
        Ok((record, n == 1))
    }

    /// Get-or-create a *typed* concept by `(project, handle)`.
    ///
    /// Like [`Self::ensure_concept`] but carries a node `kind` (`person`,
    /// `meeting`, …). On conflict: status is never downgraded (same contract),
    /// and `kind` is **COALESCE-backfilled** — a node first observed untyped
    /// gets its kind set by a later file-seed, while an already-typed node
    /// keeps its kind (never overwritten). Returns `(record, created)`.
    pub fn ensure_typed_concept(
        &self,
        project: &str,
        handle: &str,
        initial_status: ConceptStatus,
        kind: Option<&str>,
    ) -> Result<(ConceptRecord, bool)> {
        let now = Utc::now();
        let conn = self.lock();
        // `DO UPDATE` makes rows-affected unreliable for "created", so probe
        // existence first.
        let created = get_concept_exact(&conn, project, handle)?.is_none();
        let confidence = if matches!(initial_status, ConceptStatus::Candidate) {
            CANDIDATE_CONFIDENCE
        } else {
            0.0
        };
        conn.execute(
            "INSERT INTO concepts (project, handle, status, confidence, kind, created_at, updated_at)
             VALUES (?, ?, ?, ?, ?, ?, ?)
             ON CONFLICT(project, handle) DO UPDATE SET
                 kind = COALESCE(concepts.kind, excluded.kind)",
            params![
                project,
                handle,
                initial_status.as_str(),
                confidence,
                kind,
                now.to_rfc3339(),
                now.to_rfc3339(),
            ],
        )?;
        let record = get_concept_exact(&conn, project, handle)?.ok_or_else(|| {
            StoreError::InvalidEnumValue {
                field: "concept.handle".into(),
                value: format!("{project}/{handle} vanished after ensure_typed"),
            }
        })?;
        Ok((record, created))
    }

    /// Fetch a concept by exact `(project, handle)`.
    pub fn get_concept(&self, project: &str, handle: &str) -> Result<Option<ConceptRecord>> {
        let conn = self.lock();
        get_concept_exact(&conn, project, handle)
    }

    /// Resolve a name within `project` to its live concept. Matches a
    /// `handle` first, then an inbound **alias** (so `ss` lands on
    /// `slipstream`), preferring the exact scope then the global (`""`)
    /// concept. Follows `merged_into` **forward** so a merged handle/alias
    /// resolves to its canonical concept rather than the tombstone (with a
    /// cycle guard). This is the read path callers should use.
    pub fn resolve_concept(&self, project: &str, handle: &str) -> Result<Option<ConceptRecord>> {
        let conn = self.lock();
        let mut current = handle.to_string();
        let mut seen = std::collections::HashSet::new();
        loop {
            if !seen.insert(current.clone()) {
                return Ok(None); // merge cycle — give up rather than loop
            }
            match resolve_one(&conn, project, &current)? {
                Some(rec) if rec.status == ConceptStatus::Merged => match rec.merged_into.clone() {
                    Some(canonical) => current = canonical, // follow the tombstone forward
                    None => return Ok(Some(rec)),
                },
                other => return Ok(other),
            }
        }
    }

    /// List concepts, optionally filtered by project and/or status,
    /// newest-touched first. `project_filter = None` lists all projects.
    pub fn list_concepts(
        &self,
        project_filter: Option<&str>,
        status: Option<ConceptStatus>,
    ) -> Result<Vec<ConceptRecord>> {
        let conn = self.lock();
        let mut sql = String::from(
            "SELECT id, project, handle, summary, status, confidence, merged_into,
                    created_at, updated_at, kind FROM concepts",
        );
        let mut clauses = Vec::new();
        if project_filter.is_some() {
            // "Visible from scope p" = the project's own concepts + globals,
            // mirroring resolve_concept. So Some("") lists only globals, and
            // Some("acme") lists acme + globals. None lists every project.
            clauses.push("(project = ? OR project = '')");
        }
        if status.is_some() {
            clauses.push("status = ?");
        }
        if !clauses.is_empty() {
            sql.push_str(" WHERE ");
            sql.push_str(&clauses.join(" AND "));
        }
        sql.push_str(" ORDER BY updated_at DESC");
        let mut stmt = conn.prepare(&sql)?;
        // Bind in clause order.
        type Rows = std::result::Result<Vec<ConceptRecord>, rusqlite::Error>;
        let rows: Rows = match (project_filter, status) {
            (Some(p), Some(s)) => stmt
                .query_map(params![p, s.as_str()], row_to_concept)?
                .collect(),
            (Some(p), None) => stmt.query_map(params![p], row_to_concept)?.collect(),
            (None, Some(s)) => stmt
                .query_map(params![s.as_str()], row_to_concept)?
                .collect(),
            (None, None) => stmt.query_map([], row_to_concept)?.collect(),
        };
        Ok(rows?)
    }

    /// Set a concept's status (and optionally confidence). Bumps
    /// `updated_at`. Errors if `(project, handle)` does not exist.
    pub fn set_concept_status(
        &self,
        project: &str,
        handle: &str,
        status: ConceptStatus,
        confidence: Option<f32>,
    ) -> Result<()> {
        let now = Utc::now();
        let conn = self.lock();
        let n = match confidence {
            Some(c) => conn.execute(
                "UPDATE concepts SET status = ?, confidence = ?, updated_at = ?
                 WHERE project = ? AND handle = ?",
                params![status.as_str(), c, now.to_rfc3339(), project, handle],
            )?,
            None => conn.execute(
                "UPDATE concepts SET status = ?, updated_at = ? WHERE project = ? AND handle = ?",
                params![status.as_str(), now.to_rfc3339(), project, handle],
            )?,
        };
        if n == 0 {
            return Err(not_found(project, handle));
        }
        Ok(())
    }

    /// Set a concept's summary text (evidence-based snippet in v1; no LLM).
    pub fn set_concept_summary(&self, project: &str, handle: &str, summary: &str) -> Result<()> {
        let now = Utc::now();
        let conn = self.lock();
        let n = conn.execute(
            "UPDATE concepts SET summary = ?, updated_at = ? WHERE project = ? AND handle = ?",
            params![summary, now.to_rfc3339(), project, handle],
        )?;
        if n == 0 {
            return Err(not_found(project, handle));
        }
        Ok(())
    }

    /// Upsert an alias. Re-observation bumps `touch_count` + `last_seen_at`.
    pub fn touch_alias(&self, concept_id: i64, alias: &str, source: AliasSource) -> Result<()> {
        let now = Utc::now();
        let conn = self.lock();
        conn.execute(
            "INSERT INTO concept_aliases
                 (concept_id, alias, source, confidence, first_seen_at, last_seen_at, touch_count)
             VALUES (?, ?, ?, ?, ?, ?, 1)
             ON CONFLICT(concept_id, alias) DO UPDATE SET
                 last_seen_at = excluded.last_seen_at,
                 touch_count  = touch_count + 1",
            params![
                concept_id,
                alias,
                source.as_str(),
                CANDIDATE_CONFIDENCE,
                now.to_rfc3339(),
                now.to_rfc3339(),
            ],
        )?;
        Ok(())
    }

    /// Attach (or re-touch) coordinate-anchored evidence for a concept.
    /// Keyed on `(concept_id, source, source_id)` — re-observing the same
    /// place bumps `touch_count` and refreshes the resolvable pointer
    /// instead of duplicating. `anchor_vec` is only written when supplied.
    pub fn attach_concept_evidence(&self, ev: &EvidenceAttach) -> Result<()> {
        let now = Utc::now();
        let conn = self.lock();
        conn.execute(
            "INSERT INTO concept_evidence
                 (concept_id, project, source, source_id, last_resolved_chunk_id,
                  content_sha256, anchor_vec, score, reason, relation_state,
                  first_seen_at, last_seen_at, touch_count)
             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'active', ?, ?, 1)
             ON CONFLICT(concept_id, source, source_id) DO UPDATE SET
                 -- Keep (pointer, sha, anchor) describing the SAME chunk:
                 -- replace the triple together only when the new observation
                 -- is corpus-backed (carries an anchor_vec); otherwise leave
                 -- the prior coherent triple untouched. Avoids a pointer that
                 -- names one chunk while sha/anchor describe another.
                 last_resolved_chunk_id = CASE WHEN excluded.anchor_vec IS NOT NULL
                     THEN excluded.last_resolved_chunk_id ELSE last_resolved_chunk_id END,
                 content_sha256 = CASE WHEN excluded.anchor_vec IS NOT NULL
                     THEN excluded.content_sha256 ELSE content_sha256 END,
                 anchor_vec = CASE WHEN excluded.anchor_vec IS NOT NULL
                     THEN excluded.anchor_vec ELSE anchor_vec END,
                 score          = excluded.score,
                 reason         = COALESCE(excluded.reason, reason),
                 relation_state = 'active',
                 last_seen_at   = excluded.last_seen_at,
                 touch_count    = touch_count + 1",
            params![
                ev.concept_id,
                ev.project,
                ev.source,
                ev.source_id,
                ev.chunk_id,
                ev.content_sha256,
                ev.anchor_vec.map(f32_vec_to_bytes),
                ev.score,
                ev.reason,
                now.to_rfc3339(),
                now.to_rfc3339(),
            ],
        )?;
        Ok(())
    }

    /// List a concept's aliases, most-touched first.
    pub fn list_aliases(&self, concept_id: i64) -> Result<Vec<ConceptAlias>> {
        let conn = self.lock();
        let mut stmt = conn.prepare(
            "SELECT alias, source, confidence, first_seen_at, last_seen_at, touch_count
             FROM concept_aliases WHERE concept_id = ? ORDER BY touch_count DESC, alias ASC",
        )?;
        let rows = stmt
            .query_map(params![concept_id], row_to_alias)?
            .collect::<std::result::Result<Vec<_>, rusqlite::Error>>()?;
        Ok(rows)
    }

    /// List a concept's evidence, active first then by score.
    pub fn list_concept_evidence(&self, concept_id: i64) -> Result<Vec<ConceptEvidence>> {
        let conn = self.lock();
        let mut stmt = conn.prepare(
            "SELECT source, source_id, last_resolved_chunk_id, content_sha256, score, reason,
                    relation_state, first_seen_at, last_seen_at, touch_count
             FROM concept_evidence WHERE concept_id = ?
             ORDER BY (relation_state = 'active') DESC, score DESC NULLS LAST, touch_count DESC",
        )?;
        let rows = stmt
            .query_map(params![concept_id], row_to_evidence)?
            .collect::<std::result::Result<Vec<_>, rusqlite::Error>>()?;
        Ok(rows)
    }

    /// Add (or re-touch) a concept→concept relation edge by handle, resolved
    /// within `project` (global fallback). Both endpoints must exist.
    /// Returns the edge id; re-observation bumps `touch_count`.
    ///
    /// Note: resolving handles here re-applies the project→global preference,
    /// so callers that have already resolved a *specific* (possibly global)
    /// endpoint should use [`Self::add_concept_edge_by_id`] to avoid a
    /// same-handle local concept shadowing the intended global one.
    #[allow(clippy::too_many_arguments)]
    pub fn add_concept_edge(
        &self,
        project: &str,
        from_handle: &str,
        relation: &str,
        to_handle: &str,
        confidence: f32,
        source: EdgeSource,
        by: Option<&str>,
        evidence_json: Option<&str>,
    ) -> Result<i64> {
        let conn = self.lock();
        let from_id = resolve_id(&conn, project, from_handle)?;
        let to_id = resolve_id(&conn, project, to_handle)?;
        insert_concept_edge(
            &conn,
            from_id,
            relation,
            to_id,
            confidence,
            source,
            by,
            evidence_json,
        )
        .map(|(id, _)| id)
    }

    /// Add (or re-touch) an edge between two concepts identified by their
    /// surrogate ids — no handle re-resolution, so the exact endpoints the
    /// caller resolved are used (the scope-correct path for cross-project /
    /// global edges). Returns `(edge_id, created)` where `created` is true
    /// only on first insert (so callers can gate a one-time chain event).
    #[allow(clippy::too_many_arguments)]
    pub fn add_concept_edge_by_id(
        &self,
        from_id: i64,
        relation: &str,
        to_id: i64,
        confidence: f32,
        source: EdgeSource,
        by: Option<&str>,
        evidence_json: Option<&str>,
    ) -> Result<(i64, bool)> {
        let conn = self.lock();
        insert_concept_edge(
            &conn,
            from_id,
            relation,
            to_id,
            confidence,
            source,
            by,
            evidence_json,
        )
    }

    /// List edges incident on `(project, handle)` in the given direction.
    pub fn list_concept_edges(
        &self,
        project: &str,
        handle: &str,
        direction: EdgeDirection,
    ) -> Result<Vec<ConceptEdge>> {
        let conn = self.lock();
        let Some(id) = resolve_id_opt(&conn, project, handle)? else {
            return Ok(Vec::new());
        };
        let sql = match direction {
            EdgeDirection::From => {
                "SELECT e.id, cf.handle, e.relation, ct.handle, e.confidence, e.evidence_json,
                        e.first_seen_at, e.last_seen_at, e.touch_count, e.source, e.[by]
                 FROM concept_edges e
                 JOIN concepts cf ON cf.id = e.from_concept
                 JOIN concepts ct ON ct.id = e.to_concept
                 WHERE e.from_concept = ? ORDER BY e.touch_count DESC, e.id ASC"
            }
            EdgeDirection::To => {
                "SELECT e.id, cf.handle, e.relation, ct.handle, e.confidence, e.evidence_json,
                        e.first_seen_at, e.last_seen_at, e.touch_count, e.source, e.[by]
                 FROM concept_edges e
                 JOIN concepts cf ON cf.id = e.from_concept
                 JOIN concepts ct ON ct.id = e.to_concept
                 WHERE e.to_concept = ? ORDER BY e.touch_count DESC, e.id ASC"
            }
        };
        let mut stmt = conn.prepare(sql)?;
        let rows = stmt
            .query_map(params![id], row_to_edge)?
            .collect::<std::result::Result<Vec<_>, rusqlite::Error>>()?;
        Ok(rows)
    }

    /// Edges incident on `(project, handle)` paired with each edge's
    /// **derived conductance** as of `now` — the one-hop traversal read for
    /// the relational substrate. Conductance is not stored; it is computed
    /// from `confidence` and `last_seen_at` recency (see
    /// [`crate::activation::edge_conductance`]) so it cannot drift from the
    /// edge's use history. Sorted strongest-conductance first.
    pub fn neighbors(
        &self,
        project: &str,
        handle: &str,
        direction: EdgeDirection,
        now: DateTime<Utc>,
    ) -> Result<Vec<(ConceptEdge, f32)>> {
        let mut out: Vec<(ConceptEdge, f32)> = self
            .list_concept_edges(project, handle, direction)?
            .into_iter()
            .map(|e| {
                let c = crate::activation::edge_conductance(&e, now);
                (e, c)
            })
            .collect();
        out.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.0.id.cmp(&b.0.id))
        });
        Ok(out)
    }

    /// Neighbour concept **ids** of `concept_id` (both edge directions), each
    /// paired with the edge's derived conductance as of `now` — the id-space
    /// traversal read for diffusion (relational-substrate slice 2).
    ///
    /// Works entirely in id-space (never re-resolving handles), so a
    /// cross-scope edge authored by id (slice 4) traverses to the exact
    /// endpoint rather than a same-handle look-alike in another scope. One
    /// indexed scan over `concept_edges` (both endpoint indexes). Self-loops
    /// are excluded (the table CHECK bars them anyway). **Terminal neighbours**
    /// (`merged` / `rejected`) are excluded — diffusion must not flow into or
    /// surface a tombstoned/anti-pattern concept (terminal states are never
    /// touched). Multiple edges between the same pair surface as separate rows;
    /// the caller's max-accumulation collapses them.
    pub fn neighbors_by_id(&self, concept_id: i64, now: DateTime<Utc>) -> Result<Vec<(i64, f32)>> {
        let conn = self.lock();
        let mut stmt = conn.prepare(
            "SELECT CASE WHEN e.from_concept = ?1 THEN e.to_concept ELSE e.from_concept END AS nbr,
                    e.confidence, e.last_seen_at
             FROM concept_edges e
             JOIN concepts c
               ON c.id = (CASE WHEN e.from_concept = ?1 THEN e.to_concept ELSE e.from_concept END)
             WHERE (e.from_concept = ?1 OR e.to_concept = ?1)
               AND c.status NOT IN ('merged', 'rejected')",
        )?;
        let rows = stmt
            .query_map(params![concept_id], |row| {
                let nbr: i64 = row.get(0)?;
                let confidence: f32 = row.get(1)?;
                let last_seen_at: String = row.get(2)?;
                Ok((nbr, confidence, last_seen_at))
            })?
            .collect::<std::result::Result<Vec<_>, rusqlite::Error>>()?;
        let mut out = Vec::with_capacity(rows.len());
        for (nbr, confidence, lsa) in rows {
            if nbr == concept_id {
                continue; // defensive: the CHECK already bars self-edges
            }
            let last_seen_at = parse_ts(&lsa)?;
            out.push((
                nbr,
                crate::activation::conductance_of(confidence, last_seen_at, now),
            ));
        }
        Ok(out)
    }

    /// Merge `from_handle` into `into_handle` (both resolved within
    /// `project`): move aliases + evidence to the canonical concept,
    /// **rewrite incident edges** onto the canonical (dropping duplicates
    /// and self-loops), record the merged handle as a `user` alias, and
    /// tombstone the source row (`status = merged`, `merged_into` set, kept
    /// so stale references resolve forward).
    pub fn merge_concept(&self, project: &str, from_handle: &str, into_handle: &str) -> Result<()> {
        if from_handle == into_handle {
            return Err(StoreError::InvalidEnumValue {
                field: "concept.merge".into(),
                value: "cannot merge a concept into itself".into(),
            });
        }
        let now = Utc::now();
        let mut conn = self.lock();
        let tx = conn.transaction()?;
        let from_id = resolve_id(&tx, project, from_handle)?;
        let into_id = resolve_id(&tx, project, into_handle)?;
        if from_id == into_id {
            return Err(StoreError::InvalidEnumValue {
                field: "concept.merge".into(),
                value: format!("{from_handle} and {into_handle} resolve to the same concept"),
            });
        }

        // Move aliases + evidence (INSERT OR IGNORE preserves the canonical's
        // existing rows on the UNIQUE keys), then drop the source's.
        tx.execute(
            "INSERT OR IGNORE INTO concept_aliases
                 (concept_id, alias, source, confidence, first_seen_at, last_seen_at, touch_count)
             SELECT ?, alias, source, confidence, first_seen_at, last_seen_at, touch_count
             FROM concept_aliases WHERE concept_id = ?",
            params![into_id, from_id],
        )?;
        tx.execute(
            "DELETE FROM concept_aliases WHERE concept_id = ?",
            params![from_id],
        )?;
        tx.execute(
            "INSERT OR IGNORE INTO concept_evidence
                 (concept_id, project, source, source_id, last_resolved_chunk_id, content_sha256,
                  anchor_vec, score, reason, relation_state, first_seen_at, last_seen_at, touch_count)
             SELECT ?, project, source, source_id, last_resolved_chunk_id, content_sha256,
                    anchor_vec, score, reason, relation_state, first_seen_at, last_seen_at, touch_count
             FROM concept_evidence WHERE concept_id = ?",
            params![into_id, from_id],
        )?;
        tx.execute(
            "DELETE FROM concept_evidence WHERE concept_id = ?",
            params![from_id],
        )?;

        // Rewrite incident edges onto the canonical. `UPDATE OR IGNORE`
        // leaves a row in place when the repoint would violate
        // UNIQUE(from, relation, to) or the from<>to CHECK; those leftovers
        // (duplicates / self-loops) still reference `from_id`, so the
        // follow-up DELETE prunes them. Net: the canonical keeps one edge
        // per (relation, other-endpoint), no stranded or self edges remain.
        tx.execute(
            "UPDATE OR IGNORE concept_edges SET from_concept = ? WHERE from_concept = ?",
            params![into_id, from_id],
        )?;
        tx.execute(
            "UPDATE OR IGNORE concept_edges SET to_concept = ? WHERE to_concept = ?",
            params![into_id, from_id],
        )?;
        tx.execute(
            "DELETE FROM concept_edges WHERE from_concept = ? OR to_concept = ?",
            params![from_id, from_id],
        )?;

        // The merged handle becomes an alias of the canonical.
        tx.execute(
            "INSERT OR IGNORE INTO concept_aliases
                 (concept_id, alias, source, confidence, first_seen_at, last_seen_at, touch_count)
             VALUES (?, ?, 'user', ?, ?, ?, 1)",
            params![
                into_id,
                from_handle,
                CANDIDATE_CONFIDENCE,
                now.to_rfc3339(),
                now.to_rfc3339(),
            ],
        )?;

        // Tombstone the source row.
        tx.execute(
            "UPDATE concepts SET status = 'merged', merged_into = ?, updated_at = ? WHERE id = ?",
            params![into_handle, now.to_rfc3339(), from_id],
        )?;
        tx.commit()?;
        Ok(())
    }

    /// Append a narrative note (note/decision/fact/open_question) to a
    /// concept. A durable, timestamped provenance row.
    pub fn add_concept_note(&self, concept_id: i64, kind: &str, text: &str) -> Result<i64> {
        let now = Utc::now();
        let conn = self.lock();
        conn.execute(
            "INSERT INTO concept_notes (concept_id, kind, text, created_at) VALUES (?, ?, ?, ?)",
            params![concept_id, kind, text, now.to_rfc3339()],
        )?;
        Ok(conn.last_insert_rowid())
    }

    /// List a concept's narrative notes, newest first.
    pub fn list_concept_notes(&self, concept_id: i64) -> Result<Vec<ConceptNote>> {
        let conn = self.lock();
        let mut stmt = conn.prepare(
            "SELECT id, kind, text, created_at FROM concept_notes
             WHERE concept_id = ? ORDER BY created_at DESC, id DESC",
        )?;
        let rows = stmt
            .query_map(params![concept_id], |r| {
                Ok(ConceptNote {
                    id: r.get(0)?,
                    kind: r.get(1)?,
                    text: r.get(2)?,
                    created_at: parse_ts(&r.get::<_, String>(3)?)?,
                })
            })?
            .collect::<std::result::Result<Vec<_>, rusqlite::Error>>()?;
        Ok(rows)
    }

    // ---------- reconciler support ----------

    /// Raw rows the reconciler needs: every non-orphaned evidence row with
    /// its implicit `rowid` and decoded `anchor_vec`.
    pub fn evidence_to_reconcile(&self) -> Result<Vec<EvidenceReconcileRow>> {
        let conn = self.lock();
        let mut stmt = conn.prepare(
            "SELECT rowid, concept_id, source, source_id, last_resolved_chunk_id, anchor_vec
             FROM concept_evidence WHERE relation_state != 'orphaned'",
        )?;
        let rows = stmt
            .query_map([], |r| {
                let anchor_bytes: Option<Vec<u8>> = r.get(5)?;
                let anchor_vec = match anchor_bytes {
                    Some(b) => Some(bytes_to_f32_vec(&b).map_err(to_sql_err)?),
                    None => None,
                };
                Ok(EvidenceReconcileRow {
                    rowid: r.get(0)?,
                    concept_id: r.get(1)?,
                    source: r.get(2)?,
                    source_id: r.get(3)?,
                    last_resolved_chunk_id: r.get(4)?,
                    anchor_vec,
                })
            })?
            .collect::<std::result::Result<Vec<_>, rusqlite::Error>>()?;
        Ok(rows)
    }

    /// Re-point an evidence row to a freshly resolved chunk. Coordinates +
    /// pointer + content hash are overwritten; `anchor_vec` is written only
    /// when supplied (so re-resolution preserves the original semantic
    /// anchor while legacy backfill seeds it). Sets `relation_state`.
    pub fn update_evidence_resolution(
        &self,
        rowid: i64,
        source: &str,
        source_id: &str,
        chunk_id: Option<&str>,
        content_sha256: Option<&str>,
        anchor_vec: Option<&[f32]>,
        state: EvidenceState,
    ) -> Result<()> {
        let now = Utc::now();
        let conn = self.lock();
        conn.execute(
            "UPDATE concept_evidence SET
                 source = ?, source_id = ?, last_resolved_chunk_id = ?,
                 content_sha256 = COALESCE(?, content_sha256),
                 anchor_vec = COALESCE(?, anchor_vec),
                 relation_state = ?, last_seen_at = ?
             WHERE rowid = ?",
            params![
                source,
                source_id,
                chunk_id,
                content_sha256,
                anchor_vec.map(f32_vec_to_bytes),
                state.as_str(),
                now.to_rfc3339(),
                rowid,
            ],
        )?;
        Ok(())
    }

    /// Mark an evidence row orphaned (gate, don't delete).
    pub fn mark_evidence_orphaned(&self, rowid: i64) -> Result<()> {
        let now = Utc::now();
        let conn = self.lock();
        conn.execute(
            "UPDATE concept_evidence SET relation_state = 'orphaned', last_seen_at = ?
             WHERE rowid = ?",
            params![now.to_rfc3339(), rowid],
        )?;
        Ok(())
    }
}

/// A raw evidence row for the reconciler (carries the implicit `rowid`).
#[derive(Debug, Clone)]
pub struct EvidenceReconcileRow {
    pub rowid: i64,
    pub concept_id: i64,
    pub source: String,
    pub source_id: String,
    pub last_resolved_chunk_id: Option<String>,
    pub anchor_vec: Option<Vec<f32>>,
}

/// Outcome of a [`reconcile_concept_evidence`] sweep.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ReconcileStats {
    pub checked: usize,
    pub still_valid: usize,
    pub backfilled: usize,
    pub re_resolved: usize,
    pub orphaned: usize,
}

/// Keep concept evidence linked as `chunk_id`s churn. For each active
/// evidence row:
/// 1. if `last_resolved_chunk_id` still resolves in the corpus, it's valid
///    (a legacy row with no coordinate is *backfilled* from that chunk);
/// 2. else re-resolve via the durable `(source, source_id)` coordinate —
///    pick the current chunk nearest the cached `anchor_vec` (or the first
///    if no anchor);
/// 3. else mark `orphaned` (gate, don't delete) — the concept keeps its
///    provenance trail.
///
/// Deterministic and idempotent. Mirrors `re_anchor_threads_from_corpus`
/// (boot pass) and rides the same corpus/ingest handles the daemon holds.
pub async fn reconcile_concept_evidence(
    threads: &ThreadsDb,
    corpus: &CorpusStore,
    ingest: &IngestDb,
) -> Result<ReconcileStats> {
    let mut stats = ReconcileStats::default();
    for row in threads.evidence_to_reconcile()? {
        stats.checked += 1;

        // 1. Current pointer still valid?
        if let Some(cid) = row.last_resolved_chunk_id.as_deref() {
            let map = corpus.fetch_chunks_by_ids(&[cid.to_string()]).await?;
            if let Some((chunk, emb)) = map.get(cid) {
                if row.source.is_empty() {
                    // Legacy row: learn the durable coordinate from the chunk.
                    threads.update_evidence_resolution(
                        row.rowid,
                        chunk.source.as_str(),
                        &chunk.source_id,
                        Some(cid),
                        Some(&chunk.sha256),
                        emb.as_deref(),
                        EvidenceState::Active,
                    )?;
                    stats.backfilled += 1;
                } else {
                    stats.still_valid += 1;
                }
                continue;
            }
        }

        // 2. Pointer dead → re-resolve by the durable coordinate.
        if !row.source.is_empty() {
            let candidates = ingest.chunk_ids_for_source_id(&row.source, &row.source_id)?;
            if !candidates.is_empty() {
                let map = corpus.fetch_chunks_by_ids(&candidates).await?;
                if let Some((best_id, sha)) =
                    pick_nearest(&candidates, &map, row.anchor_vec.as_deref())
                {
                    threads.update_evidence_resolution(
                        row.rowid,
                        &row.source,
                        &row.source_id,
                        Some(&best_id),
                        sha.as_deref(),
                        None, // preserve the original semantic anchor
                        EvidenceState::Active,
                    )?;
                    stats.re_resolved += 1;
                    continue;
                }
            }
        }

        // 3. Unresolvable → orphan.
        threads.mark_evidence_orphaned(row.rowid)?;
        stats.orphaned += 1;
    }
    Ok(stats)
}

/// Pick the candidate chunk nearest the `anchor` embedding (cosine), or the
/// first present candidate when no anchor is cached. Returns
/// `(chunk_id, content_sha256)`.
fn pick_nearest(
    ordered_ids: &[String],
    map: &HashMap<String, (ostk_recall_core::Chunk, Option<Vec<f32>>)>,
    anchor: Option<&[f32]>,
) -> Option<(String, Option<String>)> {
    match anchor {
        Some(a) => ordered_ids
            .iter()
            .filter_map(|id| map.get(id).map(|(c, e)| (id, c, e)))
            .filter_map(|(id, c, e)| e.as_ref().map(|emb| (id, c, cosine(a, emb))))
            .max_by(|x, y| x.2.total_cmp(&y.2))
            .map(|(id, c, _)| (id.clone(), Some(c.sha256.clone()))),
        None => ordered_ids.iter().find_map(|id| {
            map.get(id)
                .map(|(c, _)| (id.clone(), Some(c.sha256.clone())))
        }),
    }
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let mut dot = 0.0f32;
    let mut na = 0.0f32;
    let mut nb = 0.0f32;
    for i in 0..a.len() {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    if na == 0.0 || nb == 0.0 {
        0.0
    } else {
        dot / (na.sqrt() * nb.sqrt())
    }
}

// ---------------------------------------------------------------------
// Row mappers + locked helpers
// ---------------------------------------------------------------------

fn not_found(project: &str, handle: &str) -> StoreError {
    StoreError::InvalidEnumValue {
        field: "concept.handle".into(),
        value: format!("{project}/{handle} not found"),
    }
}

fn to_sql_err(e: StoreError) -> rusqlite::Error {
    rusqlite::Error::FromSqlConversionFailure(0, rusqlite::types::Type::Blob, Box::new(e))
}

fn parse_ts(s: &str) -> rusqlite::Result<DateTime<Utc>> {
    DateTime::parse_from_rfc3339(s)
        .map(|dt| dt.with_timezone(&Utc))
        .map_err(|e| {
            rusqlite::Error::FromSqlConversionFailure(0, rusqlite::types::Type::Text, Box::new(e))
        })
}

fn row_to_concept(row: &rusqlite::Row) -> rusqlite::Result<ConceptRecord> {
    let status_s: String = row.get(4)?;
    let status = ConceptStatus::parse(&status_s).map_err(to_sql_err)?;
    Ok(ConceptRecord {
        id: row.get(0)?,
        project: row.get(1)?,
        handle: row.get(2)?,
        summary: row.get(3)?,
        status,
        confidence: row.get(5)?,
        merged_into: row.get(6)?,
        created_at: parse_ts(&row.get::<_, String>(7)?)?,
        updated_at: parse_ts(&row.get::<_, String>(8)?)?,
        kind: row.get(9)?,
    })
}

fn row_to_alias(row: &rusqlite::Row) -> rusqlite::Result<ConceptAlias> {
    let source_s: String = row.get(1)?;
    let source = AliasSource::parse(&source_s).map_err(to_sql_err)?;
    Ok(ConceptAlias {
        alias: row.get(0)?,
        source,
        confidence: row.get(2)?,
        first_seen_at: parse_ts(&row.get::<_, String>(3)?)?,
        last_seen_at: parse_ts(&row.get::<_, String>(4)?)?,
        touch_count: row.get(5)?,
    })
}

fn row_to_evidence(row: &rusqlite::Row) -> rusqlite::Result<ConceptEvidence> {
    let state_s: String = row.get(6)?;
    let relation_state = EvidenceState::parse(&state_s).map_err(to_sql_err)?;
    Ok(ConceptEvidence {
        source: row.get(0)?,
        source_id: row.get(1)?,
        last_resolved_chunk_id: row.get(2)?,
        content_sha256: row.get(3)?,
        score: row.get(4)?,
        reason: row.get(5)?,
        relation_state,
        first_seen_at: parse_ts(&row.get::<_, String>(7)?)?,
        last_seen_at: parse_ts(&row.get::<_, String>(8)?)?,
        touch_count: row.get(9)?,
    })
}

fn row_to_edge(row: &rusqlite::Row) -> rusqlite::Result<ConceptEdge> {
    let source = EdgeSource::parse(&row.get::<_, String>(9)?).map_err(|e| {
        rusqlite::Error::FromSqlConversionFailure(9, rusqlite::types::Type::Text, Box::new(e))
    })?;
    Ok(ConceptEdge {
        id: row.get(0)?,
        from_handle: row.get(1)?,
        relation: row.get(2)?,
        to_handle: row.get(3)?,
        confidence: row.get(4)?,
        evidence_json: row.get(5)?,
        first_seen_at: parse_ts(&row.get::<_, String>(6)?)?,
        last_seen_at: parse_ts(&row.get::<_, String>(7)?)?,
        touch_count: row.get(8)?,
        source,
        by: row.get(10)?,
    })
}

/// Insert (or re-touch) a concept edge by surrogate ids. Shared core of
/// [`ThreadsDb::add_concept_edge`] and [`ThreadsDb::add_concept_edge_by_id`].
/// ON CONFLICT leaves `source`/`by` untouched — an edge's origin is immutable;
/// re-observation is *use* (bump `touch_count`, refresh `last_seen_at`), not
/// re-authoring. Returns `(edge_id, created)`.
#[allow(clippy::too_many_arguments)]
fn insert_concept_edge(
    conn: &rusqlite::Connection,
    from_id: i64,
    relation: &str,
    to_id: i64,
    confidence: f32,
    source: EdgeSource,
    by: Option<&str>,
    evidence_json: Option<&str>,
) -> Result<(i64, bool)> {
    if from_id == to_id {
        return Err(StoreError::InvalidEnumValue {
            field: "concept_edge".into(),
            value: "self-loop not allowed".into(),
        });
    }
    let now = Utc::now();
    let existed = conn
        .query_row(
            "SELECT 1 FROM concept_edges
             WHERE from_concept = ? AND relation = ? AND to_concept = ?",
            params![from_id, relation, to_id],
            |_| Ok(()),
        )
        .optional()?
        .is_some();
    let id: i64 = conn.query_row(
        "INSERT INTO concept_edges
             (from_concept, relation, to_concept, confidence, evidence_json,
              first_seen_at, last_seen_at, touch_count, source, [by])
         VALUES (?, ?, ?, ?, ?, ?, ?, 1, ?, ?)
         ON CONFLICT(from_concept, relation, to_concept) DO UPDATE SET
             last_seen_at  = excluded.last_seen_at,
             touch_count   = touch_count + 1,
             confidence    = MAX(confidence, excluded.confidence),
             evidence_json = COALESCE(excluded.evidence_json, evidence_json)
         RETURNING id",
        params![
            from_id,
            relation,
            to_id,
            confidence,
            evidence_json,
            now.to_rfc3339(),
            now.to_rfc3339(),
            source.as_str(),
            by,
        ],
        |r| r.get(0),
    )?;
    Ok((id, !existed))
}

fn get_concept_exact(
    conn: &rusqlite::Connection,
    project: &str,
    handle: &str,
) -> Result<Option<ConceptRecord>> {
    let row = conn
        .query_row(
            "SELECT id, project, handle, summary, status, confidence, merged_into,
                    created_at, updated_at, kind
             FROM concepts WHERE project = ? AND handle = ?",
            params![project, handle],
            row_to_concept,
        )
        .optional()?;
    Ok(row)
}

/// One resolution step: match `name` as a concept handle first, then as an
/// inbound alias, preferring the exact `project` over the global (`""`)
/// row. Does NOT follow merges — the caller's loop does that.
fn resolve_one(
    conn: &rusqlite::Connection,
    project: &str,
    name: &str,
) -> Result<Option<ConceptRecord>> {
    let by_handle = conn
        .query_row(
            "SELECT id, project, handle, summary, status, confidence, merged_into,
                    created_at, updated_at, kind
             FROM concepts
             WHERE handle = ? AND (project = ? OR project = '')
             ORDER BY (project = ?) DESC, id ASC LIMIT 1",
            params![name, project, project],
            row_to_concept,
        )
        .optional()?;
    if by_handle.is_some() {
        return Ok(by_handle);
    }
    let by_alias = conn
        .query_row(
            "SELECT c.id, c.project, c.handle, c.summary, c.status, c.confidence, c.merged_into,
                    c.created_at, c.updated_at, c.kind
             FROM concept_aliases a JOIN concepts c ON c.id = a.concept_id
             WHERE a.alias = ? AND (c.project = ? OR c.project = '')
             ORDER BY (c.project = ?) DESC, c.id ASC LIMIT 1",
            params![name, project, project],
            row_to_concept,
        )
        .optional()?;
    Ok(by_alias)
}

/// Resolve a handle to its concept id within `project` (global fallback).
fn resolve_id_opt(conn: &rusqlite::Connection, project: &str, name: &str) -> Result<Option<i64>> {
    // Same alias + merge-forward resolution as `resolve_concept`, returning
    // just the id — so edges connect to the canonical, alias-resolved concept.
    let mut current = name.to_string();
    let mut seen = std::collections::HashSet::new();
    loop {
        if !seen.insert(current.clone()) {
            return Ok(None);
        }
        match resolve_one(conn, project, &current)? {
            Some(rec) if rec.status == ConceptStatus::Merged => match rec.merged_into.clone() {
                Some(canonical) => current = canonical,
                None => return Ok(Some(rec.id)),
            },
            Some(rec) => return Ok(Some(rec.id)),
            None => return Ok(None),
        }
    }
}

fn resolve_id(conn: &rusqlite::Connection, project: &str, handle: &str) -> Result<i64> {
    resolve_id_opt(conn, project, handle)?.ok_or_else(|| not_found(project, handle))
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn db() -> (TempDir, ThreadsDb) {
        let tmp = TempDir::new().unwrap();
        let db = ThreadsDb::open(tmp.path()).unwrap();
        (tmp, db)
    }

    const G: &str = GLOBAL_PROJECT;

    fn attach(db: &ThreadsDb, id: i64, source_id: &str, chunk: &str) {
        db.attach_concept_evidence(&EvidenceAttach {
            concept_id: id,
            project: G,
            source: "code",
            source_id,
            chunk_id: Some(chunk),
            content_sha256: Some("sha-1"),
            anchor_vec: Some(&[1.0, 0.0, 0.0]),
            score: Some(0.8),
            reason: Some("test"),
        })
        .unwrap();
    }

    #[test]
    fn kind_column_present_after_open() {
        let (_t, db) = db();
        // ensure_typed_concept round-trips the kind through the new column.
        let (rec, created) = db
            .ensure_typed_concept(G, "tori", ConceptStatus::Proposed, Some("person"))
            .unwrap();
        assert!(created);
        assert_eq!(rec.kind.as_deref(), Some("person"));
        assert_eq!(rec.status, ConceptStatus::Proposed);
    }

    #[test]
    fn ensure_typed_concept_is_idempotent() {
        let (_t, db) = db();
        db.ensure_typed_concept(G, "tori", ConceptStatus::Proposed, Some("person"))
            .unwrap();
        let (rec, created) = db
            .ensure_typed_concept(G, "tori", ConceptStatus::Proposed, Some("person"))
            .unwrap();
        assert!(!created, "second ensure does not re-create");
        assert_eq!(rec.kind.as_deref(), Some("person"));
        assert_eq!(db.list_concepts(Some(G), None).unwrap().len(), 1);
    }

    #[test]
    fn ensure_typed_concept_backfills_null_kind() {
        let (_t, db) = db();
        // First observed untyped (kind NULL)...
        db.ensure_concept(G, "tori", ConceptStatus::Candidate)
            .unwrap();
        // ...then a file-seed assigns the kind via COALESCE.
        let (rec, created) = db
            .ensure_typed_concept(G, "tori", ConceptStatus::Proposed, Some("person"))
            .unwrap();
        assert!(!created, "already existed");
        assert_eq!(rec.kind.as_deref(), Some("person"), "NULL kind backfilled");
        // Status not downgraded/changed by the typed ensure.
        assert_eq!(rec.status, ConceptStatus::Candidate);
    }

    #[test]
    fn ensure_typed_concept_preserves_existing_kind() {
        let (_t, db) = db();
        db.ensure_typed_concept(G, "tori", ConceptStatus::Proposed, Some("person"))
            .unwrap();
        // A later seed with a different kind must NOT overwrite (COALESCE keeps
        // the non-null original).
        let (rec, _) = db
            .ensure_typed_concept(G, "tori", ConceptStatus::Proposed, Some("meeting"))
            .unwrap();
        assert_eq!(rec.kind.as_deref(), Some("person"), "first kind wins");
    }

    #[test]
    fn ensure_is_idempotent_and_never_downgrades() {
        let (_t, db) = db();
        let (rec, created) = db
            .ensure_concept(G, "mish", ConceptStatus::Candidate)
            .unwrap();
        assert!(created);
        assert_eq!(rec.project, "");
        db.set_concept_status(G, "mish", ConceptStatus::Active, Some(1.0))
            .unwrap();
        let (rec2, created2) = db
            .ensure_concept(G, "mish", ConceptStatus::Candidate)
            .unwrap();
        assert!(!created2);
        assert_eq!(rec2.status, ConceptStatus::Active);
    }

    #[test]
    fn project_scope_isolates_then_falls_back_to_global() {
        let (_t, db) = db();
        db.ensure_concept(G, "client", ConceptStatus::Active)
            .unwrap();
        db.ensure_concept("acme", "client", ConceptStatus::Candidate)
            .unwrap();
        // Distinct rows: same handle, different project.
        assert_eq!(db.list_concepts(None, None).unwrap().len(), 2);
        // Exact scope resolves the project-local concept.
        assert_eq!(
            db.resolve_concept("acme", "client")
                .unwrap()
                .unwrap()
                .project,
            "acme"
        );
        // A project with no local concept falls back to the global row.
        assert_eq!(
            db.resolve_concept("other", "client")
                .unwrap()
                .unwrap()
                .project,
            ""
        );
    }

    #[test]
    fn evidence_keys_on_coordinate_and_retouches() {
        let (_t, db) = db();
        let (rec, _) = db
            .ensure_concept(G, "mish", ConceptStatus::Candidate)
            .unwrap();
        attach(&db, rec.id, "~/projects/mish", "chunk-a");
        // Same coordinate, new chunk id (a re-chunk) → re-touch, not a dup row.
        attach(&db, rec.id, "~/projects/mish", "chunk-b");
        let ev = db.list_concept_evidence(rec.id).unwrap();
        assert_eq!(ev.len(), 1, "coordinate is the key, not chunk_id");
        assert_eq!(ev[0].last_resolved_chunk_id.as_deref(), Some("chunk-b"));
        assert_eq!(ev[0].touch_count, 2);
        assert_eq!(ev[0].relation_state, EvidenceState::Active);
    }

    /// Seed the two endpoints of the north-star self-referential edge.
    fn seed_north_star(db: &ThreadsDb) {
        for h in ["ostk-recall", "ostk"] {
            db.ensure_concept(G, h, ConceptStatus::Active).unwrap();
        }
    }

    #[test]
    fn add_concept_edge_by_id_reports_created_then_retouch() {
        let (_t, db) = db();
        let (a, _) = db
            .ensure_concept(G, "a-node", ConceptStatus::Active)
            .unwrap();
        let (b, _) = db
            .ensure_concept(G, "b-node", ConceptStatus::Active)
            .unwrap();
        let (id1, created) = db
            .add_concept_edge_by_id(
                a.id,
                "rel",
                b.id,
                0.1,
                EdgeSource::Authored,
                Some("scanner"),
                None,
            )
            .unwrap();
        assert!(created, "first insert is created");
        let (id2, created2) = db
            .add_concept_edge_by_id(a.id, "rel", b.id, 0.1, EdgeSource::Observed, None, None)
            .unwrap();
        assert_eq!(id1, id2, "same edge row");
        assert!(!created2, "re-touch is not a creation");
        // Self-loop by id is rejected.
        assert!(
            db.add_concept_edge_by_id(a.id, "rel", a.id, 0.1, EdgeSource::Authored, None, None)
                .is_err()
        );
    }

    #[test]
    fn authored_edge_enters_low_and_carries_provenance() {
        let (_t, db) = db();
        seed_north_star(&db);
        db.add_concept_edge(
            G,
            "ostk-recall",
            "memory_layer_for",
            "ostk",
            AUTHORED_EDGE_CONFIDENCE,
            EdgeSource::Authored,
            Some("claude"),
            Some("ostk-recall is the virtual memory layer for ostk"),
        )
        .unwrap();
        let edges = db
            .list_concept_edges(G, "ostk-recall", EdgeDirection::From)
            .unwrap();
        assert_eq!(
            edges.len(),
            1,
            "the north-star edge is present and readable"
        );
        let e = &edges[0];
        assert_eq!(e.to_handle, "ostk");
        assert_eq!(e.relation, "memory_layer_for");
        assert!(
            (e.confidence - AUTHORED_EDGE_CONFIDENCE).abs() < 1e-6,
            "authored edge enters at the low prior, not full strength"
        );
        assert_eq!(e.source, EdgeSource::Authored);
        assert_eq!(e.by.as_deref(), Some("claude"));
        assert_eq!(e.touch_count, 1);
    }

    #[test]
    fn reobservation_is_use_and_preserves_origin() {
        let (_t, db) = db();
        seed_north_star(&db);
        let add = |source, by| {
            db.add_concept_edge(
                G,
                "ostk-recall",
                "memory_layer_for",
                "ostk",
                AUTHORED_EDGE_CONFIDENCE,
                source,
                by,
                None,
            )
            .unwrap()
        };
        add(EdgeSource::Authored, Some("claude"));
        // Re-observed by another mind: use, not re-authoring. Origin is immutable.
        add(EdgeSource::Observed, Some("codex"));
        let e = db
            .list_concept_edges(G, "ostk-recall", EdgeDirection::From)
            .unwrap()
            .remove(0);
        assert_eq!(e.touch_count, 2, "re-observation bumps use");
        assert_eq!(
            e.source,
            EdgeSource::Authored,
            "first author wins; an edge's origin never changes"
        );
        assert_eq!(
            e.by.as_deref(),
            Some("claude"),
            "the original author is preserved across re-observation"
        );
    }

    #[test]
    fn neighbors_pairs_edges_with_low_conductance() {
        let (_t, db) = db();
        seed_north_star(&db);
        db.add_concept_edge(
            G,
            "ostk-recall",
            "memory_layer_for",
            "ostk",
            AUTHORED_EDGE_CONFIDENCE,
            EdgeSource::Authored,
            None,
            None,
        )
        .unwrap();
        let ns = db
            .neighbors(G, "ostk-recall", EdgeDirection::From, Utc::now())
            .unwrap();
        assert_eq!(ns.len(), 1);
        let (edge, conductance) = &ns[0];
        assert_eq!(edge.to_handle, "ostk");
        // Freshly authored: recency ≈ 1, so conductance ≈ confidence, and low.
        assert!(
            *conductance > 0.0 && *conductance <= AUTHORED_EDGE_CONFIDENCE + 1e-4,
            "authored edge enters at low conductance, got {conductance}"
        );
    }

    #[test]
    fn merge_rewrites_edges_and_drops_self_loops() {
        let (_t, db) = db();
        for h in ["mish", "mish-shell", "ostk", "slipstream"] {
            db.ensure_concept(G, h, ConceptStatus::Active).unwrap();
        }
        // dup (mish-shell) has an edge to ostk; canonical (mish) also → ostk.
        db.add_concept_edge(
            G,
            "mish-shell",
            "absorbed_into",
            "ostk",
            0.5,
            EdgeSource::Observed,
            None,
            None,
        )
        .unwrap();
        db.add_concept_edge(
            G,
            "mish",
            "absorbed_into",
            "ostk",
            0.6,
            EdgeSource::Observed,
            None,
            None,
        )
        .unwrap();
        // dup also pairs_with mish itself → would become a self-loop on merge.
        db.add_concept_edge(
            G,
            "mish-shell",
            "pairs_with",
            "mish",
            0.3,
            EdgeSource::Observed,
            None,
            None,
        )
        .unwrap();
        // and a unique edge that should survive the rewrite.
        db.add_concept_edge(
            G,
            "mish-shell",
            "pairs_with",
            "slipstream",
            0.4,
            EdgeSource::Observed,
            None,
            None,
        )
        .unwrap();

        db.merge_concept(G, "mish-shell", "mish").unwrap();

        let merged = db.get_concept(G, "mish-shell").unwrap().unwrap();
        assert_eq!(merged.status, ConceptStatus::Merged);
        assert_eq!(merged.merged_into.as_deref(), Some("mish"));

        // The merged handle resolves FORWARD: querying it returns the
        // canonical's edges, and none reference the tombstone (all rewritten).
        let from = db
            .list_concept_edges(G, "mish-shell", EdgeDirection::From)
            .unwrap();
        assert!(
            !from.is_empty(),
            "merged handle forwards to canonical edges"
        );
        assert!(
            from.iter().all(|e| e.from_handle == "mish"),
            "no edge references the tombstone"
        );

        // Canonical: absorbed_into→ostk (deduped to one), pairs_with→slipstream
        // survived, and the would-be self-loop (mish pairs_with mish) is gone.
        let mish_from = db
            .list_concept_edges(G, "mish", EdgeDirection::From)
            .unwrap();
        assert!(
            mish_from
                .iter()
                .any(|e| e.relation == "absorbed_into" && e.to_handle == "ostk")
        );
        assert!(
            mish_from
                .iter()
                .any(|e| e.relation == "pairs_with" && e.to_handle == "slipstream")
        );
        assert!(
            !mish_from.iter().any(|e| e.to_handle == "mish"),
            "self-loop must be pruned"
        );
        assert_eq!(
            mish_from
                .iter()
                .filter(|e| e.relation == "absorbed_into")
                .count(),
            1,
            "duplicate edge deduped"
        );
    }

    #[test]
    fn resolve_follows_aliases_and_merges() {
        let (_t, db) = db();
        let (slip, _) = db
            .ensure_concept(G, "slipstream", ConceptStatus::Active)
            .unwrap();
        db.touch_alias(slip.id, "ss", AliasSource::User).unwrap();
        // Inbound alias resolves to the concept (ss → slipstream).
        assert_eq!(
            db.resolve_concept(G, "ss").unwrap().unwrap().handle,
            "slipstream"
        );
        // Merge-forward: a merged handle resolves to its canonical, not the
        // tombstone.
        db.ensure_concept(G, "slip", ConceptStatus::Candidate)
            .unwrap();
        db.merge_concept(G, "slip", "slipstream").unwrap();
        let r = db.resolve_concept(G, "slip").unwrap().unwrap();
        assert_eq!(r.handle, "slipstream");
        assert_eq!(r.status, ConceptStatus::Active);
    }

    #[test]
    fn list_scope_includes_globals() {
        let (_t, db) = db();
        db.ensure_concept(G, "mish", ConceptStatus::Active).unwrap();
        db.ensure_concept("acme", "auth", ConceptStatus::Proposed)
            .unwrap();
        // Global scope sees only globals; acme sees acme + globals; None = all.
        assert_eq!(db.list_concepts(Some(G), None).unwrap().len(), 1);
        assert_eq!(db.list_concepts(Some("acme"), None).unwrap().len(), 2);
        assert_eq!(db.list_concepts(None, None).unwrap().len(), 2);
    }

    #[test]
    fn evidence_triple_stays_coherent_on_retouch() {
        let (_t, db) = db();
        let (rec, _) = db
            .ensure_concept(G, "mish", ConceptStatus::Candidate)
            .unwrap();
        // Corpus-backed attach: pointer + sha + anchor describe chunk-a.
        attach(&db, rec.id, "~/projects/mish", "chunk-a");
        // A no-corpus re-touch (anchor None) must NOT advance the pointer past
        // the anchor — the triple stays coherent on chunk-a.
        db.attach_concept_evidence(&EvidenceAttach {
            concept_id: rec.id,
            project: G,
            source: "code",
            source_id: "~/projects/mish",
            chunk_id: Some("chunk-b"),
            content_sha256: None,
            anchor_vec: None,
            score: Some(0.9),
            reason: None,
        })
        .unwrap();
        let ev = db.list_concept_evidence(rec.id).unwrap();
        assert_eq!(ev[0].last_resolved_chunk_id.as_deref(), Some("chunk-a"));
        assert_eq!(ev[0].content_sha256.as_deref(), Some("sha-1"));
        assert_eq!(ev[0].touch_count, 2);
    }

    #[test]
    fn notes_are_durable_rows() {
        let (_t, db) = db();
        let (rec, _) = db.ensure_concept(G, "ostk", ConceptStatus::Active).unwrap();
        db.add_concept_note(rec.id, "decision", "absorbed mish + slipstream")
            .unwrap();
        db.add_concept_note(rec.id, "fact", "one binary, no IPC")
            .unwrap();
        let notes = db.list_concept_notes(rec.id).unwrap();
        assert_eq!(notes.len(), 2);
        assert!(notes.iter().any(|n| n.kind == "decision"));
    }

    #[test]
    fn extract_filters_structural_basenames() {
        let no_syms: Vec<String> = vec![];
        let hits = vec![
            HitView {
                source_id: "crates/store/src/lib.rs",
                symbols: &no_syms,
            },
            HitView {
                source_id: "~/projects/mish",
                symbols: &no_syms,
            },
            HitView {
                source_id: "crates/x/tests/mod.rs",
                symbols: &no_syms,
            },
        ];
        let terms = extract_concept_terms("mish", &hits);
        let handles: Vec<&str> = terms.iter().map(|t| t.handle.as_str()).collect();
        assert!(handles.contains(&"mish"));
        assert!(!handles.contains(&"lib"), "structural basename filtered");
        assert!(!handles.contains(&"mod"), "structural basename filtered");
    }

    #[test]
    fn migrates_legacy_shape_preserving_rows() {
        use rusqlite::Connection;
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("threads.sqlite");
        // Hand-build the pre-hardening shape: global-unique handle + a
        // chunk_id-keyed evidence table, with one concept + evidence row.
        {
            let conn = Connection::open(&path).unwrap();
            conn.execute_batch(
                "CREATE TABLE concepts (
                     id INTEGER PRIMARY KEY AUTOINCREMENT, handle TEXT NOT NULL UNIQUE,
                     summary TEXT, status TEXT NOT NULL DEFAULT 'candidate',
                     confidence REAL NOT NULL DEFAULT 0.0, merged_into TEXT,
                     created_at TEXT NOT NULL, updated_at TEXT NOT NULL);
                 CREATE TABLE concept_evidence (
                     concept_id INTEGER NOT NULL REFERENCES concepts(id) ON DELETE CASCADE,
                     chunk_id TEXT NOT NULL, score REAL, reason TEXT,
                     first_seen_at TEXT NOT NULL, last_seen_at TEXT NOT NULL,
                     touch_count INTEGER NOT NULL DEFAULT 1, UNIQUE(concept_id, chunk_id));
                 INSERT INTO concepts (handle, status, confidence, created_at, updated_at)
                     VALUES ('mish','active',1.0,'2026-06-01T00:00:00+00:00','2026-06-01T00:00:00+00:00');
                 INSERT INTO concept_evidence
                     (concept_id, chunk_id, score, reason, first_seen_at, last_seen_at, touch_count)
                     VALUES (1,'chunk-legacy',0.8,'old','2026-06-01T00:00:00+00:00','2026-06-01T00:00:00+00:00',3);",
            )
            .unwrap();
        }
        // Open through ThreadsDb → runs the FK-safe rebuild migration.
        let db = ThreadsDb::open(tmp.path()).unwrap();

        // Concept preserved, now global-scoped (project='').
        let rec = db.get_concept(GLOBAL_PROJECT, "mish").unwrap().unwrap();
        assert_eq!(rec.status, ConceptStatus::Active);
        assert_eq!(rec.project, "");
        // The new (project, handle) UNIQUE is live: a project-scoped `mish`
        // can coexist with the migrated global one.
        db.ensure_concept("acme", "mish", ConceptStatus::Candidate)
            .unwrap();
        assert_eq!(db.list_concepts(None, None).unwrap().len(), 2);

        // Evidence preserved: chunk_id became the resolvable pointer, the
        // coordinate is the empty backfill sentinel, counts intact.
        let ev = db.list_concept_evidence(rec.id).unwrap();
        assert_eq!(ev.len(), 1);
        assert_eq!(
            ev[0].last_resolved_chunk_id.as_deref(),
            Some("chunk-legacy")
        );
        assert_eq!(ev[0].touch_count, 3);
        assert!(
            ev[0].source.is_empty(),
            "legacy coordinate awaits reconciler backfill"
        );
        assert_eq!(ev[0].relation_state, EvidenceState::Active);
    }

    #[test]
    fn orphan_marking_gates_not_deletes() {
        let (_t, db) = db();
        let (rec, _) = db
            .ensure_concept(G, "mish", ConceptStatus::Candidate)
            .unwrap();
        attach(&db, rec.id, "~/projects/mish", "chunk-a");
        let rows = db.evidence_to_reconcile().unwrap();
        assert_eq!(rows.len(), 1);
        db.mark_evidence_orphaned(rows[0].rowid).unwrap();
        // Still present (gated), just not active, and excluded from reconcile.
        let ev = db.list_concept_evidence(rec.id).unwrap();
        assert_eq!(ev.len(), 1);
        assert_eq!(ev[0].relation_state, EvidenceState::Orphaned);
        assert!(db.evidence_to_reconcile().unwrap().is_empty());
    }
}
