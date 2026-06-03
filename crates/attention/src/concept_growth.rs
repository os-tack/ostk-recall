//! Phase 2 — ambient salience-gated concept-growth (building blocks).
//!
//! Pure, testable pieces the [`crate::observer::TurnObserver`] composes into its
//! concept-growth phase. The substrate already extracts the right per-turn
//! signal (resonance-gated familiarity) but only ever fed the *thread* graph;
//! this module is the half that lets a turn grow the **concept** graph:
//!
//! - [`ConceptGrowthConfig`] — runtime mirror of `core`'s `AmbientGrowthConfig`
//!   (`core` can't depend on `attention`; a CLI guard test keeps them in step).
//! - [`AmbientGazetteer`] — an aho-corasick matcher over the known concept
//!   surface forms, mirroring the scanner-side gazetteer in `cli::seed` but with
//!   no "self" node to exclude (the observer is not a graph node).
//! - [`ConceptGrowthCache`] — the anchor codebook (a concept's anchor embedding
//!   is the latent half of the resonance gate, `cosine(anchor, turn_rolling_vec)`).
//! - [`TermRecurrence`] — the streaming mentions-vs-resonance split for the
//!   node-minting half (a term must recur across *resonant* turns to mint).
//! - [`select_survivors`] — gazetteer matches → resonance-gated, top-K survivors.

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, AtomicUsize};

use aho_corasick::{AhoCorasick, MatchKind};
use ostk_recall_store::{ConceptStatus, ThreadsDb};
use tokio::sync::RwLock;

/// Relation minted for an ambient co-occurrence. Distinct from `mentions`
/// (directional doc→entity) and `related` (reserved by the latent promoter) so
/// the provenance of an observer-grown peer link stays legible. Diffusion
/// (`neighbors_by_id`) filters on status, not relation, so it participates
/// immediately.
pub(crate) const CO_MENTION_RELATION: &str = "co_occurs";

/// `by=` provenance stamped on every ambient-grown edge.
pub(crate) const AMBIENT_EDGE_BY: &str = "observer";

/// Minimum normalized chars for a gazetteer surface form (mirrors
/// `cli::seed::MIN_MENTION_CHARS`) — drops `id`/`ok`-class noise.
const MIN_GAZ_FORM_CHARS: usize = 3;

/// Runtime mirror of `ostk_recall_core::config::AmbientGrowthConfig`.
///
/// `core` cannot depend on `attention`, so `serve` maps the `[ambient_growth]`
/// block onto this at startup; a guard test in the CLI crate keeps the defaults
/// in lock-step (the same split as `LensSettings` ↔ `LensConfig`).
#[derive(Debug, Clone, Copy)]
pub struct ConceptGrowthConfig {
    /// Cosine a named concept's anchor must clear against the turn rolling
    /// vector to be a salient (not merely mentioned) survivor.
    pub resonance_floor: f32,
    /// Max co-resonant survivors kept per turn — edges minted ≤ `C(edge_top_k, 2)`.
    pub edge_top_k: usize,
    /// Minimum survivors a turn must produce to mint any edge; also the bar that
    /// makes a turn "resonant" for the node-recurrence accumulator.
    pub min_survivors: usize,
    /// Resonant turns an unknown term must recur across before it mints a node.
    pub node_mint_min_resonant_turns: u32,
    /// Rebuild the anchor codebook + gazetteer every N observed turns.
    pub codebook_rebuild_turns: u64,
    /// Per-session cap on minted nodes (runaway-spam guard).
    pub node_mint_cap_per_session: usize,
}

impl Default for ConceptGrowthConfig {
    fn default() -> Self {
        // MUST match `AmbientGrowthConfig::default()` in `ostk-recall-core`.
        Self {
            resonance_floor: 0.35,
            edge_top_k: 4,
            min_survivors: 2,
            node_mint_min_resonant_turns: 3,
            codebook_rebuild_turns: 32,
            node_mint_cap_per_session: 8,
        }
    }
}

/// The observer's concept-growth read cache: anchor codebook + per-project
/// gazetteers.
///
/// `anchors` maps `concept_id` → anchor-chunk embedding (the latent half of the
/// resonance gate) and is **project-agnostic** (`concept_anchors` spans all
/// projects), rebuilt on a turn-count cadence; `anchors_built` distinguishes
/// "never built" from "built but empty graph".
///
/// `gazetteers` is the lexical half and **is** scope-sensitive: keyed by
/// `scope.project` (project + globals, or all-projects when `None`). A single
/// observer instance streams turns from different `event.project`s
/// (`run_subscribed`), so reusing one project's matcher/`known` oracle for
/// another would miss the active project's concepts and re-mint existing ones.
/// Built lazily per project and cleared whenever the anchor codebook rebuilds,
/// so each refreshes on the same cadence.
#[derive(Default)]
pub(crate) struct ConceptGrowthCache {
    pub anchors: HashMap<i64, Vec<f32>>,
    pub anchors_built: bool,
    pub gazetteers: HashMap<Option<String>, AmbientGazetteer>,
}

/// Streaming recurrence accumulator for one unknown term — the mentions-vs-
/// resonance split mirrored from the thread path, but across *turns*. `mentions`
/// advances every turn the term appears; `resonance` advances only on turns that
/// were salient (≥ `min_survivors` co-resonant existing concepts). A term mints
/// a node only on `resonance`, never raw `mentions` — the occurrence≠salience law.
#[derive(Debug, Clone, Default)]
pub(crate) struct TermRecurrence {
    pub mentions: u32,
    pub resonance: u32,
    /// Co-resonant existing concepts (id→handle) gathered across the resonant
    /// turns that grew this term — the mint-time connection targets.
    pub co_resonant: BTreeMap<i64, String>,
}

/// The cross-turn concept-growth state a `TurnObserver` accumulates: the
/// anchor/gazetteer cache, its rebuild-cadence counter, and the streaming
/// node-recurrence accumulator.
///
/// **Why this is separable.** In `serve`, each watch trigger spawns a *fresh*
/// `TurnObserver` (the daemon is scan-scoped for drain ordering), so this
/// in-memory state would reset every trigger — and a term recurring across
/// *separate* live turns would never reach the recurrence gate (the node-minting
/// half would be inert). Hoisting it into `ServeContext` and sharing it across
/// triggers via [`crate::TurnObserver::with_concept_growth_runtime`] makes
/// recurrence, the codebook cache, AND the per-session node-mint cap persist for
/// the daemon's life — so `node_mint_cap_per_session` is genuinely per serve
/// session, not per scan trigger. Cheap to clone — all `Arc`.
#[derive(Clone, Default)]
pub struct ConceptGrowthRuntime {
    pub(crate) growth_cache: Arc<RwLock<ConceptGrowthCache>>,
    pub(crate) turns_since_build: Arc<AtomicU64>,
    pub(crate) term_recurrence: Arc<RwLock<HashMap<String, TermRecurrence>>>,
    pub(crate) node_mints_this_session: Arc<AtomicUsize>,
}

// --- gazetteer (mirrors cli::seed, minus the self-node exclusion) -----

#[derive(Default)]
struct FormClaims {
    project_handle: BTreeSet<i64>,
    global_handle: BTreeSet<i64>,
    project_alias: BTreeSet<i64>,
    global_alias: BTreeSet<i64>,
}

enum Pick {
    One(i64),
    Ambiguous,
    None,
}

impl FormClaims {
    /// First non-empty tier, in `resolve_one` precedence order; `Ambiguous` when
    /// that tier names more than one distinct concept (dropped, not guessed).
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

/// A matcher over the scope-visible concept surface forms (handles + aliases).
///
/// Match targets are TRUST-gated to `Active`/`Proposed` concepts (a `Candidate`
/// is too low-trust to shape topology). The `known` oracle is broader —
/// normalized forms of *every* concept incl. `Candidate`/terminal/ambiguous —
/// so node-minting never re-materializes an existing name regardless of status.
/// `handles` maps id→handle for audit events. Empty matches nothing.
#[derive(Default)]
pub(crate) struct AmbientGazetteer {
    ac: Option<AhoCorasick>,
    /// `ids[pattern] = concept id`.
    ids: Vec<i64>,
    /// Canonical handle per concept id (for `ConceptConnected` events).
    pub handles: HashMap<i64, String>,
    /// Normalized handle + alias forms of every concept — the node-mint oracle.
    pub known: HashSet<String>,
}

impl AmbientGazetteer {
    /// Distinct concept ids named in `body_norm` (already normalized). Word
    /// boundaries are enforced against the space-delimited normalized text, so
    /// `ostk` never matches inside `ostkx`. Unlike the scanner gazetteer there
    /// is no self-node to exclude — the observer is not a graph node.
    pub fn matches(&self, body_norm: &str) -> Vec<i64> {
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
                found.insert(self.ids[m.pattern().as_usize()]);
            }
        }
        found.into_iter().collect()
    }
}

/// Lowercase, replace every non-alphanumeric char with a space, collapse runs.
/// Applied identically to gazetteer forms and turn text so hyphens and markdown
/// fold away and `ostk-recall` ⇒ `ostk recall` matches hyphenated and spaced prose.
/// Mirrors `cli::seed::normalize`.
pub(crate) fn normalize(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut prev_space = true; // leading run trimmed
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
    (norm.chars().count() >= MIN_GAZ_FORM_CHARS).then_some(norm)
}

/// Build the gazetteer over concepts visible from `project_filter`
/// (`None` = all projects; `Some("")` = globals only; `Some("p")` = p + globals
/// — the `list_concepts` contract). Only `Active`/`Proposed` concepts become
/// match targets (trust gate); `Candidate` and terminal concepts still populate
/// `known` so their names are never re-minted. Best-effort: a ledger error
/// yields an empty (matches-nothing) gazetteer.
pub(crate) fn build_ambient_gazetteer(
    threads: &ThreadsDb,
    project_filter: Option<&str>,
) -> AmbientGazetteer {
    let concepts = threads
        .list_concepts(project_filter, None)
        .unwrap_or_default();
    let mut claims: HashMap<String, FormClaims> = HashMap::new();
    let mut handles: HashMap<i64, String> = HashMap::new();
    let mut known: HashSet<String> = HashSet::new();
    for c in &concepts {
        let aliases = threads.list_aliases(c.id).unwrap_or_default();
        // Capture handle + alias forms of EVERY concept (incl. Candidate and
        // terminal) into `known` BEFORE the match-target gate, so the node-mint
        // oracle never re-materializes an existing name regardless of status.
        if let Some(norm) = gaz_form(&c.handle) {
            known.insert(norm);
        }
        for a in &aliases {
            if let Some(norm) = gaz_form(&a.alias) {
                known.insert(norm);
            }
        }
        // Match targets are TRUST-gated: only `Active`/`Proposed` concepts shape
        // topology. A `Candidate` is "known enough to avoid duplicate minting,
        // not trusted enough to earn `co_occurs` edges" — otherwise recurring
        // operational lexis (`file`, `signal`, `serve`) would re-touch low-trust
        // candidate noise into durable structure. (Terminal is excluded too.)
        // Future calibration: a repeatedly-salient Candidate could be *upgraded*
        // to Proposed rather than ignored — deliberately out of scope here.
        if !matches!(c.status, ConceptStatus::Active | ConceptStatus::Proposed) {
            continue;
        }
        handles.insert(c.id, c.handle.clone());
        let project_scoped = !c.project.is_empty();
        if let Some(norm) = gaz_form(&c.handle) {
            let entry = claims.entry(norm).or_default();
            if project_scoped {
                entry.project_handle.insert(c.id);
            } else {
                entry.global_handle.insert(c.id);
            }
        }
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
    for (form, fc) in claims {
        if let Pick::One(id) = fc.pick() {
            ids.push(id);
            patterns.push(form);
        }
        // Ambiguous / None: dropped, not guessed.
    }

    let ac = if patterns.is_empty() {
        None
    } else {
        AhoCorasick::builder()
            .match_kind(MatchKind::LeftmostLongest)
            .build(&patterns)
            .ok()
    };
    AmbientGazetteer {
        ac,
        ids,
        handles,
        known,
    }
}

/// Resonance-gate and top-K the gazetteer-matched concept ids against the turn
/// rolling vector. Returns `(id, resonance)` survivors sorted by resonance
/// descending (id tie-break), capped at `edge_top_k`. Ids with no codebook
/// anchor are dropped — they cannot be salience-gated. Pure.
pub(crate) fn select_survivors(
    matched_ids: &[i64],
    anchors: &HashMap<i64, Vec<f32>>,
    rolling_vec: &[f32],
    floor: f32,
    top_k: usize,
) -> Vec<(i64, f32)> {
    let mut survivors: Vec<(i64, f32)> = matched_ids
        .iter()
        .filter_map(|id| {
            let sem = anchors.get(id)?;
            let res = crate::cosine_similarity(sem, rolling_vec);
            (res >= floor).then_some((*id, res))
        })
        .collect();
    survivors.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.0.cmp(&b.0))
    });
    survivors.truncate(top_k);
    survivors
}

#[cfg(test)]
#[allow(clippy::float_cmp)]
mod tests {
    use super::*;

    use std::sync::Arc;

    use ostk_recall_store::{ConceptStatus, NoopChainSink, ThreadsDb};
    use tempfile::TempDir;

    const G: &str = "";

    fn db() -> (TempDir, ThreadsDb) {
        let tmp = TempDir::new().unwrap();
        let db = ThreadsDb::open_with_sink(tmp.path(), Arc::new(NoopChainSink)).unwrap();
        (tmp, db)
    }

    #[test]
    fn normalize_folds_hyphens_markdown_and_case() {
        assert_eq!(normalize("**ostk-recall**"), "ostk recall");
        assert_eq!(normalize("  Hello,  World! "), "hello world");
        assert_eq!(normalize("a_b-c"), "a b c");
    }

    #[test]
    fn select_survivors_gates_on_floor_and_drops_unanchored() {
        let mut anchors: HashMap<i64, Vec<f32>> = HashMap::new();
        anchors.insert(1, vec![1.0, 0.0, 0.0]); // cosine 1.0
        anchors.insert(2, vec![1.0, 0.0, 0.0]); // cosine 1.0
        anchors.insert(3, vec![0.0, 1.0, 0.0]); // cosine 0.0 (orthogonal)
        // id 4 has no anchor entry → dropped (cannot be salience-gated).
        let rolling = vec![1.0, 0.0, 0.0];
        let got = select_survivors(&[1, 2, 3, 4], &anchors, &rolling, 0.5, 4);
        let ids: Vec<i64> = got.iter().map(|(id, _)| *id).collect();
        assert_eq!(ids, vec![1, 2]);
    }

    #[test]
    fn select_survivors_keeps_top_k_by_resonance() {
        let mut anchors: HashMap<i64, Vec<f32>> = HashMap::new();
        anchors.insert(1, vec![1.0, 0.0]); // cosine 1.0
        anchors.insert(2, vec![0.9, 0.1]); // a bit less
        anchors.insert(3, vec![0.8, 0.2]); // less still
        let got = select_survivors(&[1, 2, 3], &anchors, &[1.0, 0.0], 0.3, 2);
        assert_eq!(got.len(), 2, "capped at top_k");
        assert_eq!(got[0].0, 1, "highest resonance first");
        assert_eq!(got[1].0, 2);
    }

    #[test]
    fn gazetteer_resolves_scope_visible_concepts() {
        let (_t, db) = db();
        db.ensure_concept(G, "ostk-recall", ConceptStatus::Active)
            .unwrap();
        db.ensure_concept("haystack", "haystack-doc", ConceptStatus::Active)
            .unwrap();
        let body = normalize("today i touched ostk-recall and haystack-doc");

        // Scope carries the project → its concepts + globals are visible.
        let scoped = build_ambient_gazetteer(&db, Some("haystack"));
        assert_eq!(
            scoped.matches(&body).len(),
            2,
            "project + global concept both resolve under the project scope"
        );

        // Globals-only scope → the project concept is invisible.
        let globals = build_ambient_gazetteer(&db, Some(""));
        assert_eq!(
            globals.matches(&body).len(),
            1,
            "Some(\"\") lists globals only"
        );
    }

    #[test]
    fn gazetteer_enforces_word_boundaries() {
        let (_t, db) = db();
        db.ensure_concept(G, "ostk", ConceptStatus::Active).unwrap();
        let gaz = build_ambient_gazetteer(&db, None);
        assert_eq!(gaz.matches(&normalize("i use ostk daily")).len(), 1);
        assert!(
            gaz.matches(&normalize("ostkx is different")).is_empty(),
            "no match inside a longer token"
        );
    }

    #[test]
    fn gazetteer_excludes_terminal_targets_but_keeps_known_oracle() {
        let (_t, db) = db();
        db.ensure_concept(G, "rejected-name", ConceptStatus::Rejected)
            .unwrap();
        let gaz = build_ambient_gazetteer(&db, None);
        // Terminal concept is not a match target...
        assert!(gaz.matches(&normalize("see rejected-name")).is_empty());
        // ...but its normalized form IS in the known oracle (so node-minting
        // never re-materializes it).
        assert!(gaz.known.contains("rejected name"));
    }

    #[test]
    fn gazetteer_trust_gates_match_targets_to_active_and_proposed() {
        let (_t, db) = db();
        db.ensure_concept(G, "active-one", ConceptStatus::Active)
            .unwrap();
        db.ensure_concept(G, "proposed-two", ConceptStatus::Proposed)
            .unwrap();
        db.ensure_concept(G, "candidate-three", ConceptStatus::Candidate)
            .unwrap();
        let gaz = build_ambient_gazetteer(&db, None);

        // Active + Proposed are match targets; Candidate is not (too low-trust
        // to shape topology).
        let ids = gaz.matches(&normalize(
            "saw active-one and proposed-two next to candidate-three",
        ));
        assert_eq!(ids.len(), 2, "only Active + Proposed match");

        // The Candidate still populates the known oracle, so node-minting won't
        // re-materialize it.
        assert!(gaz.known.contains("candidate three"));
        assert!(gaz.known.contains("active one"));
    }
}
