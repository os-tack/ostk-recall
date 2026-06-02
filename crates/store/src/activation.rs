//! Concept activation — the working-memory frame's salience math.
//!
//! `memory-activation-frame.md` slice 1. One activation idea — *rank what
//! matters now, and expose the math* — projected two ways from a single
//! source of truth:
//!
//! - **Frame** ([`ConceptActivationReader::concept_activations`]): rank the
//!   *active concept objects* for `memory_surface(now)`, each carrying its
//!   decomposed `why` (the `abi-as-sovereign-boundary` law: *argue with the
//!   math, not the vibe*).
//! - **Lens** ([`ConceptActivationReader::concept_support_by_coord`]): score
//!   *candidate chunks* by the activation of the most-active concept that
//!   cites them as evidence — the fuel the `query` crate's `ConceptSupport`
//!   rank feature normalizes to light the dormant lens concept slot.
//!
//! Activation is **derived from `chain_log`**, not a parallel
//! `memory_activations` table — concept salience reads the same cognition
//! stream that chunk freshness (P7b access ledger) and thread resonance
//! already do (`chain-as-cognition-history`). It reuses the existing curves
//! rather than inventing a formula: the ACT-R base-activation decay (mirrors
//! [`crate::threads`] `FreshnessFactory`) over **distinct** recall queries —
//! the `salience-vs-familiarity` gate. A concept hit 50× by one chatty query
//! is not 50× active; only *distinct information* (distinct `query_hash`)
//! accrues. That is the exact bug that bit the thread layer; it is not
//! re-introduced here.

// `Mutex<Connection>` guard held across prepare + query, like the rest of
// this crate (see `threads.rs`); the early-drop lint fights that pattern.
#![allow(clippy::significant_drop_tightening)]

use std::collections::{HashMap, HashSet};

use chrono::{DateTime, Duration, Utc};

use crate::concepts::{ConceptEdge, ConceptRecord, ConceptStatus, EdgeDirection, EvidenceState};
use crate::corpus::Result;
use crate::threads::ThreadsDb;

/// ACT-R decay exponent (shared with `FreshnessFactory`; higher = faster fade).
pub const ACT_R_DECAY_D: f32 = 0.5;
/// Recency time-constants (hours) for the focus / note lifts.
pub const FOCUS_TAU_HOURS: f32 = 24.0;
pub const NOTE_TAU_HOURS: f32 = 72.0;
/// Recency time-constant (hours) for edge conductance.
///
/// An untouched edge fades toward zero over days; re-observation (which
/// resets `last_seen_at`) pulls it back up. Longer than the focus τ — a
/// relationship is stickier than a momentary pin.
pub const EDGE_TAU_HOURS: f32 = 72.0;

/// Activation-sum weights.
///
/// `confidence` (the durable ledger trust) is the base; each lift is a bounded
/// `[0, 1]` bonus scaled here. Tunable seam — kept compiled-in for slice 1
/// (no `[ranking]`-style block yet).
pub const W_ACCESS: f32 = 0.5;
pub const W_FOCUS: f32 = 0.3;
pub const W_EDGE: f32 = 0.2;
pub const W_NOTE: f32 = 0.2;

/// How far back the concept chain is consulted by default.
pub const DEFAULT_WINDOW_DAYS: i64 = 30;

/// The decomposed score breakdown for one active concept — the mandatory `why`.
///
/// Per `abi-as-sovereign-boundary`: every field that moved `activation` is
/// exposed so the surface can be argued with, not just trusted.
#[derive(Debug, Clone, PartialEq)]
pub struct ConceptWhy {
    /// Durable ledger trust (active concepts ≈ 1.0). The base term.
    pub confidence: f32,
    /// Squashed ACT-R base activation over **distinct** recall queries.
    pub decayed_access: f32,
    /// Recency lift from the most recent `ConceptFocused` (pin), `[0, 1]`.
    pub focus_lift: f32,
    /// Bridge lift from incident edges to other *active* concepts, `[0, 1]`.
    pub edge_lift: f32,
    /// Recency lift from the most recent `ConceptNoteAdded`, `[0, 1]`.
    pub note_recency: f32,
    /// Distinct `query_hash` seen — the salience signal (NOT raw count).
    pub distinct_queries: u32,
    /// Distinct access `reason`/surface seen.
    pub distinct_sources: u32,
    /// Seconds since the most recent concept event of any kind, if any.
    pub time_since_touch_secs: Option<i64>,
}

/// One ranked active concept plus its `why`. The frame projection.
#[derive(Debug, Clone, PartialEq)]
pub struct ConceptActivation {
    pub project: String,
    pub handle: String,
    /// `confidence + Σ weighted lifts`. Absolute scale is not meaningful
    /// across corpora; the ordering and the `why` breakdown are.
    pub activation: f32,
    pub why: ConceptWhy,
    /// Typed-node kind (`person`, …) or `None` if untyped. Surfaced on the
    /// frame so `memory_surface(now)` shows what *kind* of thing each active
    /// concept is.
    pub kind: Option<String>,
}

/// Per-coordinate lens support: the most-active active concept citing a
/// `(source, source_id)` coordinate as evidence. The lens projection.
#[derive(Debug, Clone, PartialEq)]
pub struct ConceptSupport {
    pub handle: String,
    pub activation: f32,
}

/// Diffusion output (relational-substrate slice 2): spreading activation over
/// the concept edge graph from attention-lit seeds, bridged to chunk
/// coordinates plus the resolved chunk ids worth injecting as lens candidates.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct RelationalSupport {
    /// `(source, source_id)` → max diffused activation of a concept whose
    /// evidence cites that coordinate. The scoring map for `relational_lift`.
    pub coord_activation: HashMap<(String, String), f32>,
    /// Resolved chunk ids of diffused-neighbour evidence, to inject as lens
    /// candidates so neighbour chunks can *surface*, not merely be re-ranked.
    pub inject_chunk_ids: Vec<String>,
}

/// Diffusion knobs (slice 2), tunable. Hops kept shallow + frontier capped so
/// one lens refresh is a handful of indexed adjacency reads.
pub const REL_MAX_HOPS: usize = 2;
/// Max nodes expanded per hop (strongest-weight first).
pub const REL_FRONTIER_K: usize = 16;
/// Per-hop multiplicative decay applied on top of edge conductance.
pub const REL_HOP_DECAY: f32 = 0.5;
/// Inflow below this is not propagated (prunes the long dim tail).
pub const REL_MIN_ACTIVATION: f32 = 0.05;
/// A concept seeds diffusion only if its dynamic (attention) signal exceeds
/// this — keeps diffusion attention-shaped, not a global graph glow.
pub const REL_SEED_THRESHOLD: f32 = 0.05;

/// ACT-R base activation `B = ln(1 + Σ age_hours^{-d})`. `ln_1p` is the
/// accurate small-sum form; ages are pre-floored by the caller.
#[must_use]
pub fn act_r_base(ages_hours: &[f32], d: f32) -> f32 {
    if ages_hours.is_empty() {
        return 0.0;
    }
    let sum: f32 = ages_hours.iter().map(|a| a.powf(-d)).sum();
    sum.ln_1p()
}

/// Squash an unbounded non-negative raw activation into `[0, 1)`.
#[must_use]
fn squash(raw: f32) -> f32 {
    raw / (1.0 + raw)
}

/// Read-only concept-activation surface over `chain_log` + the ledger.
///
/// Synchronous, like [`crate::threads::ChainLogReader`] — one indexed
/// `chain_log` scan under the connection mutex plus a few small ledger reads.
/// The trait
/// lets the `query` crate hold an `Arc<dyn ConceptActivationReader>` (the
/// lens concept feature) without depending on the concrete [`ThreadsDb`].
pub trait ConceptActivationReader: Send + Sync {
    /// Frame: active concepts in `project` scope (`None` = all projects;
    /// `Some("")` = global only; `Some("p")` = p + global), ranked by
    /// activation descending, each with its `why`. Events before `since`
    /// are ignored.
    fn concept_activations(
        &self,
        project: Option<&str>,
        since: DateTime<Utc>,
    ) -> Result<Vec<ConceptActivation>>;

    /// Lens: for every evidence coordinate of an *active* concept, the
    /// highest-activation concept citing it. Keyed `(source, source_id)` —
    /// the durable coordinate, matching `Chunk { source, source_id }`. A
    /// fresh corpus with no active concepts returns an empty map, so the
    /// `ConceptSupport` feature contributes nothing and the lens concept
    /// slot skips cleanly.
    fn concept_support_by_coord(
        &self,
        since: DateTime<Utc>,
    ) -> Result<HashMap<(String, String), ConceptSupport>>;

    /// Diffusion (relational-substrate slice 2): spreading activation from
    /// attention-lit seed concepts over the edge graph, bridged to chunk
    /// coordinates + injectable chunk ids. Default impl is empty so non-graph
    /// readers and the explicit path degrade cleanly (the relational lens slot
    /// skips). [`ThreadsDb`] overrides it.
    fn relational_support(&self, _since: DateTime<Utc>) -> Result<RelationalSupport> {
        Ok(RelationalSupport::default())
    }
}

/// Per-concept signals harvested from one chain scan, keyed `(project, handle)`.
#[derive(Default)]
struct ConceptSignals {
    /// Most-recent access ts per distinct `query_hash` (the salience gate:
    /// distinct queries, not raw recurrence).
    access_by_query: HashMap<String, DateTime<Utc>>,
    /// Distinct access `reason`/surface strings.
    sources: HashSet<String>,
    /// Most recent focus ts, if any.
    last_focus: Option<DateTime<Utc>>,
    /// Most recent note ts, if any.
    last_note: Option<DateTime<Utc>>,
    /// Most recent event of any kind, for `time_since_touch_secs`.
    last_touch: Option<DateTime<Utc>>,
}

impl ConceptSignals {
    fn note_touch(slot: &mut Option<DateTime<Utc>>, ts: DateTime<Utc>) {
        if slot.is_none_or(|prev| ts > prev) {
            *slot = Some(ts);
        }
    }
}

impl ThreadsDb {
    /// Scan `chain_log` for the activation-bearing concept events since
    /// `since`, grouping by `(project, handle)`. Reads the row `ts` column
    /// directly (NOT via `ChainEvent::from_row`, which synthesizes
    /// `Utc::now()` and would collapse every age — the same trap the access
    /// ledger documents).
    fn scan_concept_signals(
        &self,
        since: DateTime<Utc>,
    ) -> Result<HashMap<(String, String), ConceptSignals>> {
        let mut out: HashMap<(String, String), ConceptSignals> = HashMap::new();
        let conn = self.lock();
        let mut stmt = conn.prepare(
            "SELECT ts, kind, payload FROM chain_log \
             WHERE ts >= ?1 AND kind IN \
             ('concept_accessed', 'concept_focused', 'concept_note_added') \
             ORDER BY ts ASC",
        )?;
        let rows = stmt
            .query_map([since.to_rfc3339()], |r| {
                Ok((
                    r.get::<_, String>(0)?,
                    r.get::<_, String>(1)?,
                    r.get::<_, String>(2)?,
                ))
            })?
            .collect::<std::result::Result<Vec<_>, rusqlite::Error>>()?;
        for (ts_s, kind, payload) in rows {
            let Ok(ts) = DateTime::parse_from_rfc3339(&ts_s) else {
                continue;
            };
            let ts = ts.with_timezone(&Utc);
            let Ok(v) = serde_json::from_str::<serde_json::Value>(&payload) else {
                continue;
            };
            let project = v.get("project").and_then(|x| x.as_str()).unwrap_or("");
            let Some(handle) = v.get("handle").and_then(|x| x.as_str()) else {
                continue;
            };
            let sig = out
                .entry((project.to_string(), handle.to_string()))
                .or_default();
            match kind.as_str() {
                "concept_accessed" => {
                    let qh = v
                        .get("query_hash")
                        .and_then(|x| x.as_str())
                        .unwrap_or("")
                        .to_string();
                    // Distinct-query gate: keep the most recent ts per
                    // query_hash. 50 hits from one query collapse to one.
                    sig.access_by_query
                        .entry(qh)
                        .and_modify(|prev| {
                            if ts > *prev {
                                *prev = ts;
                            }
                        })
                        .or_insert(ts);
                    if let Some(reason) = v.get("reason").and_then(|x| x.as_str()) {
                        sig.sources.insert(reason.to_string());
                    }
                }
                "concept_focused" => ConceptSignals::note_touch(&mut sig.last_focus, ts),
                "concept_note_added" => ConceptSignals::note_touch(&mut sig.last_note, ts),
                _ => {}
            }
            ConceptSignals::note_touch(&mut sig.last_touch, ts);
        }
        Ok(out)
    }

    /// Compute `(record, activation)` for every active concept in scope.
    /// Shared core behind both trait projections.
    // The weighted activation sum reads clearest as `base + Σ wᵢ·liftᵢ`;
    // `mul_add` would obscure the formula the `why` mirrors for marginal flops.
    #[allow(clippy::suboptimal_flops)]
    fn activations_internal(
        &self,
        project: Option<&str>,
        since: DateTime<Utc>,
    ) -> Result<Vec<(ConceptRecord, ConceptActivation)>> {
        let active = self.list_concepts(project, Some(ConceptStatus::Active))?;
        if active.is_empty() {
            return Ok(Vec::new());
        }
        let signals = self.scan_concept_signals(since)?;
        // Active handle set for the edge bridge — an edge lifts only when its
        // neighbour is *also* active (a live association, not a dangling one).
        let active_handles: HashSet<String> = active.iter().map(|c| c.handle.clone()).collect();

        let now = Utc::now();
        let mut out = Vec::with_capacity(active.len());
        for rec in active {
            let key = (rec.project.clone(), rec.handle.clone());
            let sig = signals.get(&key);

            // --- decayed access (distinct-query ACT-R) -------------------
            let (decayed_access, distinct_queries, distinct_sources) = match sig {
                Some(s) if !s.access_by_query.is_empty() => {
                    let ages: Vec<f32> = s
                        .access_by_query
                        .values()
                        .map(|ts| age_hours_floored(now, *ts, 1.0))
                        .collect();
                    let distinct_queries =
                        u32::try_from(s.access_by_query.len()).unwrap_or(u32::MAX);
                    let distinct_sources = u32::try_from(s.sources.len()).unwrap_or(u32::MAX);
                    (
                        squash(act_r_base(&ages, ACT_R_DECAY_D)),
                        distinct_queries,
                        distinct_sources,
                    )
                }
                _ => (0.0, 0, 0),
            };

            // --- focus / note recency lifts ------------------------------
            let focus_lift = sig
                .and_then(|s| s.last_focus)
                .map_or(0.0, |ts| recency_lift(now, ts, FOCUS_TAU_HOURS));
            let note_recency = sig
                .and_then(|s| s.last_note)
                .map_or(0.0, |ts| recency_lift(now, ts, NOTE_TAU_HOURS));

            // --- edge bridge to active neighbours ------------------------
            let edge_lift = self.edge_lift_for(&rec, &active_handles)?;

            let time_since_touch_secs = sig
                .and_then(|s| s.last_touch)
                .map(|ts| (now - ts).num_seconds().max(0));

            let activation = rec.confidence
                + W_ACCESS * decayed_access
                + W_FOCUS * focus_lift
                + W_EDGE * edge_lift
                + W_NOTE * note_recency;

            let why = ConceptWhy {
                confidence: rec.confidence,
                decayed_access,
                focus_lift,
                edge_lift,
                note_recency,
                distinct_queries,
                distinct_sources,
                time_since_touch_secs,
            };
            let activation_row = ConceptActivation {
                project: rec.project.clone(),
                handle: rec.handle.clone(),
                activation,
                why,
                kind: rec.kind.clone(),
            };
            out.push((rec, activation_row));
        }

        out.sort_by(|a, b| {
            b.1.activation
                .partial_cmp(&a.1.activation)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.1.handle.cmp(&b.1.handle))
        });
        Ok(out)
    }

    /// Normalized incident-edge mass to *active* neighbours, `[0, 1]`.
    fn edge_lift_for(&self, rec: &ConceptRecord, active_handles: &HashSet<String>) -> Result<f32> {
        let mut conf_sum = 0.0_f32;
        for dir in [EdgeDirection::From, EdgeDirection::To] {
            for e in self.list_concept_edges(&rec.project, &rec.handle, dir)? {
                let neighbour = match dir {
                    EdgeDirection::From => &e.to_handle,
                    EdgeDirection::To => &e.from_handle,
                };
                if neighbour != &rec.handle && active_handles.contains(neighbour) {
                    conf_sum += e.confidence.max(0.0);
                }
            }
        }
        // Saturating: many strong live edges → near 1, none → 0.
        Ok(1.0 - (-conf_sum).exp())
    }
}

/// ACT-R age in hours, floored so a just-touched concept can't blow the
/// `age^{-d}` term up and a future ts (clock skew) can't go negative.
fn age_hours_floored(now: DateTime<Utc>, ts: DateTime<Utc>, floor_hours: f32) -> f32 {
    #[allow(clippy::cast_precision_loss)]
    let hours = (now - ts).num_seconds() as f32 / 3600.0;
    hours.max(floor_hours)
}

/// Exponential recency lift `exp(-age_hours / tau)` in `(0, 1]`. Floors age
/// at one minute so "just now" reads ≈ 1 without dividing by zero.
fn recency_lift(now: DateTime<Utc>, ts: DateTime<Utc>, tau_hours: f32) -> f32 {
    let age = age_hours_floored(now, ts, 1.0 / 60.0);
    (-age / tau_hours).exp().clamp(0.0, 1.0)
}

/// Derived **conductance** of an edge — how readily diffusion current flows
/// through it — as of `now`.
///
/// Never stored: it is the edge's `confidence` prior gated by `last_seen_at`
/// recency, so it cannot drift from the edge's actual use history. An
/// authored edge enters low ([`AUTHORED_EDGE_CONFIDENCE`]
/// ≈ 0.1) and decays as it goes untouched; re-observation resets
/// `last_seen_at` (recency → ~1), pulling conductance back up. This is "use
/// sets the conductance" made arithmetic — decay is free from the timestamp.
///
/// [`AUTHORED_EDGE_CONFIDENCE`]: crate::concepts::AUTHORED_EDGE_CONFIDENCE
#[must_use]
pub fn edge_conductance(edge: &ConceptEdge, now: DateTime<Utc>) -> f32 {
    conductance_of(edge.confidence, edge.last_seen_at, now)
}

/// Conductance from the raw `(confidence, last_seen_at)` pair — lets the
/// id-space traversal read ([`crate::ThreadsDb::neighbors_by_id`]) compute it
/// without materializing a full [`ConceptEdge`]. Same formula as
/// [`edge_conductance`].
#[must_use]
pub fn conductance_of(confidence: f32, last_seen_at: DateTime<Utc>, now: DateTime<Utc>) -> f32 {
    let recency = recency_lift(now, last_seen_at, EDGE_TAU_HOURS);
    (confidence.max(0.0) * recency).clamp(0.0, 1.0)
}

impl ConceptActivationReader for ThreadsDb {
    fn concept_activations(
        &self,
        project: Option<&str>,
        since: DateTime<Utc>,
    ) -> Result<Vec<ConceptActivation>> {
        Ok(self
            .activations_internal(project, since)?
            .into_iter()
            .map(|(_, a)| a)
            .collect())
    }

    fn concept_support_by_coord(
        &self,
        since: DateTime<Utc>,
    ) -> Result<HashMap<(String, String), ConceptSupport>> {
        // All projects: the ambient lens has no project text query.
        let activations = self.activations_internal(None, since)?;
        let mut out: HashMap<(String, String), ConceptSupport> = HashMap::new();
        for (rec, act) in activations {
            for ev in self.list_concept_evidence(rec.id)? {
                let coord = (ev.source.clone(), ev.source_id.clone());
                let support = ConceptSupport {
                    handle: rec.handle.clone(),
                    activation: act.activation,
                };
                out.entry(coord)
                    .and_modify(|cur| {
                        if act.activation > cur.activation {
                            *cur = support.clone();
                        }
                    })
                    .or_insert(support);
            }
        }
        Ok(out)
    }

    fn relational_support(&self, since: DateTime<Utc>) -> Result<RelationalSupport> {
        // 1. Seeds = attention-lit concepts. Weight on the DYNAMIC signal only
        //    (decayed_access + focus_lift), never durable `confidence` — diffuse
        //    from what attention is touching now, not from every Active concept
        //    (which would glow the whole graph).
        let seeds: Vec<(i64, f32)> = self
            .activations_internal(None, since)?
            .into_iter()
            .filter_map(|(rec, act)| {
                let dynamic = act.why.decayed_access + act.why.focus_lift;
                (dynamic > REL_SEED_THRESHOLD).then_some((rec.id, dynamic))
            })
            .collect();
        if seeds.is_empty() {
            return Ok(RelationalSupport::default());
        }

        // 2. Spreading activation in id-space (cross-scope safe).
        let now = Utc::now();
        let seed_ids: HashSet<i64> = seeds.iter().map(|(id, _)| *id).collect();
        let mut weight: HashMap<i64, f32> = seeds.into_iter().collect();
        let mut frontier: Vec<i64> = weight.keys().copied().collect();
        for _hop in 0..REL_MAX_HOPS {
            // Expand the strongest FRONTIER_K nodes by their *current* weight.
            frontier.sort_by(|a, b| {
                weight[b]
                    .partial_cmp(&weight[a])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            frontier.truncate(REL_FRONTIER_K);
            // Aggregate the best inflow each neighbour receives THIS hop (max
            // across all expanding sources), then commit + re-queue only the
            // neighbours whose weight strictly improves. Taking the per-hop max
            // (not the first-seen value) means a node reached weakly and then
            // strongly expands with its strongest inflow, and re-queue-on-improve
            // lets a stronger later path correct an earlier weak one. Weights
            // strictly increase and each hop's inflow is < its source's weight,
            // so this converges (and REL_MAX_HOPS bounds it regardless).
            let mut best: HashMap<i64, f32> = HashMap::new();
            for src_id in &frontier {
                let src_w = weight[src_id];
                for (nbr_id, cond) in self.neighbors_by_id(*src_id, now)? {
                    let inflow = src_w * cond * REL_HOP_DECAY;
                    if inflow < REL_MIN_ACTIVATION {
                        continue;
                    }
                    let e = best.entry(nbr_id).or_insert(0.0);
                    if inflow > *e {
                        *e = inflow;
                    }
                }
            }
            let mut next: Vec<i64> = Vec::new();
            for (nbr_id, inflow) in best {
                let cur = weight.get(&nbr_id).copied().unwrap_or(0.0);
                if inflow > cur {
                    weight.insert(nbr_id, inflow);
                    next.push(nbr_id);
                }
            }
            if next.is_empty() {
                break;
            }
            frontier = next;
        }

        // 3. Bridge each diffused concept to its evidence coordinates and the
        //    resolved chunk ids to inject (max activation per coordinate).
        let mut coord_activation: HashMap<(String, String), f32> = HashMap::new();
        let mut inject_chunk_ids: Vec<String> = Vec::new();
        let mut seen_chunks: HashSet<String> = HashSet::new();
        for (cid, w) in &weight {
            // Seeds are the *current focus*; their own chunks already surface
            // via attention. relational_lift is the NEIGHBOUR signal — emitting
            // seed coords here would let the (highest-weight) seed dominate the
            // feature's min-max normalization and push real neighbours to ~0.
            if seed_ids.contains(cid) {
                continue;
            }
            for ev in self.list_concept_evidence(*cid)? {
                // Only live evidence bridges — skip orphaned / inactive rows so
                // diffusion can't inject a chunk an evidence link has retired.
                if ev.relation_state != EvidenceState::Active {
                    continue;
                }
                coord_activation
                    .entry((ev.source.clone(), ev.source_id.clone()))
                    .and_modify(|cur| {
                        if *w > *cur {
                            *cur = *w;
                        }
                    })
                    .or_insert(*w);
                if let Some(chunk_id) = ev.last_resolved_chunk_id {
                    if seen_chunks.insert(chunk_id.clone()) {
                        inject_chunk_ids.push(chunk_id);
                    }
                }
            }
        }
        Ok(RelationalSupport {
            coord_activation,
            inject_chunk_ids,
        })
    }
}

/// Convenience: the default look-back window as a `since` cutoff from now.
#[must_use]
pub fn default_since(now: DateTime<Utc>) -> DateTime<Utc> {
    now - Duration::days(DEFAULT_WINDOW_DAYS)
}

/// The default `since` cutoff measured from the current instant. Lets callers
/// that don't otherwise depend on `chrono` (e.g. the MCP frame) pass a window
/// without importing it.
#[must_use]
pub fn default_since_now() -> DateTime<Utc> {
    default_since(Utc::now())
}

#[cfg(test)]
mod tests {
    // Exact float asserts on small, deterministic activation values.
    #![allow(clippy::float_cmp)]
    use std::sync::Arc;

    use tempfile::TempDir;

    use super::*;
    use crate::concepts::{EdgeSource, EvidenceAttach, GLOBAL_PROJECT};
    use crate::threads::{ChainSink, SqliteChainSink};

    const G: &str = GLOBAL_PROJECT;

    // A live SQLite chain sink over the same `threads.sqlite`, so emitted
    // concept events land in `chain_log` and the reader can scan them — the
    // real production path (WAL: the reader's connection sees the sink's
    // committed rows), not a stub.
    fn db() -> (TempDir, ThreadsDb) {
        let tmp = TempDir::new().unwrap();
        let sink: Arc<dyn ChainSink> = Arc::new(SqliteChainSink::open(tmp.path()).unwrap());
        let db = ThreadsDb::open_with_sink(tmp.path(), sink).unwrap();
        (tmp, db)
    }

    fn active(db: &ThreadsDb, handle: &str) {
        db.ensure_concept(G, handle, ConceptStatus::Active).unwrap();
    }

    fn test_edge(confidence: f32, last_seen: DateTime<Utc>) -> ConceptEdge {
        ConceptEdge {
            id: 1,
            from_handle: "a".into(),
            relation: "memory_layer_for".into(),
            to_handle: "b".into(),
            confidence,
            evidence_json: None,
            first_seen_at: last_seen,
            last_seen_at: last_seen,
            touch_count: 1,
            source: crate::concepts::EdgeSource::Authored,
            by: None,
        }
    }

    #[test]
    fn edge_conductance_low_when_authored_and_unused() {
        let now = Utc::now();
        let edge = test_edge(crate::concepts::AUTHORED_EDGE_CONFIDENCE, now);
        let c = edge_conductance(&edge, now);
        // confidence(0.1) × recency(~1) — low, never 1.0.
        assert!(
            c > 0.0 && c <= 0.1 + 1e-4,
            "fresh authored edge is low, got {c}"
        );
    }

    #[test]
    fn edge_conductance_decays_with_disuse() {
        let now = Utc::now();
        let fresh = test_edge(0.6, now);
        let stale = test_edge(0.6, now - Duration::days(30));
        let cf = edge_conductance(&fresh, now);
        let cs = edge_conductance(&stale, now);
        assert!(cs < cf, "an untouched edge decays: stale {cs} < fresh {cf}");
        assert!(cs >= 0.0, "conductance never goes negative");
    }

    #[test]
    fn no_active_concepts_is_empty() {
        let (_t, db) = db();
        // A candidate exists but is not active → not surfaced.
        db.ensure_concept(G, "mish", ConceptStatus::Candidate)
            .unwrap();
        let since = default_since(Utc::now());
        assert!(db.concept_activations(None, since).unwrap().is_empty());
        assert!(db.concept_support_by_coord(since).unwrap().is_empty());
    }

    #[test]
    fn distinct_query_gate_not_raw_count() {
        let (_t, db) = db();
        active(&db, "chatty");
        active(&db, "varied");
        // `chatty`: 5 accesses, all ONE query_hash → 1 distinct.
        for _ in 0..5 {
            db.record_concept_accessed(G, "chatty", "recall:path", "qh-same")
                .unwrap();
        }
        // `varied`: 2 accesses, two distinct query_hashes → 2 distinct.
        db.record_concept_accessed(G, "varied", "recall:path", "qh-a")
            .unwrap();
        db.record_concept_accessed(G, "varied", "recall:symbol", "qh-b")
            .unwrap();

        let acts = db
            .concept_activations(None, default_since(Utc::now()))
            .unwrap();
        let chatty = acts.iter().find(|a| a.handle == "chatty").unwrap();
        let varied = acts.iter().find(|a| a.handle == "varied").unwrap();
        assert_eq!(
            chatty.why.distinct_queries, 1,
            "5 same-query hits gate to 1"
        );
        assert_eq!(varied.why.distinct_queries, 2);
        assert_eq!(varied.why.distinct_sources, 2);
        // The salience gate: more distinct information ⇒ more access decay,
        // even though `chatty` had more raw events.
        assert!(
            varied.why.decayed_access > chatty.why.decayed_access,
            "distinct information must beat raw recurrence (salience-vs-familiarity)"
        );
    }

    #[test]
    fn focus_lifts_activation_and_why() {
        let (_t, db) = db();
        active(&db, "pinned");
        active(&db, "unpinned");
        db.record_concept_focused(G, "pinned").unwrap();
        let acts = db
            .concept_activations(None, default_since(Utc::now()))
            .unwrap();
        let pinned = acts.iter().find(|a| a.handle == "pinned").unwrap();
        let unpinned = acts.iter().find(|a| a.handle == "unpinned").unwrap();
        assert!(pinned.why.focus_lift > 0.5, "just-focused ≈ 1.0");
        assert_eq!(unpinned.why.focus_lift, 0.0);
        assert!(pinned.activation > unpinned.activation);
        // Sorted descending → the focused concept leads.
        assert_eq!(acts.first().unwrap().handle, "pinned");
    }

    #[test]
    fn edge_lift_only_for_active_neighbours() {
        let (_t, db) = db();
        active(&db, "mish");
        active(&db, "slipstream");
        db.ensure_concept(G, "dormant", ConceptStatus::Candidate)
            .unwrap();
        // mish—slipstream (both active) and mish—dormant (neighbour inactive).
        db.add_concept_edge(
            G,
            "mish",
            "pairs_with",
            "slipstream",
            0.6,
            EdgeSource::Observed,
            None,
            None,
        )
        .unwrap();
        db.add_concept_edge(
            G,
            "mish",
            "pairs_with",
            "dormant",
            0.6,
            EdgeSource::Observed,
            None,
            None,
        )
        .unwrap();
        let acts = db
            .concept_activations(None, default_since(Utc::now()))
            .unwrap();
        let mish = acts.iter().find(|a| a.handle == "mish").unwrap();
        // Only the active-neighbour edge contributes.
        assert!(mish.why.edge_lift > 0.0, "active neighbour lifts");
        assert!(mish.why.edge_lift < 0.5, "single 0.6-conf edge, bounded");
    }

    #[test]
    fn support_by_coord_picks_highest_activation_concept() {
        let (_t, db) = db();
        let (low, _) = db.ensure_concept(G, "low", ConceptStatus::Active).unwrap();
        let (high, _) = db.ensure_concept(G, "high", ConceptStatus::Active).unwrap();
        // Lift `high` above `low` via focus.
        db.record_concept_focused(G, "high").unwrap();
        // Both cite the SAME coordinate as evidence.
        let coord = ("thread", "shared-src");
        for id in [low.id, high.id] {
            db.attach_concept_evidence(&EvidenceAttach {
                concept_id: id,
                project: G,
                source: coord.0,
                source_id: coord.1,
                chunk_id: Some("chunk-x"),
                content_sha256: None,
                anchor_vec: None,
                score: Some(0.9),
                reason: Some("test"),
            })
            .unwrap();
        }
        let map = db
            .concept_support_by_coord(default_since(Utc::now()))
            .unwrap();
        let support = map
            .get(&("thread".to_string(), "shared-src".to_string()))
            .expect("coordinate present");
        assert_eq!(
            support.handle, "high",
            "winner is the higher-activation concept"
        );
    }

    // --- Slice 2: diffusion (neighbors_by_id + relational_support) ----------

    fn ev(db: &ThreadsDb, id: i64, source_id: &str, chunk: Option<&str>) {
        db.attach_concept_evidence(&EvidenceAttach {
            concept_id: id,
            project: G,
            source: "code",
            source_id,
            chunk_id: chunk,
            content_sha256: None,
            anchor_vec: None,
            score: Some(0.5),
            reason: Some("test"),
        })
        .unwrap();
    }

    fn id_of(db: &ThreadsDb, project: &str, handle: &str) -> i64 {
        db.get_concept(project, handle).unwrap().unwrap().id
    }

    #[test]
    fn neighbors_by_id_both_directions_by_id_not_handle() {
        let (_t, db) = db();
        for h in ["a", "b", "c"] {
            active(&db, h);
        }
        // A decoy in another scope sharing handle "b" — must never be returned.
        db.ensure_concept("memories", "b", ConceptStatus::Active)
            .unwrap();
        let (a, b, c) = (id_of(&db, G, "a"), id_of(&db, G, "b"), id_of(&db, G, "c"));
        let decoy = id_of(&db, "memories", "b");
        db.add_concept_edge_by_id(a, "rel", b, 1.0, EdgeSource::Observed, None, None)
            .unwrap(); // outgoing
        db.add_concept_edge_by_id(c, "rel", a, 1.0, EdgeSource::Observed, None, None)
            .unwrap(); // incoming
        let nbrs = db.neighbors_by_id(a, Utc::now()).unwrap();
        let ids: Vec<i64> = nbrs.iter().map(|(id, _)| *id).collect();
        assert!(ids.contains(&b) && ids.contains(&c), "both directions");
        assert!(!ids.contains(&a), "excludes self");
        assert!(
            !ids.contains(&decoy),
            "traverses by id, never the same-handle decoy in another scope"
        );
        assert!(
            nbrs.iter().all(|(_, cond)| *cond > 0.0),
            "conductance paired"
        );
    }

    #[test]
    fn relational_support_lights_neighbor_with_decayed_inflow() {
        let (_t, db) = db();
        active(&db, "tori");
        active(&db, "ostk-recall");
        // A recent access gives `tori` dynamic (attention) activation → a seed.
        db.record_concept_accessed(G, "tori", "recall:path", "qh-1")
            .unwrap();
        let (tori, orec) = (id_of(&db, G, "tori"), id_of(&db, G, "ostk-recall"));
        // Strong fresh edge so the neighbour clears the prune floor.
        db.add_concept_edge_by_id(
            tori,
            "works_on",
            orec,
            1.0,
            EdgeSource::Authored,
            None,
            None,
        )
        .unwrap();
        ev(&db, tori, "tori-src", Some("tori-chunk"));
        ev(&db, orec, "or-src", Some("or-chunk"));
        ev(&db, orec, "or-unresolved", None); // unresolved → not injectable

        let sup = db.relational_support(default_since(Utc::now())).unwrap();
        // The seed's own coord is NOT emitted — it surfaces via attention; the
        // relational signal is purely the diffused neighbour.
        assert!(
            !sup.coord_activation
                .contains_key(&("code".to_string(), "tori-src".to_string())),
            "seed coord is excluded from the relational map"
        );
        let nbr_w = *sup
            .coord_activation
            .get(&("code".to_string(), "or-src".to_string()))
            .expect("neighbour coord present (diffusion reached it)");
        assert!(nbr_w > REL_MIN_ACTIVATION, "neighbour lit above the floor");
        assert!(
            (0.18..0.22).contains(&nbr_w),
            "≈ seed_dyn × conductance × HOP_DECAY, got {nbr_w}"
        );
        // Injection: the neighbour's resolved chunk only — not the seed's
        // (excluded), not the unresolved row (no chunk to inject).
        assert_eq!(sup.inject_chunk_ids, vec!["or-chunk".to_string()]);
        assert!(
            sup.coord_activation
                .contains_key(&("code".to_string(), "or-unresolved".to_string())),
            "unresolved evidence still scores by coord, just isn't injected"
        );
    }

    #[test]
    fn relational_support_seeds_on_dynamic_signal_not_durable() {
        let (_t, db) = db();
        // Active (durable confidence) but no access/focus → NOT a seed.
        active(&db, "lonely");
        active(&db, "x");
        let (lonely, x) = (id_of(&db, G, "lonely"), id_of(&db, G, "x"));
        db.add_concept_edge_by_id(lonely, "rel", x, 1.0, EdgeSource::Observed, None, None)
            .unwrap();
        ev(&db, x, "x-src", Some("x-chunk"));
        let sup = db.relational_support(default_since(Utc::now())).unwrap();
        assert!(
            sup.coord_activation.is_empty(),
            "a durable-only Active concept does not seed diffusion (no global glow)"
        );
        assert!(sup.inject_chunk_ids.is_empty());
    }

    #[test]
    fn relational_support_two_hops_not_three() {
        let (_t, db) = db();
        for h in ["a", "b", "c", "d"] {
            active(&db, h);
        }
        db.record_concept_accessed(G, "a", "recall:path", "qh-1")
            .unwrap();
        let ids: Vec<i64> = ["a", "b", "c", "d"]
            .iter()
            .map(|h| id_of(&db, G, h))
            .collect();
        // Strong fresh chain a→b→c→d.
        for w in ids.windows(2) {
            db.add_concept_edge_by_id(w[0], "rel", w[1], 1.0, EdgeSource::Observed, None, None)
                .unwrap();
        }
        ev(&db, ids[1], "b-src", Some("b-chunk"));
        ev(&db, ids[2], "c-src", Some("c-chunk"));
        ev(&db, ids[3], "d-src", Some("d-chunk"));
        let sup = db.relational_support(default_since(Utc::now())).unwrap();
        assert!(
            sup.coord_activation
                .contains_key(&("code".to_string(), "b-src".to_string())),
            "hop 1 reached"
        );
        assert!(
            sup.coord_activation
                .contains_key(&("code".to_string(), "c-src".to_string())),
            "hop 2 reached"
        );
        assert!(
            !sup.coord_activation
                .contains_key(&("code".to_string(), "d-src".to_string())),
            "hop 3 NOT reached (MAX_HOPS = 2)"
        );
    }

    #[test]
    fn relational_support_empty_without_active_seeds() {
        let (_t, db) = db();
        db.ensure_concept(G, "cand", ConceptStatus::Candidate)
            .unwrap();
        let sup = db.relational_support(default_since(Utc::now())).unwrap();
        assert!(sup.coord_activation.is_empty() && sup.inject_chunk_ids.is_empty());
    }

    #[test]
    fn neighbors_by_id_excludes_terminal() {
        let (_t, db) = db();
        active(&db, "a");
        active(&db, "ok");
        db.ensure_concept(G, "ghost", ConceptStatus::Active)
            .unwrap();
        let (a, ok, ghost) = (
            id_of(&db, G, "a"),
            id_of(&db, G, "ok"),
            id_of(&db, G, "ghost"),
        );
        db.add_concept_edge_by_id(a, "rel", ok, 1.0, EdgeSource::Observed, None, None)
            .unwrap();
        db.add_concept_edge_by_id(a, "rel", ghost, 1.0, EdgeSource::Observed, None, None)
            .unwrap();
        db.set_concept_status(G, "ghost", ConceptStatus::Rejected, None)
            .unwrap();
        let ids: Vec<i64> = db
            .neighbors_by_id(a, Utc::now())
            .unwrap()
            .into_iter()
            .map(|(id, _)| id)
            .collect();
        assert!(ids.contains(&ok), "live neighbour present");
        assert!(
            !ids.contains(&ghost),
            "rejected (terminal) neighbour excluded"
        );
    }

    #[test]
    fn relational_support_excludes_terminal_neighbour() {
        let (_t, db) = db();
        active(&db, "tori");
        db.ensure_concept(G, "ghost", ConceptStatus::Active)
            .unwrap();
        db.record_concept_accessed(G, "tori", "recall:path", "qh-1")
            .unwrap();
        let (tori, ghost) = (id_of(&db, G, "tori"), id_of(&db, G, "ghost"));
        db.add_concept_edge_by_id(tori, "rel", ghost, 1.0, EdgeSource::Observed, None, None)
            .unwrap();
        ev(&db, ghost, "ghost-src", Some("ghost-chunk"));
        db.set_concept_status(G, "ghost", ConceptStatus::Rejected, None)
            .unwrap();
        let sup = db.relational_support(default_since(Utc::now())).unwrap();
        assert!(
            sup.coord_activation.is_empty() && sup.inject_chunk_ids.is_empty(),
            "diffusion does not flow into a terminal (rejected) neighbour"
        );
    }

    #[test]
    fn relational_support_skips_orphaned_evidence() {
        let (_t, db) = db();
        active(&db, "tori");
        active(&db, "x");
        db.record_concept_accessed(G, "tori", "recall:path", "qh-1")
            .unwrap();
        let (tori, x) = (id_of(&db, G, "tori"), id_of(&db, G, "x"));
        db.add_concept_edge_by_id(tori, "rel", x, 1.0, EdgeSource::Observed, None, None)
            .unwrap();
        ev(&db, x, "x-src", Some("x-chunk"));
        // Orphan the neighbour's only evidence row.
        let rows = db.evidence_to_reconcile().unwrap();
        let rowid = rows.iter().find(|r| r.source_id == "x-src").unwrap().rowid;
        db.mark_evidence_orphaned(rowid).unwrap();
        let sup = db.relational_support(default_since(Utc::now())).unwrap();
        assert!(
            sup.coord_activation.is_empty() && sup.inject_chunk_ids.is_empty(),
            "orphaned evidence is not bridged"
        );
    }

    #[test]
    fn relational_support_neighbour_inflow_is_max_not_summed() {
        // Two seeds both reach neighbour `c` in one hop. `c` takes the MAX
        // inflow (the stronger seed), never the sum.
        let (_t, db) = db();
        active(&db, "s1");
        active(&db, "s2");
        active(&db, "c");
        // s1 is the stronger seed (3 distinct accesses vs s2's 1).
        for qh in ["q1", "q2", "q3"] {
            db.record_concept_accessed(G, "s1", "recall:path", qh)
                .unwrap();
        }
        db.record_concept_accessed(G, "s2", "recall:path", "q9")
            .unwrap();
        let (s1, s2, c) = (id_of(&db, G, "s1"), id_of(&db, G, "s2"), id_of(&db, G, "c"));
        db.add_concept_edge_by_id(s1, "rel", c, 1.0, EdgeSource::Observed, None, None)
            .unwrap();
        db.add_concept_edge_by_id(s2, "rel", c, 1.0, EdgeSource::Observed, None, None)
            .unwrap();
        ev(&db, c, "c-src", Some("c-chunk"));

        // s1's dynamic seed weight, to compare against.
        let acts = db
            .concept_activations(None, default_since(Utc::now()))
            .unwrap();
        let s1_dyn = {
            let a = acts.iter().find(|a| a.handle == "s1").unwrap();
            a.why.decayed_access + a.why.focus_lift
        };
        let s2_dyn = {
            let a = acts.iter().find(|a| a.handle == "s2").unwrap();
            a.why.decayed_access + a.why.focus_lift
        };
        assert!(s1_dyn > s2_dyn, "s1 is the stronger seed");

        let sup = db.relational_support(default_since(Utc::now())).unwrap();
        let c_w = *sup
            .coord_activation
            .get(&("code".to_string(), "c-src".to_string()))
            .expect("neighbour reached");
        // conductance ≈ 1 on the fresh edges, so the max path ≈ s1_dyn × 0.5.
        let expected_max = s1_dyn * REL_HOP_DECAY;
        assert!(
            (c_w - expected_max).abs() < 0.03,
            "neighbour took the MAX source inflow ≈ {expected_max}, got {c_w}"
        );
        assert!(
            c_w < (s1_dyn + s2_dyn) * REL_HOP_DECAY - 0.05,
            "neighbour inflow is NOT the sum of both sources"
        );
    }
}
