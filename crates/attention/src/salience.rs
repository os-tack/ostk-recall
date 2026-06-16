//! Autonomous-salience scorer factors (THESIS axis 1–3).
//!
//! Today's scorer (`compute_score_parts`) rewards "looks like now, a lot."
//! This module supplies the three precomputed per-handle scalars that evolve
//! it into `salience = specificity × value × recency − negative_penalty`:
//!
//! - **specificity** (axis 1, this increment): `1 − H/H_max` over a handle's
//!   co-occurrence distribution across distinct source documents. A term that
//!   resonates with *everything* (high entropy) carries no information; one
//!   that concentrates (low entropy) discriminates. This is IDF restated as
//!   normalized entropy — the principled, continuous form of the binary
//!   `is_stop_handle` cliff.
//! - **value** (axis 3) and `neg_penalty` (axis 2) ship in later increments;
//!   their fields default to the neutral identity here so the scorer is a
//!   pass-through until they are populated.
//!
//! Everything in this module is pure and unit-testable in isolation. The
//! `shannon_entropy` primitive is shared with the self-audit metric (axis 4),
//! so it lives here as the single implementation.

use std::collections::HashMap;

use chrono::{DateTime, Utc};
use ostk_recall_core::config::SalienceSettings;
use ostk_recall_store::{
    ACT_R_DECAY_D, AccessWeights, EvidenceLink, RelationState, UsedAccess, act_r_base,
    age_hours_floored, squash,
};

/// Precomputed per-handle salience factors. The neutral identity
/// (`{1.0, 1.0, 0.0}`) leaves `compute_score_parts` bit-identical to the v1
/// scorer, so an absent factor (handle not in the map) is a no-op.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SalienceFactors {
    /// Axis 1: co-occurrence specificity in `[0,1]`. `1.0` = maximally
    /// discriminating / neutral; `→0` = resonates with everything (diffuse).
    pub specificity: f32,
    /// Axis 3: value in `[value_neutral, 1.0]`, monotone in positive evidence.
    /// v1 is a constant `1.0` (attribution-only pass-through).
    pub value: f32,
    /// Axis 2: negative-transfer proximity in `[0,1]`. `0.0` = no proximity to
    /// any rejected/dormant exemplar (neutral).
    pub neg_penalty: f32,
}

impl Default for SalienceFactors {
    fn default() -> Self {
        Self {
            specificity: 1.0,
            value: 1.0,
            neg_penalty: 0.0,
        }
    }
}

/// Hot-path `Copy` snapshot of the `[salience]` knobs the scorer reads on
/// every `compute_score_parts` call. Derived once from [`SalienceSettings`]
/// at construction so the scorer never touches the full config or a lock.
// The master flag + three per-axis toggles are deliberately separate bools
// mirroring the `[salience]` config keys — collapsing them into a bitflag or
// enum would obscure the 1:1 mapping the operator tunes.
#[allow(clippy::struct_excessive_bools)]
#[derive(Debug, Clone, Copy)]
pub struct SalienceScorer {
    pub scorer_v2: bool,
    pub specificity_enabled: bool,
    pub value_enabled: bool,
    pub negative_enabled: bool,
    pub specificity_lift_cutoff: f32,
    pub neg_gamma: f32,
    pub negative_lift_cutoff: f32,
    pub damper_floor: f32,
}

impl Default for SalienceScorer {
    /// Neutral: flag off ⇒ every axis gate is inert and the scorer is v1.
    fn default() -> Self {
        Self {
            scorer_v2: false,
            specificity_enabled: false,
            value_enabled: false,
            negative_enabled: false,
            specificity_lift_cutoff: 0.2,
            neg_gamma: 0.8,
            negative_lift_cutoff: 0.5,
            damper_floor: 0.02,
        }
    }
}

/// Sanitize a config knob into `[lo, hi]`, mapping NaN to `default`. The
/// scorer reads these on every `compute_score_parts` call and feeds
/// `damper_floor` straight into `f32::clamp(eps, 1.0)`, which **panics** when
/// `eps > 1.0` or `eps` is NaN (`min > max`). A misconfigured `[salience]`
/// block must never panic the scorer — least of all flag-off, where the
/// `damp` clamp still runs (review Finding 1). Sanitizing here, at the single
/// hot-path snapshot, guarantees `compute_score_parts` only ever sees a valid
/// range regardless of what the operator wrote.
fn sanitize(v: f32, lo: f32, hi: f32, default: f32) -> f32 {
    if v.is_nan() { default } else { v.clamp(lo, hi) }
}

impl From<&SalienceSettings> for SalienceScorer {
    fn from(s: &SalienceSettings) -> Self {
        // When the master flag is off, force every axis gate off too so the
        // scorer is unconditionally v1 regardless of the per-axis toggles.
        //
        // Knob sanitization (review Finding 1): every `f32` the scorer feeds
        // into a `clamp`/multiply is range-checked here so a bad config can't
        // panic or distort the hot path. `damper_floor` is held strictly below
        // 1.0 (it is the *floor* of a `[eps, 1.0]` clamp — `eps == 1.0` would
        // pin every damper to 1.0, and `eps > 1.0` panics); the lift cutoffs
        // and γ are clamped to their documented `[0, 1]` ranges. Defaults
        // mirror `SalienceScorer::default()` / the config defaults.
        Self {
            scorer_v2: s.scorer_v2,
            specificity_enabled: s.scorer_v2 && s.specificity_enabled,
            value_enabled: s.scorer_v2 && s.value_enabled,
            negative_enabled: s.scorer_v2 && s.negative_enabled,
            specificity_lift_cutoff: sanitize(s.specificity_lift_cutoff, 0.0, 1.0, 0.2),
            neg_gamma: sanitize(s.neg_gamma, 0.0, 1.0, 0.8),
            negative_lift_cutoff: sanitize(s.negative_lift_cutoff, 0.0, 1.0, 0.5),
            // Strictly < 1.0 so `clamp(damper_floor, 1.0)` is always valid.
            damper_floor: sanitize(s.damper_floor, 0.0, 1.0 - f32::EPSILON, 0.02),
        }
    }
}

/// Source-document identity for a corpus chunk, used as the entropy histogram
/// bin (specificity, axis 1) and — via `source` + `source_id` — the
/// judgment-propagation coordinate (value, axis 3). Cross-project spread is
/// *more* diffuse, so specificity keys on the `(project, source_id)` pair (R1
/// §5-Q2) — a handle that appears in both haystack and ostk-recall gets
/// strictly more bins than one confined to a single project.
///
/// `source` is the chunk's `Source::as_str()` (e.g. `"ostk_decision"`,
/// `"code"`). Value's `value_judgment` reads it to (i) detect evidence links
/// resolving to a curated `ostk_decision`/`ostk_needle` chunk and (ii) build
/// the `(source, source_id)` coordinate that intersects
/// `concept_support_by_coord` (R2 §b). Specificity ignores it.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct SourceMeta {
    pub project: Option<String>,
    pub source: String,
    pub source_id: String,
}

/// Shannon entropy `H = −Σ pᵢ ln pᵢ` of a weight distribution.
///
/// Normalizes the (non-negative) weights to a probability distribution
/// internally and returns the *raw* entropy in nats; callers divide by their
/// own `ln(N_eff)` (specificity wants `1 − H/ln(N_eff)`; the surface-health
/// metric wants `H/ln(N)`), so this stays normalization-agnostic. Empty,
/// single-bin, or all-zero input → `0.0` (a degenerate distribution carries
/// no uncertainty).
#[must_use]
pub fn shannon_entropy(weights: &[f32]) -> f32 {
    let total: f32 = weights.iter().filter(|w| **w > 0.0).sum();
    if total <= 0.0 {
        return 0.0;
    }
    let mut h = 0.0f32;
    for &w in weights {
        if w > 0.0 {
            let p = w / total;
            h -= p * p.ln();
        }
    }
    // Clamp tiny negative values from float rounding (e.g. a single bin with
    // p = 1.0 yields −1.0·ln(1.0) = 0.0, but accumulation can underflow).
    h.max(0.0)
}

/// Specificity `= 1 − H/ln(N_eff)` over a handle's co-occurrence distribution.
///
/// Builds a `(project, source_id)` histogram from the handle's Active evidence
/// links (one bin per distinct source document, counted by resonating chunks),
/// computes [`shannon_entropy`], and normalizes by `ln(N_eff)` where `N_eff`
/// is the number of populated bins. Result is in `[0,1]`: `1.0` = all
/// resonances concentrate in one document (maximally discriminating); `→0` =
/// resonances spread evenly across many documents (diffuse coherent noise).
///
/// Guards (so the multiplier never punishes thin-but-real concepts):
/// - `N_eff ≤ 1` ⇒ `1.0` (one document is maximally specific; `ln(1) = 0`,
///   avoid div-by-zero).
/// - total resonating chunks `< specificity_min_evidence` ⇒ `1.0` (neutral —
///   not enough evidence to call it diffuse; mirrors `STOPWORD_MENTION_MIN`).
/// - no evidence / no resolvable source ⇒ `1.0` (neutral, never a damp).
// `n_eff` is a small bin count (number of distinct source docs a handle
// resonated with); the `as f32` for `ln(N_eff)` is well within mantissa
// range. `implicit_hasher`: the boot caller always passes a default-hasher
// `HashMap`, so generalizing the signature buys nothing.
/// Entropy CORE of the specificity axis — the source-agnostic `1 − H/ln(N_eff)`
/// over a per-handle `(project, source_id)` co-occurrence histogram.
///
/// `bin_weights` is the per-document count (one entry per distinct source doc
/// the handle co-occurs with); `total` is their sum (the total co-occurrence
/// count). Both the corpus-token co-occurrence path (the live input, recal §2)
/// and the legacy evidence-link path build this histogram and call here, so the
/// entropy math + its guards live in one place:
/// - `total < specificity_min_evidence` ⇒ `1.0` (not enough signal to judge —
///   never punish a thin-but-real concept on noise);
/// - `N_eff ≤ 1` (one bin) ⇒ `1.0` (one document is maximally specific);
/// - otherwise `1 − H/ln(N_eff)` in `[0,1]`: peaked → high, flat → low.
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn specificity_from_histogram(bin_weights: &[f32], total: u32, cfg: &SalienceSettings) -> f32 {
    if total < cfg.specificity_min_evidence {
        return 1.0; // not enough co-occurrence to judge — neutral
    }
    let n_eff = bin_weights.iter().filter(|w| **w > 0.0).count();
    if n_eff <= 1 {
        return 1.0; // single document ⇒ maximally specific
    }
    let h = shannon_entropy(bin_weights);
    let h_max = (n_eff as f32).ln();
    if h_max <= 0.0 {
        return 1.0;
    }
    (1.0 - h / h_max).clamp(0.0, 1.0)
}

/// Specificity over a handle's PROJECT-count distribution, normalized by the
/// GLOBAL project count (recal approach 1, with the lead's formula correction).
///
/// `project_counts` is the handle's per-project match count (one entry per
/// distinct project the handle's phrase appears in); `n_projects_global` is the
/// total number of distinct projects in the corpus. Returns
/// `1 − H(project_dist) / ln(n_projects_global)` in `[0,1]`.
///
/// **Why GLOBAL normalization, not `ln(N_eff)` (load-bearing):** normalizing by
/// the handle's OWN span discards the span signal — a concept even across 6
/// projects and plumbing even across 14 both give `1 − ln(k)/ln(k) = 0`, so the
/// 6-vs-14 discrimination the diagnostic proved is normalized away. Against the
/// fixed global denominator `ln(N_global)`, span survives:
/// concentrated-in-1 → `H=0` → `1.0` (specific); even-across-6 →
/// `1 − ln 6/ln 23 ≈ 0.43`; even-across-14 → `1 − ln 14/ln 23 ≈ 0.16` (diffuse).
/// Captures BOTH span and skew, globally comparable across handles.
///
/// No `min_evidence` guard: for an IDF-like signal the sparse end is correct on
/// its own — a handle in 1 project → `H=0` → `1.0` (maximally specific); 0
/// projects (empty) → `1.0` (caller's default; coined-unwritten handle → rests
/// on the other axes).
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn specificity_from_project_dist(project_counts: &[f32], n_projects_global: usize) -> f32 {
    let n_eff = project_counts.iter().filter(|w| **w > 0.0).count();
    if n_eff <= 1 {
        return 1.0; // in ≤1 project ⇒ maximally specific
    }
    // RAW project SPAN, not entropy. The diagnostic proved distinct project
    // COUNT separates concepts (≤6) from plumbing (8–14) — but entropy weights
    // by match VOLUME, and every handle is bursty in its home project, so the
    // skew re-conflates them (concepts and plumbing both land ~0.6–0.7). Span
    // ignores volume: a handle in `s` of `N` projects scores `1 − (s−1)/(N−1)`
    // — in 1 project → 1.0 (specific), in all → 0.0 (generic), linear between.
    // This is the signal the diagnostic measured directly.
    let n_global = n_projects_global.max(n_eff);
    if n_global <= 1 {
        return 1.0;
    }
    let span_frac = (n_eff as f32 - 1.0) / (n_global as f32 - 1.0);
    (1.0 - span_frac).clamp(0.0, 1.0)
}

/// Specificity over a handle's **evidence-link** co-occurrence distribution
/// (legacy input — kept for tests and as a fallback). Builds the `(project,
/// source_id)` histogram from the handle's Active evidence links, then defers
/// to [`specificity_from_histogram`].
///
/// NOTE: on the real ledger this input is too sparse to fire (evidence_links
/// is a curated high-cosine weaver artifact, < `specificity_min_evidence` per
/// handle), which is why the boot pass now feeds the entropy core from
/// `CorpusStore::token_cooccurrence` instead (recal §2). This fn remains the
/// reference implementation the unit tests pin.
#[must_use]
#[allow(clippy::implicit_hasher)]
pub fn specificity_from_evidence(
    evidence: &[EvidenceLink],
    chunk_meta: &HashMap<String, SourceMeta>,
    cfg: &SalienceSettings,
) -> f32 {
    let mut hist: HashMap<&SourceMeta, f32> = HashMap::new();
    let mut total_chunks: u32 = 0;
    for link in evidence {
        // Only Active links count — a broken/moved reference no longer
        // attests co-occurrence.
        if link.relation_state != RelationState::Active {
            continue;
        }
        // The resonating chunk-id is `last_resolved_chunk_id` (current) or the
        // original path (the chunk-id at weave time); prefer the resolved one.
        let chunk_id = link
            .last_resolved_chunk_id
            .as_deref()
            .or_else(|| link.original_path.to_str());
        let Some(chunk_id) = chunk_id else { continue };
        let Some(meta) = chunk_meta.get(chunk_id) else {
            continue;
        };
        *hist.entry(meta).or_insert(0.0) += 1.0;
        total_chunks += 1;
    }
    let weights: Vec<f32> = hist.values().copied().collect();
    specificity_from_histogram(&weights, total_chunks, cfg)
}

// --- value / use-feedback + judgment propagation (THESIS axis 3 / R2) --
//
// Ground salience in what was actually USED. Two composable terms, combined
// into a bounded positive lift that is **monotone in positive evidence**
// (design §4.5 — the lead-gate invariant): evidence only raises value toward
// 1.0, never below the no-evidence neutral point.
//
//   value_use      — the click-through loop (WEAK fidelity): a thread whose
//                    evidence chunks were explicitly recalled / operator-
//                    selected. REUSES `activation.rs` curves verbatim
//                    (act_r_base / squash / age_hours_floored / ACT_R_DECAY_D /
//                    AccessWeights) — no new curve.
//   value_judgment — curated-confidence propagation: a thread whose Active
//                    evidence resolves to an `ostk_decision`/`ostk_needle`
//                    chunk (j_evidence), or whose evidence coordinates a live
//                    active concept also cites (j_concept). No new join table.
//
//   positive = clamp01(w_use·value_use + w_judg·value_judgment)
//   value    = value_neutral + (1 − value_neutral)·positive
//
// **v1 ships `value_neutral = 1.0`** ⇒ value is a constant-1.0 pass-through
// (attribution-only): the join runs, the `why` carries the positive evidence,
// but the multiplier is inert. There is no valid *negative* (wasted) signal in
// v1 — that needs the deferred `ThreadSurfaced` event (design §4.4) — so value
// cannot damp below neutral and the invariant holds trivially. The
// ceiling-raiser regime (`value_neutral < 1.0`) unlocks in the second
// increment.

/// The curated source kinds whose evidence links propagate operator judgment
/// confidence onto a co-occurring thread (R2 §b). Matched against a chunk's
/// `Source::as_str()`.
const JUDGMENT_SOURCES: &[&str] = &["ostk_decision", "ostk_needle"];

/// `value_use(H)` — the click-through loop, `[0,1)`.
///
/// `squash(Σ over distinct (query_hash, kind) of weight_of(kind)·act_r_base(age))`.
/// The distinct-query gate already collapsed the buckets upstream
/// (`ThreadsDb::surfaced_vs_used`); here each surviving [`UsedAccess`] bucket
/// contributes its `AccessWeights` weight aged by the ACT-R base-activation
/// curve. Empty list (no used accesses) ⇒ `0.0` (no signal — NOT a damp).
#[must_use]
pub fn value_use(used: &[UsedAccess], now: DateTime<Utc>) -> f32 {
    if used.is_empty() {
        return 0.0;
    }
    let weights = AccessWeights::default();
    // Σ weight_of(kind) · age^{-d}, then squash to [0,1). One age term per
    // distinct bucket — `act_r_base` takes the ages; we pre-scale each age's
    // contribution by its access weight by summing weighted single-age bases
    // (a heavier access kind contributes a larger base-activation increment).
    let raw: f32 = used
        .iter()
        .map(|a| {
            let age = age_hours_floored(now, a.ts, 1.0);
            weights.weight_of(a.kind) * act_r_base(&[age], ACT_R_DECAY_D)
        })
        .sum();
    squash(raw)
}

/// `value_judgment(H)` — curated-confidence propagation, `[0,1]`.
///
/// `1 − exp(−Σ)` (the `edge_lift_for` saturation — many weak judgment links ≈
/// one strong one, bounded to 1) over two contributions:
/// - **j_evidence**: each Active evidence link of `H` whose resolved chunk has
///   `source ∈ {ostk_decision, ostk_needle}` contributes its link `similarity`
///   (the weaver-recorded cosine; `0.5` fallback when absent). READS the
///   already-materialized `Derived` links — no new similarity computation.
/// - **j_concept**: each of `H`'s evidence coordinates `(source, source_id)`
///   that a live **active** concept also cites contributes that concept's
///   support activation (`active_coords` = `concept_support_by_coord`, where
///   active concepts carry `confidence = 1.0`).
///
/// `0.0` when `H` has no judgment-sourced evidence and touches no active
/// coordinate (the common case — most threads neither).
// `chunk_meta` is always a default-hasher map from the boot caller;
// generalizing the hasher buys nothing (implicit_hasher).
#[must_use]
#[allow(clippy::implicit_hasher)]
pub fn value_judgment(
    evidence: &[EvidenceLink],
    chunk_meta: &HashMap<String, SourceMeta>,
    active_coords: &HashMap<(String, String), f32>,
) -> f32 {
    let mut sum = 0.0_f32;
    for link in evidence {
        if link.relation_state != RelationState::Active {
            continue;
        }
        let chunk_id = link
            .last_resolved_chunk_id
            .as_deref()
            .or_else(|| link.original_path.to_str());
        let Some(chunk_id) = chunk_id else { continue };
        let Some(meta) = chunk_meta.get(chunk_id) else {
            continue;
        };
        // (i) j_evidence: link resolves to a curated decision/needle chunk.
        if JUDGMENT_SOURCES.contains(&meta.source.as_str()) {
            // The weaver records the resonance cosine on the link; fall back to
            // a mid weight when an older curated link carries no similarity.
            sum += link.similarity.unwrap_or(0.5).max(0.0);
        }
        // (ii) j_concept: this coordinate is cited by a live active concept.
        if let Some(activation) = active_coords.get(&(meta.source.clone(), meta.source_id.clone())) {
            sum += activation.max(0.0);
        }
    }
    if sum <= 0.0 {
        return 0.0;
    }
    1.0 - (-sum).exp()
}

/// `value(H) ∈ [value_neutral, 1.0]` — the axis-3 floor multiplier (design
/// §4.3 / §4.5). Monotone non-decreasing in positive evidence by construction.
///
/// `positive = clamp01(w_use·value_use + w_judg·value_judgment)`;
/// `value = value_neutral + (1 − value_neutral)·positive`. With the shipped
/// `value_neutral = 1.0` this is a constant `1.0` regardless of evidence (the
/// protective pass-through — positive evidence is attribution-only); with
/// `value_neutral < 1.0` (the deferred ceiling-raiser) evidence raises a
/// proven handle back toward 1.0 while an unproven one sits at the neutral
/// floor.
#[must_use]
pub fn value_from(
    used: &[UsedAccess],
    evidence: &[EvidenceLink],
    chunk_meta: &HashMap<String, SourceMeta>,
    active_coords: &HashMap<(String, String), f32>,
    now: DateTime<Utc>,
    cfg: &SalienceSettings,
) -> f32 {
    let v_use = value_use(used, now);
    let v_judg = value_judgment(evidence, chunk_meta, active_coords);
    let positive = (cfg.value_w_use * v_use + cfg.value_w_judg * v_judg).clamp(0.0, 1.0);
    let v_neutral = cfg.value_neutral.clamp(0.0, 1.0);
    (v_neutral + (1.0 - v_neutral) * positive).clamp(0.0, 1.0)
}

// --- negative-transfer (THESIS axis 2 / R3) ----------------------------
//
// Damp a handle's salience by its embedding-proximity to the set of
// rejected concepts + demoted/dormant thread handles. The two corrections
// R3 proved MANDATORY (raw cosine to a single centroid is useless —
// separation +0.0195, the `potion` space is anisotropic):
//   1. mean-CENTER the space first (5× separation improvement on its own);
//   2. kNN to the nearest exemplars (k=3, AUC 0.917) — a single centroid
//      still overlaps, and k=1 (0.957) is brittle (the `ostk-cache`
//      centered-cosine-1.0 collision with rejected sub-terms).
// The penalty is consumed as a SOFT, bounded multiplicative damp in
// `compute_score_parts` (`1 − γ·penalty`, γ < 1), never a hard gate, so a
// real concept reusing rejected sub-vocab stays recoverable on fresh
// resonance.

/// L2-normalize a vector. Zero-norm / empty → returned as-is (all-zero stays
/// all-zero; downstream `cosine_similarity` treats it as "no contribution").
#[must_use]
fn normalize(v: &[f32]) -> Vec<f32> {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        v.iter().map(|x| x / norm).collect()
    } else {
        v.to_vec()
    }
}

/// Mean-center then re-normalize: `normalize(normalize(v) − global_mean)`.
///
/// R3 §TL;DR: this is MANDATORY. The embedding space has a strong common
/// component (live global-mean anchor norm ≈ 0.62), so raw cosine to any
/// centroid is dominated by that shared direction and discriminates nothing.
/// Subtracting the (already-normalized) global mean removes it. A `global_mean`
/// of the wrong dimension (or empty) is a no-op beyond the inner normalize —
/// the safe choice, matching `cosine_similarity`'s dimension-mismatch contract.
#[must_use]
pub fn center(v: &[f32], global_mean: &[f32]) -> Vec<f32> {
    let n = normalize(v);
    if global_mean.len() != n.len() {
        return n;
    }
    let centered: Vec<f32> = n
        .iter()
        .zip(global_mean.iter())
        .map(|(x, m)| x - m)
        .collect();
    normalize(&centered)
}

/// Negative-transfer penalty in `[0,1]` for a handle's anchor (R3 §3).
///
/// The τ-floored, rescaled mean of the top-`k` cosines to the centered
/// negative exemplars. `0.0` = far from every rejected/dormant exemplar
/// (neutral); `1.0` = a near-perfect twin of the rejected centroid.
///
/// - `exemplars` are **already centered** (built once at boot — design §2.2
///   pass B); only the query `anchor` is centered here.
/// - `k = cfg.negative_knn_k` (default 3). k=1 is intentionally NOT the
///   default: it is brittle to a single colliding exemplar (the `ostk-cache`
///   case, R3 §3).
/// - `τ = cfg.negative_tau` (default 0.45): proximity below τ is ignored
///   (centered noise sits ~0..0.4), proximity above τ scales linearly to 1.0.
///
/// Neutral (`0.0`) when there are no exemplars or the anchor is empty — so a
/// cold substrate, or a thread with no anchor, is never penalized.
#[must_use]
pub fn negative_penalty(
    anchor: &[f32],
    global_mean: &[f32],
    exemplars: &[Vec<f32>],
    cfg: &SalienceSettings,
) -> f32 {
    if exemplars.is_empty() || anchor.is_empty() {
        return 0.0;
    }
    let a = center(anchor, global_mean);
    let mut sims: Vec<f32> = exemplars
        .iter()
        .map(|e| crate::cosine_similarity(&a, e))
        .collect();
    // Descending: largest cosines first, take the k nearest.
    sims.sort_unstable_by(|x, y| y.partial_cmp(x).unwrap_or(std::cmp::Ordering::Equal));
    let k = cfg.negative_knn_k.clamp(1, sims.len());
    #[allow(clippy::cast_precision_loss)] // k is small (default 3)
    let prox = sims[..k].iter().sum::<f32>() / k as f32;
    let tau = cfg.negative_tau;
    // τ-floored linear rescale to [0,1]. Guard τ ≈ 1.0 (degenerate config).
    let denom = (1.0 - tau).max(f32::EPSILON);
    ((prox - tau) / denom).clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use chrono::Utc;
    use ostk_recall_store::{AssociationType, EvidenceLink, RelationState};

    use super::*;

    // --- shannon_entropy -------------------------------------------------

    #[test]
    fn entropy_single_bin_is_zero() {
        assert_eq!(shannon_entropy(&[5.0]), 0.0);
        assert_eq!(shannon_entropy(&[5.0, 0.0, 0.0]), 0.0);
    }

    #[test]
    fn entropy_empty_or_zero_is_zero() {
        assert_eq!(shannon_entropy(&[]), 0.0);
        assert_eq!(shannon_entropy(&[0.0, 0.0]), 0.0);
    }

    #[test]
    fn entropy_uniform_is_ln_n() {
        // N uniform bins ⇒ H = ln(N).
        let four = [1.0, 1.0, 1.0, 1.0];
        assert!((shannon_entropy(&four) - 4.0f32.ln()).abs() < 1e-5);
        let ten = [2.0; 10]; // equal weights, any magnitude
        assert!((shannon_entropy(&ten) - 10.0f32.ln()).abs() < 1e-5);
    }

    #[test]
    fn entropy_peaked_is_low() {
        // One dominant bin + a thin tail ⇒ entropy well below the uniform max.
        let peaked = [100.0, 1.0, 1.0, 1.0];
        let uniform = [1.0, 1.0, 1.0, 1.0];
        assert!(shannon_entropy(&peaked) < shannon_entropy(&uniform) * 0.3);
    }

    // --- specificity_from_evidence --------------------------------------

    fn cfg() -> SalienceSettings {
        // The entropy-math tests use small synthetic histograms; pin a low
        // `specificity_min_evidence` so they exercise the FORMULA, not the
        // production token-hit floor (20). The `*_below_min_evidence_*` test
        // sets its own threshold to exercise the floor itself.
        let mut s = SalienceSettings::default();
        s.specificity_min_evidence = 2;
        s
    }

    fn link(chunk_id: &str) -> EvidenceLink {
        let now = Utc::now();
        EvidenceLink {
            id: 0,
            thread_handle: ostk_recall_core::ThreadHandle::new("x").unwrap(),
            original_path: PathBuf::from(chunk_id),
            current_path: None,
            content_hash: None,
            last_resolved_chunk_id: Some(chunk_id.to_string()),
            relation_state: RelationState::Active,
            association_type: AssociationType::Derived,
            category: "code".to_string(),
            similarity: Some(0.9),
            created_at: now,
            updated_at: now,
            touch_count: 1,
            last_touched_at: now,
        }
    }

    fn meta(project: &str, source_id: &str) -> SourceMeta {
        meta_src(project, "code", source_id)
    }

    fn meta_src(project: &str, source: &str, source_id: &str) -> SourceMeta {
        SourceMeta {
            project: Some(project.to_string()),
            source: source.to_string(),
            source_id: source_id.to_string(),
        }
    }

    #[test]
    fn specificity_from_project_dist_global_norm_captures_span() {
        // recal approach 1 + the global-normalization fix. The live diagnostic
        // measured concepts in ≤6 projects, plumbing in 8–14, out of N=23.
        // Normalizing by the GLOBAL count (not the handle's own span) must keep
        // the span signal: a concept spread evenly across 6 projects out-ranks
        // plumbing spread across 14 — both would be 0 under ln(N_eff) norm.
        let global = 23;
        let concept = vec![1.0_f32; 6]; // cognitive-memory: 6 projects, even
        let plumbing = vec![1.0_f32; 14]; // in-memory: 14 projects, even
        let s_concept = specificity_from_project_dist(&concept, global);
        let s_plumbing = specificity_from_project_dist(&plumbing, global);
        // RAW SPAN: 1 − (span−1)/(N−1). concept 6/23 → 1−5/22 ≈ 0.773;
        // plumbing 14/23 → 1−13/22 ≈ 0.409. Span ignores match volume, so the
        // 6-vs-14 separation survives regardless of skew.
        assert!((s_concept - (1.0 - 5.0 / 22.0)).abs() < 1e-5);
        assert!((s_plumbing - (1.0 - 13.0 / 22.0)).abs() < 1e-5);
        assert!(
            s_concept > s_plumbing + 0.2,
            "raw span must keep the 6-vs-14 separation: concept {s_concept} vs plumbing {s_plumbing}"
        );
        // Concentrated in one project ⇒ maximally specific (regardless of volume).
        assert_eq!(specificity_from_project_dist(&[9.0], global), 1.0);
        // A SKEWED 14-project handle still reads generic by span — the property
        // entropy lost: volume concentration must NOT rescue a broad-span term.
        let skewed14: Vec<f32> = {
            let mut v = vec![1.0_f32; 13];
            v.push(900.0); // bursty home project
            v
        };
        let s_skewed = specificity_from_project_dist(&skewed14, global);
        assert!(
            (s_skewed - s_plumbing).abs() < 1e-5,
            "span ignores volume: a skewed 14-project handle scores like an even \
             14-project one ({s_skewed} vs {s_plumbing}) — the skew can't rescue it"
        );
        // Empty (coined-unwritten handle) ⇒ neutral 1.0 (rests on other axes).
        assert_eq!(specificity_from_project_dist(&[], global), 1.0);
    }

    #[test]
    fn specificity_from_histogram_discriminates_plumbing_vs_concept() {
        // recal §2: the corpus-token co-occurrence histogram is the live input.
        // PLUMBING (`top-level`/`in-memory`) phrase-matches across MANY docs →
        // flat histogram → LOW specificity. A CONCEPT (`cognitive-memory`)
        // concentrates in a few docs → peaked → HIGH. Drive the entropy core
        // directly on the two histogram shapes.
        let c = cfg(); // min_evidence 2

        // Plumbing: 100 hits spread evenly across 50 distinct docs.
        let plumbing: Vec<f32> = vec![2.0; 50];
        let s_plumbing = specificity_from_histogram(&plumbing, 100, &c);
        assert!(
            s_plumbing < 0.05,
            "plumbing spread across 50 docs should be ~0 specificity, got {s_plumbing}"
        );

        // Concept: 100 hits, 90 in one doc + a thin 10 across 5 others.
        let mut concept = vec![90.0];
        concept.extend(std::iter::repeat(2.0).take(5));
        let s_concept = specificity_from_histogram(&concept, 100, &c);
        assert!(
            s_concept > 0.7 && s_concept > s_plumbing,
            "concentrated concept should be HIGH and well above plumbing: \
             concept {s_concept} vs plumbing {s_plumbing}"
        );

        // Single doc ⇒ maximally specific.
        assert_eq!(specificity_from_histogram(&[100.0], 100, &c), 1.0);
        // Below the floor ⇒ neutral regardless of spread.
        let mut hi = cfg();
        hi.specificity_min_evidence = 50;
        assert_eq!(specificity_from_histogram(&[1.0, 1.0, 1.0], 3, &hi), 1.0);
    }

    #[test]
    fn specificity_single_document_is_one() {
        // 6 resonances (clears min_evidence), all in ONE source doc → 1.0.
        let ev: Vec<_> = (0..6).map(|i| link(&format!("c{i}"))).collect();
        let mut cm = HashMap::new();
        for i in 0..6 {
            cm.insert(format!("c{i}"), meta("haystack", "src/derefer.rs"));
        }
        assert_eq!(specificity_from_evidence(&ev, &cm, &cfg()), 1.0);
    }

    #[test]
    fn specificity_uniform_spread_is_near_zero() {
        // 8 resonances, each in a DISTINCT source doc → maximally diffuse.
        let ev: Vec<_> = (0..8).map(|i| link(&format!("c{i}"))).collect();
        let mut cm = HashMap::new();
        for i in 0..8 {
            cm.insert(format!("c{i}"), meta("p", &format!("doc{i}.jsonl")));
        }
        let s = specificity_from_evidence(&ev, &cm, &cfg());
        assert!(s < 0.05, "uniform-8 spread should be ~0, got {s}");
    }

    #[test]
    fn specificity_peaked_ranks_above_uniform_same_bins() {
        // 10 resonances, 8 in one doc + 2 scattered → distribution [8,1,1]
        // over 3 bins. The monotonic property we care about: a *peaked*
        // 3-bin distribution is strictly more specific than a *uniform* 3-bin
        // one (which is 0.0), and well clear of the diffuse floor. (The raw
        // value here is ~0.42 — moderate by construction, since two of ten
        // resonances do leak elsewhere; the invariant is the ordering.)
        let peaked: Vec<_> = (0..10).map(|i| link(&format!("c{i}"))).collect();
        let mut cm = HashMap::new();
        for i in 0..8 {
            cm.insert(format!("c{i}"), meta("p", "main-doc.md"));
        }
        cm.insert("c8".to_string(), meta("p", "other-a.md"));
        cm.insert("c9".to_string(), meta("p", "other-b.md"));
        let s_peaked = specificity_from_evidence(&peaked, &cm, &cfg());

        // Uniform over the same 3 docs (specificity 0.0 by construction).
        let uniform: Vec<_> = (0..9).map(|i| link(&format!("u{i}"))).collect();
        let mut um = HashMap::new();
        for i in 0..3 {
            for j in 0..3 {
                um.insert(format!("u{}", i * 3 + j), meta("p", &format!("d{i}.md")));
            }
        }
        let s_uniform = specificity_from_evidence(&uniform, &um, &cfg());

        assert!(s_uniform < 0.05, "uniform-3 should be ~0, got {s_uniform}");
        assert!(
            s_peaked > 0.3 && s_peaked > s_uniform,
            "peaked [8,1,1] must rank above uniform-3: peaked {s_peaked} vs uniform {s_uniform}"
        );
    }

    #[test]
    fn specificity_below_min_evidence_is_neutral() {
        // 3 resonances across 3 docs — would read diffuse, but with a
        // min_evidence floor ABOVE the count there isn't enough signal to
        // judge → neutral 1.0. (Exercises the floor itself; uses an explicit
        // threshold of 5 rather than `cfg()`'s test-low 2.)
        let mut c = cfg();
        c.specificity_min_evidence = 5;
        let ev: Vec<_> = (0..3).map(|i| link(&format!("c{i}"))).collect();
        let mut cm = HashMap::new();
        for i in 0..3 {
            cm.insert(format!("c{i}"), meta("p", &format!("doc{i}.md")));
        }
        assert_eq!(specificity_from_evidence(&ev, &cm, &c), 1.0);
    }

    #[test]
    fn specificity_ignores_non_active_and_unresolvable_links() {
        // A broken link and a link whose chunk has no metadata are both
        // skipped; remaining 5 Active+resolvable resonances in one doc → 1.0.
        let mut ev: Vec<_> = (0..5).map(|i| link(&format!("c{i}"))).collect();
        let mut broken = link("c-broken");
        broken.relation_state = RelationState::BrokenReference;
        ev.push(broken);
        ev.push(link("c-no-meta")); // no chunk_meta entry
        let mut cm = HashMap::new();
        for i in 0..5 {
            cm.insert(format!("c{i}"), meta("p", "one.md"));
        }
        cm.insert("c-broken".to_string(), meta("p", "broken.md"));
        assert_eq!(specificity_from_evidence(&ev, &cm, &cfg()), 1.0);
    }

    // --- value / use-feedback + judgment (THESIS axis 3 / R2) -----------

    use ostk_recall_store::{AccessKind, UsedAccess};

    fn used(kind: AccessKind) -> UsedAccess {
        UsedAccess {
            kind,
            ts: Utc::now(),
        }
    }

    /// An Active evidence link resolving to `chunk_id`, with the given recorded
    /// similarity (the weaver cosine value_judgment reads).
    fn judg_link(chunk_id: &str, similarity: f32) -> EvidenceLink {
        let mut l = link(chunk_id);
        l.similarity = Some(similarity);
        l
    }

    #[test]
    fn value_monotone_in_positive_evidence() {
        // THE invariant (design §4.5): A has positive evidence (a used access
        // AND a decision-sourced link); B has none. value(A) ≥ value(B) for
        // EVERY value_neutral — the regression guard against "average a zero
        // down." Asserted on the value scalar directly (the scorer's `val`
        // multiplier is monotone in it, so score monotonicity follows — and is
        // covered end-to-end by the lib scorer tests).
        let a_used = vec![used(AccessKind::ExplicitRecall)];
        let a_ev = vec![judg_link("dec-1", 0.9)];
        let mut cm = HashMap::new();
        cm.insert("dec-1".to_string(), meta_src("haystack", "ostk_decision", "some_decision"));
        let empty_coords: HashMap<(String, String), f32> = HashMap::new();
        let now = Utc::now();

        for vn in [1.0_f32, 0.7] {
            let mut c = cfg();
            c.value_neutral = vn;
            let va = value_from(&a_used, &a_ev, &cm, &empty_coords, now, &c);
            // B: no used accesses, no evidence at all.
            let vb = value_from(&[], &[], &cm, &empty_coords, now, &c);
            assert!(
                va >= vb,
                "value monotone in evidence at value_neutral={vn}: A {va} < B {vb}"
            );
            assert!(vb >= vn - 1e-6, "unevidenced handle sits at the neutral floor");
        }
    }

    #[test]
    fn value_v1_is_pass_through() {
        // Shipped v1 config (value_neutral = 1.0): any handle — judged, used,
        // or neither — gets exactly 1.0, so the floor multiplier is inert (no
        // double-count with specificity/negative).
        let c = cfg(); // default value_neutral = 1.0
        assert!((c.value_neutral - 1.0).abs() < f32::EPSILON, "v1 default");
        let now = Utc::now();
        let coords: HashMap<(String, String), f32> = HashMap::new();
        let mut cm = HashMap::new();
        cm.insert("dec-1".to_string(), meta_src("p", "ostk_decision", "d"));

        // Heavily-evidenced handle.
        let proven = value_from(
            &[used(AccessKind::ExplicitRecall), used(AccessKind::OperatorSelected)],
            &[judg_link("dec-1", 0.95)],
            &cm,
            &coords,
            now,
            &c,
        );
        // Evidence-less handle.
        let bare = value_from(&[], &[], &cm, &coords, now, &c);
        assert!((proven - 1.0).abs() < 1e-6, "v1 proven handle is 1.0, got {proven}");
        assert!((bare - 1.0).abs() < 1e-6, "v1 bare handle is 1.0, got {bare}");
    }

    #[test]
    fn value_judgment_raises_proven_handle_when_unlocked() {
        // With value_neutral = 0.7 (the deferred ceiling-raiser regime): a
        // thread whose Active evidence resolves to an ostk_decision chunk is
        // raised ABOVE 0.7 (toward 1.0); a plumbing-only thread stays at 0.7;
        // the judged thread out-ranks. Lift direction is UP.
        let mut c = cfg();
        c.value_neutral = 0.7;
        let now = Utc::now();
        let coords: HashMap<(String, String), f32> = HashMap::new();

        // Judged: evidence link to a decision chunk.
        let mut cm = HashMap::new();
        cm.insert("dec-1".to_string(), meta_src("haystack", "ostk_decision", "dereference_or_void"));
        let judged = value_from(&[], &[judg_link("dec-1", 0.9)], &cm, &coords, now, &c);

        // Plumbing: an Active link to an ordinary code chunk (no judgment, no use).
        let mut pm = HashMap::new();
        pm.insert("code-1".to_string(), meta_src("p", "code", "src/turn.rs"));
        let plumbing = value_from(&[], &[judg_link("code-1", 0.9)], &pm, &coords, now, &c);

        assert!(judged > 0.7, "decision-cited handle is raised above neutral: {judged}");
        assert!((plumbing - 0.7).abs() < 1e-6, "plumbing-only stays at neutral: {plumbing}");
        assert!(judged > plumbing, "judged out-ranks plumbing: {judged} vs {plumbing}");
    }

    #[test]
    fn value_judgment_lifts_via_active_concept_coordinate() {
        // The j_concept path: a thread whose evidence coordinate is cited by a
        // live active concept (concept_support_by_coord) inherits that support
        // even with no decision/needle link. Unlocked regime.
        let mut c = cfg();
        c.value_neutral = 0.7;
        let now = Utc::now();
        let mut cm = HashMap::new();
        cm.insert("ch-1".to_string(), meta_src("p", "thread", "cognitive-memory"));
        let mut coords: HashMap<(String, String), f32> = HashMap::new();
        coords.insert(("thread".to_string(), "cognitive-memory".to_string()), 1.2);

        let lifted = value_from(&[], &[judg_link("ch-1", 0.8)], &cm, &coords, now, &c);
        let bare = value_from(&[], &[], &cm, &coords, now, &c);
        assert!(lifted > bare, "active-concept coordinate lifts the handle: {lifted} vs {bare}");
        assert!(lifted > 0.7, "lift is above the neutral floor: {lifted}");
    }

    #[test]
    fn value_distinct_query_gate() {
        // The distinct-query gate is enforced UPSTREAM in surfaced_vs_used (one
        // UsedAccess bucket per distinct (query_hash, kind)); value_use sees
        // only the gated buckets. So 50 same-query accesses arrive as ONE
        // bucket and a 2-distinct handle out-scores it — holds at any
        // value_neutral. Model on activation.rs::distinct_query_gate_not_raw_count.
        let now = Utc::now();
        let one_bucket = vec![used(AccessKind::ExplicitRecall)]; // 50 raw → 1 distinct
        let two_buckets = vec![used(AccessKind::ExplicitRecall), used(AccessKind::ExplicitRecall)];
        let v_one = value_use(&one_bucket, now);
        let v_two = value_use(&two_buckets, now);
        assert!(v_two > v_one, "2 distinct buckets beat 1: {v_two} vs {v_one}");

        // And it propagates through value_from at value_neutral < 1.0.
        let mut c = cfg();
        c.value_neutral = 0.7;
        let cm: HashMap<String, SourceMeta> = HashMap::new();
        let coords: HashMap<(String, String), f32> = HashMap::new();
        let val_one = value_from(&one_bucket, &[], &cm, &coords, now, &c);
        let val_two = value_from(&two_buckets, &[], &cm, &coords, now, &c);
        assert!(val_two >= val_one, "gate propagates to value: {val_two} vs {val_one}");
    }

    // --- SalienceScorer derivation --------------------------------------

    #[test]
    fn scorer_flag_off_forces_all_axes_inert() {
        let mut s = SalienceSettings::default();
        s.specificity_enabled = true;
        s.value_enabled = true;
        s.negative_enabled = true;
        // scorer_v2 still false (default) ⇒ all gates forced off.
        let snap = SalienceScorer::from(&s);
        assert!(!snap.scorer_v2);
        assert!(!snap.specificity_enabled);
        assert!(!snap.value_enabled);
        assert!(!snap.negative_enabled);
    }

    #[test]
    fn scorer_flag_on_respects_per_axis_toggles() {
        let mut s = SalienceSettings::default();
        s.scorer_v2 = true;
        s.specificity_enabled = true;
        s.value_enabled = false;
        s.negative_enabled = false;
        let snap = SalienceScorer::from(&s);
        assert!(snap.specificity_enabled);
        assert!(!snap.value_enabled);
        assert!(!snap.negative_enabled);
    }

    #[test]
    fn scorer_from_sanitizes_panic_inducing_knobs() {
        // review Finding 1: a misconfigured `damper_floor` (> 1.0 or NaN)
        // feeds `f32::clamp(eps, 1.0)` in `compute_score_parts`, which panics
        // when `min > max` / NaN — and that clamp runs even flag-off. The
        // `From` snapshot must sanitize every knob the scorer touches so the
        // hot path only ever sees a valid range.
        for bad in [1.5_f32, 9.9, f32::NAN, f32::INFINITY, -3.0] {
            let mut s = SalienceSettings::default();
            s.damper_floor = bad;
            s.neg_gamma = bad;
            s.negative_lift_cutoff = bad;
            s.specificity_lift_cutoff = bad;
            let snap = SalienceScorer::from(&s);
            // damper_floor must be a usable `clamp` floor: in [0, 1).
            assert!(
                (0.0..1.0).contains(&snap.damper_floor),
                "damper_floor must be sanitized into [0,1), got {} from {bad}",
                snap.damper_floor
            );
            // The other knobs land in their documented [0,1] ranges, never NaN.
            for v in [snap.neg_gamma, snap.negative_lift_cutoff, snap.specificity_lift_cutoff] {
                assert!(v.is_finite() && (0.0..=1.0).contains(&v), "knob {v} not in [0,1]");
            }
            // The whole point: `clamp(eps, 1.0)` is now always valid (no panic).
            let _ = (1.0f32 - snap.neg_gamma * 1.0).clamp(snap.damper_floor, 1.0);
        }
    }

    #[test]
    fn scorer_from_preserves_sane_knobs() {
        // Sanitization is a no-op for in-range values (no silent retuning).
        let mut s = SalienceSettings::default();
        s.scorer_v2 = true;
        s.damper_floor = 0.05;
        s.neg_gamma = 0.7;
        s.negative_lift_cutoff = 0.4;
        s.specificity_lift_cutoff = 0.15;
        let snap = SalienceScorer::from(&s);
        assert!((snap.damper_floor - 0.05).abs() < 1e-7);
        assert!((snap.neg_gamma - 0.7).abs() < 1e-7);
        assert!((snap.negative_lift_cutoff - 0.4).abs() < 1e-7);
        assert!((snap.specificity_lift_cutoff - 0.15).abs() < 1e-7);
    }

    // --- negative-transfer (center + negative_penalty) ------------------

    /// Build a global mean from a set of raw vectors (mirrors the boot pass's
    /// `mean_normalized`), so the centering in tests matches production.
    fn mean_of(vecs: &[Vec<f32>]) -> Vec<f32> {
        let dim = vecs[0].len();
        let mut acc = vec![0.0f32; dim];
        for v in vecs {
            let nv = normalize(v);
            for (a, x) in acc.iter_mut().zip(nv.iter()) {
                *a += *x;
            }
        }
        let n = vecs.len() as f32;
        for a in &mut acc {
            *a /= n;
        }
        acc
    }

    #[test]
    fn negative_penalty_neutral_when_no_exemplars() {
        // Empty exemplar set (cold substrate / flag-off boot) ⇒ 0.0, never a
        // damp. Same for an empty anchor (a thread with no anchor seeded).
        let cfg = SalienceSettings::default();
        assert_eq!(negative_penalty(&[1.0, 0.0, 0.0], &[], &[], &cfg), 0.0);
        assert_eq!(
            negative_penalty(&[], &[0.1, 0.1, 0.1], &[vec![1.0, 0.0, 0.0]], &cfg),
            0.0
        );
    }

    #[test]
    fn negative_penalty_damps_twin_not_clean() {
        // A handle near the exemplars earns a high penalty; one far from them
        // earns less. Exemplars are centered (as the boot pass stores them).
        // The negatives span two sub-clusters so the centered residuals carry
        // real direction (a single tight cluster centers to ~noise — the same
        // anisotropy the centering corrects in the wild, just degenerate at
        // tiny dim). τ = 0 here: the discriminator under test is the ORDERING,
        // not the live-space-calibrated floor.
        let raw_negatives = vec![
            vec![1.0, 0.2, 0.0, 0.0],
            vec![0.95, 0.1, 0.1, 0.0],
            vec![0.1, 1.0, 0.0, 0.0],
            vec![0.0, 0.95, 0.15, 0.0],
        ];
        let gmean = mean_of(&raw_negatives);
        let exemplars: Vec<Vec<f32>> = raw_negatives.iter().map(|v| center(v, &gmean)).collect();
        let mut cfg = SalienceSettings::default();
        cfg.negative_tau = 0.0;

        // A twin sits in the first negative sub-cluster's direction.
        let twin = negative_penalty(&[0.98, 0.15, 0.05, 0.0], &gmean, &exemplars, &cfg);
        // A clean handle points along the untouched 4th axis.
        let clean = negative_penalty(&[0.0, 0.0, 0.0, 1.0], &gmean, &exemplars, &cfg);
        assert!(
            twin > clean,
            "twin must earn a higher penalty than a clean handle: twin {twin} vs clean {clean}"
        );
        assert!(twin > 0.0, "the twin should be penalized at all: {twin}");
    }

    #[test]
    fn negative_penalty_k3_robust_to_single_collision() {
        // The ostk-cache mechanism (R3): with k=3 a single near-identical
        // exemplar does NOT max out the penalty — the other two neighbours
        // (which the recoverable concept is FAR from) pull the top-3 mean down.
        // Contrast k=1, which would saturate on the one collision and is why
        // the production default is 3.
        let raw_negatives = vec![
            vec![1.0, 0.0, 0.0, 0.0], // the colliding sub-term twin
            vec![0.0, 1.0, 0.0, 0.0], // unrelated rejected vocab
            vec![0.0, 0.0, 1.0, 0.0], // unrelated rejected vocab
            vec![0.0, 0.0, 0.0, 1.0], // unrelated rejected vocab
        ];
        let gmean = mean_of(&raw_negatives);
        let exemplars: Vec<Vec<f32>> = raw_negatives.iter().map(|v| center(v, &gmean)).collect();
        // The recoverable concept is the near-twin of exemplar 0 only.
        let anchor = vec![1.0, 0.0, 0.0, 0.0];

        let mut k1 = SalienceSettings::default();
        k1.negative_knn_k = 1;
        let mut k3 = SalienceSettings::default();
        k3.negative_knn_k = 3;

        let p1 = negative_penalty(&anchor, &gmean, &exemplars, &k1);
        let p3 = negative_penalty(&anchor, &gmean, &exemplars, &k3);
        assert!(
            p3 < p1,
            "k=3 must be gentler than k=1 on a single-collision concept \
             (the ostk-cache survival mechanism): k3 {p3} vs k1 {p1}"
        );
    }

    #[test]
    fn centering_is_required_for_separation() {
        // R3 §empirical: in the RAW (un-centered) space, a strong common
        // component makes a good concept and a harness twin near-indistinguishable
        // by cosine to the negatives; CENTERING separates them. We model the
        // anisotropy with a large shared offset added to every vector.
        let shared = [5.0, 5.0, 5.0, 5.0]; // dominant common component
        let add = |base: &[f32; 4]| -> Vec<f32> {
            base.iter().zip(shared.iter()).map(|(b, s)| b + s).collect()
        };
        let raw_negatives = vec![
            add(&[1.0, 0.0, 0.0, 0.0]),
            add(&[0.8, 0.2, 0.0, 0.0]),
            add(&[0.9, 0.1, 0.1, 0.0]),
        ];
        // A good concept points orthogonally (in the pre-offset basis) to the
        // negatives, but the shared offset dominates its raw cosine.
        let good = add(&[0.0, 0.0, 0.0, 1.0]);
        let twin = add(&[0.95, 0.05, 0.0, 0.0]);

        // RAW: cosine to the mean negative — both are dominated by `shared`, so
        // the good concept and the twin look nearly the same.
        let neg_mean_raw = mean_of(&raw_negatives);
        let raw_good = crate::cosine_similarity(&normalize(&good), &neg_mean_raw);
        let raw_twin = crate::cosine_similarity(&normalize(&twin), &neg_mean_raw);
        let raw_gap = raw_twin - raw_good;

        // CENTERED: subtract the global mean first, then compare. τ = 0 so the
        // comparison is the raw proximity gap, not the live-space floor.
        let all = [raw_negatives.clone(), vec![good.clone(), twin.clone()]].concat();
        let gmean = mean_of(&all);
        let cen_negs: Vec<Vec<f32>> = raw_negatives.iter().map(|v| center(v, &gmean)).collect();
        let mut cfg = SalienceSettings::default();
        cfg.negative_tau = 0.0;
        let cen_good = negative_penalty(&good, &gmean, &cen_negs, &cfg);
        let cen_twin = negative_penalty(&twin, &gmean, &cen_negs, &cfg);
        let cen_gap = cen_twin - cen_good;

        assert!(
            cen_gap > raw_gap,
            "centering must widen the twin-vs-good separation \
             (raw gap {raw_gap}, centered gap {cen_gap})"
        );
        assert!(
            cen_twin > cen_good,
            "centered: the twin must out-penalize the good concept \
             (twin {cen_twin} vs good {cen_good})"
        );
    }

    #[test]
    fn center_is_noop_on_dimension_mismatch() {
        // A wrong-dimension global mean must not corrupt the result — the
        // safe fallback is the inner normalize (matches cosine_similarity).
        let v = [3.0, 4.0]; // norm 5
        let out = center(&v, &[1.0, 1.0, 1.0]); // mismatched dim
        let n = normalize(&v);
        assert_eq!(out, n);
    }
}
