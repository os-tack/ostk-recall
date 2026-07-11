//! Self-audit salience health (THESIS axis 4 / R4).
//!
//! `dereference-or-void` + `projection-truth` applied to the surfacer's own
//! state: the substrate could not previously see its own salience drift (the
//! optimize bug + dangling anchors were found by accident, not surfaced).
//! This module computes four decomposed health metrics over the *active
//! surface* (the pages `attention_surface` returns above `ARCHIVE_THRESHOLD`)
//! and the access ledger:
//!
//! 1. **surface entropy** `H/ln(N)` of the score-share distribution — is the
//!    surface spread across distinct concepts or collapsed onto a few
//!    coherent-noise handles? (Plus a cheap embedding-free score-spread
//!    companion — the live ~0.25-wide ribbon alarm.)
//! 2. **curated:autonomous ratio** — what fraction of the surface is held up
//!    by the curated `stop_handles` / hand-authored anchors vs. earned
//!    autonomously? Health = the curated dependence falls (the THESIS goal
//!    that the hand-list becomes redundant).
//! 3. **surfaced-N-times-never-used** — handles surfaced (`LensIncluded`)
//!    repeatedly but never used (`ExplicitRecall`/`OperatorSelected`). Pure
//!    surface cost; the click-through loop's negative tail.
//! 4. **active-vs-decided drift** — Jaccard distance between the active set
//!    and the recently-judged-salient set, plus the `J \ A` forgotten tail
//!    (the top alarm: judged salient but the surfacer dropped it).
//!
//! The compute path is **pure observation, NOT flag-gated** (design §7.2): it
//! watches BOTH the old and new scorer, so it cannot be conditional on
//! `scorer_v2`. Only the `[salience.health]` thresholds are config. The four
//! metrics double as the A/B scoreboard (R4 §1, design §7.3).
//!
//! Surfaced the way `stale_ingest` is (server.rs): an always-on pull field on
//! `recall_stats` + a conditional loud-on-failure push when `unhealthy`.

use std::collections::{HashMap, HashSet};

use ostk_recall_core::AttentionPage;
use ostk_recall_core::config::SalienceHealthSettings;
use ostk_recall_core::types::{NeverUsed, SalienceHealth, SalienceHealthThresholds};
use ostk_recall_store::UseLedger;

use crate::salience::shannon_entropy;

/// Compute the salience-health block over an active surface.
///
/// Pure and reusable: `recall_stats` (pull), the conditional push wrapper, and
/// the A/B harness (called twice, once per scorer) all call this. Inputs are
/// gathered by the caller (which holds the attention/threads handles) so this
/// fn touches no lock and no DB — the same discipline `compute_score_parts`
/// follows.
///
/// - `surface`: the active surface = pages above `ARCHIVE_THRESHOLD`, already
///   ranked. Metrics 1, 2 read it directly; it defines the active set `A` for
///   metric 4.
/// - `curated`: the curated handle set (`stop_handles` ∪ hand-authored
///   anchors) — used to split the surface into curated vs. autonomous
///   (metric 2). Matched by handle string.
/// - `ledger`: per-handle `surfaced_vs_used` roll-up over the health window
///   (metric 3). A surface handle absent from the map is treated as
///   no-ledger-data (neither surfaced nor used) — not flagged.
/// - `judged`: the recently-judged-salient set `J` (active-concept handles +
///   handles touched by recent decisions), for metric 4's drift. Caller
///   gathers it from the activation reader / decision sources over the same
///   window.
/// - `cfg`: the `[salience.health]` thresholds.
#[must_use]
#[allow(clippy::implicit_hasher)]
pub fn salience_health(
    surface: &[AttentionPage],
    curated: &HashSet<String>,
    ledger: &HashMap<String, UseLedger>,
    judged: &HashSet<String>,
    cfg: &SalienceHealthSettings,
) -> SalienceHealth {
    // --- Metric 1: surface entropy + score spread --------------------------
    // Score-share entropy over the surface. Scores can dip slightly negative
    // for opposed threads; floor each share at 0 (a negative score carries no
    // positive surface mass) so `shannon_entropy` sees a clean distribution.
    let shares: Vec<f32> = surface.iter().map(|p| p.score.max(0.0)).collect();
    let n = shares.iter().filter(|s| **s > 0.0).count();
    let surface_entropy = if n >= 2 {
        let h = shannon_entropy(&shares);
        #[allow(clippy::cast_precision_loss)]
        let h_max = (n as f32).ln();
        if h_max > 0.0 {
            Some((h / h_max).clamp(0.0, 1.0))
        } else {
            None
        }
    } else {
        None
    };
    let surface_score_spread = match (
        surface
            .iter()
            .map(|p| p.score)
            .fold(f32::NEG_INFINITY, f32::max),
        surface
            .iter()
            .map(|p| p.score)
            .fold(f32::INFINITY, f32::min),
    ) {
        (max, min) if max.is_finite() && min.is_finite() => max - min,
        _ => 0.0,
    };

    // --- Metric 2: curated:autonomous ratio --------------------------------
    let mut curated_load_bearing: Vec<String> = surface
        .iter()
        .filter(|p| curated.contains(&p.handle))
        .map(|p| p.handle.clone())
        .collect();
    curated_load_bearing.sort();
    let curated_ratio = if surface.is_empty() {
        0.0
    } else {
        #[allow(clippy::cast_precision_loss)]
        let r = curated_load_bearing.len() as f32 / surface.len() as f32;
        r
    };

    // --- Metric 3: surfaced-N-times-never-used -----------------------------
    // Walk the surface (not the whole ledger) so we only flag what is actually
    // being injected into context. A handle with no evidence to join against
    // (`unattributable`) is reported separately, never miscounted as
    // never-used (R4 §5).
    let mut never_used: Vec<NeverUsed> = Vec::new();
    let mut unattributable: Vec<String> = Vec::new();
    for page in surface {
        let Some(use_ledger) = ledger.get(&page.handle) else {
            continue;
        };
        if use_ledger.unattributable {
            unattributable.push(page.handle.clone());
            continue;
        }
        if use_ledger.surfaced >= cfg.never_used_min_surfaced
            && use_ledger.distinct_used_queries == 0
        {
            never_used.push(NeverUsed {
                handle: page.handle.clone(),
                surfaced: use_ledger.surfaced,
                used: use_ledger.distinct_used_queries,
            });
        }
    }
    never_used.sort_by(|a, b| b.surfaced.cmp(&a.surfaced).then(a.handle.cmp(&b.handle)));
    unattributable.sort();

    // --- Metric 4: active-vs-decided drift ---------------------------------
    let active: HashSet<&str> = surface.iter().map(|p| p.handle.as_str()).collect();
    let judged_refs: HashSet<&str> = judged.iter().map(String::as_str).collect();
    let intersection = active.intersection(&judged_refs).count();
    let union = active.union(&judged_refs).count();
    let active_decided_drift = if union == 0 {
        None
    } else {
        #[allow(clippy::cast_precision_loss)]
        let jaccard = intersection as f32 / union as f32;
        Some(1.0 - jaccard)
    };
    // J \ A — judged salient but NOT active. The top alarm: the surfacer
    // forgot something the operator recently judged worth keeping.
    let mut drift_forgotten: Vec<String> = judged_refs
        .difference(&active)
        .map(|s| (*s).to_string())
        .collect();
    drift_forgotten.sort();

    // --- Health verdict (drives the push leg) ------------------------------
    let entropy_breach = surface_entropy.is_some_and(|h| h < cfg.min_surface_entropy);
    let drift_breach = active_decided_drift.is_some_and(|d| d > cfg.max_active_decided_drift);
    let never_used_breach = !never_used.is_empty();
    let unhealthy = entropy_breach || drift_breach || never_used_breach;

    SalienceHealth {
        surface_entropy,
        surface_score_spread,
        curated_ratio,
        curated_load_bearing,
        never_used,
        unattributable,
        active_decided_drift,
        drift_forgotten,
        thresholds: SalienceHealthThresholds {
            min_surface_entropy: cfg.min_surface_entropy,
            max_active_decided_drift: cfg.max_active_decided_drift,
            never_used_min_surfaced: cfg.never_used_min_surfaced,
            window_days: cfg.health_window_days,
        },
        unhealthy,
    }
}

#[cfg(test)]
#[allow(clippy::float_cmp)]
mod tests {
    use super::*;
    use ostk_recall_core::FoldDepth;
    use ostk_recall_core::attention::ScoreAttribution;

    fn page(handle: &str, score: f32) -> AttentionPage {
        AttentionPage {
            handle: handle.to_string(),
            depth: FoldDepth::Folded,
            score,
            why: ScoreAttribution {
                tension: 0.0,
                resonance: 0.0,
                mentions: 0,
                resonance_count: 0,
                off_diagonal_lift: 0.0,
                time_since_touch_secs: 0,
                specificity: 1.0,
                value: 1.0,
                neg_penalty: 0.0,
            },
        }
    }

    fn thresholds() -> SalienceHealthSettings {
        // Mirror the shipped `[salience.health]` defaults.
        SalienceHealthSettings {
            min_surface_entropy: 0.6,
            max_active_decided_drift: 0.7,
            never_used_min_surfaced: 10,
            health_window_days: 14,
            ttl_secs: 30,
        }
    }

    fn ledger_of(entries: &[(&str, UseLedger)]) -> HashMap<String, UseLedger> {
        entries
            .iter()
            .map(|(h, l)| ((*h).to_string(), *l))
            .collect()
    }

    #[test]
    fn surface_entropy_collapses_on_dominant_handle() {
        // One handle carries almost all the surface mass → low normalized
        // entropy; an even spread → high. (Metric 1.)
        let cfg = thresholds();
        let curated = HashSet::new();
        let ledger = HashMap::new();
        let judged = HashSet::new();

        let dominant = vec![
            page("whale", 100.0),
            page("a", 0.01),
            page("b", 0.01),
            page("c", 0.01),
        ];
        let collapsed = salience_health(&dominant, &curated, &ledger, &judged, &cfg);
        let h_low = collapsed.surface_entropy.expect("entropy defined for N>=2");
        assert!(
            h_low < 0.3,
            "a dominant handle should collapse normalized entropy, got {h_low}"
        );
        assert!(collapsed.unhealthy, "below min_surface_entropy ⇒ unhealthy");

        let even = vec![
            page("a", 1.0),
            page("b", 1.0),
            page("c", 1.0),
            page("d", 1.0),
        ];
        let spread = salience_health(&even, &curated, &ledger, &judged, &cfg);
        let h_high = spread.surface_entropy.expect("entropy defined");
        assert!(h_high > 0.99, "uniform spread ⇒ H/ln(N) ≈ 1, got {h_high}");
        assert!(
            h_high > h_low,
            "even surface is more diverse than collapsed"
        );
    }

    #[test]
    fn surface_entropy_none_for_degenerate_surface() {
        let cfg = thresholds();
        let empty: Vec<AttentionPage> = Vec::new();
        let one = vec![page("solo", 1.0)];
        let curated = HashSet::new();
        let ledger = HashMap::new();
        let judged = HashSet::new();
        assert_eq!(
            salience_health(&empty, &curated, &ledger, &judged, &cfg).surface_entropy,
            None
        );
        assert_eq!(
            salience_health(&one, &curated, &ledger, &judged, &cfg).surface_entropy,
            None,
            "a single bin carries no uncertainty"
        );
    }

    #[test]
    fn curated_ratio_counts_only_curated_handles_on_surface() {
        // Metric 2: curated handles on the surface / surface size, and the
        // load-bearing list names exactly those handles.
        let cfg = thresholds();
        let surface = vec![
            page("turn-digest", 1.2), // curated
            page("squad-lead", 1.1),  // curated
            page("cognitive-memory", 1.0),
            page("ostk-cache", 0.9),
        ];
        let curated: HashSet<String> = ["turn-digest".to_string(), "squad-lead".to_string()].into();
        let ledger = HashMap::new();
        let judged = HashSet::new();
        let out = salience_health(&surface, &curated, &ledger, &judged, &cfg);
        assert_eq!(out.curated_ratio, 0.5);
        assert_eq!(
            out.curated_load_bearing,
            vec!["squad-lead".to_string(), "turn-digest".to_string()]
        );
    }

    #[test]
    fn never_used_flags_surfaced_not_recalled() {
        // Metric 3: a handle surfaced >= N but with zero distinct used queries
        // is flagged; one with recalls is not; an unattributable one is set
        // aside, never miscounted.
        let cfg = thresholds();
        let surface = vec![
            page("noise", 1.1),
            page("used-thing", 1.0),
            page("orphan", 0.9),
            page("below-n", 0.8),
        ];
        let curated = HashSet::new();
        let judged = HashSet::new();
        let ledger = ledger_of(&[
            (
                "noise",
                UseLedger {
                    surfaced: 25,
                    used_weighted: 0.0,
                    distinct_used_queries: 0,
                    unattributable: false,
                },
            ),
            (
                "used-thing",
                UseLedger {
                    surfaced: 30,
                    used_weighted: 2.0,
                    distinct_used_queries: 3,
                    unattributable: false,
                },
            ),
            (
                "orphan",
                UseLedger {
                    surfaced: 0,
                    used_weighted: 0.0,
                    distinct_used_queries: 0,
                    unattributable: true,
                },
            ),
            (
                "below-n",
                UseLedger {
                    surfaced: 4, // below never_used_min_surfaced = 10
                    used_weighted: 0.0,
                    distinct_used_queries: 0,
                    unattributable: false,
                },
            ),
        ]);
        let out = salience_health(&surface, &curated, &ledger, &judged, &cfg);
        assert_eq!(out.never_used.len(), 1, "only `noise` qualifies");
        assert_eq!(out.never_used[0].handle, "noise");
        assert_eq!(out.never_used[0].surfaced, 25);
        assert_eq!(out.unattributable, vec!["orphan".to_string()]);
        assert!(out.unhealthy, "a never-used flag ⇒ unhealthy");
    }

    #[test]
    fn drift_reports_forgotten_judged_handle() {
        // Metric 4: a handle in the judged set J but not on the active surface
        // A appears in drift_forgotten, and the Jaccard distance reflects the
        // partial overlap.
        let cfg = thresholds();
        let surface = vec![page("cognitive-memory", 1.0), page("noise", 0.9)];
        let curated = HashSet::new();
        let ledger = HashMap::new();
        // J = {cognitive-memory, dereference-or-void}; A = {cognitive-memory,
        // noise}. ∩ = 1, ∪ = 3 ⇒ Jaccard 1/3 ⇒ drift 2/3.
        let judged: HashSet<String> = [
            "cognitive-memory".to_string(),
            "dereference-or-void".to_string(),
        ]
        .into();
        let out = salience_health(&surface, &curated, &ledger, &judged, &cfg);
        assert_eq!(
            out.drift_forgotten,
            vec!["dereference-or-void".to_string()],
            "the judged-but-inactive handle is the forgotten tail"
        );
        let drift = out.active_decided_drift.expect("both sets non-empty");
        assert!(
            (drift - 2.0 / 3.0).abs() < 1e-6,
            "drift = 1 − 1/3 = 2/3, got {drift}"
        );
    }

    #[test]
    fn drift_none_when_no_judgment_and_empty_surface() {
        let cfg = thresholds();
        let out = salience_health(&[], &HashSet::new(), &HashMap::new(), &HashSet::new(), &cfg);
        assert_eq!(out.active_decided_drift, None);
        assert!(out.drift_forgotten.is_empty());
        assert!(!out.unhealthy, "a fully-empty read is not a failure");
    }

    #[test]
    fn health_runs_under_both_scorers() {
        // The compute path is observation, not gated on scorer_v2 — the page
        // set just differs. Whatever surface it is handed, it produces a block.
        // (Modeled on design I4 test #4: feed two different surfaces standing
        // in for the two scorers and assert both produce a coherent read.)
        let cfg = thresholds();
        let curated = HashSet::new();
        let ledger = HashMap::new();
        let judged = HashSet::new();

        let scorer_v1_surface = vec![page("a", 1.05), page("b", 1.02), page("c", 1.0)];
        let scorer_v2_surface = vec![page("a", 2.0), page("b", 0.5), page("c", 0.3)];

        let v1 = salience_health(&scorer_v1_surface, &curated, &ledger, &judged, &cfg);
        let v2 = salience_health(&scorer_v2_surface, &curated, &ledger, &judged, &cfg);

        assert!(v1.surface_entropy.is_some());
        assert!(v2.surface_entropy.is_some());
        // v1's tight ribbon is MORE even (higher entropy) than v2's spread —
        // exactly the "before" baseline the A/B scoreboard captures (R4 §2).
        assert!(
            v1.surface_entropy > v2.surface_entropy,
            "the flat ribbon reads as higher score-share entropy than a \
             discriminating scorer's peaked surface"
        );
        assert!(v1.surface_score_spread < v2.surface_score_spread);
    }
}
