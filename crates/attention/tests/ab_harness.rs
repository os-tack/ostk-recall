//! Integrated A/B harness (THESIS proof bar; plan.md §"A/B harness" / P-4;
//! review.md Finding 3).
//!
//! The per-axis unit tests each prove a *fragment* (one diffuse vs one
//! concentrated handle; the bounded damp recovers; the metric computes). This
//! is the integrated fixture the plan promised and the V review found missing:
//! it loads the named coherent-noise SET vs the named real-concept SET,
//! computes each handle's `specificity` through the **real**
//! `specificity_from_evidence` (not an injected scalar), and ranks the set
//! three ways to assert the THESIS ordering:
//!
//! - **P1 (ordering):** every confirmed coherent-noise handle ranks BELOW
//!   every confirmed real concept under the new scorer.
//! - **P2 (specificity alone):** the specificity-only run (value/negative off,
//!   curated stop-set disabled) reproduces P1 — i.e. specificity reproduces the
//!   stop-set's effect with NO hand-list.
//! - **P3 (ostk-cache survives):** the named recoverable concept stays above
//!   `ARCHIVE_THRESHOLD` and above all P1 noise in the negative-enabled run.
//! - **P4 (redundancy):** disabling the curated stop-set drops `curated_ratio`
//!   on the surface without regressing the P1 ordering — the receipt that the
//!   hand-list is redundant under specificity.
//!
//! This gates every future scorer change in CI; the live MCP-driven run at V
//! is the complement, not a substitute.

use std::collections::{HashMap, HashSet};

use ostk_recall_attention::health::salience_health;
use ostk_recall_attention::salience::specificity_from_evidence;
use ostk_recall_attention::{
    AttentionForwardStore, InMemoryAttention, SalienceFactors, SourceMeta, stub_embed,
};
use ostk_recall_core::attention::{AttentionScope, PrivacyTier};
use ostk_recall_core::config::SalienceSettings;
use ostk_recall_core::{SalienceHealthSettings, ThreadHandle};
use ostk_recall_store::{AssociationType, EvidenceLink, RelationState};

// --- the named SETS (THESIS / review P1) ------------------------------------

/// Confirmed coherent-noise — harness/plumbing vocab that "looks like what's
/// happening, a lot" (THESIS). Each is given evidence spread across many
/// unrelated source docs ⇒ high co-occurrence entropy ⇒ low specificity.
const COHERENT_NOISE: &[&str] = &[
    "turn-digest",
    "squad-lead",
    "re-run",
    "non-blocking",
    "re-read",
    "pre-existing",
    "follow-up",
    "top-level",
    "system-reminder",
];

/// Confirmed real concepts — each given evidence concentrated in one
/// source doc/topic ⇒ low entropy ⇒ high specificity.
const REAL_CONCEPTS: &[&str] = &[
    "cognitive-memory",
    "ostk-cache",
    "dereference-or-void",
    "relational-substrate-docgraph",
];

fn scope() -> AttentionScope {
    AttentionScope {
        project: None,
        session_id: Some("ab".into()),
        agent: Some("substrate".into()),
        privacy_tier: PrivacyTier::T1Project,
    }
}

fn handle(s: &str) -> ThreadHandle {
    ThreadHandle::new(s).expect("valid handle")
}

fn link(handle: &str, chunk_id: &str) -> EvidenceLink {
    let now = chrono::Utc::now();
    EvidenceLink {
        id: 0,
        thread_handle: ThreadHandle::new(handle).unwrap(),
        original_path: std::path::PathBuf::from(chunk_id),
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

/// Build the realistic evidence graph for the whole set. Coherent-noise
/// handles resonate across MANY distinct `(project, source_id)` docs (diffuse,
/// high entropy); real concepts concentrate in ONE doc (peaked, low entropy).
/// Returns the `(evidence-by-handle, chunk_meta)` the real
/// `specificity_from_evidence` consumes — exactly the shape the boot pass
/// builds from `list_evidence_all` + `fetch_chunks_by_ids`.
fn build_evidence() -> (
    HashMap<String, Vec<EvidenceLink>>,
    HashMap<String, SourceMeta>,
) {
    let mut ev: HashMap<String, Vec<EvidenceLink>> = HashMap::new();
    let mut meta: HashMap<String, SourceMeta> = HashMap::new();

    // Diffuse: 12 resonances, each in a distinct doc across two projects.
    for h in COHERENT_NOISE {
        let mut links = Vec::new();
        for i in 0..12 {
            let cid = format!("{h}-c{i}");
            let project = if i % 2 == 0 {
                "ostk-recall"
            } else {
                "haystack"
            };
            meta.insert(
                cid.clone(),
                SourceMeta {
                    project: Some(project.to_string()),
                    source: "claude_code".to_string(),
                    source_id: format!("doc-{h}-{i}.jsonl"),
                },
            );
            links.push(link(h, &cid));
        }
        ev.insert((*h).to_string(), links);
    }
    // Concentrated: 12 resonances, all in ONE source doc (one topic/file).
    for h in REAL_CONCEPTS {
        let mut links = Vec::new();
        for i in 0..12 {
            let cid = format!("{h}-c{i}");
            meta.insert(
                cid.clone(),
                SourceMeta {
                    project: Some("ostk-recall".to_string()),
                    source: "code".to_string(),
                    source_id: format!("{h}.md"),
                },
            );
            links.push(link(h, &cid));
        }
        ev.insert((*h).to_string(), links);
    }
    (ev, meta)
}

/// Compute `SalienceFactors` for the set the way the boot pass does: real
/// `specificity_from_evidence` over the evidence graph. (Value stays neutral —
/// v1 ships `value_neutral = 1.0`; negative is injected per-run below since the
/// exemplar set is a separate store-level input.)
fn compute_factors(cfg: &SalienceSettings) -> HashMap<String, SalienceFactors> {
    let (ev, meta) = build_evidence();
    let mut factors = HashMap::new();
    for (h, links) in &ev {
        let specificity = specificity_from_evidence(links, &meta, cfg);
        factors.insert(
            h.clone(),
            SalienceFactors {
                specificity,
                ..Default::default()
            },
        );
    }
    factors
}

/// Seed every handle in the set with identical counters + identically-aligned
/// anchors, so the *live resonance term is equal across the whole set* and the
/// ONLY thing that can reorder them is the salience factors. (300/290: rate
/// 0.97 — none is a derived `is_stop_handle`, so specificity must do the work.)
async fn seed_set(store: &InMemoryAttention, sc: &AttentionScope) {
    store.attend(sc, "active working context").await.unwrap();
    let anchor = stub_embed("active working context");
    for h in COHERENT_NOISE.iter().chain(REAL_CONCEPTS) {
        store
            .seed_anchor(sc, handle(h), anchor.clone())
            .await
            .unwrap();
        store.seed_counters(sc, &handle(h), 300, 290).await.unwrap();
    }
}

fn ab_settings(specificity: bool, value: bool, negative: bool) -> SalienceSettings {
    let mut s = SalienceSettings::default();
    s.scorer_v2 = true;
    s.specificity_enabled = specificity;
    s.value_enabled = value;
    s.negative_enabled = negative;
    // The fixture seeds 12 synthetic co-occurrence docs per handle; pin the
    // min-evidence floor below that so the harness exercises the entropy
    // ORDERING, not the production token-hit floor (20). The live re-run via
    // salience-ab is what validates the real-corpus threshold.
    s.specificity_min_evidence = 5;
    s
}

/// Rank the seeded set, returning handle→score for the surface.
async fn rank(store: &InMemoryAttention, sc: &AttentionScope) -> HashMap<String, f32> {
    let pages = store.surface(sc, 100).await.unwrap();
    pages.into_iter().map(|p| (p.handle, p.score)).collect()
}

/// Assert the P1 ordering: every coherent-noise handle present on the surface
/// scores below every real concept present. (A noise handle archived off the
/// surface entirely is an even stronger pass.)
fn assert_p1_ordering(scores: &HashMap<String, f32>, label: &str) {
    let worst_concept = REAL_CONCEPTS
        .iter()
        .filter_map(|c| scores.get(*c))
        .copied()
        .fold(f32::INFINITY, f32::min);
    assert!(
        worst_concept.is_finite(),
        "[{label}] at least one real concept must be on the surface",
    );
    for n in COHERENT_NOISE {
        if let Some(noise_score) = scores.get(*n) {
            assert!(
                *noise_score < worst_concept,
                "[{label}] P1 violated: coherent-noise `{n}` ({noise_score}) \
                 must rank below the worst real concept ({worst_concept})",
            );
        }
    }
}

#[tokio::test]
async fn ab_p1_p2_specificity_alone_reproduces_stop_set_ordering() {
    // P1 + P2: the specificity-only run (value/negative OFF) with NO curated
    // stop-set must rank the whole coherent-noise set below the whole
    // real-concept set. This is the headline THESIS claim: specificity alone,
    // unsupervised, reproduces what the hand-list did.
    let cfg = ab_settings(true, false, false);
    let store = InMemoryAttention::new().with_salience_settings(&cfg); // NO with_stop_handles
    let sc = scope();
    seed_set(&store, &sc).await;
    store.set_salience_factors(compute_factors(&cfg)).await;

    let scores = rank(&store, &sc).await;
    assert_p1_ordering(&scores, "P2 specificity-alone, no hand-list");
}

#[tokio::test]
async fn ab_p1_all_axes_on_holds_ordering() {
    // P1 under the full scorer (all axes on). Value is the v1 pass-through
    // (value_neutral=1.0) and no negative exemplars are installed, so this is
    // specificity-driven — but it proves enabling the other axes does not
    // regress the ordering.
    let cfg = ab_settings(true, true, true);
    let store = InMemoryAttention::new().with_salience_settings(&cfg);
    let sc = scope();
    seed_set(&store, &sc).await;
    store.set_salience_factors(compute_factors(&cfg)).await;

    let scores = rank(&store, &sc).await;
    assert_p1_ordering(&scores, "P1 all-axes-on");
}

#[tokio::test]
async fn ab_p3_ostk_cache_survives_negative_penalty() {
    // P3: the named recoverable concept stays above ARCHIVE_THRESHOLD and above
    // all coherent-noise — even carrying a high negative penalty computed by
    // the REAL `negative_penalty()` (review Finding 2: drive the real kNN, not
    // an injected 1.0). The realistic scenario: ostk-cache is being actively
    // discussed (anchor aligned with the live attention vector) and collides in
    // centered space with a rejected sub-term, while the coherent-noise set is
    // idle (orthogonal anchors — not currently discussed). The bounded damp
    // (γ<1) must preserve enough of ostk-cache's live signal to keep it above
    // the idle noise.
    let cfg = ab_settings(true, false, true);
    let store = InMemoryAttention::new().with_salience_settings(&cfg);
    let sc = scope();

    store.attend(&sc, "active working context").await.unwrap();
    let live = stub_embed("active working context");
    let idle = stub_embed("unrelated-idle-context-zzz");

    // ostk-cache: actively resonating, high real specificity.
    store
        .seed_anchor(&sc, handle("ostk-cache"), live.clone())
        .await
        .unwrap();
    store
        .seed_counters(&sc, &handle("ostk-cache"), 22, 11)
        .await
        .unwrap();
    // Coherent noise: idle (orthogonal), high mentions / diffuse specificity.
    for n in COHERENT_NOISE {
        store
            .seed_anchor(&sc, handle(n), idle.clone())
            .await
            .unwrap();
        store
            .seed_counters(&sc, &handle(n), 300, 290)
            .await
            .unwrap();
    }

    // Drive the REAL negative_penalty(): ostk-cache sits in a NEIGHBORHOOD of
    // rejected sub-vocab — the top-k nearest exemplars are all moderately close
    // to its (centered) anchor, so the k=3 mean-proximity clears `negative_tau`
    // and produces a substantial real penalty. (A single collision averaged
    // against far exemplars would be damped below tau by k=3 — that is the
    // ostk-cache survival mechanism R3 names, and not what P3 stresses; here we
    // give it a genuinely high penalty and show the BOUNDED damp still recovers
    // it.) Exemplars are deterministic blends toward `live` plus a far one.
    let global_mean = vec![0.0_f32; live.len()];
    let blend = |a: &[f32], b: &[f32], w: f32| -> Vec<f32> {
        a.iter()
            .zip(b)
            .map(|(x, y)| w * x + (1.0 - w) * y)
            .collect()
    };
    let exemplars: Vec<Vec<f32>> = vec![
        blend(&live, &idle, 0.92), // very near ostk-cache
        blend(&live, &idle, 0.85),
        blend(&live, &idle, 0.80),
        stub_embed("another-far-rejected-term-qqq"), // far — excluded by k=3
    ];
    let cache_neg =
        ostk_recall_attention::salience::negative_penalty(&live, &global_mean, &exemplars, &cfg);
    assert!(
        cache_neg > 0.3,
        "the rejected-vocab neighborhood must produce a substantial real \
         penalty (k=3 mean above tau), got {cache_neg}",
    );

    let mut factors = compute_factors(&cfg);
    factors
        .entry("ostk-cache".to_string())
        .or_default()
        .neg_penalty = cache_neg;
    store.set_salience_factors(factors).await;

    let scores = rank(&store, &sc).await;
    let cache = scores
        .get("ostk-cache")
        .copied()
        .expect("ostk-cache must survive the bounded negative damp");
    assert!(
        cache >= ostk_recall_attention::ARCHIVE_THRESHOLD,
        "P3: ostk-cache must stay above ARCHIVE_THRESHOLD under the real \
         negative penalty ({cache_neg}), got score {cache}",
    );
    // And it still clears every coherent-noise handle on the surface.
    for n in COHERENT_NOISE {
        if let Some(noise) = scores.get(*n) {
            assert!(
                cache > *noise,
                "P3: penalized-but-recoverable ostk-cache ({cache}) must outrank \
                 idle coherent-noise `{n}` ({noise})",
            );
        }
    }
}

#[tokio::test]
async fn ab_p4_curated_stopset_is_redundant() {
    // P4: with the curated stop-set DISABLED, curated_ratio on the surface
    // falls (to ~0 — nothing is hand-listed) while the P1 ordering still holds.
    // The receipt that specificity makes the hand-list redundant. We compare
    // the health curated_ratio with vs without the stop-set, holding the
    // (specificity-driven) scorer constant.
    let cfg = ab_settings(true, false, false);
    let sc = scope();
    let factors = compute_factors(&cfg);
    let curated_handles: Vec<String> = COHERENT_NOISE.iter().map(|s| (*s).to_string()).collect();

    // (a) stop-set ENABLED: the noise handles are forced_stop / hand-listed.
    let with_stopset = InMemoryAttention::new()
        .with_salience_settings(&cfg)
        .with_stop_handles(curated_handles.clone());
    seed_set(&with_stopset, &sc).await;
    with_stopset.set_salience_factors(factors.clone()).await;
    let surf_with = with_stopset.surface(&sc, 100).await.unwrap();
    let curated_with: HashSet<String> = with_stopset.curated_handles();
    let health_with = salience_health(
        &surf_with,
        &curated_with,
        &HashMap::new(),
        &HashSet::new(),
        &health_thresholds(),
    );

    // (b) stop-set DISABLED: specificity alone must hold the ordering, and the
    // curated dependence drops to zero.
    let without_stopset = InMemoryAttention::new().with_salience_settings(&cfg); // no stop-set
    seed_set(&without_stopset, &sc).await;
    without_stopset.set_salience_factors(factors).await;
    let surf_without = without_stopset.surface(&sc, 100).await.unwrap();
    let scores_without: HashMap<String, f32> = surf_without
        .iter()
        .map(|p| (p.handle.clone(), p.score))
        .collect();
    let curated_without: HashSet<String> = without_stopset.curated_handles();
    let health_without = salience_health(
        &surf_without,
        &curated_without,
        &HashMap::new(),
        &HashSet::new(),
        &health_thresholds(),
    );

    // P4a: curated_ratio falls when the hand-list is removed.
    assert!(
        health_without.curated_ratio < health_with.curated_ratio,
        "P4: curated_ratio must fall when the stop-set is disabled: with={} without={}",
        health_with.curated_ratio,
        health_without.curated_ratio,
    );
    assert_eq!(
        health_without.curated_ratio, 0.0,
        "P4: with no stop-set, nothing on the surface is curated",
    );
    // P4b: ...without regressing the P1 ordering (specificity carries it alone).
    assert_p1_ordering(&scores_without, "P4 stop-set-disabled");
}

fn health_thresholds() -> SalienceHealthSettings {
    SalienceHealthSettings {
        min_surface_entropy: 0.6,
        max_active_decided_drift: 0.7,
        never_used_min_surfaced: 10,
        health_window_days: 14,
        ttl_secs: 30,
    }
}
