//! P6A gate test — `attention_affinity_score` correctness.
//!
//! Direct score function (P3A-style); the `RankFeatureFactory` /
//! `RankFeatureInstance` wrapping ships in P3B/P6-full. The
//! function's contract:
//!
//! - Returns `0.0` when either input is missing (empty-mind scope
//!   or non-projected candidate).
//! - Returns `cosine(scope_vector, dense_embedding).max(0.0)`,
//!   clamped at the lower bound so anti-resonant candidates don't
//!   contribute negatively.
//! - Output is in `[0.0, 1.0]` for the typical case (the rank
//!   engine also clamps to `[0, 1]` defensively).

use ostk_recall_core::{Chunk, FacetSet, Links, Source};
use ostk_recall_query::candidate::Candidate;
use ostk_recall_query::context::AttentionContext;
use ostk_recall_query::rank::attention_affinity_score;

fn make_candidate(chunk_id: &str, embedding: Option<Vec<f32>>) -> Candidate {
    let chunk = Chunk {
        chunk_id: chunk_id.into(),
        source: Source::Markdown,
        project: Some("test".into()),
        source_id: "test-source".into(),
        source_config_id: "test:cfg".into(),
        chunk_index: 0,
        ts: None,
        role: None,
        text: "test content".into(),
        sha256: "sha-test".into(),
        links: Links::default(),
        facets: FacetSet::default(),
        embedding_input_sha256: "emb-test".into(),
        extra: serde_json::Value::Null,
    };
    let mut c = Candidate::for_chunk(chunk);
    c.dense_embedding = embedding;
    c
}

#[test]
fn aligned_vectors_score_near_one() {
    let cand = make_candidate("c1", Some(vec![1.0, 0.0, 0.0]));
    let ctx = AttentionContext::with_scope_vector(vec![1.0, 0.0, 0.0]);
    let score = attention_affinity_score(&cand, &ctx);
    assert!(
        (score - 1.0).abs() < 1e-6,
        "identical unit vectors should score 1.0, got {score}"
    );
}

#[test]
fn orthogonal_vectors_score_zero() {
    let cand = make_candidate("c1", Some(vec![1.0, 0.0, 0.0]));
    let ctx = AttentionContext::with_scope_vector(vec![0.0, 1.0, 0.0]);
    let score = attention_affinity_score(&cand, &ctx);
    assert!(
        score.abs() < 1e-6,
        "orthogonal vectors should score 0, got {score}"
    );
}

#[test]
fn anti_resonant_vectors_clamp_to_zero() {
    // cosine = -1 → score = 0 (clamped). Anti-resonant content
    // shouldn't get a *negative* lift, only the absence of a
    // positive one.
    let cand = make_candidate("c1", Some(vec![1.0, 0.0]));
    let ctx = AttentionContext::with_scope_vector(vec![-1.0, 0.0]);
    let score = attention_affinity_score(&cand, &ctx);
    assert_eq!(score, 0.0, "anti-resonant should clamp to 0, got {score}");
}

#[test]
fn missing_scope_vector_returns_zero() {
    let cand = make_candidate("c1", Some(vec![1.0, 0.0]));
    let ctx = AttentionContext::empty();
    assert_eq!(attention_affinity_score(&cand, &ctx), 0.0);
}

#[test]
fn missing_dense_embedding_returns_zero() {
    let cand = make_candidate("c1", None);
    let ctx = AttentionContext::with_scope_vector(vec![1.0, 0.0]);
    assert_eq!(attention_affinity_score(&cand, &ctx), 0.0);
}

#[test]
fn dim_mismatch_returns_zero() {
    let cand = make_candidate("c1", Some(vec![1.0, 0.0, 0.0, 0.0]));
    let ctx = AttentionContext::with_scope_vector(vec![1.0, 0.0]);
    let score = attention_affinity_score(&cand, &ctx);
    assert_eq!(score, 0.0, "dim mismatch must not propagate as a score");
}

#[test]
fn with_rolling_builder_drives_affinity_score() {
    // Review-fix regression test: `AttentionContext::with_rolling`
    // must populate `scope_vector` so callers using the builder see
    // non-zero affinity. A rolling-only setter would silently make
    // every candidate score 0 — exactly the failure mode P9b-min's
    // enrich step is most likely to trip into.
    let cand = make_candidate("c1", Some(vec![1.0, 0.0, 0.0]));
    let ctx = AttentionContext::default().with_rolling(vec![1.0, 0.0, 0.0]);
    let score = attention_affinity_score(&cand, &ctx);
    assert!(
        (score - 1.0).abs() < 1e-6,
        "with_rolling must propagate to scope_vector for affinity scoring, got {score}"
    );
}

#[test]
fn with_rolling_preserves_existing_scope_vector() {
    // Review-fix regression test: pin precedence must survive a
    // `.with_rolling()` chained on after `with_scope_vector`. P9b's
    // enrich step composes contexts as
    // `with_scope_vector(pinned).with_rolling(rolling_snapshot)`;
    // a naive setter that always overwrites `scope_vector` would
    // silently demote the pin to the rolling channel.
    let pin = vec![1.0, 0.0, 0.0];
    let rolling = vec![0.0, 1.0, 0.0];
    let ctx = AttentionContext::with_scope_vector(pin.clone()).with_rolling(rolling.clone());
    assert_eq!(
        ctx.scope_vector.as_deref(),
        Some(pin.as_slice()),
        "pin must survive a subsequent with_rolling"
    );
    assert_eq!(
        ctx.rolling_vec.as_deref(),
        Some(rolling.as_slice()),
        "rolling channel must still be populated"
    );
}

#[test]
fn affinity_scores_against_pin_when_both_set() {
    // The behavioural counterpart to the field-level test above:
    // when `with_scope_vector` and `with_rolling` are both used,
    // affinity scoring follows the pin (which is what
    // `attention_affinity_score` reads).
    let cand = make_candidate("c1", Some(vec![1.0, 0.0, 0.0]));
    // Pin aligns with the candidate; rolling is orthogonal — the
    // delta is large enough that misreading one for the other can't
    // be hidden by float fuzz.
    let pin = vec![1.0, 0.0, 0.0];
    let rolling = vec![0.0, 1.0, 0.0];
    let ctx = AttentionContext::with_scope_vector(pin).with_rolling(rolling);
    let score = attention_affinity_score(&cand, &ctx);
    assert!(
        (score - 1.0).abs() < 1e-6,
        "affinity must score against the pinned scope_vector, not rolling (got {score})"
    );
}

#[test]
fn score_is_in_unit_interval() {
    // Random-ish vectors — verify the score is always in [0, 1].
    let cand = make_candidate("c1", Some(vec![0.6, 0.8, 0.0]));
    let ctx = AttentionContext::with_scope_vector(vec![0.8, 0.6, 0.0]);
    let score = attention_affinity_score(&cand, &ctx);
    assert!(
        (0.0..=1.0).contains(&score),
        "score must be in [0, 1], got {score}"
    );
}
