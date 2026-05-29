//! P6A gate test — rolling EMA correctness.
//!
//! `attend()` must:
//! - seed `rolling_vec` with the first turn's embed verbatim,
//! - subsequent calls blend per `rolling = normalize((1-λ)·prev + λ·new)`,
//! - drift smoothly toward the latest input rather than snap to it.
//!
//! These are unit-level claims on `InMemoryAttention`; no observer,
//! no chain sink. The observer-mediated chain emission lives in
//! `observer_owns_chain_emit.rs`.

use ostk_recall_attention::{
    AttendOutcome, AttentionForwardStore, DEFAULT_ATTENTION_LAMBDA, InMemoryAttention,
    cosine_similarity, stub_embed,
};
use ostk_recall_core::attention::{AttentionScope, PrivacyTier};

fn scope() -> AttentionScope {
    AttentionScope {
        project: Some("p6a".into()),
        session_id: Some("rolling".into()),
        agent: Some("claude".into()),
        privacy_tier: PrivacyTier::T1Project,
    }
}

#[tokio::test]
async fn first_attend_seeds_rolling_with_embed() {
    let attn = InMemoryAttention::new();
    let outcome = attn.attend(&scope(), "first turn alpha").await.unwrap();

    let AttendOutcome::Updated {
        rolling_vec,
        lambda,
    } = outcome
    else {
        panic!("expected Updated outcome on first attend");
    };

    // Lambda reported on the outcome matches the runtime's setting.
    assert!((lambda - DEFAULT_ATTENTION_LAMBDA).abs() < f32::EPSILON);

    // Seed turn: rolling == embed, byte-exact (no blend happened).
    let expected = stub_embed("first turn alpha");
    assert_eq!(rolling_vec.len(), expected.len());
    for (a, b) in rolling_vec.iter().zip(expected.iter()) {
        assert!((a - b).abs() < 1e-6, "seed differs: {a} vs {b}");
    }

    // scope_vector reflects the rolling channel (no pin set).
    let sv = attn.scope_vector(&scope()).await.unwrap().unwrap();
    assert_eq!(sv, rolling_vec);
}

#[tokio::test]
async fn second_attend_emas_toward_new() {
    let attn = InMemoryAttention::new();
    attn.attend(&scope(), "alpha").await.unwrap();
    let outcome = attn.attend(&scope(), "beta differs").await.unwrap();

    let AttendOutcome::Updated { rolling_vec, .. } = outcome else {
        panic!("expected Updated");
    };

    let alpha = stub_embed("alpha");
    let beta = stub_embed("beta differs");

    // Rolling should now sit between alpha and beta, biased toward
    // alpha because λ=0.3 (alpha keeps 70% of its mass).
    let sim_alpha = cosine_similarity(&rolling_vec, &alpha);
    let sim_beta = cosine_similarity(&rolling_vec, &beta);
    assert!(
        sim_alpha > sim_beta,
        "λ=0.3 should leave rolling closer to the older alpha turn (a={sim_alpha}, b={sim_beta})"
    );

    // Rolling is NOT a snap to the latest input (the original
    // pre-P6 behaviour). transient_vec is — the cold-start fallback
    // path. We verify the differentiation here: rolling ≠ beta.
    let exact_beta_match = rolling_vec
        .iter()
        .zip(beta.iter())
        .all(|(a, b)| (a - b).abs() < 1e-6);
    assert!(
        !exact_beta_match,
        "rolling should not snap to latest input (that's transient's job)"
    );
}

#[tokio::test]
async fn rolling_is_l2_normalized() {
    let attn = InMemoryAttention::new();
    attn.attend(&scope(), "alpha one").await.unwrap();
    attn.attend(&scope(), "beta two").await.unwrap();
    let outcome = attn.attend(&scope(), "gamma three").await.unwrap();
    let AttendOutcome::Updated { rolling_vec, .. } = outcome else {
        panic!("expected Updated");
    };

    let norm: f32 = rolling_vec.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!(
        (norm - 1.0).abs() < 1e-5,
        "rolling vector should be unit-norm after EMA blend, got norm={norm}"
    );
}

#[tokio::test]
async fn lambda_one_snaps_to_latest() {
    // λ=1.0 collapses EMA to "take the latest" — useful as a
    // sanity check on the blend formula. (Production never sets
    // this; the clamp prevents λ=0 disabling the channel entirely.)
    let attn = InMemoryAttention::new().with_lambda(1.0);
    attn.attend(&scope(), "old").await.unwrap();
    let outcome = attn.attend(&scope(), "new content").await.unwrap();
    let AttendOutcome::Updated { rolling_vec, .. } = outcome else {
        panic!("expected Updated");
    };

    // With λ=1, the blend is `normalize(0 * old + 1 * new) = normalize(new)`.
    // stub_embed already produces vectors with stable structure, but
    // not unit-norm — the EMA path always normalizes. Compare via
    // cosine similarity instead of byte-exact.
    let new_embed = stub_embed("new content");
    let sim = cosine_similarity(&rolling_vec, &new_embed);
    assert!(
        sim > 0.9999,
        "λ=1 should collapse rolling onto the latest embed direction (sim={sim})"
    );
}

#[tokio::test]
async fn lambda_clamps_below_zero_and_above_one() {
    // λ=0 would freeze the rolling channel forever after the first
    // turn — the clamp must prevent that. λ>1 amplifies noise; same.
    let frozen = InMemoryAttention::new().with_lambda(-0.5);
    assert!(frozen.lambda() > 0.0, "lambda must clamp above 0");
    let amplified = InMemoryAttention::new().with_lambda(7.0);
    assert!(amplified.lambda() <= 1.0, "lambda must clamp to ≤ 1");
}
