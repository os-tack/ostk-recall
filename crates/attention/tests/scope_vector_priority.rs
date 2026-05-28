//! P6A gate test — `scope_vector()` priority chain.
//!
//! `effective_vec()` (which `scope_vector()` delegates to) returns
//! values in this priority order, per `p6-attention-ema.md`:
//!
//! 1. `pinned_focus` — operator's stated lens; wins absolutely.
//! 2. `rolling_vec` — EMA-blended attention; the new P6A channel.
//! 3. `transient_vec` — last-turn embed; cold-start fallback.

use ostk_recall_attention::{
    AttendOutcome, AttentionForwardStore, InMemoryAttention, cosine_similarity, stub_embed,
};
use ostk_recall_core::attention::{AttentionScope, PrivacyTier};

fn scope() -> AttentionScope {
    AttentionScope {
        project: Some("p6a".into()),
        session_id: Some("priority".into()),
        agent: Some("claude".into()),
        privacy_tier: PrivacyTier::T1Project,
    }
}

#[tokio::test]
async fn no_pin_returns_rolling_after_first_attend() {
    let attn = InMemoryAttention::new();
    let outcome = attn.attend(&scope(), "alpha").await.unwrap();
    let AttendOutcome::Updated { rolling_vec, .. } = outcome else {
        panic!("expected Updated");
    };

    // First-attend special case: rolling == transient == embed.
    // scope_vector returns one of them — confirm it matches rolling
    // (the priority-chain pick when no pin exists).
    let sv = attn.scope_vector(&scope()).await.unwrap().unwrap();
    assert_eq!(sv, rolling_vec);
}

#[tokio::test]
async fn no_pin_returns_rolling_not_transient_when_they_diverge() {
    let attn = InMemoryAttention::new();
    attn.attend(&scope(), "alpha").await.unwrap();
    let last = attn.attend(&scope(), "beta divergent").await.unwrap();
    let AttendOutcome::Updated {
        rolling_vec: post_blend,
        ..
    } = last
    else {
        panic!("expected Updated");
    };

    // After two attend()s with λ < 1, rolling ≠ transient (the EMA
    // smooths). scope_vector should return rolling — that's what
    // `apply_attention_bias` and the future lens loop consume.
    let sv = attn.scope_vector(&scope()).await.unwrap().unwrap();
    assert_eq!(
        sv, post_blend,
        "scope_vector should follow the rolling channel, not the latest-turn transient"
    );

    // And it must NOT equal the latest embed (that would mean the
    // chain accidentally picked transient).
    let beta = stub_embed("beta divergent");
    let sim_to_transient = cosine_similarity(&sv, &beta);
    assert!(
        sim_to_transient < 0.9999,
        "scope_vector must not snap to last-turn embed (transient)"
    );
}

#[tokio::test]
async fn pinned_focus_wins_over_rolling() {
    let attn = InMemoryAttention::new();
    attn.attend(&scope(), "ambient conversation").await.unwrap();
    attn.attend(&scope(), "more ambient noise").await.unwrap();
    let outcome = attn.focus(&scope(), "pinned query".into()).await.unwrap();
    let pin_vec = outcome.pinned.unwrap().vec;

    let sv = attn.scope_vector(&scope()).await.unwrap().unwrap();
    assert_eq!(
        sv, pin_vec,
        "pinned focus must override the rolling channel"
    );
}

#[tokio::test]
async fn unfocus_falls_back_to_rolling_not_transient() {
    let attn = InMemoryAttention::new();
    attn.attend(&scope(), "first").await.unwrap();
    let after_blend = attn.attend(&scope(), "second different").await.unwrap();
    let AttendOutcome::Updated {
        rolling_vec: rolling_after_blend,
        ..
    } = after_blend
    else {
        panic!("expected Updated");
    };

    attn.focus(&scope(), "operator pin".into()).await.unwrap();
    attn.unfocus(&scope()).await.unwrap();

    // After unfocus the pin is gone, but rolling is still present.
    // Priority chain falls through to rolling — NOT transient.
    let sv = attn.scope_vector(&scope()).await.unwrap().unwrap();
    assert_eq!(
        sv, rolling_after_blend,
        "after unfocus, scope_vector should pick rolling (pin → rolling), not transient"
    );
}

#[tokio::test]
async fn empty_mind_returns_none() {
    let attn = InMemoryAttention::new();
    // No attend, no pin — empty mind.
    let sv = attn.scope_vector(&scope()).await.unwrap();
    assert!(sv.is_none(), "empty-mind scope should return None");
}
