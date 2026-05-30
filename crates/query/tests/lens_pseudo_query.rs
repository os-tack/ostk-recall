//! P9b-full gate — pseudo-query construction in `AttentionContext::enrich_for_lens`.
//!
//! The pseudo-query is the synthetic text the MaxSim (P4) rank feature
//! consumes in the ambient/lens path (there is no user query string). It is
//! built from the dominant concept label + recent entity handles. Until P7
//! (entities) and P8 (concepts) land, both inputs are empty, so the
//! pseudo-query must be `None` — MaxSim then contributes 0 for every
//! candidate rather than ranking on an empty string.

use ostk_recall_query::context::AttentionContext;

#[tokio::test]
async fn empty_inputs_yield_no_pseudo_query() {
    // Today's steady state: no entity ring (P7), no concept overlay (P8).
    let ctx = AttentionContext::enrich_for_lens(
        Some(vec![1.0, 0.0]),
        Some(vec![1.0, 0.0]),
        false,
        None,
        Vec::new(),
        None,
    )
    .await;
    assert_eq!(
        ctx.pseudo_query, None,
        "no entities + no concept ⇒ pseudo_query None (MaxSim contributes 0)"
    );
    assert!(ctx.recent_entities.is_empty());
    assert_eq!(ctx.dominant_concept_label, None);
}

#[tokio::test]
async fn concept_and_entities_compose_into_pseudo_query() {
    let ctx = AttentionContext::enrich_for_lens(
        Some(vec![0.0, 1.0]),
        None,
        false,
        None,
        vec!["path:auth.rs".to_string(), "decision:→1840".to_string()],
        Some("auth-overhaul".to_string()),
    )
    .await;
    let pq = ctx.pseudo_query.expect("concept + entities ⇒ Some pseudo_query");
    // Concept label leads, entity handles follow (construction order).
    assert!(pq.contains("auth-overhaul"), "concept label present: {pq}");
    assert!(pq.contains("path:auth.rs"), "entity present: {pq}");
    assert!(pq.contains("decision:→1840"), "entity present: {pq}");
}

#[tokio::test]
async fn blank_strings_are_ignored() {
    // Defensive: empty/blank tokens must not produce a whitespace-only
    // pseudo-query that would look non-empty to MaxSim.
    let ctx = AttentionContext::enrich_for_lens(
        None,
        None,
        false,
        None,
        vec![String::new()],
        Some(String::new()),
    )
    .await;
    assert_eq!(
        ctx.pseudo_query, None,
        "blank concept + blank entity ⇒ pseudo_query None"
    );
}

#[tokio::test]
async fn enrich_sets_pinned_and_carries_vectors() {
    let ctx = AttentionContext::enrich_for_lens(
        Some(vec![1.0, 2.0, 3.0]),
        Some(vec![0.5, 0.5, 0.0]),
        true,
        None,
        Vec::new(),
        None,
    )
    .await;
    assert!(ctx.pinned, "explicit pin flag carried through enrich");
    assert_eq!(ctx.scope_vector.as_deref(), Some([1.0, 2.0, 3.0].as_slice()));
    assert_eq!(ctx.rolling_vec.as_deref(), Some([0.5, 0.5, 0.0].as_slice()));
}
