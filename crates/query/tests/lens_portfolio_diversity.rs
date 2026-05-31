//! P9b-full gate — portfolio allocation across slots.
//!
//! Verifies the full `default_slots()` allocator:
//! - slots are filled by **weighted contribution** dominance (not raw),
//! - the canonical slot order is honored (attention → freshness → entity →
//!   concept → diversity_jump) with first-match-wins,
//! - the entity + concept slots **skip cleanly** because their features
//!   (`entity_salience`, `concept_support`) are unregistered until P7/P8,
//! - the diversity-jump slot picks the most *lateral* candidate (lowest
//!   cosine to the already-selected set), not just the next-highest score.

use std::collections::BTreeMap;

use ostk_recall_core::{Chunk, FacetSet, Links, Source};
use ostk_recall_query::candidate::Candidate;
use ostk_recall_query::lens::{LensConfig, allocate_portfolio, default_slots};
use ostk_recall_query::rank::{FeatureAttribution, RankedHit};

fn chunk(id: &str) -> Chunk {
    Chunk {
        chunk_id: id.into(),
        source: Source::Markdown,
        project: Some("test".into()),
        source_id: format!("{id}.md"),
        source_config_id: "test:cfg".into(),
        chunk_index: 0,
        ts: None,
        role: None,
        text: format!("body text for {id}"),
        sha256: format!("sha-{id}"),
        links: Links::default(),
        facets: FacetSet::default(),
        embedding_input_sha256: format!("emb-{id}"),
        extra: serde_json::Value::Null,
    }
}

/// Build a ranked hit with attention (weight 1.0) + freshness (weight 0.5)
/// contributions — the P9b-full lens default weights — and an embedding for
/// the diversity metric.
fn hit(id: &str, emb: Vec<f32>, attn_raw: f32, fresh_raw: f32) -> RankedHit {
    let a = FeatureAttribution::new(attn_raw, 1.0);
    let f = FeatureAttribution::new(fresh_raw, 0.5);
    let total = a.contribution + f.contribution;
    let mut features: BTreeMap<&'static str, FeatureAttribution> = BTreeMap::new();
    features.insert("attention_affinity", a);
    features.insert("freshness", f);
    let mut c = Candidate::for_chunk(chunk(id));
    c.dense_embedding = Some(emb);
    RankedHit {
        candidate: c,
        total,
        features,
    }
}

#[test]
fn default_portfolio_fills_attention_and_freshness_skips_entity_concept() {
    let hits = vec![
        // Two attention-dominant chunks (highest totals → the 2 attention seats).
        hit("a1", vec![1.0, 0.0, 0.0], 0.9, 0.1),
        hit("a2", vec![1.0, 0.0, 0.0], 0.8, 0.2),
        // Freshness-dominant (attention share well below 0.30).
        hit("f1", vec![1.0, 0.0, 0.0], 0.1, 0.9),
        // A lateral candidate (orthogonal embedding) for the diversity slot.
        hit("d1", vec![0.0, 0.0, 1.0], 0.3, 0.3),
        // Similar-to-selected, low score: eligible nowhere but the diversity
        // slot, where it must LOSE to the lateral d1.
        hit("e1", vec![1.0, 0.0, 0.0], 0.05, 0.05),
    ];

    let entries = allocate_portfolio(&hits, &default_slots(), &LensConfig::default());
    let slots: Vec<&str> = entries.iter().map(|e| e.slot_name).collect();

    // attention ×2, freshness ×1, diversity_jump ×1. Entity/concept skip.
    assert_eq!(
        slots,
        vec!["attention", "attention", "freshness", "diversity_jump"],
        "canonical slot order; entity/concept produce no entries"
    );
    assert!(
        entries
            .iter()
            .all(|e| e.slot_name != "entity" && e.slot_name != "concept"),
        "entity/concept slots skip cleanly when their features are unregistered"
    );

    let by_slot = |s: &str| -> Vec<&str> {
        entries
            .iter()
            .filter(|e| e.slot_name == s)
            .map(|e| e.chunk_id.as_str())
            .collect()
    };
    assert_eq!(by_slot("attention"), vec!["a1", "a2"], "top-2 by total");
    assert_eq!(by_slot("freshness"), vec!["f1"], "freshness-dominant pick");
    assert_eq!(
        by_slot("diversity_jump"),
        vec!["d1"],
        "diversity slot picks the lateral candidate, not the similar low-score e1"
    );
}

#[test]
fn dominance_uses_weighted_contribution_not_raw() {
    // A chunk with a high RAW freshness (0.9) but tiny WEIGHT-scaled
    // contribution must not be treated as freshness-dominant if attention
    // dominates the weighted total. Here attention raw=0.95 weight=1.0
    // (contrib 0.95); freshness raw=0.9 weight=0.05 (contrib 0.045). Total
    // 0.995 → attention share 0.955, freshness share 0.045 (< 0.30). So this
    // chunk is attention-dominant, never freshness-dominant, despite the high
    // raw freshness.
    let a = FeatureAttribution::new(0.95, 1.0);
    let f = FeatureAttribution::new(0.9, 0.05);
    let total = a.contribution + f.contribution;
    let mut features: BTreeMap<&'static str, FeatureAttribution> = BTreeMap::new();
    features.insert("attention_affinity", a);
    features.insert("freshness", f);
    let mut c = Candidate::for_chunk(chunk("x1"));
    c.dense_embedding = Some(vec![1.0, 0.0]);
    let h = RankedHit {
        candidate: c,
        total,
        features,
    };

    let entries = allocate_portfolio(&[h], &default_slots(), &LensConfig::default());
    // The single chunk takes the attention slot (first match). The freshness
    // slot finds nothing dominant, and there are no other candidates for the
    // remaining slots — so attention is the only entry.
    assert_eq!(entries.len(), 1, "only the attention slot fills");
    assert_eq!(entries[0].slot_name, "attention");
    assert!(
        entries.iter().all(|e| e.slot_name != "freshness"),
        "high raw but low weighted freshness must not win the freshness slot"
    );
}
