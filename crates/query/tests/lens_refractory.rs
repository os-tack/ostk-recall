//! P9b-full gate — refractory penalty.
//!
//! A chunk surfaced in a recent lens (`LensIncluded` event) is decayed by
//! `weight * exp(-Δt / τ)` so the portfolio rotates instead of repeating
//! itself. Verifies: the penalty is an attributed *negative* `refractory`
//! row that preserves `total = Σ contribution`; a recently-included chunk is
//! demoted below a fresh one; more-recent inclusion decays harder; and
//! `weight = 0` is a no-op.

use std::collections::{HashMap, HashSet};

use chrono::{DateTime, Duration, Utc};
use ostk_recall_core::{Chunk, FacetSet, Links, Source};
use ostk_recall_store::{AccessKind, ChainLogReader};

use ostk_recall_query::candidate::Candidate;
use ostk_recall_query::lens::apply_refractory;
use ostk_recall_query::rank::{FeatureAttribution, RankedHit};

/// In-memory `ChainLogReader` returning canned access events — gives exact
/// control over event timestamps without depending on how the SQLite sink
/// stamps rows.
struct MockLedger {
    events: HashMap<String, Vec<(AccessKind, DateTime<Utc>)>>,
}

impl ChainLogReader for MockLedger {
    fn access_history(
        &self,
        chunk_ids: &[String],
        _since: DateTime<Utc>,
    ) -> ostk_recall_store::corpus::Result<HashMap<String, Vec<(AccessKind, DateTime<Utc>)>>> {
        let wanted: HashSet<&str> = chunk_ids.iter().map(String::as_str).collect();
        Ok(self
            .events
            .iter()
            .filter(|(id, _)| wanted.contains(id.as_str()))
            .map(|(id, evs)| (id.clone(), evs.clone()))
            .collect())
    }
}

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

/// Single attention_affinity feature (weight 1.0) ⇒ `total == attn_raw`.
fn hit(id: &str, attn_raw: f32) -> RankedHit {
    let attr = FeatureAttribution::new(attn_raw, 1.0);
    let total = attr.contribution;
    let mut features = std::collections::BTreeMap::new();
    features.insert("attention_affinity", attr);
    RankedHit {
        candidate: Candidate::for_chunk(chunk(id)),
        total,
        features,
    }
}

#[test]
fn recently_lens_included_chunk_is_penalized_and_demoted() {
    let now = Utc::now();
    let mut events = HashMap::new();
    events.insert(
        "c_stale".to_string(),
        vec![(AccessKind::LensIncluded, now - Duration::minutes(5))],
    );
    let ledger = MockLedger { events };

    // Equal attention (0.8). Before refractory the tie breaks on chunk_id
    // (c_fresh < c_stale); after refractory c_stale sinks below c_fresh.
    let ranked = vec![hit("c_stale", 0.8), hit("c_fresh", 0.8)];
    let out = apply_refractory(ranked, &ledger, 3600, 0.5);

    assert_eq!(
        out[0].candidate.chunk.chunk_id, "c_fresh",
        "fresh ranks first"
    );
    assert_eq!(
        out[1].candidate.chunk.chunk_id, "c_stale",
        "penalized sinks"
    );

    let stale = out
        .iter()
        .find(|h| h.candidate.chunk.chunk_id == "c_stale")
        .unwrap();
    let refr = stale
        .features
        .get("refractory")
        .expect("penalized chunk carries an attributed refractory row");
    assert!(
        refr.contribution < 0.0,
        "refractory contribution is negative"
    );
    assert!(stale.total < 0.8, "total visibly reduced: {}", stale.total);
    // Σ-contribution invariant survives the post-rank adjustment.
    let sum: f32 = stale.features.values().map(|a| a.contribution).sum();
    assert!(
        (stale.total - sum).abs() < 1e-5,
        "total ({}) == Σ contribution ({sum})",
        stale.total
    );

    let fresh = out
        .iter()
        .find(|h| h.candidate.chunk.chunk_id == "c_fresh")
        .unwrap();
    assert!(
        fresh.features.get("refractory").is_none(),
        "never-included chunk gets no penalty"
    );
    assert!((fresh.total - 0.8).abs() < 1e-6);
}

#[test]
fn more_recent_inclusion_decays_harder() {
    let now = Utc::now();
    let mut events = HashMap::new();
    events.insert(
        "recent".to_string(),
        vec![(AccessKind::LensIncluded, now - Duration::minutes(1))],
    );
    events.insert(
        "older".to_string(),
        vec![(AccessKind::LensIncluded, now - Duration::minutes(50))],
    );
    let ledger = MockLedger { events };

    let out = apply_refractory(
        vec![hit("recent", 0.9), hit("older", 0.9)],
        &ledger,
        3600,
        0.5,
    );
    let recent = out
        .iter()
        .find(|h| h.candidate.chunk.chunk_id == "recent")
        .unwrap();
    let older = out
        .iter()
        .find(|h| h.candidate.chunk.chunk_id == "older")
        .unwrap();
    assert!(
        recent.total < older.total,
        "more-recent inclusion decays harder: recent {} should be < older {}",
        recent.total,
        older.total
    );
}

#[test]
fn zero_weight_is_a_noop() {
    let now = Utc::now();
    let mut events = HashMap::new();
    events.insert("c1".to_string(), vec![(AccessKind::LensIncluded, now)]);
    let ledger = MockLedger { events };
    let out = apply_refractory(vec![hit("c1", 0.5)], &ledger, 3600, 0.0);
    assert!(
        out[0].features.get("refractory").is_none(),
        "weight 0 disables the refractory stage"
    );
    assert!((out[0].total - 0.5).abs() < 1e-6);
}

#[test]
fn non_lens_access_kinds_do_not_penalize() {
    // Only LensIncluded events decay the lens. An ExplicitRecall (proven-useful
    // memory) must NOT be refractory-penalized.
    let now = Utc::now();
    let mut events = HashMap::new();
    events.insert(
        "c1".to_string(),
        vec![(AccessKind::ExplicitRecall, now - Duration::minutes(2))],
    );
    let ledger = MockLedger { events };
    let out = apply_refractory(vec![hit("c1", 0.7)], &ledger, 3600, 0.5);
    assert!(
        out[0].features.get("refractory").is_none(),
        "ExplicitRecall is not a refractory signal"
    );
    assert!((out[0].total - 0.7).abs() < 1e-6);
}
