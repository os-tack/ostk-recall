//! P7b gate: the ACT-R Freshness feature computes
//! `B_i = ln(1 + Σ weight_k · age_hours_k^{-0.5})` against a fixed access
//! history (via a fake `ChainLogReader`), min-max normalizes across the
//! pool, and handles the degenerate cases (flat pool → 0.5, no
//! events → 0.0, future ts → age floored to 1h).

use std::collections::HashMap;
use std::sync::Arc;

use chrono::{DateTime, Duration, Utc};
use ostk_recall_core::{Chunk, FacetSet, Links, Source};
use ostk_recall_query::{
    AttentionContext, Candidate, FreshnessFactory, QueryContext, RankFeatureFactory,
};
use ostk_recall_store::{AccessKind, AccessWeights, ChainLogReader};

/// A `ChainLogReader` that returns a fixed, pre-baked history regardless
/// of `since` (the test controls ages directly via the event timestamps).
struct FakeChainLog {
    history: HashMap<String, Vec<(AccessKind, DateTime<Utc>)>>,
}

impl ChainLogReader for FakeChainLog {
    fn access_history(
        &self,
        chunk_ids: &[String],
        _since: DateTime<Utc>,
    ) -> std::result::Result<
        HashMap<String, Vec<(AccessKind, DateTime<Utc>)>>,
        ostk_recall_store::StoreError,
    > {
        let wanted: std::collections::HashSet<&str> =
            chunk_ids.iter().map(String::as_str).collect();
        Ok(self
            .history
            .iter()
            .filter(|(k, _)| wanted.contains(k.as_str()))
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect())
    }
}

fn chunk_at(id: &str, ts: Option<DateTime<Utc>>) -> Candidate {
    Candidate::for_chunk(Chunk {
        chunk_id: id.to_string(),
        source: Source::Markdown,
        project: None,
        source_id: format!("src/{id}"),
        source_config_id: "test:cfg".to_string(),
        chunk_index: 0,
        ts,
        role: None,
        text: format!("text {id}"),
        sha256: format!("sha-{id}"),
        links: Links::default(),
        facets: FacetSet::default(),
        embedding_input_sha256: format!("emb-{id}"),
        extra: serde_json::Value::Null,
    })
}

/// Build a freshness instance, run prepare+score over `cands` with the
/// given fake history, and return (chunk_id → score).
async fn run(
    cands: Vec<Candidate>,
    history: HashMap<String, Vec<(AccessKind, DateTime<Utc>)>>,
) -> HashMap<String, f32> {
    let reader: Arc<dyn ChainLogReader> = Arc::new(FakeChainLog { history });
    let attn = AttentionContext::empty().with_chain_log(reader);
    let factory = FreshnessFactory::default(); // weights default, d=0.5, 30d window
    let mut inst = factory.build_instance();
    let mut cands = cands;
    inst.prepare(&mut cands, &QueryContext::Ambient, &attn)
        .await
        .unwrap();
    cands
        .iter()
        .map(|c| {
            (
                c.chunk.chunk_id.clone(),
                inst.score(c, &QueryContext::Ambient, &attn),
            )
        })
        .collect()
}

#[tokio::test]
async fn freshness_normalizes_act_r_activation_across_pool() {
    // Ages chosen so age^-0.5 is clean: 1h→1.0, 4h→0.5, 100h→0.1.
    // Default weights: Creation=1.0, ExplicitRecall=1.0.
    let now = Utc::now();
    let h = |hours: i64| now - Duration::hours(hours);

    // "hot": creation 1h ago + an explicit recall 1h ago.
    //   sum = 1.0·1.0 + 1.0·1.0 = 2.0 → B = ln(3) ≈ 1.0986
    // "warm": creation 4h ago only. sum = 1.0·0.5 = 0.5 → B = ln(1.5) ≈ 0.4055
    // "cold": creation 100h ago only. sum = 1.0·0.1 = 0.1 → B = ln(1.1) ≈ 0.0953
    let cands = vec![
        chunk_at("hot", Some(h(1))),
        chunk_at("warm", Some(h(4))),
        chunk_at("cold", Some(h(100))),
    ];
    let mut history = HashMap::new();
    history.insert("hot".to_string(), vec![(AccessKind::ExplicitRecall, h(1))]);

    let scores = run(cands, history).await;

    let b_hot = (3.0_f32).ln();
    let b_warm = (1.5_f32).ln();
    let b_cold = (1.1_f32).ln();
    let norm = |b: f32| (b - b_cold) / (b_hot - b_cold);

    assert!(
        (scores["hot"] - norm(b_hot)).abs() < 1e-4,
        "hot={}",
        scores["hot"]
    );
    assert!(
        (scores["warm"] - norm(b_warm)).abs() < 1e-4,
        "warm={}",
        scores["warm"]
    );
    assert!(
        (scores["cold"] - norm(b_cold)).abs() < 1e-4,
        "cold={}",
        scores["cold"]
    );
    // Endpoints: most-active → 1.0, least → 0.0.
    assert!((scores["hot"] - 1.0).abs() < 1e-4);
    assert!(scores["cold"].abs() < 1e-4);
    // Monotonic.
    assert!(scores["hot"] > scores["warm"] && scores["warm"] > scores["cold"]);
}

#[tokio::test]
async fn freshness_flat_pool_is_neutral_half() {
    // All candidates identical age, no ledger events → equal B_i → flat
    // pool → every score 0.5 (no discriminating information).
    let now = Utc::now();
    let cands = vec![
        chunk_at("a", Some(now - Duration::hours(10))),
        chunk_at("b", Some(now - Duration::hours(10))),
    ];
    let scores = run(cands, HashMap::new()).await;
    assert!((scores["a"] - 0.5).abs() < 1e-6);
    assert!((scores["b"] - 0.5).abs() < 1e-6);
}

#[tokio::test]
async fn freshness_no_ts_no_events_is_pool_floor() {
    // "ghost" has no ts and no ledger events → B_i = 0.0 (pool min);
    // "real" has a creation access → higher. ghost normalizes to 0.0.
    let now = Utc::now();
    let cands = vec![
        chunk_at("ghost", None),
        chunk_at("real", Some(now - Duration::hours(4))),
    ];
    let scores = run(cands, HashMap::new()).await;
    assert!(scores["ghost"].abs() < 1e-6, "ghost={}", scores["ghost"]);
    assert!(
        (scores["real"] - 1.0).abs() < 1e-6,
        "real={}",
        scores["real"]
    );
}

#[tokio::test]
async fn freshness_future_ts_age_floored_to_one_hour() {
    // A future creation ts (clock skew) must not produce a negative age
    // or NaN — age floors to 1h, same as a just-created chunk. With a
    // peer at 100h, the future-ts chunk is the freshest → 1.0.
    let now = Utc::now();
    let cands = vec![
        chunk_at("future", Some(now + Duration::hours(5))),
        chunk_at("old", Some(now - Duration::hours(100))),
    ];
    let scores = run(cands, HashMap::new()).await;
    assert!(scores["future"].is_finite());
    assert!((scores["future"] - 1.0).abs() < 1e-4);
    assert!(scores["old"].abs() < 1e-4);
}
