//! P3B gate: the two-phase `prepare()` → `score()` contract.
//!
//! Proves (1) `prepare()` runs exactly ONCE per `rank()` call (not once
//! per candidate) via a factory-held atomic counter, and (2) `prepare()`
//! runs BEFORE any `score()` and its pool-level result is visible to
//! `score()` — a feature whose `score()` depends on a value computed in
//! `prepare()` returns the prepared value, not the un-prepared default.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use async_trait::async_trait;
use ostk_recall_core::{Chunk, FacetSet, Links, Source};
use ostk_recall_query::{
    AttentionContext, Candidate, QueryContext, RankEngine, RankFeatureFactory, RankFeatureInstance,
    error::Result,
};

fn cand(id: &str) -> Candidate {
    Candidate::for_chunk(Chunk {
        chunk_id: id.to_string(),
        source: Source::Markdown,
        project: None,
        source_id: format!("src/{id}"),
        source_config_id: "test:cfg".to_string(),
        chunk_index: 0,
        ts: None,
        role: None,
        text: format!("text {id}"),
        sha256: format!("sha-{id}"),
        links: Links::default(),
        facets: FacetSet::default(),
        embedding_input_sha256: format!("emb-{id}"),
        extra: serde_json::Value::Null,
    })
}

/// Factory holds a shared counter so the test can observe how many times
/// `prepare()` ran across one `rank()`. The instance computes a
/// pool-level value (candidate count) in `prepare()` and scores from it.
struct PoolCountFactory {
    prepares: Arc<AtomicUsize>,
}

struct PoolCountInstance {
    prepares: Arc<AtomicUsize>,
    /// `None` until `prepare()` runs — proves order.
    pool_size: Option<usize>,
}

impl RankFeatureFactory for PoolCountFactory {
    fn name(&self) -> &'static str {
        "pool_count"
    }
    fn build_instance(&self) -> Box<dyn RankFeatureInstance> {
        Box::new(PoolCountInstance {
            prepares: Arc::clone(&self.prepares),
            pool_size: None,
        })
    }
}

#[async_trait]
impl RankFeatureInstance for PoolCountInstance {
    fn name(&self) -> &'static str {
        "pool_count"
    }

    async fn prepare(
        &mut self,
        candidates: &mut [Candidate],
        _q: &QueryContext,
        _a: &AttentionContext,
    ) -> Result<()> {
        self.prepares.fetch_add(1, Ordering::SeqCst);
        self.pool_size = Some(candidates.len());
        Ok(())
    }

    #[allow(clippy::cast_precision_loss)]
    fn score(&self, _c: &Candidate, _q: &QueryContext, _a: &AttentionContext) -> f32 {
        // Returns the pool fraction iff prepare ran; 0.0 otherwise.
        // With 4 candidates → 0.4 for every candidate.
        self.pool_size.map_or(0.0, |n| (n as f32) / 10.0)
    }
}

#[tokio::test]
async fn prepare_runs_once_per_query_before_scoring() {
    let prepares = Arc::new(AtomicUsize::new(0));
    let engine = RankEngine::new().with_factory(
        Arc::new(PoolCountFactory {
            prepares: Arc::clone(&prepares),
        }),
        1.0,
    );

    let cands = vec![cand("a"), cand("b"), cand("c"), cand("d")];
    let hits = engine
        .rank(cands, &QueryContext::Ambient, &AttentionContext::empty())
        .await
        .unwrap();

    // prepare() ran exactly once for the whole pool — NOT once per candidate.
    assert_eq!(
        prepares.load(Ordering::SeqCst),
        1,
        "prepare must run once per rank() call, not per candidate"
    );

    // Every candidate's score reflects the pool-level value computed in
    // prepare() (4 candidates → 0.4). A 0.0 here would mean score() ran
    // before prepare() populated pool_size.
    assert_eq!(hits.len(), 4);
    for h in &hits {
        let raw = h.features.get("pool_count").unwrap().raw;
        assert!(
            (raw - 0.4).abs() < 1e-6,
            "score must see prepare()'s pool_size; got {raw}"
        );
    }
}
