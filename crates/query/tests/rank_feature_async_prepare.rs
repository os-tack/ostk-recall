//! P3B gate: `prepare()` is genuinely async — it can `.await` (the whole
//! reason the trait method is async, so enrichment features like MaxSim
//! /Freshness/ConceptSupport can await Lance/SQLite I/O). This feature's
//! `prepare()` awaits a yield point before computing its state; `score()`
//! returns that state, proving the awaited prepare completed before
//! scoring.
//!
//! The companion "no `block_on`" invariant is enforced as a source audit
//! (`grep -rn "block_on" crates/query/src` → zero), not a runtime
//! assertion — you can't assert the absence of a call at runtime.

use std::sync::Arc;

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

struct AsyncPrepareFactory;
struct AsyncPrepareInstance {
    ready: bool,
}

impl RankFeatureFactory for AsyncPrepareFactory {
    fn name(&self) -> &'static str {
        "async_prepare"
    }
    fn build_instance(&self) -> Box<dyn RankFeatureInstance> {
        Box::new(AsyncPrepareInstance { ready: false })
    }
}

#[async_trait]
impl RankFeatureInstance for AsyncPrepareInstance {
    fn name(&self) -> &'static str {
        "async_prepare"
    }

    async fn prepare(
        &mut self,
        _candidates: &mut [Candidate],
        _q: &QueryContext,
        _a: &AttentionContext,
    ) -> Result<()> {
        // A real await point — stands in for Lance/SQLite I/O. The
        // method compiling with `.await` inside IS the async proof; the
        // yield forces the future to actually suspend and resume.
        tokio::task::yield_now().await;
        self.ready = true;
        Ok(())
    }

    fn score(&self, _c: &Candidate, _q: &QueryContext, _a: &AttentionContext) -> f32 {
        if self.ready { 1.0 } else { 0.0 }
    }
}

#[tokio::test]
async fn prepare_can_await_before_scoring() {
    let engine = RankEngine::new().with_factory(Arc::new(AsyncPrepareFactory), 1.0);
    let hits = engine
        .rank(
            vec![cand("a"), cand("b")],
            &QueryContext::Ambient,
            &AttentionContext::empty(),
        )
        .await
        .unwrap();
    assert_eq!(hits.len(), 2);
    for h in &hits {
        assert_eq!(
            h.features.get("async_prepare").unwrap().raw,
            1.0,
            "score must observe the awaited prepare() having completed"
        );
    }
}
