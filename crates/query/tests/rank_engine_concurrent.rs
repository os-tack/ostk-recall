//! P3B gate (critical): concurrent `rank()` calls must not cross-
//! contaminate per-query feature state. Each query carries a unique
//! token; the feature stores that token in `prepare()` and scores a
//! candidate `1.0` iff its chunk_id matches the query's OWN token. If
//! instance state leaked between concurrent calls, some candidate would
//! be scored against the wrong token and drop to `0.0`.
//!
//! A per-query token discriminator beats a candidate-count probe: counts
//! collide across calls and hide leaks, whereas a mismatched token is
//! always observable.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use async_trait::async_trait;
use ostk_recall_core::{Chunk, FacetSet, Links, Source};
use ostk_recall_query::{
    AttentionContext, Candidate, QueryContext, RankEngine, RankFeatureFactory, RankFeatureInstance,
    error::Result,
};
use tokio::task::JoinSet;

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

struct TokenEchoFactory {
    prepares: Arc<AtomicUsize>,
}

struct TokenEchoInstance {
    prepares: Arc<AtomicUsize>,
    token: String,
}

impl RankFeatureFactory for TokenEchoFactory {
    fn name(&self) -> &'static str {
        "token_echo"
    }
    fn build_instance(&self) -> Box<dyn RankFeatureInstance> {
        Box::new(TokenEchoInstance {
            prepares: Arc::clone(&self.prepares),
            token: String::new(),
        })
    }
}

#[async_trait]
impl RankFeatureInstance for TokenEchoInstance {
    fn name(&self) -> &'static str {
        "token_echo"
    }

    async fn prepare(
        &mut self,
        _candidates: &mut [Candidate],
        query: &QueryContext,
        _attn: &AttentionContext,
    ) -> Result<()> {
        // Read THIS query's token into per-instance state.
        self.token = query.query_text().unwrap_or_default().to_string();
        self.prepares.fetch_add(1, Ordering::SeqCst);
        // Force a suspension so the runtime interleaves the concurrent
        // calls — any shared-state bug manifests here.
        tokio::task::yield_now().await;
        Ok(())
    }

    fn score(&self, candidate: &Candidate, _q: &QueryContext, _a: &AttentionContext) -> f32 {
        // 1.0 iff this candidate belongs to the query this instance prepared.
        if candidate.chunk.chunk_id.starts_with(&self.token) {
            1.0
        } else {
            0.0
        }
    }
}

const N: usize = 8;

#[tokio::test]
async fn concurrent_rank_calls_do_not_cross_contaminate() {
    let prepares = Arc::new(AtomicUsize::new(0));
    let engine = Arc::new(RankEngine::new().with_factory(
        Arc::new(TokenEchoFactory {
            prepares: Arc::clone(&prepares),
        }),
        1.0,
    ));

    let mut set: JoinSet<(usize, Vec<f32>)> = JoinSet::new();
    for i in 0..N {
        let engine = Arc::clone(&engine);
        set.spawn(async move {
            let token = format!("q{i}");
            let query = QueryContext::explicit(token.clone(), vec![0.1, 0.2]);
            let cands = vec![cand(&format!("{token}-a")), cand(&format!("{token}-b"))];
            let hits = engine
                .rank(cands, &query, &AttentionContext::empty())
                .await
                .unwrap();
            let scores: Vec<f32> = hits
                .iter()
                .map(|h| h.features.get("token_echo").unwrap().raw)
                .collect();
            (i, scores)
        });
    }

    let mut completed = 0;
    while let Some(res) = set.join_next().await {
        let (i, scores) = res.unwrap();
        completed += 1;
        // Every candidate of query i matched query i's OWN token. A leak
        // (instance i scoring against query j's token) would yield 0.0.
        assert!(
            scores.iter().all(|s| (*s - 1.0).abs() < 1e-6),
            "query {i} saw cross-contaminated token state: {scores:?}"
        );
    }
    assert_eq!(completed, N, "all {N} concurrent rank calls completed");
    assert_eq!(
        prepares.load(Ordering::SeqCst),
        N,
        "exactly one prepare() per rank() call"
    );
}
