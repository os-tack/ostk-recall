//! ostk-recall-query — vector + lexical retrieval against the corpus store.
//!
//! Two public entry points:
//! - The free [`recall`] function (preferred for new callers, e.g. the
//!   haystack kernel page-fault path).
//! - [`QueryEngine`], a thin orchestrator that owns the store/embedder
//!   handles and delegates to the free functions. Retained as the stable
//!   shape for the MCP server and integration tests; new code should
//!   prefer the free function.

use std::sync::Arc;

use ostk_recall_store::{EventsDb, IngestDb};

pub mod audit;
pub mod candidate;
pub mod concept;
pub mod context;
pub mod error;
pub mod freshness;
pub mod hybrid;
pub mod lanes;
pub mod lens;
pub mod link;
pub mod rank;
pub mod relational;
pub mod rerank;
mod row;
pub mod stats;
pub mod synthesis;
pub mod types;

pub use candidate::Candidate;
pub use concept::{ConceptSupportFactory, ConceptSupportInstance};
pub use context::{AttentionContext, QueryContext};
pub use error::{QueryError, Result};
pub use freshness::{FreshnessFactory, FreshnessInstance};
pub use hybrid::recall;
pub use lanes::{ambient_candidates, explicit_candidates, relational_candidates};
pub use ostk_recall_core::{
    AuditResult, Chunk, Links, RecallHit, RecallIntent, RecallLinkResult, RecallParams,
    RecallStats, RerankerStats, Source, SourceCount, SynthesizedPage,
};
pub use ostk_recall_pipeline::ChunkEmbedder;
pub use ostk_recall_store::CorpusStore;
pub use rank::{
    FeatureAttribution, FnFactory, RankEngine, RankFeatureFactory, RankFeatureInstance, RankedHit,
    build_engine_from_weights,
};
pub use relational::{
    CodebookEntry, ConceptCodebook, LatentNeighbor, PromotionReport, RelationalLiftFactory,
    RelationalLiftInstance, augment_relational_support_latent, latent_neighbors,
    promote_latent_edges,
};
#[cfg(feature = "reranker")]
pub use rerank::Reranker;
pub use rerank::{RerankerError, RerankerLike};
pub use synthesis::Synthesizer;

/// Orchestrator that owns the resources needed to answer recall queries.
///
/// Wraps the free [`recall`] function plus the `audit`/`link`/`stats`
/// modules so the MCP server and integration tests have a stable
/// surface. The free function is the canonical entry point for new
/// callers — `QueryEngine` exists for ergonomic resource ownership.
pub struct QueryEngine {
    store: Arc<CorpusStore>,
    ingest: Arc<IngestDb>,
    events: Option<Arc<EventsDb>>,
    embedder: Arc<dyn ChunkEmbedder>,
    model: String,
    reranker: Option<Arc<dyn RerankerLike>>,
    /// →1957: corpus root, so stats can surface the watcher's
    /// `watch_status.json` snapshot. Optional — tests and embedded
    /// callers without a watcher skip it.
    corpus_root: Option<std::path::PathBuf>,
}

impl QueryEngine {
    pub fn new(
        store: Arc<CorpusStore>,
        ingest: Arc<IngestDb>,
        events: Option<Arc<EventsDb>>,
        embedder: Arc<dyn ChunkEmbedder>,
        model: impl Into<String>,
    ) -> Self {
        Self {
            store,
            ingest,
            events,
            embedder,
            model: model.into(),
            reranker: None,
            corpus_root: None,
        }
    }

    /// Builder: record the corpus root so `recall_stats` can read the
    /// watcher's `watch_status.json` snapshot (→1957).
    #[must_use]
    pub fn with_corpus_root(mut self, root: std::path::PathBuf) -> Self {
        self.corpus_root = Some(root);
        self
    }

    /// Builder: attach a cross-encoder reranker. When set, hybrid recall
    /// fetches a wider candidate pool from RRF and re-orders the top via
    /// the supplied reranker before truncating to `limit`.
    #[must_use]
    pub fn with_reranker(mut self, reranker: Arc<dyn RerankerLike>) -> Self {
        self.reranker = Some(reranker);
        self
    }

    pub fn reranker(&self) -> Option<&Arc<dyn RerankerLike>> {
        self.reranker.as_ref()
    }

    pub const fn has_audit(&self) -> bool {
        self.events.is_some()
    }

    pub const fn store(&self) -> &Arc<CorpusStore> {
        &self.store
    }

    pub const fn ingest(&self) -> &Arc<IngestDb> {
        &self.ingest
    }

    pub const fn events(&self) -> Option<&Arc<EventsDb>> {
        self.events.as_ref()
    }

    pub const fn embedder(&self) -> &Arc<dyn ChunkEmbedder> {
        &self.embedder
    }

    pub fn model(&self) -> &str {
        &self.model
    }

    /// Hybrid dense + BM25 retrieval. Delegates to the free [`recall`]
    /// function so engine-bound and free callers share one code path.
    pub async fn recall(&self, params: RecallParams) -> Result<Vec<RecallHit>> {
        hybrid::recall(
            self.store.as_ref(),
            self.embedder.as_ref(),
            self.reranker.as_deref(),
            &params,
        )
        .await
    }

    /// Point lookup by chunk id, plus parents.
    pub async fn recall_link(&self, chunk_id: &str) -> Result<RecallLinkResult> {
        link::recall_link(self.store.connection(), chunk_id).await
    }

    /// Corpus stats.
    pub async fn recall_stats(&self) -> Result<RecallStats> {
        stats::recall_stats(
            &self.store,
            &self.ingest,
            self.events.as_deref(),
            &self.model,
            self.reranker.as_deref(),
            self.corpus_root.as_deref(),
        )
        .await
    }

    /// Raw SELECT over `audit_events`. Errors if no `EventsDb` is bound.
    pub fn recall_audit(&self, sql: &str) -> Result<AuditResult> {
        let events = self.events.as_ref().ok_or(QueryError::NoEventsStore)?;
        audit::recall_audit(events, sql)
    }
}

#[cfg(test)]
mod tests;
