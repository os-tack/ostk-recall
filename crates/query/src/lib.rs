//! ostk-recall-query — vector + lexical retrieval against the corpus store.
//!
//! Public entrypoint is [`QueryEngine`], which bundles a `CorpusStore`, the
//! `IngestDb` (used for stats), an optional `EventsDb` (used for `recall_audit`),
//! and an embedder implementing [`ChunkEmbedder`].

use std::sync::Arc;

use ostk_recall_pipeline::ChunkEmbedder;
use ostk_recall_store::{CorpusStore, EventsDb, IngestDb};

pub mod audit;
pub mod error;
pub mod hybrid;
pub mod link;
pub mod rerank;
mod row;
pub mod stats;
pub mod types;

pub use error::{QueryError, Result};
pub use rerank::{Reranker, RerankerError, RerankerLike};
pub use types::{
    AuditResult, RecallHit, RecallLinkResult, RecallParams, RecallStats, RerankerStats, SourceCount,
};

/// Orchestrator that owns the resources needed to answer recall queries.
pub struct QueryEngine {
    store: Arc<CorpusStore>,
    ingest: Arc<IngestDb>,
    events: Option<Arc<EventsDb>>,
    embedder: Arc<dyn ChunkEmbedder>,
    model: String,
    reranker: Option<Arc<dyn RerankerLike>>,
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
        }
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

    /// Hybrid dense + BM25 retrieval. If a reranker is attached, the top
    /// RRF candidates are re-scored by the cross-encoder before truncation.
    pub async fn recall(&self, params: RecallParams) -> Result<Vec<RecallHit>> {
        hybrid::recall(
            self.store.connection(),
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
            &self.model,
            self.reranker.as_deref(),
        )
        .await
    }

    /// Raw `DuckDB` SELECT over `audit_events`.
    pub fn recall_audit(&self, sql: &str) -> Result<AuditResult> {
        let events = self.events.as_ref().ok_or(QueryError::NoEventsStore)?;
        audit::recall_audit(events, sql)
    }
}

#[cfg(test)]
mod tests;
