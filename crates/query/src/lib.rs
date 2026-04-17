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
mod row;
pub mod stats;
pub mod types;

pub use error::{QueryError, Result};
pub use types::{AuditResult, RecallHit, RecallLinkResult, RecallParams, RecallStats, SourceCount};

/// Orchestrator that owns the resources needed to answer recall queries.
pub struct QueryEngine {
    store: Arc<CorpusStore>,
    ingest: Arc<IngestDb>,
    events: Option<Arc<EventsDb>>,
    embedder: Arc<dyn ChunkEmbedder>,
    model: String,
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
        }
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

    /// Hybrid dense + BM25 retrieval.
    pub async fn recall(&self, params: RecallParams) -> Result<Vec<RecallHit>> {
        hybrid::recall(self.store.connection(), self.embedder.as_ref(), &params).await
    }

    /// Point lookup by chunk id, plus parents.
    pub async fn recall_link(&self, chunk_id: &str) -> Result<RecallLinkResult> {
        link::recall_link(self.store.connection(), chunk_id).await
    }

    /// Corpus stats.
    pub async fn recall_stats(&self) -> Result<RecallStats> {
        stats::recall_stats(&self.store, &self.ingest, &self.model).await
    }

    /// Raw `DuckDB` SELECT over `audit_events`.
    pub fn recall_audit(&self, sql: &str) -> Result<AuditResult> {
        let events = self.events.as_ref().ok_or(QueryError::NoEventsStore)?;
        audit::recall_audit(events, sql)
    }
}

#[cfg(test)]
mod tests;
