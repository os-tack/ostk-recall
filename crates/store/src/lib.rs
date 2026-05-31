//! Storage layer: `LanceDB` table (vector + Tantivy FTS) + `SQLite` files
//! (`events.sqlite` for audit firehose, `ingest.sqlite` for idempotency,
//! `threads.sqlite` for the attention-substrate threads/evidence ledger).

pub mod corpus;
pub mod events;
pub mod ingest;
pub mod manifest;
pub mod schema;
pub mod threads;

pub use corpus::{ActivityBurst, CorpusStore, StoreError};
pub use events::{AuditEventRow, EventsDb};
pub use ingest::{IngestChunkRow, IngestDb};
pub use manifest::rebuild_ingest_manifest;
pub use schema::{CORPUS_TABLE, corpus_schema};
pub use threads::{
    AccessKind, AccessWeights, AssociationType, ChainEvent, ChainLogReader, ChainSink,
    EvidenceLink, NoopChainSink, ProposedThreadRecord, RelationState, SqliteChainSink,
    TensionState, ThreadRecord, ThreadThreadLink, ThreadsDb,
};
