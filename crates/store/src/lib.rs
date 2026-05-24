//! Storage layer: `LanceDB` table (vector + Tantivy FTS) + `SQLite` files
//! (`events.sqlite` for audit firehose, `ingest.sqlite` for idempotency,
//! `threads.sqlite` for the attention-substrate threads/evidence ledger).

pub mod corpus;
pub mod events;
pub mod ingest;
pub mod schema;
pub mod threads;

pub use corpus::{CorpusStore, StoreError};
pub use events::{AuditEventRow, EventsDb};
pub use ingest::{IngestChunkRow, IngestDb};
pub use schema::{CORPUS_TABLE, corpus_schema};
pub use threads::{
    AssociationType, ChainEvent, ChainSink, EvidenceLink, NoopChainSink, RelationState,
    TensionState, ThreadRecord, ThreadsDb,
};
