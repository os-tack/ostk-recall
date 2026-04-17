//! Storage layer: `LanceDB` table (vector + Tantivy FTS) + two `DuckDB` files
//! (`events.duckdb` for audit firehose, `ingest.duckdb` for idempotency).

pub mod corpus;
pub mod events;
pub mod ingest;
pub mod schema;

pub use corpus::{CorpusStore, StoreError};
pub use events::{AuditEventRow, EventsDb};
pub use ingest::{IngestChunkRow, IngestDb};
pub use schema::{CORPUS_TABLE, corpus_schema};
