//! Manifest rebuild — reconstruct `ingest.sqlite` from `corpus.lance` alone.
//!
//! P0 ships a `manifest rebuild` recovery path so a corpus that lost its
//! SQLite ledger (disk failure, partial delete, copy-without-sidecar) can
//! be made consistent again without re-embedding. The Lance corpus row
//! carries every field needed: `chunk_id`, `source`, `source_id`,
//! `source_config_id`, `chunk_index`, `sha256`.

use arrow_array::{Array, RecordBatch, StringArray, UInt32Array};
use futures::TryStreamExt;
use lancedb::query::{ExecutableQuery, QueryBase, Select};

use crate::corpus::{CorpusStore, Result, StoreError};
use crate::ingest::{IngestChunkRow, IngestDb};
use crate::schema::CORPUS_TABLE;

/// Rebuild `ingest.sqlite` rows from `corpus.lance`. Returns the number
/// of `ingest_chunks` rows written.
///
/// Skips rows whose `source_config_id` is NULL (legacy v0.5 rows that
/// haven't been migrated). Callers should typically open a fresh
/// `ingest.sqlite` and pass it in; rebuilding on top of an existing
/// ledger is supported (`INSERT OR REPLACE`) but the use case is
/// usually "lost or corrupted ledger".
///
/// `run_id` is the synthesized provenance tag stamped on every row.
/// Operators rerun a full scan after rebuild to update `last_run_id` to
/// the live run.
pub async fn rebuild_ingest_manifest(
    corpus: &CorpusStore,
    ingest: &IngestDb,
    run_id: &str,
) -> Result<usize> {
    let table = corpus
        .connection()
        .open_table(CORPUS_TABLE)
        .execute()
        .await?;
    let stream = table
        .query()
        .select(Select::Columns(vec![
            "chunk_id".into(),
            "source".into(),
            "source_id".into(),
            "source_config_id".into(),
            "chunk_index".into(),
            "sha256".into(),
        ]))
        .execute()
        .await?;

    let batches: Vec<RecordBatch> = stream
        .try_collect()
        .await
        .map_err(|e| StoreError::Lance(e))?;

    let mut written = 0usize;
    for batch in &batches {
        let chunk_ids = batch
            .column_by_name("chunk_id")
            .ok_or_else(|| StoreError::InvalidEnumValue {
                field: "chunk_id".into(),
                value: "missing column".into(),
            })?
            .as_any()
            .downcast_ref::<StringArray>()
            .expect("chunk_id is Utf8");
        let sources = batch
            .column_by_name("source")
            .expect("source column")
            .as_any()
            .downcast_ref::<StringArray>()
            .expect("source is Utf8");
        let source_ids = batch
            .column_by_name("source_id")
            .expect("source_id column")
            .as_any()
            .downcast_ref::<StringArray>()
            .expect("source_id is Utf8");
        let source_cfg_ids = batch
            .column_by_name("source_config_id")
            .expect("source_config_id column")
            .as_any()
            .downcast_ref::<StringArray>()
            .expect("source_config_id is Utf8");
        let chunk_indices = batch
            .column_by_name("chunk_index")
            .expect("chunk_index column")
            .as_any()
            .downcast_ref::<UInt32Array>()
            .expect("chunk_index is UInt32");
        let shas = batch
            .column_by_name("sha256")
            .expect("sha256 column")
            .as_any()
            .downcast_ref::<StringArray>()
            .expect("sha256 is Utf8");

        for i in 0..batch.num_rows() {
            // Skip pre-migration rows where source_config_id is NULL.
            if source_cfg_ids.is_null(i) {
                continue;
            }
            let row = IngestChunkRow {
                chunk_id: chunk_ids.value(i).to_string(),
                source: sources.value(i).to_string(),
                source_id: source_ids.value(i).to_string(),
                source_config_id: source_cfg_ids.value(i).to_string(),
                chunk_index: chunk_indices.value(i),
                content_sha256: shas.value(i).to_string(),
                // Rebuild can't reconstruct the embedding-input hash from
                // corpus.lance alone (the header version + model id aren't
                // stored). Leave empty so the next scan force-recomputes
                // it; users who never re-tag won't notice.
                embedding_input_sha256: String::new(),
            };
            ingest.record_chunk(&row, Some(run_id))?;
            // Reconstruct an ingest_sources row keyed on
            // (source, source_id, source_config_id). Mtime/size are
            // unknown after rebuild; stamp 0/0 — a follow-up scan will
            // refresh them on the next file visit.
            ingest.update_source_metadata(
                &row.source,
                &row.source_config_id,
                &row.source_id,
                0,
                0,
                "",
                run_id,
            )?;
            written += 1;
        }
    }
    Ok(written)
}
