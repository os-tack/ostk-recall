//! Manifest rebuild — reconstruct `ingest.sqlite` from `corpus.lance` alone.
//!
//! P0 ships a `manifest rebuild` recovery path so a corpus that lost its
//! SQLite ledger (disk failure, partial delete, copy-without-sidecar) can
//! be made consistent again without re-embedding. The Lance corpus row
//! carries every field needed: `chunk_id`, `source`, `source_id`,
//! `source_config_id`, `chunk_index`, `sha256`.

use std::collections::{HashMap, HashSet};

use arrow_array::{Array, RecordBatch, StringArray, UInt32Array};
use futures::TryStreamExt;
use lancedb::query::{ExecutableQuery, QueryBase, Select};

use crate::corpus::{CorpusStore, Result, StoreError};
use crate::ingest::{IngestChunkRow, IngestDb};
use crate::schema::CORPUS_TABLE;

#[derive(Debug, Clone)]
struct CorpusManifestRow {
    chunk_id: String,
    source: String,
    source_id: String,
    source_config_id: String,
    chunk_index: u32,
    content_sha256: String,
}

/// Summary for a targeted manifest drift repair.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ManifestRepairReport {
    pub corpus_rows: usize,
    pub corpus_unique_chunk_ids: usize,
    pub ingest_rows: usize,
    pub missing_ingest_rows: usize,
    pub written_ingest_rows: usize,
    pub extra_ingest_rows: usize,
    pub duplicate_corpus_chunk_ids: usize,
    pub duplicate_corpus_rows: usize,
    pub unrepairable_missing_rows: usize,
    pub unrepairable_duplicate_chunk_ids: usize,
    pub deduped_corpus_chunk_ids: usize,
    pub deleted_corpus_rows: u64,
    pub reinserted_corpus_rows: usize,
    pub dry_run: bool,
}

impl ManifestRepairReport {
    /// True when drift can be repaired without deleting ingest ledger rows.
    #[must_use]
    pub const fn repairable_without_ingest_delete(&self) -> bool {
        self.unrepairable_duplicate_chunk_ids == 0
            && self.extra_ingest_rows == 0
            && self.unrepairable_missing_rows == 0
    }
}

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

/// Repair row-count drift by inserting only the `ingest_chunks` rows that are
/// present in `corpus.lance` but missing from `ingest.sqlite`.
///
/// This is intentionally narrower than [`rebuild_ingest_manifest`]: it leaves
/// existing ledger rows untouched, preserving their `embedding_input_sha256`
/// and source cursor metadata. It is the right tool when `verify` reports
/// `corpus_rows > ingest_rows` by a small amount after an interrupted write.
///
/// If the count mismatch is caused by duplicate Lance `chunk_id` rows, extra
/// SQLite rows, or legacy corpus rows without `source_config_id`, the report
/// says so and no insert-only repair is attempted.
pub async fn repair_ingest_manifest_drift(
    corpus: &CorpusStore,
    ingest: &IngestDb,
    run_id: &str,
    dry_run: bool,
) -> Result<ManifestRepairReport> {
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
    let batches: Vec<RecordBatch> = stream.try_collect().await.map_err(StoreError::Lance)?;

    let mut corpus_rows = 0usize;
    let mut corpus_ids: HashSet<String> = HashSet::new();
    let mut repairable_rows: HashMap<String, CorpusManifestRow> = HashMap::new();
    let mut duplicate_ids: HashSet<String> = HashSet::new();
    let mut duplicate_rows = 0usize;

    for batch in &batches {
        let chunk_ids = required_string_array(batch, "chunk_id")?;
        let sources = required_string_array(batch, "source")?;
        let source_ids = required_string_array(batch, "source_id")?;
        let source_cfg_ids = required_string_array(batch, "source_config_id")?;
        let chunk_indices = batch
            .column_by_name("chunk_index")
            .ok_or_else(|| StoreError::InvalidEnumValue {
                field: "chunk_index".into(),
                value: "missing column".into(),
            })?
            .as_any()
            .downcast_ref::<UInt32Array>()
            .expect("chunk_index is UInt32");
        let shas = required_string_array(batch, "sha256")?;

        for i in 0..batch.num_rows() {
            corpus_rows += 1;
            let chunk_id = chunk_ids.value(i).to_string();
            if !corpus_ids.insert(chunk_id.clone()) {
                duplicate_ids.insert(chunk_id.clone());
                duplicate_rows += 1;
            }

            if source_cfg_ids.is_null(i) {
                continue;
            }

            repairable_rows
                .entry(chunk_id.clone())
                .or_insert_with(|| CorpusManifestRow {
                    chunk_id,
                    source: sources.value(i).to_string(),
                    source_id: source_ids.value(i).to_string(),
                    source_config_id: source_cfg_ids.value(i).to_string(),
                    chunk_index: chunk_indices.value(i),
                    content_sha256: shas.value(i).to_string(),
                });
        }
    }

    let ingest_ids_vec = ingest.all_chunk_ids()?;
    let ingest_rows = ingest_ids_vec.len();
    let ingest_ids: HashSet<String> = ingest_ids_vec.into_iter().collect();

    let missing_ids: Vec<String> = corpus_ids
        .difference(&ingest_ids)
        .map(ToOwned::to_owned)
        .collect();
    let extra_ingest_rows = ingest_ids.difference(&corpus_ids).count();
    let unrepairable_missing_rows = missing_ids
        .iter()
        .filter(|id| !repairable_rows.contains_key(*id))
        .count();
    let mut duplicate_id_vec: Vec<String> = duplicate_ids.iter().cloned().collect();
    duplicate_id_vec.sort();

    let mut report = ManifestRepairReport {
        corpus_rows,
        corpus_unique_chunk_ids: corpus_ids.len(),
        ingest_rows,
        missing_ingest_rows: missing_ids.len(),
        written_ingest_rows: 0,
        extra_ingest_rows,
        duplicate_corpus_chunk_ids: duplicate_ids.len(),
        duplicate_corpus_rows: duplicate_rows,
        unrepairable_missing_rows,
        unrepairable_duplicate_chunk_ids: 0,
        deduped_corpus_chunk_ids: 0,
        deleted_corpus_rows: 0,
        reinserted_corpus_rows: 0,
        dry_run,
    };

    if !report.repairable_without_ingest_delete() || dry_run {
        return Ok(report);
    }

    let mut dedup_chunks = Vec::new();
    let mut dedup_embeddings = Vec::new();
    if !duplicate_id_vec.is_empty() {
        let fetched = corpus.fetch_chunks_by_ids(&duplicate_id_vec).await?;
        for id in &duplicate_id_vec {
            match fetched.get(id) {
                Some((chunk, Some(embedding))) => {
                    dedup_chunks.push(chunk.clone());
                    dedup_embeddings.push(embedding.clone());
                }
                _ => {
                    report.unrepairable_duplicate_chunk_ids += 1;
                }
            }
        }
        if report.unrepairable_duplicate_chunk_ids > 0 {
            return Ok(report);
        }
    }

    if !dedup_chunks.is_empty() {
        report.deleted_corpus_rows = corpus.delete_chunks(&duplicate_id_vec).await?;
        report.reinserted_corpus_rows = corpus.upsert(&dedup_chunks, &dedup_embeddings).await?;
        report.deduped_corpus_chunk_ids = dedup_chunks.len();
    }

    for id in &missing_ids {
        let Some(row) = repairable_rows.get(id) else {
            continue;
        };
        ingest.record_chunk(
            &IngestChunkRow {
                chunk_id: row.chunk_id.clone(),
                source: row.source.clone(),
                source_id: row.source_id.clone(),
                source_config_id: row.source_config_id.clone(),
                chunk_index: row.chunk_index,
                content_sha256: row.content_sha256.clone(),
                // We cannot reconstruct the embedding-input hash from Lance
                // alone. Preserve existing rows by only inserting missing
                // rows; a future scan can refresh these small repairs.
                embedding_input_sha256: String::new(),
            },
            Some(run_id),
        )?;
        ingest.update_source_metadata(
            &row.source,
            &row.source_config_id,
            &row.source_id,
            0,
            0,
            "",
            run_id,
        )?;
        report.written_ingest_rows += 1;
    }

    Ok(report)
}

fn required_string_array<'a>(batch: &'a RecordBatch, name: &str) -> Result<&'a StringArray> {
    let col = batch
        .column_by_name(name)
        .ok_or_else(|| StoreError::InvalidEnumValue {
            field: name.into(),
            value: "missing column".into(),
        })?;
    col.as_any().downcast_ref::<StringArray>().ok_or_else(|| {
        StoreError::Arrow(arrow::error::ArrowError::CastError(format!(
            "{name} expected to be Utf8"
        )))
    })
}
