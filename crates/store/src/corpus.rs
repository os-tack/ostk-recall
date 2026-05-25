use std::path::{Path, PathBuf};
use std::sync::Arc;

use arrow_array::{
    BooleanArray, FixedSizeListArray, RecordBatch, RecordBatchIterator, RecordBatchReader,
    StringArray, TimestampMicrosecondArray, UInt32Array,
};
use arrow_schema::Schema;
use futures::TryStreamExt;
use lancedb::Connection;
use lancedb::index::Index;
use lancedb::index::scalar::FtsIndexBuilder;
use lancedb::query::{ExecutableQuery, QueryBase, Select};
use lancedb::table::OptimizeAction;
use ostk_recall_core::Chunk;
use thiserror::Error;

use crate::schema::{CORPUS_TABLE, corpus_schema};

#[derive(Debug, Error)]
pub enum StoreError {
    #[error("lancedb: {0}")]
    Lance(#[from] lancedb::Error),

    #[error("arrow: {0}")]
    Arrow(#[from] arrow::error::ArrowError),

    #[error("sqlite: {0}")]
    Sqlite(rusqlite::Error),

    #[error("UNIQUE constraint violated on {table}: {constraint}")]
    UniqueViolation { table: String, constraint: String },

    #[error("invalid {field} value {value:?}")]
    InvalidEnumValue { field: String, value: String },

    #[error("schema mismatch: table dim {have}, embedder dim {want}")]
    DimMismatch { have: usize, want: usize },

    #[error("io: {0}")]
    Io(#[from] std::io::Error),

    #[error("json: {0}")]
    Json(#[from] serde_json::Error),
}

impl From<rusqlite::Error> for StoreError {
    fn from(e: rusqlite::Error) -> Self {
        if let rusqlite::Error::SqliteFailure(code, ref msg) = e {
            if code.code == rusqlite::ErrorCode::ConstraintViolation
                && code.extended_code == rusqlite::ffi::SQLITE_CONSTRAINT_UNIQUE
            {
                let constraint = msg.clone().unwrap_or_else(|| "UNIQUE".into());
                let table = constraint
                    .split('.')
                    .next()
                    .map_or_else(|| "unknown".into(), str::to_string);
                return Self::UniqueViolation { table, constraint };
            }
        }
        Self::Sqlite(e)
    }
}

pub type Result<T> = std::result::Result<T, StoreError>;

/// LanceDB-backed corpus store.
pub struct CorpusStore {
    conn: Connection,
    dim: usize,
    root: PathBuf,
}

impl CorpusStore {
    /// Open (or create) the `LanceDB` connection at `<root>/corpus.lance`. The
    /// corpus table is created with the given embedding dim if absent.
    pub async fn open_or_create(root: &Path, dim: usize) -> Result<Self> {
        tokio::fs::create_dir_all(root).await?;
        let db_dir = root.join("corpus.lance");
        let uri = db_dir.to_string_lossy().into_owned();
        let conn = lancedb::connect(&uri).execute().await?;

        let table_names = conn.table_names().execute().await?;
        if !table_names.iter().any(|n| n == CORPUS_TABLE) {
            tracing::info!(dim, uri = %uri, "creating corpus table");
            let schema = corpus_schema(dim);
            let empty = RecordBatchIterator::new(
                std::iter::empty::<arrow::error::Result<RecordBatch>>(),
                schema.clone(),
            );
            let reader: Box<dyn RecordBatchReader + Send> = Box::new(empty);
            conn.create_table(CORPUS_TABLE, reader).execute().await?;
        }

        Ok(Self {
            conn,
            dim,
            root: root.to_path_buf(),
        })
    }

    pub const fn dim(&self) -> usize {
        self.dim
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub const fn connection(&self) -> &Connection {
        &self.conn
    }

    /// Count of rows in the corpus table.
    pub async fn row_count(&self) -> Result<usize> {
        let table = self.conn.open_table(CORPUS_TABLE).execute().await?;
        let n = table.count_rows(None).await?;
        Ok(n)
    }

    /// Ensure the Tantivy FTS index exists on the `text` column. Idempotent —
    /// no-op if already present. `full_text_search` in lancedb requires this.
    pub async fn ensure_fts_index(&self) -> Result<()> {
        let table = self.conn.open_table(CORPUS_TABLE).execute().await?;
        let indices = table.list_indices().await?;
        if indices
            .iter()
            .any(|ix| ix.columns.iter().any(|c| c == "text"))
        {
            return Ok(());
        }
        tracing::info!("creating FTS index on corpus.text");
        table
            .create_index(&["text"], Index::FTS(FtsIndexBuilder::default()))
            .execute()
            .await?;
        Ok(())
    }

    /// Upsert a batch of chunks + their embeddings. The two slices must align.
    pub async fn upsert(&self, chunks: &[Chunk], embeddings: &[Vec<f32>]) -> Result<usize> {
        debug_assert_eq!(chunks.len(), embeddings.len());
        if chunks.is_empty() {
            return Ok(0);
        }

        let schema = corpus_schema(self.dim);
        let batch = build_record_batch(&schema, chunks, embeddings, self.dim)?;
        let table = self.conn.open_table(CORPUS_TABLE).execute().await?;

        let reader = RecordBatchIterator::new(
            std::iter::once(Ok::<_, arrow::error::ArrowError>(batch)),
            schema,
        );
        let boxed_reader: Box<dyn RecordBatchReader + Send> = Box::new(reader);

        let mut builder = table.merge_insert(&["chunk_id"]);
        builder
            .when_matched_update_all(None)
            .when_not_matched_insert_all();
        builder.execute(boxed_reader).await?;

        Ok(chunks.len())
    }

    /// Return all `chunk_id`s in the corpus whose `project` column equals
    /// `project`. Used by the `--reingest` path to collect ids for
    /// cross-store cleanup before the `LanceDB` delete fires.
    pub async fn chunk_ids_for_project(&self, project: &str) -> Result<Vec<String>> {
        let table = self.conn.open_table(CORPUS_TABLE).execute().await?;
        let filter = format!("project = '{}'", escape_sql(project));
        let stream = table
            .query()
            .only_if(filter)
            .select(Select::Columns(vec!["chunk_id".into()]))
            .execute()
            .await?;
        let batches: Vec<RecordBatch> = stream.try_collect().await?;
        let mut ids = Vec::new();
        for batch in &batches {
            let col = batch.column_by_name("chunk_id").ok_or_else(|| {
                StoreError::Arrow(arrow::error::ArrowError::SchemaError(
                    "chunk_id column missing in projection".into(),
                ))
            })?;
            let arr = col.as_any().downcast_ref::<StringArray>().ok_or_else(|| {
                StoreError::Arrow(arrow::error::ArrowError::CastError(
                    "chunk_id expected to be Utf8".into(),
                ))
            })?;
            let rows = batch.num_rows();
            ids.reserve(rows);
            for i in 0..rows {
                ids.push(arr.value(i).to_string());
            }
        }
        Ok(ids)
    }

    /// Delete every corpus row whose `project` column equals `project`.
    /// Returns the number of rows removed (computed as `before - after`).
    pub async fn delete_by_project(&self, project: &str) -> Result<u64> {
        let table = self.conn.open_table(CORPUS_TABLE).execute().await?;
        let before = table.count_rows(None).await?;
        let filter = format!("project = '{}'", escape_sql(project));
        table.delete(&filter).await?;
        let after = table.count_rows(None).await?;
        Ok(u64::try_from(before.saturating_sub(after)).unwrap_or(0))
    }

    /// Delete a batch of chunks by their unique `chunk_id`.
    pub async fn delete_chunks(&self, ids: &[String]) -> Result<u64> {
        if ids.is_empty() {
            return Ok(0);
        }
        let table = self.conn.open_table(CORPUS_TABLE).execute().await?;
        let before = table.count_rows(None).await?;

        let ids_joined = ids
            .iter()
            .map(|id| format!("'{}'", escape_sql(id)))
            .collect::<Vec<_>>()
            .join(", ");
        let filter = format!("chunk_id IN ({ids_joined})");

        table.delete(&filter).await?;
        let after = table.count_rows(None).await?;
        Ok(u64::try_from(before.saturating_sub(after)).unwrap_or(0))
    }

    /// Fetch chunk embeddings by id. Returns a map keyed by `chunk_id`;
    /// ids absent from the corpus are simply omitted (no error).
    ///
    /// Used by the auto-weaver (Phase 7) to compute resonance between
    /// freshly-ingested chunks and the anchor vectors of existing
    /// threads. The vector column is `FixedSizeList<Float32, dim>`; rows
    /// are decoded one slice at a time.
    pub async fn fetch_embeddings(
        &self,
        ids: &[String],
    ) -> Result<std::collections::HashMap<String, Vec<f32>>> {
        use arrow_array::Array;
        use std::collections::HashMap;

        let mut out: HashMap<String, Vec<f32>> = HashMap::new();
        if ids.is_empty() {
            return Ok(out);
        }
        let table = self.conn.open_table(CORPUS_TABLE).execute().await?;

        let ids_joined = ids
            .iter()
            .map(|id| format!("'{}'", escape_sql(id)))
            .collect::<Vec<_>>()
            .join(", ");
        let filter = format!("chunk_id IN ({ids_joined})");
        let stream = table
            .query()
            .only_if(filter)
            .select(Select::Columns(vec!["chunk_id".into(), "embedding".into()]))
            .execute()
            .await?;
        let batches: Vec<RecordBatch> = stream.try_collect().await?;
        for batch in &batches {
            let id_col = batch.column_by_name("chunk_id").ok_or_else(|| {
                StoreError::Arrow(arrow::error::ArrowError::SchemaError(
                    "chunk_id column missing in projection".into(),
                ))
            })?;
            let ids_arr = id_col
                .as_any()
                .downcast_ref::<StringArray>()
                .ok_or_else(|| {
                    StoreError::Arrow(arrow::error::ArrowError::CastError(
                        "chunk_id expected to be Utf8".into(),
                    ))
                })?;
            let emb_col = batch.column_by_name("embedding").ok_or_else(|| {
                StoreError::Arrow(arrow::error::ArrowError::SchemaError(
                    "embedding column missing in projection".into(),
                ))
            })?;
            let emb_arr = emb_col
                .as_any()
                .downcast_ref::<FixedSizeListArray>()
                .ok_or_else(|| {
                    StoreError::Arrow(arrow::error::ArrowError::CastError(
                        "embedding expected to be FixedSizeList".into(),
                    ))
                })?;
            let f32_values = emb_arr
                .values()
                .as_any()
                .downcast_ref::<arrow_array::Float32Array>()
                .ok_or_else(|| {
                    StoreError::Arrow(arrow::error::ArrowError::CastError(
                        "embedding inner expected to be Float32".into(),
                    ))
                })?;
            let dim = usize::try_from(emb_arr.value_length()).map_err(|_| {
                StoreError::Arrow(arrow::error::ArrowError::SchemaError(format!(
                    "embedding value_length not representable as usize: {}",
                    emb_arr.value_length()
                )))
            })?;
            for i in 0..batch.num_rows() {
                if emb_arr.is_null(i) {
                    continue;
                }
                let start = i * dim;
                let slice = &f32_values.values()[start..start + dim];
                out.insert(ids_arr.value(i).to_string(), slice.to_vec());
            }
        }
        Ok(out)
    }

    /// Fetch the `text` column for a batch of chunks by their `chunk_id`.
    /// Mirrors `fetch_embeddings` — used by ambient consumers (e.g.
    /// `TurnObserver`) that subscribe to `IngestEvent` and need the
    /// original chunk text to observe.
    pub async fn fetch_texts(
        &self,
        ids: &[String],
    ) -> Result<std::collections::HashMap<String, String>> {
        use std::collections::HashMap;

        use arrow_array::Array;

        let mut out: HashMap<String, String> = HashMap::new();
        if ids.is_empty() {
            return Ok(out);
        }
        let table = self.conn.open_table(CORPUS_TABLE).execute().await?;

        let ids_joined = ids
            .iter()
            .map(|id| format!("'{}'", escape_sql(id)))
            .collect::<Vec<_>>()
            .join(", ");
        let filter = format!("chunk_id IN ({ids_joined})");
        let stream = table
            .query()
            .only_if(filter)
            .select(Select::Columns(vec!["chunk_id".into(), "text".into()]))
            .execute()
            .await?;
        let batches: Vec<RecordBatch> = stream.try_collect().await?;
        for batch in &batches {
            let id_col = batch.column_by_name("chunk_id").ok_or_else(|| {
                StoreError::Arrow(arrow::error::ArrowError::SchemaError(
                    "chunk_id column missing in projection".into(),
                ))
            })?;
            let ids_arr = id_col
                .as_any()
                .downcast_ref::<StringArray>()
                .ok_or_else(|| {
                    StoreError::Arrow(arrow::error::ArrowError::CastError(
                        "chunk_id expected to be Utf8".into(),
                    ))
                })?;
            let text_col = batch.column_by_name("text").ok_or_else(|| {
                StoreError::Arrow(arrow::error::ArrowError::SchemaError(
                    "text column missing in projection".into(),
                ))
            })?;
            let text_arr = text_col
                .as_any()
                .downcast_ref::<StringArray>()
                .ok_or_else(|| {
                    StoreError::Arrow(arrow::error::ArrowError::CastError(
                        "text expected to be Utf8".into(),
                    ))
                })?;
            for i in 0..batch.num_rows() {
                if text_arr.is_null(i) {
                    continue;
                }
                out.insert(ids_arr.value(i).to_string(), text_arr.value(i).to_string());
            }
        }
        Ok(out)
    }

    /// Sample non-stale chunks ingested since `since`, up to `limit` rows.
    /// Returns `(chunk_id, embedding)` pairs for use with emergent
    /// clustering. Ordering is whatever Lance returns — callers that
    /// care about strict recency should request a smaller window.
    ///
    /// Empty result is fine; the caller treats "no recent activity" as
    /// "nothing to surface."
    pub async fn sample_recent_chunks(
        &self,
        since: chrono::DateTime<chrono::Utc>,
        limit: usize,
    ) -> Result<Vec<(String, Vec<f32>)>> {
        use arrow_array::Array;

        if limit == 0 {
            return Ok(Vec::new());
        }
        let table = self.conn.open_table(CORPUS_TABLE).execute().await?;
        // Lance accepts SQL-flavoured timestamp literals via cast.
        let filter = format!(
            "stale = false AND ts >= TIMESTAMP '{}'",
            since.format("%Y-%m-%d %H:%M:%S")
        );
        let stream = table
            .query()
            .only_if(filter)
            .limit(limit)
            .select(Select::Columns(vec![
                "chunk_id".into(),
                "embedding".into(),
            ]))
            .execute()
            .await?;
        let batches: Vec<RecordBatch> = stream.try_collect().await?;
        let mut out: Vec<(String, Vec<f32>)> = Vec::new();
        for batch in &batches {
            let id_col = batch.column_by_name("chunk_id").ok_or_else(|| {
                StoreError::Arrow(arrow::error::ArrowError::SchemaError(
                    "chunk_id column missing in projection".into(),
                ))
            })?;
            let ids_arr = id_col
                .as_any()
                .downcast_ref::<StringArray>()
                .ok_or_else(|| {
                    StoreError::Arrow(arrow::error::ArrowError::CastError(
                        "chunk_id expected to be Utf8".into(),
                    ))
                })?;
            let emb_col = batch.column_by_name("embedding").ok_or_else(|| {
                StoreError::Arrow(arrow::error::ArrowError::SchemaError(
                    "embedding column missing in projection".into(),
                ))
            })?;
            let emb_arr = emb_col
                .as_any()
                .downcast_ref::<FixedSizeListArray>()
                .ok_or_else(|| {
                    StoreError::Arrow(arrow::error::ArrowError::CastError(
                        "embedding expected to be FixedSizeList".into(),
                    ))
                })?;
            let f32_values = emb_arr
                .values()
                .as_any()
                .downcast_ref::<arrow_array::Float32Array>()
                .ok_or_else(|| {
                    StoreError::Arrow(arrow::error::ArrowError::CastError(
                        "embedding inner expected to be Float32".into(),
                    ))
                })?;
            let dim = usize::try_from(emb_arr.value_length()).map_err(|_| {
                StoreError::Arrow(arrow::error::ArrowError::SchemaError(format!(
                    "embedding value_length not representable as usize: {}",
                    emb_arr.value_length()
                )))
            })?;
            for i in 0..batch.num_rows() {
                if emb_arr.is_null(i) {
                    continue;
                }
                let start = i * dim;
                let slice = &f32_values.values()[start..start + dim];
                out.push((ids_arr.value(i).to_string(), slice.to_vec()));
                if out.len() >= limit {
                    return Ok(out);
                }
            }
        }
        Ok(out)
    }

    /// Mark a batch of chunks as stale by their unique `chunk_id`.
    pub async fn mark_chunks_stale(&self, ids: &[String]) -> Result<u64> {
        if ids.is_empty() {
            return Ok(0);
        }
        let table = self.conn.open_table(CORPUS_TABLE).execute().await?;

        let ids_joined = ids
            .iter()
            .map(|id| format!("'{}'", escape_sql(id)))
            .collect::<Vec<_>>()
            .join(", ");
        let filter = format!("chunk_id IN ({ids_joined})");

        table
            .update()
            .only_if(&filter)
            .column("stale", "true")
            .execute()
            .await?;
        Ok(ids.len() as u64)
    }

    /// Count rows matching `stale = false AND <filter>`. Used for
    /// before/after diffs on maintenance sweeps where we don't have ids
    /// up-front.
    pub async fn count_active(&self, filter: &str) -> Result<u64> {
        let table = self.conn.open_table(CORPUS_TABLE).execute().await?;
        let full = format!("stale = false AND ({filter})");
        let mut stream = table
            .query()
            .only_if(&full)
            .select(Select::Columns(vec!["chunk_id".to_string()]))
            .execute()
            .await?;
        let mut n: u64 = 0;
        while let Some(batch) = stream.try_next().await? {
            n += batch.num_rows() as u64;
        }
        Ok(n)
    }

    /// Run Lance's full optimize pass on the corpus table — merges small
    /// fragments, reindexes new data into existing indices, and prunes
    /// versions older than the lance default retention window (so files
    /// from an in-flight reader/writer are not removed). After a bulk
    /// mutation such as [`Self::mark_tool_blocks_stale`] this is what
    /// actually collapses the per-batch fragments and lets serve return
    /// to idle.
    pub async fn optimize_all(&self) -> Result<()> {
        let table = self.conn.open_table(CORPUS_TABLE).execute().await?;
        table.optimize(OptimizeAction::All).await?;
        Ok(())
    }

    /// Aggregate non-stale chunks ingested since `since` into per-source
    /// activity bursts — one row per `(project, source_id)` pair, with
    /// the chunk count, time bounds, and up to `samples_per_group` text
    /// snippets pulled from the most recent chunks in the group.
    ///
    /// This is the data layer for the activity-burst attention surface
    /// (`thread_attention` in attention-mcp): "where did most of the
    /// ingest happen in the last N hours, and what does it look like."
    /// Robust to the "thoughts are unique" problem because it doesn't
    /// touch embeddings — pure timestamp + source aggregation.
    pub async fn activity_bursts(
        &self,
        since: chrono::DateTime<chrono::Utc>,
        samples_per_group: usize,
    ) -> Result<Vec<ActivityBurst>> {
        use std::collections::HashMap;

        use arrow_array::Array;

        let table = self.conn.open_table(CORPUS_TABLE).execute().await?;
        let filter = format!(
            "stale = false AND ts >= TIMESTAMP '{}'",
            since.format("%Y-%m-%d %H:%M:%S")
        );
        let stream = table
            .query()
            .only_if(filter)
            .select(Select::Columns(vec![
                "chunk_id".into(),
                "project".into(),
                "source_id".into(),
                "ts".into(),
                "text".into(),
            ]))
            .execute()
            .await?;
        let batches: Vec<RecordBatch> = stream.try_collect().await?;

        // Key = (project, source_id). project may be null → use empty
        // string so the group still aggregates (rather than dropping).
        let mut groups: HashMap<(String, String), ActivityBurst> = HashMap::new();
        for batch in &batches {
            let id_arr = column_as_str(batch, "chunk_id")?;
            let proj_arr = column_as_str(batch, "project")?;
            let src_arr = column_as_str(batch, "source_id")?;
            let text_arr = column_as_str(batch, "text")?;
            let ts_col = batch.column_by_name("ts").ok_or_else(|| {
                StoreError::Arrow(arrow::error::ArrowError::SchemaError(
                    "ts column missing in projection".into(),
                ))
            })?;
            let ts_arr = ts_col
                .as_any()
                .downcast_ref::<TimestampMicrosecondArray>()
                .ok_or_else(|| {
                    StoreError::Arrow(arrow::error::ArrowError::CastError(
                        "ts expected to be TimestampMicrosecond".into(),
                    ))
                })?;

            for i in 0..batch.num_rows() {
                if ts_arr.is_null(i) {
                    continue;
                }
                let micros = ts_arr.value(i);
                let Some(ts) = chrono::DateTime::<chrono::Utc>::from_timestamp_micros(micros)
                else {
                    continue;
                };
                let project = if proj_arr.is_null(i) {
                    String::new()
                } else {
                    proj_arr.value(i).to_string()
                };
                let source_id = src_arr.value(i).to_string();
                let chunk_id = id_arr.value(i).to_string();
                let text = text_arr.value(i).to_string();

                let entry = groups
                    .entry((project.clone(), source_id.clone()))
                    .or_insert_with(|| ActivityBurst {
                        project,
                        source_id,
                        count: 0,
                        min_ts: ts,
                        max_ts: ts,
                        samples: Vec::new(),
                    });
                entry.count += 1;
                if ts < entry.min_ts {
                    entry.min_ts = ts;
                }
                if ts > entry.max_ts {
                    entry.max_ts = ts;
                }
                // Keep up to `samples_per_group` of the most-recent
                // (chunk_id, ts, text) tuples. We sort by ts at the end.
                entry.samples.push((chunk_id, ts, text));
            }
        }

        // Finalise: keep only the N most-recent samples per group.
        let mut out: Vec<ActivityBurst> = groups.into_values().collect();
        for burst in &mut out {
            burst
                .samples
                .sort_by(|a, b| b.1.cmp(&a.1));
            burst.samples.truncate(samples_per_group);
        }
        Ok(out)
    }

    /// Mark every active chunk whose `extra_json` reports a `block_kind`
    /// of `tool_use` or `tool_result` as stale. Returns the row count
    /// matched before the update (so the caller can report what changed).
    ///
    /// The filter relies on the JSON serializer keeping the literal
    /// `"block_kind":"tool_use"` / `"block_kind":"tool_result"` substring
    /// — the only writer is [`crate::schema`] via
    /// `crates/scan/src/anthropic_session.rs::build_chunks`, which emits
    /// that exact shape with no whitespace.
    pub async fn mark_tool_blocks_stale(&self) -> Result<u64> {
        let filter = r#"extra_json LIKE '%"block_kind":"tool_use"%' OR extra_json LIKE '%"block_kind":"tool_result"%'"#;
        let matched = self.count_active(filter).await?;
        if matched == 0 {
            return Ok(0);
        }
        let table = self.conn.open_table(CORPUS_TABLE).execute().await?;
        let full = format!("stale = false AND ({filter})");
        table
            .update()
            .only_if(&full)
            .column("stale", "true")
            .execute()
            .await?;
        Ok(matched)
    }

    /// Mark every active chunk whose `text` begins with Claude Code's
    /// slash-command surface scaffolding (`<local-command-…>`,
    /// `<command-name>`, `<command-message>`, `<command-args>`,
    /// `</command-name>`) as stale. These chunks carry `block_kind=user`
    /// so [`Self::mark_tool_blocks_stale`] does not catch them, but they
    /// are procedurally identical: high-volume meta scaffolding with no
    /// thinking content.
    pub async fn mark_local_command_wrappers_stale(&self) -> Result<u64> {
        let filter = "text LIKE '<local-command-%' \
            OR text LIKE '<command-name>%' \
            OR text LIKE '</command-name>%' \
            OR text LIKE '<command-message>%' \
            OR text LIKE '<command-args>%'";
        let matched = self.count_active(filter).await?;
        if matched == 0 {
            return Ok(0);
        }
        let table = self.conn.open_table(CORPUS_TABLE).execute().await?;
        let full = format!("stale = false AND ({filter})");
        table
            .update()
            .only_if(&full)
            .column("stale", "true")
            .execute()
            .await?;
        Ok(matched)
    }
}

/// Activity burst — one `(project, source_id)` group with chunk count,
/// time bounds, and a small sample of `(chunk_id, ts, text)` tuples
/// (most recent first). Caller (e.g. attention crate) computes the
/// recency-weighted score and ranks.
#[derive(Debug, Clone)]
pub struct ActivityBurst {
    pub project: String,
    pub source_id: String,
    pub count: usize,
    pub min_ts: chrono::DateTime<chrono::Utc>,
    pub max_ts: chrono::DateTime<chrono::Utc>,
    pub samples: Vec<(String, chrono::DateTime<chrono::Utc>, String)>,
}

fn column_as_str<'a>(batch: &'a RecordBatch, name: &str) -> Result<&'a StringArray> {
    let col = batch.column_by_name(name).ok_or_else(|| {
        StoreError::Arrow(arrow::error::ArrowError::SchemaError(format!(
            "{name} column missing in projection"
        )))
    })?;
    col.as_any().downcast_ref::<StringArray>().ok_or_else(|| {
        StoreError::Arrow(arrow::error::ArrowError::CastError(format!(
            "{name} expected to be Utf8"
        )))
    })
}

/// Escape single quotes for inlining into a `LanceDB` filter expression.
/// Duplicates `'` -> `''`; `LanceDB`'s filter parser is SQL-like.
fn escape_sql(value: &str) -> String {
    value.replace('\'', "''")
}

fn build_record_batch(
    schema: &Arc<Schema>,
    chunks: &[Chunk],
    embeddings: &[Vec<f32>],
    dim: usize,
) -> Result<RecordBatch> {
    use arrow_array::builder::{FixedSizeListBuilder, Float32Builder};

    let chunk_id = StringArray::from_iter_values(chunks.iter().map(|c| c.chunk_id.as_str()));
    let source = StringArray::from_iter_values(chunks.iter().map(|c| c.source.as_str()));
    let project: StringArray = chunks.iter().map(|c| c.project.as_deref()).collect();
    let source_id = StringArray::from_iter_values(chunks.iter().map(|c| c.source_id.as_str()));
    let chunk_index = UInt32Array::from_iter_values(chunks.iter().map(|c| c.chunk_index));
    let ts = TimestampMicrosecondArray::from(
        chunks
            .iter()
            .map(|c| c.ts.map(|t| t.timestamp_micros()))
            .collect::<Vec<_>>(),
    )
    .with_timezone("UTC");
    let role: StringArray = chunks.iter().map(|c| c.role.as_deref()).collect();
    let text = StringArray::from_iter_values(chunks.iter().map(|c| c.text.as_str()));
    let sha256 = StringArray::from_iter_values(chunks.iter().map(|c| c.sha256.as_str()));

    let links_json_strings: Vec<String> = chunks
        .iter()
        .map(|c| serde_json::to_string(&c.links).expect("links serialize"))
        .collect();
    let links_json = StringArray::from_iter_values(links_json_strings.iter().map(String::as_str));

    let extra_json_strings: Vec<String> = chunks
        .iter()
        .map(|c| serde_json::to_string(&c.extra).expect("extra serialize"))
        .collect();
    let extra_json = StringArray::from_iter_values(extra_json_strings.iter().map(String::as_str));

    // New chunks are NEVER stale by default.
    let stale = BooleanArray::from(vec![false; chunks.len()]);

    let mut builder = FixedSizeListBuilder::new(
        Float32Builder::new(),
        i32::try_from(dim).expect("dim in i32"),
    );
    for v in embeddings {
        if v.len() != dim {
            return Err(StoreError::DimMismatch {
                have: v.len(),
                want: dim,
            });
        }
        let vb = builder.values();
        for &x in v {
            vb.append_value(x);
        }
        builder.append(true);
    }
    let embedding = builder.finish();

    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(chunk_id),
            Arc::new(source),
            Arc::new(project),
            Arc::new(source_id),
            Arc::new(chunk_index),
            Arc::new(ts),
            Arc::new(role),
            Arc::new(text),
            Arc::new(sha256),
            Arc::new(links_json),
            Arc::new(extra_json),
            Arc::new(stale),
            Arc::new(embedding) as Arc<FixedSizeListArray>,
        ],
    )?;
    Ok(batch)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ostk_recall_core::{Chunk, Links, Source};
    use tempfile::TempDir;

    fn sample_chunk(id: &str, text: &str) -> Chunk {
        Chunk {
            chunk_id: id.to_string(),
            source: Source::Markdown,
            project: Some("test".into()),
            source_id: "file.md".into(),
            chunk_index: 0,
            ts: None,
            role: None,
            text: text.to_string(),
            sha256: Chunk::content_hash(text),
            links: Links::default(),
            extra: serde_json::Value::Null,
        }
    }

    #[tokio::test]
    async fn open_or_create_creates_table() {
        let tmp = TempDir::new().unwrap();
        let store = CorpusStore::open_or_create(tmp.path(), 64).await.unwrap();
        assert_eq!(store.dim(), 64);
        assert_eq!(store.row_count().await.unwrap(), 0);
    }

    #[tokio::test]
    async fn upsert_and_count() {
        let tmp = TempDir::new().unwrap();
        let store = CorpusStore::open_or_create(tmp.path(), 4).await.unwrap();

        let chunks = vec![
            sample_chunk("a", "alpha text"),
            sample_chunk("b", "beta text"),
        ];
        let embs = vec![vec![1.0, 0.0, 0.0, 0.0], vec![0.0, 1.0, 0.0, 0.0]];

        let n = store.upsert(&chunks, &embs).await.unwrap();
        assert_eq!(n, 2);
        assert_eq!(store.row_count().await.unwrap(), 2);

        // Re-upsert same ids → still 2 rows (merge_insert).
        store.upsert(&chunks, &embs).await.unwrap();
        assert_eq!(store.row_count().await.unwrap(), 2);
    }

    #[tokio::test]
    async fn upsert_rejects_wrong_dim() {
        let tmp = TempDir::new().unwrap();
        let store = CorpusStore::open_or_create(tmp.path(), 4).await.unwrap();
        let chunks = vec![sample_chunk("a", "x")];
        let embs = vec![vec![1.0, 2.0]]; // dim=2 != store dim=4
        let err = store.upsert(&chunks, &embs).await.unwrap_err();
        assert!(matches!(err, StoreError::DimMismatch { .. }));
    }

    async fn read_stale_for(store: &CorpusStore, chunk_id: &str) -> bool {
        use arrow_array::BooleanArray;
        let table = store.conn.open_table(CORPUS_TABLE).execute().await.unwrap();
        let filter = format!("chunk_id = '{chunk_id}'");
        let stream = table
            .query()
            .only_if(filter)
            .select(Select::Columns(vec!["chunk_id".into(), "stale".into()]))
            .execute()
            .await
            .unwrap();
        let batches: Vec<RecordBatch> = stream.try_collect().await.unwrap();
        for batch in &batches {
            let stale_col = batch.column_by_name("stale").expect("stale column present");
            let arr = stale_col
                .as_any()
                .downcast_ref::<BooleanArray>()
                .expect("stale is Boolean");
            if batch.num_rows() > 0 {
                return arr.value(0);
            }
        }
        panic!("no row found for chunk_id={chunk_id}");
    }

    #[tokio::test]
    async fn fetch_embeddings_returns_known_vectors() {
        let tmp = TempDir::new().unwrap();
        let store = CorpusStore::open_or_create(tmp.path(), 4).await.unwrap();

        let chunks = vec![
            sample_chunk("a", "alpha"),
            sample_chunk("b", "beta"),
            sample_chunk("c", "gamma"),
        ];
        let embs = vec![
            vec![1.0, 0.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0, 0.0],
            vec![0.5, 0.5, 0.5, 0.5],
        ];
        store.upsert(&chunks, &embs).await.unwrap();

        // Fetch a subset (and one missing id); only present rows come back.
        let map = store
            .fetch_embeddings(&["a".into(), "c".into(), "missing".into()])
            .await
            .unwrap();
        assert_eq!(map.len(), 2);
        assert_eq!(map.get("a").unwrap(), &vec![1.0, 0.0, 0.0, 0.0]);
        assert_eq!(map.get("c").unwrap(), &vec![0.5, 0.5, 0.5, 0.5]);
        assert!(!map.contains_key("missing"));

        // Empty input is a no-op.
        let empty = store.fetch_embeddings(&[]).await.unwrap();
        assert!(empty.is_empty());
    }

    #[tokio::test]
    async fn mark_chunks_stale() {
        let tmp = TempDir::new().unwrap();
        let store = CorpusStore::open_or_create(tmp.path(), 4).await.unwrap();

        let chunks = vec![
            sample_chunk("a", "alpha"),
            sample_chunk("b", "beta"),
            sample_chunk("c", "gamma"),
        ];
        let embs = vec![
            vec![1.0, 0.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0, 0.0],
            vec![0.0, 0.0, 1.0, 0.0],
        ];

        store.upsert(&chunks, &embs).await.unwrap();

        // Sanity: nothing is stale on insert.
        assert!(!read_stale_for(&store, "a").await);
        assert!(!read_stale_for(&store, "b").await);
        assert!(!read_stale_for(&store, "c").await);

        let n = store
            .mark_chunks_stale(&["a".into(), "c".into()])
            .await
            .unwrap();
        assert_eq!(n, 2);

        // Targeted rows flipped, untouched row stayed false.
        assert!(read_stale_for(&store, "a").await, "a should be stale");
        assert!(!read_stale_for(&store, "b").await, "b should NOT be stale");
        assert!(read_stale_for(&store, "c").await, "c should be stale");
    }
}
