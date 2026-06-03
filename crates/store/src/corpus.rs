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
use lancedb::index::scalar::{BTreeIndexBuilder, BitmapIndexBuilder, FtsIndexBuilder};
use lancedb::query::{ExecutableQuery, QueryBase, Select};
use lancedb::table::OptimizeAction;
use ostk_recall_core::{Chunk, SourceKind};
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

/// One source-kind batch selected for a windowed corpus pass.
#[derive(Debug, Clone)]
pub struct CorpusWindowBatch {
    pub source: SourceKind,
    pub chunk_ids: Vec<String>,
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

    /// Current Lance dataset version of the corpus table. Each commit
    /// (`merge_insert`, delete, optimize) bumps this. Opening the table
    /// fresh returns the latest version, so callers see commits made by
    /// other handles. Used to verify commit batching and prune passes.
    pub async fn version(&self) -> Result<u64> {
        let table = self.conn.open_table(CORPUS_TABLE).execute().await?;
        Ok(table.version().await?)
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

    /// Ensure a BTree scalar index exists on `chunk_id`. Idempotent.
    ///
    /// `Pipeline::embed_and_persist` lands every batch through
    /// `table.merge_insert(&["chunk_id"])`, and several recall paths
    /// (`fetch_texts`, `fetch_embeddings`, membrane filters) emit
    /// `chunk_id IN (…)` queries. Without a scalar index on this column
    /// lance resolves both shapes by full-table filtered scan, so each
    /// ingest batch reads tens of megabytes to find one row. The BTree
    /// turns those lookups into O(log N).
    pub async fn ensure_chunk_id_index(&self) -> Result<()> {
        let table = self.conn.open_table(CORPUS_TABLE).execute().await?;
        let indices = table.list_indices().await?;
        if indices
            .iter()
            .any(|ix| ix.columns.iter().any(|c| c == "chunk_id"))
        {
            return Ok(());
        }
        tracing::info!("creating BTree index on corpus.chunk_id");
        table
            .create_index(&["chunk_id"], Index::BTree(BTreeIndexBuilder::default()))
            .execute()
            .await?;
        Ok(())
    }

    /// Ensure a Bitmap scalar index exists on `project`. Idempotent.
    ///
    /// `project` is a low-cardinality column (one value per
    /// `[[sources]]` entry, typically ~20) and is filtered on every
    /// orphan-sweep and reingest path. Bitmap is the natural shape for
    /// that distribution.
    pub async fn ensure_project_index(&self) -> Result<()> {
        let table = self.conn.open_table(CORPUS_TABLE).execute().await?;
        let indices = table.list_indices().await?;
        if indices
            .iter()
            .any(|ix| ix.columns.iter().any(|c| c == "project"))
        {
            return Ok(());
        }
        tracing::info!("creating Bitmap index on corpus.project");
        table
            .create_index(&["project"], Index::Bitmap(BitmapIndexBuilder::default()))
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

    /// Fetch up to `limit` `(chunk_id, embedding, extra_json)` triples for
    /// active (non-stale) chunks, no id filter. Sibling of
    /// [`Self::fetch_embeddings`] — the read side P8's concept refresh needs
    /// (`fetch_all_embeddings` with a bound) and what the HDBSCAN scale probe
    /// uses. No ordering guarantee beyond Lance scan order.
    ///
    /// When `exclude_apparatus` is true, structural apparatus is filtered out
    /// SQL-side so the `limit` lands on real-cognition chunks. The exclusion is
    /// derived from the same `ostk_recall_core::APPARATUS_*` consts that back
    /// `is_structural_apparatus`, so the SQL and Rust encodings can't drift.
    pub async fn fetch_sample_embeddings(
        &self,
        limit: usize,
        exclude_apparatus: bool,
    ) -> Result<Vec<(String, Vec<f32>, String)>> {
        use arrow_array::Array;

        let mut filter = String::from("stale = false");
        if exclude_apparatus {
            // The consts are fixed, non-quote-bearing identifiers, so direct
            // interpolation into the LIKE clauses is injection-safe.
            let mut excl: Vec<String> = Vec::new();
            for bk in ostk_recall_core::APPARATUS_BLOCK_KINDS {
                excl.push(format!("extra_json LIKE '%\"block_kind\":\"{bk}\"%'"));
            }
            for prefix in ostk_recall_core::APPARATUS_TEXT_PREFIXES {
                excl.push(format!("text LIKE '{prefix}%'"));
            }
            if !excl.is_empty() {
                filter.push_str(&format!(" AND NOT ({})", excl.join(" OR ")));
            }
        }

        let table = self.conn.open_table(CORPUS_TABLE).execute().await?;
        let stream = table
            .query()
            .only_if(&filter)
            .select(Select::Columns(vec![
                "chunk_id".into(),
                "embedding".into(),
                "extra_json".into(),
            ]))
            .limit(limit)
            .execute()
            .await?;
        let batches: Vec<RecordBatch> = stream.try_collect().await?;
        let mut out: Vec<(String, Vec<f32>, String)> = Vec::new();
        for batch in &batches {
            let ids_arr = batch
                .column_by_name("chunk_id")
                .and_then(|c| c.as_any().downcast_ref::<StringArray>())
                .ok_or_else(|| {
                    StoreError::Arrow(arrow::error::ArrowError::SchemaError(
                        "chunk_id column missing or not Utf8".into(),
                    ))
                })?;
            let extra_arr = batch
                .column_by_name("extra_json")
                .and_then(|c| c.as_any().downcast_ref::<StringArray>());
            let emb_arr = batch
                .column_by_name("embedding")
                .and_then(|c| c.as_any().downcast_ref::<FixedSizeListArray>())
                .ok_or_else(|| {
                    StoreError::Arrow(arrow::error::ArrowError::SchemaError(
                        "embedding column missing or not FixedSizeList".into(),
                    ))
                })?;
            let f32_values = emb_arr
                .values()
                .as_any()
                .downcast_ref::<arrow_array::Float32Array>()
                .ok_or_else(|| {
                    StoreError::Arrow(arrow::error::ArrowError::CastError(
                        "embedding inner not Float32".into(),
                    ))
                })?;
            let dim = usize::try_from(emb_arr.value_length()).unwrap_or(0);
            for i in 0..batch.num_rows() {
                if dim == 0 || emb_arr.is_null(i) {
                    continue;
                }
                let start = i * dim;
                let slice = &f32_values.values()[start..start + dim];
                let extra = extra_arr
                    .filter(|a| !a.is_null(i))
                    .map(|a| a.value(i).to_string())
                    .unwrap_or_default();
                out.push((ids_arr.value(i).to_string(), slice.to_vec(), extra));
            }
        }
        Ok(out)
    }

    /// Sibling of [`Self::fetch_sample_embeddings`] returning `(chunk_id, text)`
    /// instead of the stored embedding — for callers that re-embed (the
    /// contextual-embedder probe) or need text for concept labeling. Same
    /// apparatus-exclusion semantics, derived from the same core consts.
    pub async fn fetch_sample_texts(
        &self,
        limit: usize,
        exclude_apparatus: bool,
    ) -> Result<Vec<(String, String)>> {
        self.fetch_sample_texts_inner(limit, exclude_apparatus, None)
            .await
    }

    /// Like [`Self::fetch_sample_texts`], but constrained to one concrete
    /// stored source. Probe-only callers use this to test whether concept
    /// clustering works per source when global clustering collapses into the
    /// dominant corpus basin.
    pub async fn fetch_sample_texts_for_source(
        &self,
        limit: usize,
        exclude_apparatus: bool,
        source: ostk_recall_core::Source,
    ) -> Result<Vec<(String, String)>> {
        self.fetch_sample_texts_inner(limit, exclude_apparatus, Some(source.as_str()))
            .await
    }

    async fn fetch_sample_texts_inner(
        &self,
        limit: usize,
        exclude_apparatus: bool,
        source: Option<&str>,
    ) -> Result<Vec<(String, String)>> {
        use arrow_array::Array;

        let mut filter = String::from("stale = false");
        if let Some(source) = source {
            filter.push_str(&format!(" AND source = '{}'", escape_sql(source)));
        }
        if exclude_apparatus {
            let mut excl: Vec<String> = Vec::new();
            for bk in ostk_recall_core::APPARATUS_BLOCK_KINDS {
                excl.push(format!("extra_json LIKE '%\"block_kind\":\"{bk}\"%'"));
            }
            for prefix in ostk_recall_core::APPARATUS_TEXT_PREFIXES {
                excl.push(format!("text LIKE '{prefix}%'"));
            }
            if !excl.is_empty() {
                filter.push_str(&format!(" AND NOT ({})", excl.join(" OR ")));
            }
        }

        let table = self.conn.open_table(CORPUS_TABLE).execute().await?;
        let stream = table
            .query()
            .only_if(&filter)
            .select(Select::Columns(vec!["chunk_id".into(), "text".into()]))
            .limit(limit)
            .execute()
            .await?;
        let batches: Vec<RecordBatch> = stream.try_collect().await?;
        let mut out: Vec<(String, String)> = Vec::new();
        for batch in &batches {
            let ids_arr = batch
                .column_by_name("chunk_id")
                .and_then(|c| c.as_any().downcast_ref::<StringArray>())
                .ok_or_else(|| {
                    StoreError::Arrow(arrow::error::ArrowError::SchemaError(
                        "chunk_id column missing or not Utf8".into(),
                    ))
                })?;
            let txt_arr = batch
                .column_by_name("text")
                .and_then(|c| c.as_any().downcast_ref::<StringArray>())
                .ok_or_else(|| {
                    StoreError::Arrow(arrow::error::ArrowError::SchemaError(
                        "text column missing or not Utf8".into(),
                    ))
                })?;
            for i in 0..batch.num_rows() {
                if txt_arr.is_null(i) {
                    continue;
                }
                out.push((ids_arr.value(i).to_string(), txt_arr.value(i).to_string()));
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

    /// Like [`Self::fetch_texts`] but also returns each chunk's `project`.
    ///
    /// Used by the ambient `TurnObserver`: a turn's concept-growth gazetteer is
    /// scoped to the chunk's *own* project, and derived-project transcript
    /// sources (e.g. claude_code sessions under a project-less source config)
    /// carry their project **per chunk**, not on the `IngestEvent` (which only
    /// has the source config's `project`). Returns `chunk_id -> (text, project)`;
    /// `project` is `None` when the row's column is null.
    pub async fn fetch_texts_with_project(
        &self,
        ids: &[String],
    ) -> Result<std::collections::HashMap<String, (String, Option<String>)>> {
        use std::collections::HashMap;

        use arrow_array::Array;

        let mut out: HashMap<String, (String, Option<String>)> = HashMap::new();
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
            .select(Select::Columns(vec![
                "chunk_id".into(),
                "text".into(),
                "project".into(),
            ]))
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
            let proj_col = batch.column_by_name("project").ok_or_else(|| {
                StoreError::Arrow(arrow::error::ArrowError::SchemaError(
                    "project column missing in projection".into(),
                ))
            })?;
            let proj_arr = proj_col
                .as_any()
                .downcast_ref::<StringArray>()
                .ok_or_else(|| {
                    StoreError::Arrow(arrow::error::ArrowError::CastError(
                        "project expected to be Utf8".into(),
                    ))
                })?;
            for i in 0..batch.num_rows() {
                if text_arr.is_null(i) {
                    continue;
                }
                let project = (!proj_arr.is_null(i)).then(|| proj_arr.value(i).to_string());
                out.insert(
                    ids_arr.value(i).to_string(),
                    (text_arr.value(i).to_string(), project),
                );
            }
        }
        Ok(out)
    }

    /// Batch-fetch full `Chunk` rows plus their dense embeddings.
    ///
    /// Returns `chunk_id -> (Chunk, Option<embedding>)`. Ids absent
    /// from the corpus are simply omitted. The embedding entry is
    /// `None` only when the row's embedding column is null (legacy /
    /// partial-ingest case).
    ///
    /// Used by `crates/query/src/lanes.rs::build_candidates` after the
    /// per-lane queries return ids only. One query per call regardless
    /// of K — the `chunk_id IN (...)` filter lets Lance push the lookup
    /// down.
    pub async fn fetch_chunks_by_ids(
        &self,
        ids: &[String],
    ) -> Result<std::collections::HashMap<String, (Chunk, Option<Vec<f32>>)>> {
        use std::collections::HashMap;

        use arrow_array::{Array, ListArray};
        use chrono::TimeZone;
        use ostk_recall_core::Links;

        let mut out: HashMap<String, (Chunk, Option<Vec<f32>>)> = HashMap::new();
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
            .select(Select::Columns(vec![
                "chunk_id".into(),
                "source".into(),
                "project".into(),
                "source_id".into(),
                "source_config_id".into(),
                "facets".into(),
                "embedding_input_sha256".into(),
                "chunk_index".into(),
                "ts".into(),
                "role".into(),
                "text".into(),
                "sha256".into(),
                "links_json".into(),
                "extra_json".into(),
                "embedding".into(),
            ]))
            .execute()
            .await?;
        let batches: Vec<RecordBatch> = stream.try_collect().await?;

        for batch in &batches {
            let n = batch.num_rows();
            if n == 0 {
                continue;
            }
            let chunk_id = downcast_str(batch, "chunk_id")?;
            let source = downcast_str(batch, "source")?;
            let project = downcast_str_opt(batch, "project");
            let source_id = downcast_str(batch, "source_id")?;
            let source_config_id = downcast_str_opt(batch, "source_config_id");
            let embedding_input_sha256 = downcast_str_opt(batch, "embedding_input_sha256");
            let chunk_index = downcast_u32(batch, "chunk_index")?;
            let ts = batch
                .column_by_name("ts")
                .and_then(|c| c.as_any().downcast_ref::<TimestampMicrosecondArray>());
            let role = downcast_str_opt(batch, "role");
            let text = downcast_str(batch, "text")?;
            let sha256 = downcast_str(batch, "sha256")?;
            let links_json = downcast_str(batch, "links_json")?;
            let extra_json = downcast_str(batch, "extra_json")?;
            let facets = batch
                .column_by_name("facets")
                .and_then(|c| c.as_any().downcast_ref::<ListArray>());
            let emb = batch
                .column_by_name("embedding")
                .and_then(|c| c.as_any().downcast_ref::<FixedSizeListArray>());
            let emb_dim = emb.map(|a| usize::try_from(a.value_length()).unwrap_or(0));
            let emb_values = emb.and_then(|a| {
                a.values()
                    .as_any()
                    .downcast_ref::<arrow_array::Float32Array>()
                    .cloned()
            });

            for i in 0..n {
                let id = chunk_id.value(i).to_string();
                let parsed_source = match parse_source(source.value(i)) {
                    Some(s) => s,
                    None => continue,
                };

                let links: Links = if links_json.is_null(i) {
                    Links::default()
                } else {
                    serde_json::from_str(links_json.value(i)).unwrap_or_default()
                };
                let extra: serde_json::Value = if extra_json.is_null(i) {
                    serde_json::Value::Null
                } else {
                    serde_json::from_str(extra_json.value(i)).unwrap_or(serde_json::Value::Null)
                };

                let facet_list: Vec<String> = if let Some(arr) = facets {
                    if arr.is_null(i) {
                        Vec::new()
                    } else {
                        let inner = arr.value(i);
                        let inner_str =
                            inner
                                .as_any()
                                .downcast_ref::<StringArray>()
                                .ok_or_else(|| {
                                    StoreError::Arrow(arrow::error::ArrowError::CastError(
                                        "facets inner expected Utf8".into(),
                                    ))
                                })?;
                        let mut v = Vec::with_capacity(inner_str.len());
                        for j in 0..inner_str.len() {
                            if !inner_str.is_null(j) {
                                v.push(inner_str.value(j).to_string());
                            }
                        }
                        v
                    }
                } else {
                    Vec::new()
                };
                let facets_set = ostk_recall_core::from_list(&facet_list);

                let ts_val = ts.and_then(|a| {
                    if a.is_null(i) {
                        None
                    } else {
                        chrono::Utc.timestamp_micros(a.value(i)).single()
                    }
                });

                let chunk = Chunk {
                    chunk_id: id.clone(),
                    source: parsed_source,
                    project: project.and_then(|a| {
                        if a.is_null(i) {
                            None
                        } else {
                            Some(a.value(i).to_string())
                        }
                    }),
                    source_id: source_id.value(i).to_string(),
                    source_config_id: source_config_id.map_or(String::new(), |a| {
                        if a.is_null(i) {
                            String::new()
                        } else {
                            a.value(i).to_string()
                        }
                    }),
                    chunk_index: chunk_index.value(i),
                    ts: ts_val,
                    role: role.and_then(|a| {
                        if a.is_null(i) {
                            None
                        } else {
                            Some(a.value(i).to_string())
                        }
                    }),
                    text: text.value(i).to_string(),
                    sha256: sha256.value(i).to_string(),
                    links,
                    facets: facets_set,
                    embedding_input_sha256: embedding_input_sha256.map_or(String::new(), |a| {
                        if a.is_null(i) {
                            String::new()
                        } else {
                            a.value(i).to_string()
                        }
                    }),
                    extra,
                };

                let embedding = match (emb, emb_values.as_ref(), emb_dim) {
                    (Some(arr), Some(values), Some(dim)) if dim > 0 && !arr.is_null(i) => {
                        let start = i * dim;
                        Some(values.values()[start..start + dim].to_vec())
                    }
                    _ => None,
                };

                out.insert(id, (chunk, embedding));
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
            .select(Select::Columns(vec!["chunk_id".into(), "embedding".into()]))
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

    /// Sibling of [`Self::sample_recent_chunks`] that also returns the
    /// `project` column alongside `(chunk_id, embedding)`. Used by the
    /// novelty surface to look up per-project baselines without a
    /// separate fetch.
    pub async fn sample_recent_chunks_with_project(
        &self,
        since: chrono::DateTime<chrono::Utc>,
        limit: usize,
    ) -> Result<Vec<(String, Option<String>, Vec<f32>)>> {
        use arrow_array::Array;

        if limit == 0 {
            return Ok(Vec::new());
        }
        let table = self.conn.open_table(CORPUS_TABLE).execute().await?;
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
                "project".into(),
                "embedding".into(),
            ]))
            .execute()
            .await?;
        let batches: Vec<RecordBatch> = stream.try_collect().await?;
        let mut out: Vec<(String, Option<String>, Vec<f32>)> = Vec::new();
        for batch in &batches {
            let id_arr = column_as_str(batch, "chunk_id")?;
            let proj_arr = column_as_str(batch, "project")?;
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
                let project = if proj_arr.is_null(i) {
                    None
                } else {
                    Some(proj_arr.value(i).to_string())
                };
                out.push((id_arr.value(i).to_string(), project, slice.to_vec()));
                if out.len() >= limit {
                    return Ok(out);
                }
            }
        }
        Ok(out)
    }

    /// Compute the component-wise mean of `embedding` over non-stale
    /// chunks ingested within the last `days` days, optionally filtered
    /// to a single `project`. Returns `Ok(None)` when fewer than 10
    /// qualifying chunks exist — the caller (novelty surface) falls
    /// back to the global baseline in that case.
    ///
    /// Streams the embedding column; never materialises individual
    /// vectors beyond a single accumulator of size `self.dim`.
    pub async fn project_baseline_mean(
        &self,
        project: Option<&str>,
        days: i64,
    ) -> Result<Option<Vec<f32>>> {
        use arrow_array::Array;

        let cutoff = chrono::Utc::now() - chrono::Duration::days(days);
        let mut filter = format!(
            "stale = false AND ts >= TIMESTAMP '{}'",
            cutoff.format("%Y-%m-%d %H:%M:%S")
        );
        if let Some(p) = project {
            filter.push_str(&format!(" AND project = '{}'", escape_sql(p)));
        }
        let table = self.conn.open_table(CORPUS_TABLE).execute().await?;
        let stream = table
            .query()
            .only_if(filter)
            .select(Select::Columns(vec!["embedding".into()]))
            .execute()
            .await?;
        let batches: Vec<RecordBatch> = stream.try_collect().await?;

        let mut sum: Vec<f64> = vec![0.0; self.dim];
        let mut count: u64 = 0;
        for batch in &batches {
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
            if dim != self.dim {
                return Err(StoreError::DimMismatch {
                    have: dim,
                    want: self.dim,
                });
            }
            for i in 0..batch.num_rows() {
                if emb_arr.is_null(i) {
                    continue;
                }
                let start = i * dim;
                let slice = &f32_values.values()[start..start + dim];
                for (slot, x) in sum.iter_mut().zip(slice.iter()) {
                    *slot += f64::from(*x);
                }
                count += 1;
            }
        }

        if count < 10 {
            return Ok(None);
        }
        // f64 accumulator → f32 mean. The mean magnitude is bounded by
        // the embedding magnitude (unit-ish), so f32 precision is fine.
        #[allow(clippy::cast_possible_truncation)]
        let inv = 1.0_f64 / count as f64;
        let mean: Vec<f32> = sum.into_iter().map(|s| (s * inv) as f32).collect();
        Ok(Some(mean))
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

    /// Aggressive optimize: compact + reindex (via [`OptimizeAction::All`]),
    /// then prune **all** historical versions, collapsing the table to its
    /// latest version. Returns the number of old versions removed.
    ///
    /// Unlike [`Self::optimize_all`], this overrides Lance's default
    /// retention: `delete_unverified = true` bypasses the 7-day
    /// in-progress-transaction guard so versions created *today* are
    /// actually removed. That is the only way to undo a version explosion
    /// from a heavy scan without waiting two weeks — but it is **only safe
    /// when no other process is writing this corpus**. Callers must
    /// guarantee exclusivity (the `optimize --aggressive` CLI path makes
    /// the operator responsible).
    pub async fn optimize_compact_and_prune(&self) -> Result<u64> {
        let table = self.conn.open_table(CORPUS_TABLE).execute().await?;
        // 1. Compact small fragments + fold new data into indices.
        table.optimize(OptimizeAction::All).await?;
        // 2. Prune every version older than "now" (keep only the latest).
        let stats = table
            .optimize(OptimizeAction::Prune {
                older_than: Some(chrono::Duration::zero()),
                delete_unverified: Some(true),
                error_if_tagged_old_versions: Some(false),
            })
            .await?;
        Ok(stats.prune.map_or(0, |p| p.old_versions))
    }

    /// Cheap, scan-time variant of [`Self::optimize_all`]: just fold any
    /// fragments appended since the last index build into the existing
    /// scalar / FTS indices. Skips `Compact` and `Prune` because both
    /// scan with the version count, which is `O(commits)` and slow on a
    /// long-running corpus — and the index-folding step is the only
    /// one that affects the next scan's lookup cost.
    pub async fn optimize_indices(&self) -> Result<()> {
        let table = self.conn.open_table(CORPUS_TABLE).execute().await?;
        table
            .optimize(OptimizeAction::Index(Default::default()))
            .await?;
        Ok(())
    }

    /// Idempotently strip lance's per-commit auto-cleanup config from
    /// the dataset manifest. Returns `true` if a write happened.
    ///
    /// Lance's `WriteParams::default()` enables an auto-cleanup hook
    /// with `interval=20 / older_than=14days` and writes those values
    /// into the manifest on first insert. Lance then fires
    /// `cleanup_old_versions` on every 20th commit (see
    /// `lance/src/io/commit.rs:972` → `auto_cleanup_hook`). The hook
    /// walks every manifest file in the dataset, decoding each one's
    /// protobuf, even when nothing meets the retention threshold —
    /// which is the steady state for a write-heavy substrate where
    /// most versions are fresh. Profiling showed ~67% of scan CPU
    /// going to this walk on a 22 000-version corpus. We never query
    /// historical versions, so cleanup belongs in the explicit
    /// `ostk-recall optimize` path, not as a per-commit tax.
    ///
    /// Deletes the two keys via `delete_config_keys`. If the keys
    /// aren't present the call is a no-op write (one tiny manifest
    /// commit). Tracked separately so callers can suppress the noise
    /// of a per-startup config sync when it's not needed.
    pub async fn ensure_auto_cleanup_disabled(&self) -> Result<bool> {
        let table = self.conn.open_table(CORPUS_TABLE).execute().await?;
        // `manifest()` / `delete_config_keys()` live on `NativeTable`, not
        // the `Table` trait. `as_native()` returns `None` only for the
        // remote-LanceDB client; we always use the embedded path so this
        // is infallible in practice.
        let Some(native) = table.as_native() else {
            return Ok(false);
        };
        let manifest = native.manifest().await?;
        let has_interval = manifest.config.contains_key("lance.auto_cleanup.interval");
        let has_older = manifest
            .config
            .contains_key("lance.auto_cleanup.older_than");
        if !has_interval && !has_older {
            return Ok(false);
        }
        tracing::info!(
            interval = has_interval,
            older_than = has_older,
            "stripping lance.auto_cleanup.* from corpus manifest"
        );
        native
            .delete_config_keys(&[
                "lance.auto_cleanup.interval",
                "lance.auto_cleanup.older_than",
            ])
            .await?;
        Ok(true)
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
                        chunk_ids: Vec::new(),
                        samples: Vec::new(),
                    });
                entry.count += 1;
                if ts < entry.min_ts {
                    entry.min_ts = ts;
                }
                if ts > entry.max_ts {
                    entry.max_ts = ts;
                }
                entry.chunk_ids.push(chunk_id.clone());
                // Keep up to `samples_per_group` of the most-recent
                // (chunk_id, ts, text) tuples. We sort by ts at the end.
                entry.samples.push((chunk_id, ts, text));
            }
        }

        // Finalise: keep only the N most-recent samples per group, and
        // give chunk_ids a stable lexicographic order so identity hashes
        // are reproducible across runs.
        let mut out: Vec<ActivityBurst> = groups.into_values().collect();
        for burst in &mut out {
            burst.samples.sort_by(|a, b| b.1.cmp(&a.1));
            burst.samples.truncate(samples_per_group);
            burst.chunk_ids.sort();
        }
        Ok(out)
    }

    /// Fetch `(chunk_id, ts)` for each requested id. Returns a map; ids
    /// not found in the corpus (or with NULL ts) are absent from the
    /// result. Used by `thread_query`'s cross-axis backfill to compute
    /// an activity score over a cluster whose surfacing primitive
    /// didn't carry timestamps.
    pub async fn fetch_timestamps(
        &self,
        chunk_ids: &[String],
    ) -> Result<std::collections::HashMap<String, chrono::DateTime<chrono::Utc>>> {
        use arrow_array::Array;
        use std::collections::HashMap;

        if chunk_ids.is_empty() {
            return Ok(HashMap::new());
        }
        let table = self.conn.open_table(CORPUS_TABLE).execute().await?;
        let quoted: Vec<String> = chunk_ids
            .iter()
            .map(|id| format!("'{}'", id.replace('\'', "''")))
            .collect();
        let filter = format!("chunk_id IN ({})", quoted.join(","));
        let stream = table
            .query()
            .only_if(filter)
            .select(Select::Columns(vec!["chunk_id".into(), "ts".into()]))
            .execute()
            .await?;
        let batches: Vec<RecordBatch> = stream.try_collect().await?;
        let mut out: HashMap<String, chrono::DateTime<chrono::Utc>> = HashMap::new();
        for batch in &batches {
            let id_arr = column_as_str(batch, "chunk_id")?;
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
                out.insert(id_arr.value(i).to_string(), ts);
            }
        }
        Ok(out)
    }

    /// Return non-stale chunk ids in a time window, grouped by source kind
    /// and split into fixed-size batches.
    ///
    /// `since = None` means the whole active corpus. `since = Some(ts)`
    /// excludes rows with NULL timestamps, which is the only defensible
    /// interpretation for a lower-bound time filter.
    pub async fn chunk_id_batches_by_source_window(
        &self,
        since: Option<chrono::DateTime<chrono::Utc>>,
        epoch_size: usize,
    ) -> Result<Vec<CorpusWindowBatch>> {
        use std::collections::HashMap;

        let epoch_size = epoch_size.max(1);
        let table = self.conn.open_table(CORPUS_TABLE).execute().await?;
        let filter = match since {
            Some(ts) => format!(
                "stale = false AND ts >= TIMESTAMP '{}'",
                ts.format("%Y-%m-%d %H:%M:%S")
            ),
            None => "stale = false".to_string(),
        };
        let stream = table
            .query()
            .only_if(filter)
            .select(Select::Columns(vec!["chunk_id".into(), "source".into()]))
            .execute()
            .await?;
        let batches: Vec<RecordBatch> = stream.try_collect().await?;

        let mut grouped: HashMap<SourceKind, Vec<String>> = HashMap::new();
        for batch in &batches {
            let id_arr = column_as_str(batch, "chunk_id")?;
            let source_arr = column_as_str(batch, "source")?;
            for i in 0..batch.num_rows() {
                let Some(source) = parse_source_kind(source_arr.value(i)) else {
                    continue;
                };
                grouped
                    .entry(source)
                    .or_default()
                    .push(id_arr.value(i).to_string());
            }
        }

        let mut out = Vec::new();
        for (source, mut ids) in grouped {
            ids.sort();
            for chunk in ids.chunks(epoch_size) {
                out.push(CorpusWindowBatch {
                    source,
                    chunk_ids: chunk.to_vec(),
                });
            }
        }
        out.sort_by(|a, b| {
            a.source
                .as_str()
                .cmp(b.source.as_str())
                .then_with(|| a.chunk_ids.first().cmp(&b.chunk_ids.first()))
        });
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
///
/// `chunk_ids` is the full membership of the burst (sorted lexicographically
/// for stable identity), separate from the truncated `samples` used for
/// human-readable snippets. v0.4.1+ cross-axis backfill in `thread_query`
/// joins on chunk_ids; the samples remain bounded by `samples_per_group`
/// to keep response payloads reasonable.
#[derive(Debug, Clone)]
pub struct ActivityBurst {
    pub project: String,
    pub source_id: String,
    pub count: usize,
    pub min_ts: chrono::DateTime<chrono::Utc>,
    pub max_ts: chrono::DateTime<chrono::Utc>,
    pub chunk_ids: Vec<String>,
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

/// Required Utf8 column — error if missing or wrong type.
fn downcast_str<'a>(batch: &'a RecordBatch, name: &str) -> Result<&'a StringArray> {
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

/// Optional Utf8 column — returns `None` if absent or the wrong type.
fn downcast_str_opt<'a>(batch: &'a RecordBatch, name: &str) -> Option<&'a StringArray> {
    batch
        .column_by_name(name)
        .and_then(|c| c.as_any().downcast_ref::<StringArray>())
}

/// Inverse of `Source::as_str`. Returns `None` for unknown strings —
/// callers skip such rows rather than fail.
fn parse_source(s: &str) -> Option<ostk_recall_core::Source> {
    use ostk_recall_core::Source;
    Some(match s {
        "markdown" => Source::Markdown,
        "code" => Source::Code,
        "claude_code" => Source::ClaudeCode,
        "ostk_decision" => Source::OstkDecision,
        "ostk_needle" => Source::OstkNeedle,
        "ostk_audit_significant" => Source::OstkAuditSignificant,
        "ostk_conversation" => Source::OstkConversation,
        "ostk_session" => Source::OstkSession,
        "ostk_memory" => Source::OstkMemory,
        "ostk_spec" => Source::OstkSpec,
        "file_glob" => Source::FileGlob,
        "zip_export" => Source::ZipExport,
        "gemini" => Source::Gemini,
        "codex" => Source::Codex,
        "thread" => Source::Thread,
        "membrane" => Source::Membrane,
        _ => return None,
    })
}

/// Map concrete stored source rows back to their scanner/source kind.
fn parse_source_kind(s: &str) -> Option<SourceKind> {
    Some(match s {
        "markdown" => SourceKind::Markdown,
        "code" => SourceKind::Code,
        "claude_code" => SourceKind::ClaudeCode,
        "ostk_decision"
        | "ostk_needle"
        | "ostk_audit_significant"
        | "ostk_conversation"
        | "ostk_session"
        | "ostk_memory"
        | "ostk_spec" => SourceKind::OstkProject,
        "file_glob" => SourceKind::FileGlob,
        "zip_export" => SourceKind::ZipExport,
        "gemini" => SourceKind::Gemini,
        "codex" => SourceKind::Codex,
        "thread" => SourceKind::Thread,
        "membrane" => SourceKind::Membrane,
        _ => return None,
    })
}

/// Required UInt32 column.
fn downcast_u32<'a>(batch: &'a RecordBatch, name: &str) -> Result<&'a UInt32Array> {
    let col = batch.column_by_name(name).ok_or_else(|| {
        StoreError::Arrow(arrow::error::ArrowError::SchemaError(format!(
            "{name} column missing in projection"
        )))
    })?;
    col.as_any().downcast_ref::<UInt32Array>().ok_or_else(|| {
        StoreError::Arrow(arrow::error::ArrowError::CastError(format!(
            "{name} expected to be UInt32"
        )))
    })
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
    // source_config_id is non-empty in normal flow but the Lance column is
    // nullable for the migration window — encode empty as NULL so v0.5
    // rows added later round-trip identically.
    let source_config_id: StringArray = chunks
        .iter()
        .map(|c| {
            if c.source_config_id.is_empty() {
                None
            } else {
                Some(c.source_config_id.as_str())
            }
        })
        .collect();
    // P1: facets serialized as `key:value` strings via to_list (sorted
    // for stable round-trip). Column is List<Utf8>.
    let facets_lists: Vec<Vec<String>> = chunks
        .iter()
        .map(|c| ostk_recall_core::to_list(&c.facets))
        .collect();
    let facets_col = {
        use arrow_array::builder::{ListBuilder, StringBuilder};
        let mut b = ListBuilder::new(StringBuilder::new());
        for list in &facets_lists {
            for v in list {
                b.values().append_value(v);
            }
            b.append(true);
        }
        b.finish()
    };
    // P1: embedding_input_sha256 — nullable; empty string maps to NULL
    // for migration symmetry.
    let embedding_input_sha256: StringArray = chunks
        .iter()
        .map(|c| {
            if c.embedding_input_sha256.is_empty() {
                None
            } else {
                Some(c.embedding_input_sha256.as_str())
            }
        })
        .collect();
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
            Arc::new(source_config_id),
            Arc::new(facets_col),
            Arc::new(embedding_input_sha256),
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
            facets: Default::default(),
            embedding_input_sha256: String::new(),
            source_config_id: "test-cfg".into(),
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
    async fn fetch_texts_with_project_returns_per_chunk_project() {
        let tmp = TempDir::new().unwrap();
        let store = CorpusStore::open_or_create(tmp.path(), 4).await.unwrap();

        let mut a = sample_chunk("a", "alpha text");
        a.project = Some("proja".into());
        let mut b = sample_chunk("b", "beta text");
        b.project = Some("projb".into());
        let mut c = sample_chunk("c", "gamma text");
        c.project = None; // null project column
        let embs = vec![
            vec![1.0, 0.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0, 0.0],
            vec![0.0, 0.0, 1.0, 0.0],
        ];
        store.upsert(&[a, b, c], &embs).await.unwrap();

        let got = store
            .fetch_texts_with_project(&["a".into(), "b".into(), "c".into()])
            .await
            .unwrap();
        assert_eq!(got["a"].0, "alpha text");
        assert_eq!(got["a"].1.as_deref(), Some("proja"));
        assert_eq!(got["b"].1.as_deref(), Some("projb"));
        assert_eq!(got["c"].0, "gamma text");
        assert_eq!(got["c"].1, None, "null project column → None");
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
    async fn optimize_compact_and_prune_removes_old_versions_keeps_data() {
        let tmp = TempDir::new().unwrap();
        let store = CorpusStore::open_or_create(tmp.path(), 4).await.unwrap();
        // Many separate upserts → many versions (mimics per-batch commits).
        for i in 0..8 {
            store
                .upsert(
                    &[sample_chunk(&format!("c{i}"), "text")],
                    &[vec![1.0, 0.0, 0.0, 0.0]],
                )
                .await
                .unwrap();
        }
        let before = store.version().await.unwrap();
        assert!(
            before >= 8,
            "expected many versions before prune, got {before}"
        );

        let pruned = store.optimize_compact_and_prune().await.unwrap();
        assert!(
            pruned > 0,
            "aggressive prune should remove historical versions, removed {pruned}"
        );
        // Prune drops history, never live rows.
        assert_eq!(store.row_count().await.unwrap(), 8);
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

    fn chunk_with(id: &str, project: &str, ts: chrono::DateTime<chrono::Utc>) -> Chunk {
        Chunk {
            chunk_id: id.to_string(),
            source: Source::Markdown,
            project: Some(project.to_string()),
            source_id: format!("{id}.md"),
            facets: Default::default(),
            embedding_input_sha256: String::new(),
            source_config_id: "test-cfg".to_string(),
            chunk_index: 0,
            ts: Some(ts),
            role: None,
            text: format!("text-{id}"),
            sha256: Chunk::content_hash(id),
            links: Links::default(),
            extra: serde_json::Value::Null,
        }
    }

    #[tokio::test]
    async fn project_baseline_mean_returns_none_below_threshold() {
        let tmp = TempDir::new().unwrap();
        let store = CorpusStore::open_or_create(tmp.path(), 4).await.unwrap();

        let now = chrono::Utc::now();
        let mut chunks = Vec::new();
        let mut embs = Vec::new();
        for i in 0..9_u32 {
            chunks.push(chunk_with(&format!("c{i}"), "alpha", now));
            embs.push(vec![1.0_f32, 0.0, 0.0, 0.0]);
        }
        store.upsert(&chunks, &embs).await.unwrap();

        let out = store.project_baseline_mean(Some("alpha"), 7).await.unwrap();
        assert!(out.is_none(), "9 chunks must be below the 10-chunk floor");
    }

    #[tokio::test]
    async fn project_baseline_mean_computes_exact_mean() {
        let tmp = TempDir::new().unwrap();
        let store = CorpusStore::open_or_create(tmp.path(), 4).await.unwrap();

        let now = chrono::Utc::now();
        let mut chunks = Vec::new();
        let mut embs = Vec::new();
        // 10 unit vectors on axis 0 + 10 on axis 1, all project=alpha.
        for i in 0..10_u32 {
            chunks.push(chunk_with(&format!("a{i}"), "alpha", now));
            embs.push(vec![1.0_f32, 0.0, 0.0, 0.0]);
        }
        for i in 0..10_u32 {
            chunks.push(chunk_with(&format!("b{i}"), "alpha", now));
            embs.push(vec![0.0_f32, 1.0, 0.0, 0.0]);
        }
        store.upsert(&chunks, &embs).await.unwrap();

        let mean = store
            .project_baseline_mean(Some("alpha"), 7)
            .await
            .unwrap()
            .expect("20 chunks → Some(mean)");
        assert_eq!(mean.len(), 4);
        // Each axis sums 10 -> mean 0.5; remaining axes 0.
        assert!((mean[0] - 0.5).abs() < 1e-6, "mean[0] = {}", mean[0]);
        assert!((mean[1] - 0.5).abs() < 1e-6, "mean[1] = {}", mean[1]);
        assert!(mean[2].abs() < 1e-6);
        assert!(mean[3].abs() < 1e-6);
    }

    #[tokio::test]
    async fn project_baseline_mean_respects_project_filter() {
        let tmp = TempDir::new().unwrap();
        let store = CorpusStore::open_or_create(tmp.path(), 4).await.unwrap();

        let now = chrono::Utc::now();
        let mut chunks = Vec::new();
        let mut embs = Vec::new();
        // 10 chunks in `alpha` on axis 0.
        for i in 0..10_u32 {
            chunks.push(chunk_with(&format!("a{i}"), "alpha", now));
            embs.push(vec![1.0_f32, 0.0, 0.0, 0.0]);
        }
        // 10 chunks in `beta` on axis 1.
        for i in 0..10_u32 {
            chunks.push(chunk_with(&format!("b{i}"), "beta", now));
            embs.push(vec![0.0_f32, 1.0, 0.0, 0.0]);
        }
        store.upsert(&chunks, &embs).await.unwrap();

        let alpha = store
            .project_baseline_mean(Some("alpha"), 7)
            .await
            .unwrap()
            .expect("alpha has 10 chunks");
        assert!((alpha[0] - 1.0).abs() < 1e-6);
        assert!(alpha[1].abs() < 1e-6);

        let beta = store
            .project_baseline_mean(Some("beta"), 7)
            .await
            .unwrap()
            .expect("beta has 10 chunks");
        assert!(beta[0].abs() < 1e-6);
        assert!((beta[1] - 1.0).abs() < 1e-6);

        // Global baseline (no project filter): mean of all 20 = [0.5, 0.5, 0, 0].
        let global = store
            .project_baseline_mean(None, 7)
            .await
            .unwrap()
            .expect("20 chunks total");
        assert!((global[0] - 0.5).abs() < 1e-6);
        assert!((global[1] - 0.5).abs() < 1e-6);
    }

    #[tokio::test]
    async fn sample_recent_chunks_with_project_returns_project_column() {
        let tmp = TempDir::new().unwrap();
        let store = CorpusStore::open_or_create(tmp.path(), 4).await.unwrap();

        let now = chrono::Utc::now();
        let chunks = vec![chunk_with("a", "alpha", now), chunk_with("b", "beta", now)];
        let embs = vec![vec![1.0, 0.0, 0.0, 0.0], vec![0.0, 1.0, 0.0, 0.0]];
        store.upsert(&chunks, &embs).await.unwrap();

        let since = now - chrono::Duration::hours(1);
        let out = store
            .sample_recent_chunks_with_project(since, 100)
            .await
            .unwrap();
        assert_eq!(out.len(), 2);
        let by_id: std::collections::HashMap<String, (Option<String>, Vec<f32>)> =
            out.into_iter().map(|(id, p, e)| (id, (p, e))).collect();
        assert_eq!(by_id.get("a").unwrap().0.as_deref(), Some("alpha"));
        assert_eq!(by_id.get("a").unwrap().1, vec![1.0, 0.0, 0.0, 0.0]);
        assert_eq!(by_id.get("b").unwrap().0.as_deref(), Some("beta"));
    }
}
