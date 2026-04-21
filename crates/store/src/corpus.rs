use std::path::{Path, PathBuf};
use std::sync::Arc;

use arrow_array::{
    FixedSizeListArray, RecordBatch, RecordBatchIterator, RecordBatchReader, StringArray,
    TimestampMicrosecondArray, UInt32Array,
};
use arrow_schema::Schema;
use futures::TryStreamExt;
use lancedb::Connection;
use lancedb::index::Index;
use lancedb::index::scalar::FtsIndexBuilder;
use lancedb::query::{ExecutableQuery, QueryBase, Select};
use ostk_recall_core::Chunk;
use thiserror::Error;

use crate::schema::{CORPUS_TABLE, corpus_schema};

#[derive(Debug, Error)]
pub enum StoreError {
    #[error("lancedb: {0}")]
    Lance(#[from] lancedb::Error),

    #[error("arrow: {0}")]
    Arrow(#[from] arrow::error::ArrowError),

    #[error("schema mismatch: table dim {have}, embedder dim {want}")]
    DimMismatch { have: usize, want: usize },

    #[error("io: {0}")]
    Io(#[from] std::io::Error),

    #[error("json: {0}")]
    Json(#[from] serde_json::Error),
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
}
