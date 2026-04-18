use std::path::Path;
use std::sync::Mutex;

use duckdb::{Connection, params};
use ostk_recall_core::Source;
use serde::{Deserialize, Serialize};

use crate::corpus::{Result, StoreError};

impl From<duckdb::Error> for StoreError {
    fn from(e: duckdb::Error) -> Self {
        Self::Lance(lancedb::Error::Other {
            message: format!("duckdb: {e}"),
            source: Some(Box::new(e)),
        })
    }
}

/// Idempotency + provenance metadata. Lives at `<root>/ingest.duckdb`.
///
/// The inner `Connection` is wrapped in a `Mutex` so this type is `Sync`
/// and can live inside an `Arc` shared between async tasks. `duckdb::
/// Connection` is already `Send` but not `Sync` (it holds `RefCell`s), so
/// the mutex is the minimal fix.
pub struct IngestDb {
    conn: Mutex<Connection>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IngestChunkRow {
    pub chunk_id: String,
    pub source: String,
    pub source_id: String,
    pub chunk_index: u32,
    pub content_sha256: String,
}

impl IngestDb {
    pub fn open(root: &Path) -> Result<Self> {
        let path = root.join("ingest.duckdb");
        let conn = Connection::open(path)?;
        Self::migrate(&conn)?;
        Ok(Self {
            conn: Mutex::new(conn),
        })
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Connection> {
        self.conn
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    fn migrate(conn: &Connection) -> Result<()> {
        conn.execute_batch(
            r"
CREATE TABLE IF NOT EXISTS ingest_runs (
    run_id       VARCHAR PRIMARY KEY,
    source       VARCHAR NOT NULL,
    scanner      VARCHAR NOT NULL,
    started_at   TIMESTAMP NOT NULL,
    finished_at  TIMESTAMP,
    items_seen   INTEGER DEFAULT 0,
    chunks_emitted INTEGER DEFAULT 0,
    chunks_upserted INTEGER DEFAULT 0,
    errors       INTEGER DEFAULT 0,
    notes        JSON
);

CREATE TABLE IF NOT EXISTS ingest_chunks (
    chunk_id      VARCHAR PRIMARY KEY,
    source        VARCHAR NOT NULL,
    source_id     VARCHAR NOT NULL,
    chunk_index   INTEGER NOT NULL,
    content_sha256 VARCHAR NOT NULL,
    upserted_at   TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_ingest_chunks_source
    ON ingest_chunks(source, source_id);
",
        )?;
        Ok(())
    }

    /// Does a chunk with this id and content hash already exist?
    #[allow(clippy::significant_drop_tightening)]
    pub fn content_already_ingested(&self, chunk_id: &str, content_sha256: &str) -> Result<bool> {
        let conn = self.lock();
        let mut stmt = conn.prepare(
            "SELECT 1 FROM ingest_chunks WHERE chunk_id = ? AND content_sha256 = ? LIMIT 1",
        )?;
        let mut rows = stmt.query(params![chunk_id, content_sha256])?;
        Ok(rows.next()?.is_some())
    }

    #[allow(clippy::significant_drop_tightening)]
    pub fn record_chunk(&self, row: &IngestChunkRow) -> Result<()> {
        let conn = self.lock();
        conn.execute(
            "INSERT OR REPLACE INTO ingest_chunks
             (chunk_id, source, source_id, chunk_index, content_sha256)
             VALUES (?, ?, ?, ?, ?)",
            params![
                row.chunk_id,
                row.source,
                row.source_id,
                i64::from(row.chunk_index),
                row.content_sha256
            ],
        )?;
        Ok(())
    }

    #[allow(clippy::significant_drop_tightening)]
    pub fn count_by_source(&self) -> Result<Vec<(String, u64)>> {
        let conn = self.lock();
        let mut stmt = conn.prepare(
            "SELECT source, COUNT(*) FROM ingest_chunks GROUP BY source ORDER BY source",
        )?;
        let rows = stmt
            .query_map([], |r| Ok((r.get::<_, String>(0)?, r.get::<_, i64>(1)?)))?
            .collect::<std::result::Result<Vec<_>, _>>()?;
        Ok(rows
            .into_iter()
            .map(|(s, n)| (s, u64::try_from(n).unwrap_or(0)))
            .collect())
    }

    #[allow(clippy::significant_drop_tightening)]
    pub fn count_for_source(&self, source: Source) -> Result<u64> {
        let conn = self.lock();
        let mut stmt = conn.prepare("SELECT COUNT(*) FROM ingest_chunks WHERE source = ?")?;
        let n: i64 = stmt.query_row(params![source.as_str()], |r| r.get(0))?;
        Ok(u64::try_from(n).unwrap_or(0))
    }

    /// Delete ingest rows matching any of the supplied chunk ids. Returns
    /// the total number of rows removed. Used by the `--reingest` path
    /// after LanceDB rows for a project have been collected — deleting
    /// here lets the next scan pass re-ingest those chunks instead of
    /// short-circuiting on the `content_already_ingested` dedupe check.
    ///
    /// `ingest_chunks` has no `project` column, so the caller (the CLI
    /// reingest wrapper) first queries LanceDB for every chunk_id whose
    /// `project` matches the flag, then passes those ids here in batches.
    #[allow(clippy::significant_drop_tightening)]
    pub fn delete_by_chunk_ids(&self, chunk_ids: &[String]) -> Result<u64> {
        if chunk_ids.is_empty() {
            return Ok(0);
        }
        let conn = self.lock();
        let mut total: u64 = 0;
        for batch in chunk_ids.chunks(1000) {
            let placeholders = std::iter::repeat_n('?', batch.len())
                .collect::<Vec<_>>()
                .into_iter()
                .map(|c| c.to_string())
                .collect::<Vec<_>>()
                .join(",");
            let sql = format!("DELETE FROM ingest_chunks WHERE chunk_id IN ({placeholders})");
            let params_refs: Vec<&dyn duckdb::ToSql> =
                batch.iter().map(|s| s as &dyn duckdb::ToSql).collect();
            let n = conn.execute(&sql, duckdb::params_from_iter(params_refs.iter().copied()))?;
            total = total.saturating_add(u64::try_from(n).unwrap_or(0));
        }
        Ok(total)
    }

    /// Latest `upserted_at` across all ingested chunks, as RFC3339 text.
    /// Returns None if the table is empty.
    #[allow(clippy::significant_drop_tightening)]
    pub fn latest_upserted_at(&self) -> Result<Option<String>> {
        let conn = self.lock();
        let mut stmt =
            conn.prepare("SELECT CAST(MAX(upserted_at) AS VARCHAR) FROM ingest_chunks")?;
        let s: Option<String> = stmt.query_row([], |r| r.get(0))?;
        Ok(s)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn ingest_roundtrip() {
        let tmp = TempDir::new().unwrap();
        let db = IngestDb::open(tmp.path()).unwrap();
        assert!(db.count_by_source().unwrap().is_empty());

        let row = IngestChunkRow {
            chunk_id: "abc".into(),
            source: "markdown".into(),
            source_id: "foo.md".into(),
            chunk_index: 0,
            content_sha256: "deadbeef".into(),
        };
        db.record_chunk(&row).unwrap();
        assert!(db.content_already_ingested("abc", "deadbeef").unwrap());
        assert!(!db.content_already_ingested("abc", "other").unwrap());
        assert_eq!(db.count_by_source().unwrap(), vec![("markdown".into(), 1)]);
    }
}
