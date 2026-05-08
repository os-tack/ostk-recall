use std::path::Path;
use std::sync::Mutex;

use rusqlite::{Connection, params};
use ostk_recall_core::Source;
use serde::{Deserialize, Serialize};

use crate::corpus::{Result, StoreError};

impl From<rusqlite::Error> for StoreError {
    fn from(e: rusqlite::Error) -> Self {
        Self::Lance(lancedb::Error::Other {
            message: format!("sqlite: {e}"),
            source: Some(Box::new(e)),
        })
    }
}

/// Idempotency + provenance metadata. Lives at <root>/ingest.sqlite.
pub struct IngestDb {
    conn: Mutex<Connection>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IngestChunkRow {
    pub chunk_id: String,
    pub source: String,
    pub project: String,
    pub source_id: String,
    pub chunk_index: u32,
    pub content_sha256: String,
}

impl IngestDb {
    pub fn open(root: &Path) -> Result<Self> {
        let path = root.join("ingest.sqlite");
        let conn = Connection::open(path)?;
        Self::setup_connection(&conn)?;
        Self::migrate(&conn)?;
        Ok(Self {
            conn: Mutex::new(conn),
        })
    }

    pub fn open_read_only(root: &Path) -> Result<Self> {
        let path = root.join("ingest.sqlite");
        let conn = Connection::open_with_flags(
            path,
            rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY | rusqlite::OpenFlags::SQLITE_OPEN_NO_MUTEX,
        )?;
        Self::setup_connection(&conn)?;
        Ok(Self {
            conn: Mutex::new(conn),
        })
    }

    fn setup_connection(conn: &Connection) -> Result<()> {
        conn.execute_batch(
            "PRAGMA journal_mode = WAL;
             PRAGMA busy_timeout = 5000;
             PRAGMA synchronous = NORMAL;"
        )?;
        Ok(())
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Connection> {
        self.conn
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    fn migrate(conn: &Connection) -> Result<()> {
        conn.execute_batch(
            r"
CREATE TABLE IF NOT EXISTS ingest_sources (
    source        TEXT NOT NULL,
    project       TEXT NOT NULL DEFAULT '',
    source_id     TEXT NOT NULL,
    mtime_micros  INTEGER NOT NULL,
    size_bytes    INTEGER NOT NULL,
    last_run_id   TEXT,
    PRIMARY KEY (source, project, source_id)
);

CREATE TABLE IF NOT EXISTS ingest_chunks (
    chunk_id      TEXT PRIMARY KEY,
    source        TEXT NOT NULL,
    project       TEXT NOT NULL DEFAULT '',
    source_id     TEXT NOT NULL,
    chunk_index   INTEGER NOT NULL,
    content_sha256 TEXT NOT NULL,
    upserted_at   TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    last_run_id   TEXT
);

CREATE INDEX IF NOT EXISTS idx_ingest_chunks_source_project
    ON ingest_chunks(source, project, source_id);
",
        )?;
        
        // Migration for existing table
        let _ = conn.execute("ALTER TABLE ingest_sources ADD COLUMN project TEXT NOT NULL DEFAULT ''", []);
        let _ = conn.execute("ALTER TABLE ingest_chunks ADD COLUMN project TEXT NOT NULL DEFAULT ''", []);
        
        Ok(())
    }

    pub fn content_already_ingested(&self, chunk_id: &str, content_sha256: &str) -> Result<bool> {
        let conn = self.lock();
        let mut stmt = conn.prepare(
            "SELECT 1 FROM ingest_chunks WHERE chunk_id = ? AND content_sha256 = ? LIMIT 1",
        )?;
        let exists = stmt.exists(params![chunk_id, content_sha256])?;
        Ok(exists)
    }

    pub fn record_chunk(&self, row: &IngestChunkRow, run_id: Option<&str>) -> Result<()> {
        let conn = self.lock();
        conn.execute(
            "INSERT OR REPLACE INTO ingest_chunks
             (chunk_id, source, project, source_id, chunk_index, content_sha256, last_run_id)
             VALUES (?, ?, ?, ?, ?, ?, ?)",
            params![
                row.chunk_id,
                row.source,
                row.project,
                row.source_id,
                row.chunk_index,
                row.content_sha256,
                run_id
            ],
        )?;
        Ok(())
    }

    pub fn get_source_metadata(&self, source: &str, project: &str, source_id: &str) -> Result<Option<(i64, i64)>> {
        let conn = self.lock();
        let mut stmt = conn.prepare(
            "SELECT mtime_micros, size_bytes FROM ingest_sources WHERE source = ? AND project = ? AND source_id = ?"
        )?;
        let res = stmt.query_row(params![source, project, source_id], |r| {
            let m: i64 = r.get(0)?;
            let s: i64 = r.get(1)?;
            Ok((m, s))
        });
        match res {
            Ok(v) => Ok(Some(v)),
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
            Err(e) => Err(e.into()),
        }
    }

    pub fn touch_source_chunks(&self, source: &str, project: &str, source_id: &str, run_id: &str) -> Result<()> {
        let conn = self.lock();
        conn.execute(
            "UPDATE ingest_sources SET last_run_id = ? WHERE source = ? AND project = ? AND source_id = ?",
            params![run_id, source, project, source_id],
        )?;
        conn.execute(
            "UPDATE ingest_chunks SET last_run_id = ? WHERE source = ? AND project = ? AND source_id = ?",
            params![run_id, source, project, source_id],
        )?;
        Ok(())
    }

    pub fn update_source_metadata(&self, source: &str, project: &str, source_id: &str, mtime: i64, size: i64, run_id: &str) -> Result<()> {
        let conn = self.lock();
        conn.execute(
            "INSERT OR REPLACE INTO ingest_sources (source, project, source_id, mtime_micros, size_bytes, last_run_id)
             VALUES (?, ?, ?, ?, ?, ?)",
            params![source, project, source_id, mtime, size, run_id],
        )?;
        Ok(())
    }

    pub fn delete_orphans(&self, source: &str, project: &str, current_run_id: &str) -> Result<Vec<String>> {
        let conn = self.lock();
        let mut ids = Vec::new();
        {
            let mut stmt = conn.prepare(
                "SELECT chunk_id FROM ingest_chunks WHERE source = ? AND project = ? AND (last_run_id != ? OR last_run_id IS NULL)"
            )?;
            let mut rows = stmt.query(params![source, project, current_run_id])?;
            while let Some(row) = rows.next()? {
                let id: String = row.get(0)?;
                ids.push(id);
            }
        }
        
        conn.execute(
            "DELETE FROM ingest_chunks WHERE source = ? AND project = ? AND (last_run_id != ? OR last_run_id IS NULL)",
            params![source, project, current_run_id]
        )?;

        conn.execute(
            "DELETE FROM ingest_sources WHERE source = ? AND project = ? AND (last_run_id != ? OR last_run_id IS NULL)",
            params![source, project, current_run_id]
        )?;
        
        Ok(ids)
    }

    pub fn mark_orphans_stale(&self, source: &str, project: &str, current_run_id: &str) -> Result<Vec<String>> {
        let conn = self.lock();
        let mut ids = Vec::new();
        let mut stmt = conn.prepare(
            "SELECT chunk_id FROM ingest_chunks WHERE source = ? AND project = ? AND (last_run_id != ? OR last_run_id IS NULL)"
        )?;
        let mut rows = stmt.query(params![source, project, current_run_id])?;
        while let Some(row) = rows.next()? {
            let id: String = row.get(0)?;
            ids.push(id);
        }
        Ok(ids)
    }

    pub fn count_by_source(&self) -> Result<Vec<(String, u64)>> {
        let conn = self.lock();
        let mut stmt = conn.prepare(
            "SELECT source, COUNT(*) FROM ingest_chunks GROUP BY source ORDER BY source",
        )?;
        let rows = stmt
            .query_map([], |r| {
                let s: String = r.get(0)?;
                let n: i64 = r.get(1)?;
                Ok((s, n as u64))
            })?
            .collect::<std::result::Result<Vec<_>, rusqlite::Error>>()?;
        Ok(rows)
    }

    pub fn count_active_by_source(&self) -> Result<Vec<(String, u64)>> {
        self.count_by_source()
    }

    pub fn count_for_source(&self, source: Source) -> Result<u64> {
        let conn = self.lock();
        let mut stmt = conn.prepare("SELECT COUNT(*) FROM ingest_chunks WHERE source = ?")?;
        let n: i64 = stmt.query_row(params![source.as_str()], |r| r.get(0))?;
        Ok(n as u64)
    }

    pub fn delete_by_chunk_ids(&self, chunk_ids: &[String]) -> Result<u64> {
        if chunk_ids.is_empty() {
            return Ok(0);
        }
        let conn = self.lock();
        let mut total: u64 = 0;
        for batch in chunk_ids.chunks(999) {
            let placeholders = std::iter::repeat("?").take(batch.len()).collect::<Vec<_>>().join(",");
            let sql = format!("DELETE FROM ingest_chunks WHERE chunk_id IN ({})", placeholders);
            let mut stmt = conn.prepare(&sql)?;
            let n = stmt.execute(rusqlite::params_from_iter(batch))?;
            total += n as u64;
        }
        Ok(total)
    }

    pub fn latest_upserted_at(&self) -> Result<Option<String>> {
        let conn = self.lock();
        let mut stmt =
            conn.prepare("SELECT MAX(upserted_at) FROM ingest_chunks")?;
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
            project: "test".into(),
            source_id: "foo.md".into(),
            chunk_index: 0,
            content_sha256: "deadbeef".into(),
        };
        db.record_chunk(&row, Some("run-1")).unwrap();
        assert!(db.content_already_ingested("abc", "deadbeef").unwrap());
        assert!(!db.content_already_ingested("abc", "other").unwrap());
        assert_eq!(db.count_by_source().unwrap(), vec![("markdown".into(), 1)]);
    }
}
