use std::path::Path;
use std::sync::Mutex;

use ostk_recall_core::Source;
use rusqlite::{Connection, params};
use serde::{Deserialize, Serialize};

use crate::corpus::Result;

/// Detect a v0.5 ingest.sqlite schema (legacy `project` PK column on
/// `ingest_sources`). Used by `IngestDb::migrate` to refuse to open old
/// corpora rather than silently mixing chunk_id formulas across schema
/// versions. The check is cheap: a single `pragma_table_info` row.
///
/// Returns false (clean) on a brand-new DB where neither table exists yet.
fn has_legacy_v05_schema(conn: &Connection) -> Result<bool> {
    let mut stmt =
        conn.prepare("SELECT 1 FROM pragma_table_info('ingest_sources') WHERE name = 'project'")?;
    let exists = stmt.exists([])?;
    Ok(exists)
}

/// Idempotency + provenance metadata. Lives at <root>/ingest.sqlite.
pub struct IngestDb {
    conn: Mutex<Connection>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IngestChunkRow {
    pub chunk_id: String,
    pub source: String,
    pub source_id: String,
    /// P0: physical-identity discriminator. Always non-empty.
    pub source_config_id: String,
    pub chunk_index: u32,
    pub content_sha256: String,
}

// Connection is the lock: every method must hold the guard across `prepare`
// + statement execution, so tightening `conn`'s scope is wrong here.
// `i64 -> u64` casts come from `SELECT COUNT(*)`, which is always non-negative.
#[allow(clippy::significant_drop_tightening, clippy::cast_sign_loss)]
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
             PRAGMA synchronous = NORMAL;",
        )?;
        Ok(())
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Connection> {
        self.conn
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    fn migrate(conn: &Connection) -> Result<()> {
        // P0 (v0.6): refuse to open v0.5 schemas. The old PK is
        // (source, project, source_id); v0.6 is (source, source_id,
        // source_config_id). Migrating in place would silently mix old
        // chunk_ids (computed without source_config_id) with new ones,
        // breaking dedupe and orphan-sweep invariants. Operators run
        // `ostk-recall reset --keep-threads` to clear the v0.5 corpus
        // before the first v0.6 scan; see rollout.md.
        if has_legacy_v05_schema(conn)? {
            return Err(rusqlite::Error::SqliteFailure(
                rusqlite::ffi::Error::new(rusqlite::ffi::SQLITE_ERROR),
                Some(
                    "ingest.sqlite is v0.5 schema (legacy `project` PK detected). \
                     The v0.6 chunk_id formula is incompatible — run \
                     `ostk-recall reset --keep-threads` to drop the v0.5 corpus and \
                     re-scan. Thread anchors are preserved by the reset; the v0.6.0 \
                     migration tool (P10) will repair them."
                        .to_string(),
                ),
            )
            .into());
        }

        conn.execute_batch(
            r"
CREATE TABLE IF NOT EXISTS ingest_sources (
    source            TEXT NOT NULL,
    source_id         TEXT NOT NULL,
    source_config_id  TEXT NOT NULL,
    mtime_micros      INTEGER NOT NULL,
    size_bytes        INTEGER NOT NULL,
    last_run_id       TEXT,
    PRIMARY KEY (source, source_id, source_config_id)
);

CREATE TABLE IF NOT EXISTS ingest_chunks (
    chunk_id          TEXT PRIMARY KEY,
    source            TEXT NOT NULL,
    source_id         TEXT NOT NULL,
    source_config_id  TEXT NOT NULL,
    chunk_index       INTEGER NOT NULL,
    content_sha256    TEXT NOT NULL,
    upserted_at       TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    last_run_id       TEXT
);

CREATE INDEX IF NOT EXISTS idx_ingest_chunks_source_cfg_id
    ON ingest_chunks(source, source_config_id, source_id);
",
        )?;

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
             (chunk_id, source, source_id, source_config_id, chunk_index, content_sha256, last_run_id)
             VALUES (?, ?, ?, ?, ?, ?, ?)",
            params![
                row.chunk_id,
                row.source,
                row.source_id,
                row.source_config_id,
                row.chunk_index,
                row.content_sha256,
                run_id
            ],
        )?;
        Ok(())
    }

    pub fn get_source_metadata(
        &self,
        source: &str,
        source_config_id: &str,
        source_id: &str,
    ) -> Result<Option<(i64, i64)>> {
        let conn = self.lock();
        let mut stmt = conn.prepare(
            "SELECT mtime_micros, size_bytes FROM ingest_sources WHERE source = ? AND source_config_id = ? AND source_id = ?"
        )?;
        let res = stmt.query_row(params![source, source_config_id, source_id], |r| {
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

    pub fn touch_source_chunks(
        &self,
        source: &str,
        source_config_id: &str,
        source_id: &str,
        run_id: &str,
    ) -> Result<()> {
        let conn = self.lock();
        conn.execute(
            "UPDATE ingest_sources SET last_run_id = ? WHERE source = ? AND source_config_id = ? AND source_id = ?",
            params![run_id, source, source_config_id, source_id],
        )?;
        conn.execute(
            "UPDATE ingest_chunks SET last_run_id = ? WHERE source = ? AND source_config_id = ? AND source_id = ?",
            params![run_id, source, source_config_id, source_id],
        )?;
        Ok(())
    }

    pub fn update_source_metadata(
        &self,
        source: &str,
        source_config_id: &str,
        source_id: &str,
        mtime: i64,
        size: i64,
        run_id: &str,
    ) -> Result<()> {
        let conn = self.lock();
        conn.execute(
            "INSERT OR REPLACE INTO ingest_sources (source, source_id, source_config_id, mtime_micros, size_bytes, last_run_id)
             VALUES (?, ?, ?, ?, ?, ?)",
            params![source, source_id, source_config_id, mtime, size, run_id],
        )?;
        Ok(())
    }

    pub fn delete_orphans(
        &self,
        source: &str,
        source_config_id: &str,
        current_run_id: &str,
    ) -> Result<Vec<String>> {
        let conn = self.lock();
        let mut ids = Vec::new();
        {
            let mut stmt = conn.prepare(
                "SELECT chunk_id FROM ingest_chunks WHERE source = ? AND source_config_id = ? AND (last_run_id != ? OR last_run_id IS NULL)"
            )?;
            let mut rows = stmt.query(params![source, source_config_id, current_run_id])?;
            while let Some(row) = rows.next()? {
                let id: String = row.get(0)?;
                ids.push(id);
            }
        }

        conn.execute(
            "DELETE FROM ingest_chunks WHERE source = ? AND source_config_id = ? AND (last_run_id != ? OR last_run_id IS NULL)",
            params![source, source_config_id, current_run_id]
        )?;

        conn.execute(
            "DELETE FROM ingest_sources WHERE source = ? AND source_config_id = ? AND (last_run_id != ? OR last_run_id IS NULL)",
            params![source, source_config_id, current_run_id]
        )?;

        Ok(ids)
    }

    pub fn mark_orphans_stale(
        &self,
        source: &str,
        source_config_id: &str,
        current_run_id: &str,
    ) -> Result<Vec<String>> {
        let conn = self.lock();
        let mut ids = Vec::new();
        let mut stmt = conn.prepare(
            "SELECT chunk_id FROM ingest_chunks WHERE source = ? AND source_config_id = ? AND (last_run_id != ? OR last_run_id IS NULL)"
        )?;
        let mut rows = stmt.query(params![source, source_config_id, current_run_id])?;
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

    /// Tombstone every chunk + source ledger row keyed by
    /// `(source, project, source_id)`. Returns the `chunk_ids` that were
    /// removed so the caller can issue the corresponding corpus delete or
    /// stale-mark.
    ///
    /// Used by [`Pipeline::scan_paths`] (gh#7) for delete-event handling:
    /// when a watcher trigger names a path that the per-source
    /// `discover_paths` no longer yields (file deleted, gitignored, or
    /// extension-filtered out), the chunk family for that path is purged
    /// from the ledger and the returned ids drive the corpus-side delete.
    /// No-op (returns empty Vec) when nothing was registered for the
    /// triple — matches the "path that was never ingested" semantics.
    pub fn tombstone_chunks_by_path(
        &self,
        source: &str,
        source_config_id: &str,
        source_id: &str,
    ) -> Result<Vec<String>> {
        let conn = self.lock();
        let mut ids = Vec::new();
        {
            let mut stmt = conn.prepare(
                "SELECT chunk_id FROM ingest_chunks WHERE source = ? AND source_config_id = ? AND source_id = ?"
            )?;
            let mut rows = stmt.query(params![source, source_config_id, source_id])?;
            while let Some(row) = rows.next()? {
                let id: String = row.get(0)?;
                ids.push(id);
            }
        }

        conn.execute(
            "DELETE FROM ingest_chunks WHERE source = ? AND source_config_id = ? AND source_id = ?",
            params![source, source_config_id, source_id],
        )?;

        conn.execute(
            "DELETE FROM ingest_sources WHERE source = ? AND source_config_id = ? AND source_id = ?",
            params![source, source_config_id, source_id],
        )?;

        Ok(ids)
    }

    pub fn delete_by_chunk_ids(&self, chunk_ids: &[String]) -> Result<u64> {
        if chunk_ids.is_empty() {
            return Ok(0);
        }
        let conn = self.lock();
        let mut total: u64 = 0;
        for batch in chunk_ids.chunks(999) {
            let placeholders = std::iter::repeat_n("?", batch.len())
                .collect::<Vec<_>>()
                .join(",");
            let sql = format!("DELETE FROM ingest_chunks WHERE chunk_id IN ({placeholders})");
            let mut stmt = conn.prepare(&sql)?;
            let n = stmt.execute(rusqlite::params_from_iter(batch))?;
            total += n as u64;
        }
        Ok(total)
    }

    pub fn latest_upserted_at(&self) -> Result<Option<String>> {
        let conn = self.lock();
        let mut stmt = conn.prepare("SELECT MAX(upserted_at) FROM ingest_chunks")?;
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
            source_config_id: "cfg-test".into(),
            chunk_index: 0,
            content_sha256: "deadbeef".into(),
        };
        db.record_chunk(&row, Some("run-1")).unwrap();
        assert!(db.content_already_ingested("abc", "deadbeef").unwrap());
        assert!(!db.content_already_ingested("abc", "other").unwrap());
        assert_eq!(db.count_by_source().unwrap(), vec![("markdown".into(), 1)]);
    }

    #[test]
    fn tombstone_chunks_by_path_returns_ids_and_clears_rows() {
        let tmp = TempDir::new().unwrap();
        let db = IngestDb::open(tmp.path()).unwrap();
        let cfg = "cfg-p1";

        for i in 0..3 {
            db.record_chunk(
                &IngestChunkRow {
                    chunk_id: format!("doomed-{i}"),
                    source: "markdown".into(),
                    source_id: "doomed.md".into(),
                    source_config_id: cfg.into(),
                    chunk_index: i,
                    content_sha256: format!("sha-{i}"),
                },
                Some("run-1"),
            )
            .unwrap();
        }
        // Sibling chunk under a different source_id must NOT be touched.
        db.record_chunk(
            &IngestChunkRow {
                chunk_id: "keep".into(),
                source: "markdown".into(),
                source_id: "keep.md".into(),
                source_config_id: cfg.into(),
                chunk_index: 0,
                content_sha256: "sha-keep".into(),
            },
            Some("run-1"),
        )
        .unwrap();
        db.update_source_metadata("markdown", cfg, "doomed.md", 0, 0, "run-1")
            .unwrap();
        db.update_source_metadata("markdown", cfg, "keep.md", 0, 0, "run-1")
            .unwrap();

        let purged = db
            .tombstone_chunks_by_path("markdown", cfg, "doomed.md")
            .unwrap();
        assert_eq!(purged.len(), 3);
        assert!(purged.iter().all(|id| id.starts_with("doomed-")));

        // Re-tombstone is a no-op (the row is already gone).
        let again = db
            .tombstone_chunks_by_path("markdown", cfg, "doomed.md")
            .unwrap();
        assert!(again.is_empty(), "second tombstone returns nothing");

        // Sibling row survived.
        assert_eq!(
            db.count_by_source().unwrap(),
            vec![("markdown".into(), 1)],
            "only the keep.md chunk remains"
        );
        // Source metadata for doomed.md is also gone.
        assert!(
            db.get_source_metadata("markdown", cfg, "doomed.md")
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn tombstone_chunks_by_path_for_unknown_path_is_noop() {
        let tmp = TempDir::new().unwrap();
        let db = IngestDb::open(tmp.path()).unwrap();
        let purged = db
            .tombstone_chunks_by_path("markdown", "cfg-p1", "never-existed.md")
            .unwrap();
        assert!(purged.is_empty());
    }

    #[test]
    fn legacy_v05_schema_refused_at_open() {
        let tmp = TempDir::new().unwrap();
        // Manually create a v0.5-shaped ingest.sqlite to simulate an
        // existing corpus.
        {
            let conn = Connection::open(tmp.path().join("ingest.sqlite")).unwrap();
            conn.execute_batch(
                "CREATE TABLE ingest_sources (
                    source TEXT NOT NULL,
                    project TEXT NOT NULL DEFAULT '',
                    source_id TEXT NOT NULL,
                    mtime_micros INTEGER NOT NULL,
                    size_bytes INTEGER NOT NULL,
                    last_run_id TEXT,
                    PRIMARY KEY (source, project, source_id)
                );",
            )
            .unwrap();
        }
        let err = match IngestDb::open(tmp.path()) {
            Ok(_) => panic!("expected v0.5 schema to be rejected"),
            Err(e) => e,
        };
        let msg = err.to_string();
        assert!(
            msg.contains("v0.5 schema") && msg.contains("reset --keep-threads"),
            "expected reset hint, got: {msg}"
        );
    }
}
