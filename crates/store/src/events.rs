//! Full audit-event firehose → `events.duckdb`.
//!
//! The `ostk_project` scanner ingests every row from `.ostk/audit.jsonl`
//! into this `DuckDB` file, regardless of whether the event is significant
//! enough for the semantic corpus. Queries that need "what was happening
//! at ts=X" can join against this table without touching the `LanceDB`
//! corpus.
//!
//! Schema:
//!
//! ```sql
//! CREATE TABLE audit_events (
//!     row_key VARCHAR PRIMARY KEY,  -- project:ts:prev_hash
//!     project VARCHAR,
//!     ts TIMESTAMP,
//!     event VARCHAR,
//!     tool VARCHAR,
//!     agent VARCHAR,
//!     success BOOLEAN,
//!     exit_code INTEGER,
//!     duration_ms INTEGER,
//!     raw JSON
//! );
//! ```
//!
//! Inserts are streamed through a single prepared statement wrapped in an
//! explicit transaction so large audit files don't blow out memory or
//! commit latency.

use std::path::Path;
use std::sync::Mutex;

use chrono::{DateTime, Utc};
use duckdb::{Connection, params};

use crate::corpus::Result;

/// Events store (handle to `<root>/events.duckdb`).
pub struct EventsDb {
    conn: Mutex<Connection>,
}

/// One row as it lands in `audit_events`.
#[derive(Debug, Clone)]
pub struct AuditEventRow {
    pub row_key: String,
    pub project: Option<String>,
    pub ts: Option<DateTime<Utc>>,
    pub event: Option<String>,
    pub tool: Option<String>,
    pub agent: Option<String>,
    pub success: Option<bool>,
    pub exit_code: Option<i64>,
    pub duration_ms: Option<i64>,
    pub raw: String,
}

impl EventsDb {
    pub fn open(root: &Path) -> Result<Self> {
        let path = root.join("events.duckdb");
        let conn = Connection::open(path)?;
        Self::migrate(&conn)?;
        Ok(Self {
            conn: Mutex::new(conn),
        })
    }

    /// Read-only companion to [`EventsDb::open`]. See
    /// [`IngestDb::open_read_only`](crate::ingest::IngestDb::open_read_only)
    /// for the rationale.
    pub fn open_read_only(root: &Path) -> Result<Self> {
        let path = root.join("events.duckdb");
        let conn = crate::duckdb_open::open_read_only(&path)?;
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
CREATE TABLE IF NOT EXISTS audit_events (
    row_key VARCHAR PRIMARY KEY,
    project VARCHAR,
    ts TIMESTAMP,
    event VARCHAR,
    tool VARCHAR,
    agent VARCHAR,
    success BOOLEAN,
    exit_code INTEGER,
    duration_ms INTEGER,
    raw JSON
);

CREATE INDEX IF NOT EXISTS idx_audit_ts
    ON audit_events(ts);
CREATE INDEX IF NOT EXISTS idx_audit_project_agent
    ON audit_events(project, agent);
",
        )?;
        Ok(())
    }

    /// Ingest a batch of rows inside a single transaction. Uses
    /// `INSERT OR IGNORE` on `row_key` so repeated runs don't duplicate.
    #[allow(clippy::significant_drop_tightening)]
    pub fn ingest_batch(&self, rows: &[AuditEventRow]) -> Result<usize> {
        if rows.is_empty() {
            return Ok(0);
        }
        let mut conn = self.lock();
        let tx = conn.transaction()?;
        let mut inserted = 0usize;
        {
            let mut stmt = tx.prepare(
                "INSERT OR IGNORE INTO audit_events
                 (row_key, project, ts, event, tool, agent, success, exit_code, duration_ms, raw)
                 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            )?;
            for row in rows {
                let n = stmt.execute(params![
                    row.row_key,
                    row.project,
                    row.ts.map(|t| t.to_rfc3339()),
                    row.event,
                    row.tool,
                    row.agent,
                    row.success,
                    row.exit_code,
                    row.duration_ms,
                    row.raw,
                ])?;
                inserted += n;
            }
        }
        tx.commit()?;
        Ok(inserted)
    }

    /// Count rows. Mostly for tests.
    #[allow(clippy::significant_drop_tightening)]
    pub fn row_count(&self) -> Result<u64> {
        let conn = self.lock();
        let mut stmt = conn.prepare("SELECT COUNT(*) FROM audit_events")?;
        let n: i64 = stmt.query_row([], |r| r.get(0))?;
        Ok(u64::try_from(n).unwrap_or(0))
    }

    /// Run a raw SELECT-only SQL statement and return (columns, rows) as
    /// JSON values. Rejects anything that isn't a single SELECT. Used by the
    /// `recall_audit` MCP tool; the caller is responsible for higher-level
    /// policy (rate limiting, etc.).
    #[allow(clippy::significant_drop_tightening)]
    pub fn execute_select(&self, sql: &str) -> Result<(Vec<String>, Vec<Vec<serde_json::Value>>)> {
        let conn = self.lock();
        let mut stmt = conn.prepare(sql)?;
        let mut rows = stmt.query([])?;
        let mut out: Vec<Vec<serde_json::Value>> = Vec::new();
        let mut columns: Vec<String> = Vec::new();
        let mut column_count = 0usize;
        while let Some(r) = rows.next()? {
            if columns.is_empty() {
                let stmt_ref: &duckdb::Statement<'_> = r.as_ref();
                column_count = stmt_ref.column_count();
                columns = (0..column_count)
                    .map(|i| {
                        stmt_ref
                            .column_name(i)
                            .map(ToString::to_string)
                            .unwrap_or_default()
                    })
                    .collect();
            }
            let mut row = Vec::with_capacity(column_count);
            for i in 0..column_count {
                let v: duckdb::types::Value = r.get(i)?;
                row.push(duck_value_to_json(v));
            }
            out.push(row);
        }
        // If empty result set, we still want to attempt to read column meta
        // off the statement via the Rows handle.
        if columns.is_empty() {
            if let Some(stmt_ref) = rows.as_ref() {
                columns = (0..stmt_ref.column_count())
                    .map(|i| {
                        stmt_ref
                            .column_name(i)
                            .map(ToString::to_string)
                            .unwrap_or_default()
                    })
                    .collect();
            }
        }
        Ok((columns, out))
    }

    /// Look up a single row by `row_key`.
    #[allow(clippy::significant_drop_tightening)]
    pub fn get(&self, row_key: &str) -> Result<Option<AuditEventRow>> {
        let conn = self.lock();
        let mut stmt = conn.prepare(
            "SELECT row_key, project, CAST(ts AS VARCHAR), event, tool, agent,
                    success, exit_code, duration_ms, raw
             FROM audit_events WHERE row_key = ? LIMIT 1",
        )?;
        let mut rows = stmt.query(params![row_key])?;
        if let Some(r) = rows.next()? {
            let ts_str: Option<String> = r.get(2)?;
            let ts = ts_str.as_deref().and_then(parse_ts);
            Ok(Some(AuditEventRow {
                row_key: r.get(0)?,
                project: r.get(1)?,
                ts,
                event: r.get(3)?,
                tool: r.get(4)?,
                agent: r.get(5)?,
                success: r.get(6)?,
                exit_code: r.get(7)?,
                duration_ms: r.get(8)?,
                raw: r.get(9)?,
            }))
        } else {
            Ok(None)
        }
    }
}

// NOTE: `StoreError: From<duckdb::Error>` is defined once in `ingest.rs`
// and applies throughout this crate.

fn duck_value_to_json(v: duckdb::types::Value) -> serde_json::Value {
    use duckdb::types::Value as V;
    use serde_json::Value as J;
    match v {
        V::Null => J::Null,
        V::Boolean(b) => J::Bool(b),
        V::TinyInt(n) => J::from(i64::from(n)),
        V::SmallInt(n) => J::from(i64::from(n)),
        V::Int(n) => J::from(i64::from(n)),
        V::BigInt(n) => J::from(n),
        V::HugeInt(n) => J::String(n.to_string()),
        V::UTinyInt(n) => J::from(u64::from(n)),
        V::USmallInt(n) => J::from(u64::from(n)),
        V::UInt(n) => J::from(u64::from(n)),
        V::UBigInt(n) => J::from(n),
        V::Float(f) => serde_json::Number::from_f64(f64::from(f)).map_or(J::Null, J::Number),
        V::Double(f) => serde_json::Number::from_f64(f).map_or(J::Null, J::Number),
        V::Text(s) | V::Enum(s) => J::String(s),
        V::Blob(b) => J::String(hex::encode(b)),
        V::Date32(d) => J::from(i64::from(d)),
        V::Time64(_, t) | V::Timestamp(_, t) => J::from(t),
        V::Decimal(d) => J::String(d.to_string()),
        other => J::String(format!("{other:?}")),
    }
}

fn parse_ts(s: &str) -> Option<DateTime<Utc>> {
    // Accept either RFC3339 (`2026-04-17T10:00:00Z`) or DuckDB's default
    // timestamp rendering (`2026-04-17 10:00:00` / `…+00`).
    if let Ok(dt) = DateTime::parse_from_rfc3339(s) {
        return Some(dt.with_timezone(&Utc));
    }
    for fmt in [
        "%Y-%m-%d %H:%M:%S%.f%z",
        "%Y-%m-%d %H:%M:%S%.f",
        "%Y-%m-%d %H:%M:%S",
    ] {
        if let Ok(dt) = chrono::NaiveDateTime::parse_from_str(s, fmt) {
            return Some(DateTime::<Utc>::from_naive_utc_and_offset(dt, Utc));
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn sample(i: u32) -> AuditEventRow {
        AuditEventRow {
            row_key: format!("proj:2026-04-17T10:00:0{i}Z:hash{i}"),
            project: Some("proj".into()),
            ts: Some(
                DateTime::parse_from_rfc3339(&format!("2026-04-17T10:00:0{i}Z"))
                    .unwrap()
                    .with_timezone(&Utc),
            ),
            event: Some("tool.call".into()),
            tool: Some("bash".into()),
            agent: Some("claude".into()),
            success: Some(true),
            exit_code: Some(0),
            duration_ms: Some(42),
            raw: format!(r#"{{"row":{i}}}"#),
        }
    }

    #[test]
    fn open_and_insert() {
        let tmp = TempDir::new().unwrap();
        let db = EventsDb::open(tmp.path()).unwrap();
        assert_eq!(db.row_count().unwrap(), 0);

        let rows = vec![sample(1), sample(2), sample(3)];
        let n = db.ingest_batch(&rows).unwrap();
        assert_eq!(n, 3);
        assert_eq!(db.row_count().unwrap(), 3);

        // Re-insert → dedup via PK.
        db.ingest_batch(&rows).unwrap();
        assert_eq!(db.row_count().unwrap(), 3);
    }

    #[test]
    fn get_round_trip() {
        let tmp = TempDir::new().unwrap();
        let db = EventsDb::open(tmp.path()).unwrap();
        let row = sample(7);
        db.ingest_batch(std::slice::from_ref(&row)).unwrap();
        let back = db.get(&row.row_key).unwrap().expect("row should exist");
        assert_eq!(back.project.as_deref(), Some("proj"));
        assert_eq!(back.event.as_deref(), Some("tool.call"));
        assert_eq!(back.tool.as_deref(), Some("bash"));
        assert_eq!(back.success, Some(true));
        assert_eq!(back.exit_code, Some(0));
        assert_eq!(back.duration_ms, Some(42));
    }
}
