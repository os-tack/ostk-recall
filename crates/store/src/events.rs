use std::path::Path;
use std::sync::Mutex;

use chrono::{DateTime, Utc};
use rusqlite::{Connection, params};

use crate::corpus::Result;

/// Events store (handle to <root>/events.sqlite).
pub struct EventsDb {
    conn: Mutex<Connection>,
}

/// One row as it lands in audit_events.
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
        let path = root.join("events.sqlite");
        let conn = Connection::open(path)?;
        Self::setup_connection(&conn)?;
        Self::migrate(&conn)?;
        Ok(Self {
            conn: Mutex::new(conn),
        })
    }

    pub fn open_read_only(root: &Path) -> Result<Self> {
        let path = root.join("events.sqlite");
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
CREATE TABLE IF NOT EXISTS audit_events (
    row_key TEXT PRIMARY KEY,
    project TEXT,
    ts TEXT,
    event TEXT,
    tool TEXT,
    agent TEXT,
    success INTEGER,
    exit_code INTEGER,
    duration_ms INTEGER,
    raw TEXT
);

CREATE INDEX IF NOT EXISTS idx_audit_ts
    ON audit_events(ts);
CREATE INDEX IF NOT EXISTS idx_audit_project_agent
    ON audit_events(project, agent);
",
        )?;
        Ok(())
    }

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
                    row.success.map(|b| if b { 1i32 } else { 0i32 }),
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

    pub fn row_count(&self) -> Result<u64> {
        let conn = self.lock();
        let mut stmt = conn.prepare("SELECT COUNT(*) FROM audit_events")?;
        let n: i64 = stmt.query_row([], |r| r.get(0))?;
        Ok(n as u64)
    }

    pub fn execute_select(&self, sql: &str) -> Result<(Vec<String>, Vec<Vec<serde_json::Value>>)> {
        let conn = self.lock();
        let mut stmt = conn.prepare(sql)?;
        let column_count = stmt.column_count();
        let columns: Vec<String> = stmt.column_names().into_iter().map(ToString::to_string).collect();
        
        let mut rows = stmt.query([])?;
        let mut out = Vec::new();
        while let Some(r) = rows.next()? {
            let mut row = Vec::with_capacity(column_count);
            for i in 0..column_count {
                let v: rusqlite::types::Value = r.get(i)?;
                row.push(sql_value_to_json(v));
            }
            out.push(row);
        }
        Ok((columns, out))
    }

    pub fn get(&self, row_key: &str) -> Result<Option<AuditEventRow>> {
        let conn = self.lock();
        let mut stmt = conn.prepare(
            "SELECT row_key, project, ts, event, tool, agent,
                    success, exit_code, duration_ms, raw
             FROM audit_events WHERE row_key = ? LIMIT 1",
        )?;
        let res = stmt.query_row(params![row_key], |r| {
            let ts_str: Option<String> = r.get(2)?;
            let ts = ts_str.as_deref().and_then(parse_ts);
            let success_int: Option<i32> = r.get(6)?;
            let success = success_int.map(|n| n != 0);
            Ok(AuditEventRow {
                row_key: r.get(0)?,
                project: r.get(1)?,
                ts,
                event: r.get(3)?,
                tool: r.get(4)?,
                agent: r.get(5)?,
                success,
                exit_code: r.get(7)?,
                duration_ms: r.get(8)?,
                raw: r.get(9)?,
            })
        });
        match res {
            Ok(v) => Ok(Some(v)),
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
            Err(e) => Err(e.into()),
        }
    }
}

fn sql_value_to_json(v: rusqlite::types::Value) -> serde_json::Value {
    use rusqlite::types::Value as V;
    use serde_json::Value as J;
    match v {
        V::Null => J::Null,
        V::Integer(n) => J::from(n),
        V::Real(f) => serde_json::Number::from_f64(f).map_or(J::Null, J::Number),
        V::Text(s) => J::String(s),
        V::Blob(b) => J::String(hex::encode(b)),
    }
}

fn parse_ts(s: &str) -> Option<DateTime<Utc>> {
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

        // Re-insert -> dedup via PK.
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
