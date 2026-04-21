//! Shared `DuckDB` open helpers.
//!
//! The serve path opens both `ingest.duckdb` and `events.duckdb` in
//! read-only mode so multiple `ostk-recall serve --stdio` processes can
//! share a corpus without fighting over the `DuckDB` single-writer lock.
//! A short retry loop handles the transient case where a writer (scan /
//! ingest) holds the lock at startup.

use std::path::Path;
use std::thread;
use std::time::Duration;

use duckdb::{AccessMode, Config, Connection};

use crate::corpus::Result;

/// 6 attempts × 250 ms ≈ 1.5 s total wait before giving up.
const OPEN_MAX_ATTEMPTS: u32 = 6;
const OPEN_RETRY_DELAY: Duration = Duration::from_millis(250);

/// Open a `DuckDB` connection at `path` in read-only mode. Retries briefly
/// on lock-contention errors so serve processes that start during a write
/// don't hard-fail. Non-lock errors bubble up on the first attempt.
pub fn open_read_only(path: &Path) -> Result<Connection> {
    let mut attempts: u32 = 0;
    loop {
        attempts += 1;
        let config = Config::default().access_mode(AccessMode::ReadOnly)?;
        match Connection::open_with_flags(path, config) {
            Ok(conn) => return Ok(conn),
            Err(e) if attempts < OPEN_MAX_ATTEMPTS && is_lock_contention(&e) => {
                tracing::debug!(
                    attempt = attempts,
                    path = %path.display(),
                    "duckdb lock contention; retrying read-only open"
                );
                thread::sleep(OPEN_RETRY_DELAY);
            }
            Err(e) => return Err(e.into()),
        }
    }
}

fn is_lock_contention(e: &duckdb::Error) -> bool {
    // DuckDB surfaces this as an IO Error with a specific message; we
    // match on the stable prefix rather than a structured code because
    // the rust binding doesn't expose one.
    e.to_string().contains("Could not set lock on file")
}
