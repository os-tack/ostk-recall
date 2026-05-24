//! Threads + evidence-links ledger (Phase 3 of the attention substrate).
//!
//! Lives at `<root>/threads.sqlite`. Holds the durable identity of every
//! thread (`threads` table) and every evidence link a thread carries
//! (`evidence_links` table). Evidence rows are keyed by
//! `(thread_handle, original_path, category)` — the `original_path`
//! never mutates, so links survive renames (the watcher updates
//! `current_path` / `relation_state` instead, per Correction 5 of the
//! implementation plan).
//!
//! Mutations emit [`ChainEvent`]s through a [`ChainSink`] held on the
//! store. Phase 3 ships [`NoopChainSink`] as the default; the real
//! signing / journal sink lands later — see `chain-as-cognition-history`.

// `Mutex<Connection>` is the canonical pattern across this crate
// (events.rs, ingest.rs). The MutexGuard must live across prepare +
// execute because rusqlite `Statement`s borrow from `Connection`.
// `single_match_else` fires on the `match Option<TensionState>` arms
// in `list_threads`, where the two-branch shape is clearer than the
// equivalent `if let`.
#![allow(clippy::significant_drop_tightening, clippy::single_match_else)]

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use chrono::{DateTime, Utc};
use ostk_recall_core::{FoldDepth, PrivacyTier, ThreadHandle};
use rusqlite::{Connection, OptionalExtension, params};

use crate::corpus::{Result, StoreError};

/// Tension axis for a thread row.
///
/// Wire form is `snake_case` to match the `tension` column.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TensionState {
    Active,
    Slack,
    Dormant,
}

impl TensionState {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Active => "active",
            Self::Slack => "slack",
            Self::Dormant => "dormant",
        }
    }

    pub fn parse(s: &str) -> Result<Self> {
        match s {
            "active" => Ok(Self::Active),
            "slack" => Ok(Self::Slack),
            "dormant" => Ok(Self::Dormant),
            other => Err(bad_enum("tension", other)),
        }
    }
}

/// Lifecycle of an evidence link relative to its `original_path`.
///
/// Once `original_path` is committed the row never disappears under us;
/// `relation_state` records what happened to the underlying content.
/// Replay reads the chain in order to recover this history.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RelationState {
    Active,
    Moved,
    BrokenReference,
    Superseded,
}

impl RelationState {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Active => "active",
            Self::Moved => "moved",
            Self::BrokenReference => "broken_reference",
            Self::Superseded => "superseded",
        }
    }

    pub fn parse(s: &str) -> Result<Self> {
        match s {
            "active" => Ok(Self::Active),
            "moved" => Ok(Self::Moved),
            "broken_reference" => Ok(Self::BrokenReference),
            "superseded" => Ok(Self::Superseded),
            other => Err(bad_enum("relation_state", other)),
        }
    }
}

/// Who created the evidence link.
///
/// `Curated` rows came from an operator (CLI / MCP / VFS write);
/// `Derived` rows came from the auto-weaver.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AssociationType {
    Curated,
    Derived,
}

impl AssociationType {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Curated => "curated",
            Self::Derived => "derived",
        }
    }

    pub fn parse(s: &str) -> Result<Self> {
        match s {
            "curated" => Ok(Self::Curated),
            "derived" => Ok(Self::Derived),
            other => Err(bad_enum("association_type", other)),
        }
    }
}

fn bad_enum(field: &str, value: &str) -> StoreError {
    StoreError::InvalidEnumValue {
        field: field.to_string(),
        value: value.to_string(),
    }
}

/// A row in the `threads` table.
#[derive(Debug, Clone)]
pub struct ThreadRecord {
    pub handle: ThreadHandle,
    pub tension: TensionState,
    pub familiarity: u32,
    pub last_touched_at: DateTime<Utc>,
    pub anchor_chunk_id: Option<String>,
    pub fold_override: Option<FoldDepth>,
    pub created_at: DateTime<Utc>,
    pub created_scope_key: Option<String>,
    pub privacy_tier: PrivacyTier,
}

/// A row in the `threads_proposed` table.
///
/// The substrate writes proposals when it notices a chunk cluster
/// without an existing thread anchor. Operators inspect proposals via
/// `thread proposed-list` and promote one into an active thread by
/// running `thread create` against the cluster's anchor chunk.
#[derive(Debug, Clone)]
pub struct ProposedThreadRecord {
    pub id: i64,
    pub proposed_handle: String,
    pub chunk_ids: Vec<String>,
    pub centroid_vec: Vec<f32>,
    pub cohesion: f32,
    pub created_at: DateTime<Utc>,
    pub promoted_to: Option<String>,
}

/// A row in the `evidence_links` table.
#[derive(Debug, Clone)]
pub struct EvidenceLink {
    pub id: i64,
    pub thread_handle: ThreadHandle,
    pub original_path: PathBuf,
    pub current_path: Option<PathBuf>,
    pub content_hash: Option<String>,
    pub last_resolved_chunk_id: Option<String>,
    pub relation_state: RelationState,
    pub association_type: AssociationType,
    pub category: String,
    pub similarity: Option<f32>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

/// Substrate-changing events emitted by ledger mutations.
///
/// The set is the one defined in `chain-as-cognition-history`: anything
/// that changes future surfacer behaviour, never operational reads.
#[derive(Debug, Clone)]
pub enum ChainEvent {
    ThreadCreate {
        handle: ThreadHandle,
        ts: DateTime<Utc>,
    },
    ThreadRename {
        old: ThreadHandle,
        new: ThreadHandle,
        ts: DateTime<Utc>,
    },
    ThreadDelete {
        handle: ThreadHandle,
        ts: DateTime<Utc>,
    },
    EvidenceAdd {
        thread: ThreadHandle,
        path: PathBuf,
        association: AssociationType,
        ts: DateTime<Utc>,
    },
    EvidenceRemove {
        thread: ThreadHandle,
        evidence_id: i64,
        ts: DateTime<Utc>,
    },
    EvidenceStateChange {
        evidence_id: i64,
        from: RelationState,
        to: RelationState,
        ts: DateTime<Utc>,
    },
    FamiliarityBatch {
        entries: Vec<(ThreadHandle, u32)>,
        turn_seq: u64,
        ts: DateTime<Utc>,
    },
    TensionTransition {
        handle: ThreadHandle,
        from: TensionState,
        to: TensionState,
        ts: DateTime<Utc>,
    },
}

/// Sink for substrate chain rows.
///
/// The real implementation signs and journals; Phase 3 only needs the
/// interface so mutation methods are chain-ready.
pub trait ChainSink: Send + Sync {
    fn append(&self, event: &ChainEvent) -> Result<()>;
}

/// Stub sink used when nothing wants the events yet.
#[derive(Debug, Default, Clone, Copy)]
pub struct NoopChainSink;

impl ChainSink for NoopChainSink {
    fn append(&self, _event: &ChainEvent) -> Result<()> {
        Ok(())
    }
}

/// Threads-ledger handle (one per `<root>/threads.sqlite`).
pub struct ThreadsDb {
    conn: Mutex<Connection>,
    sink: Arc<dyn ChainSink>,
}

impl ThreadsDb {
    /// Open (creating if absent) and run migrations.
    pub fn open(root: &Path) -> Result<Self> {
        Self::open_with_sink(root, Arc::new(NoopChainSink))
    }

    /// Open with a caller-supplied [`ChainSink`].
    pub fn open_with_sink(root: &Path, sink: Arc<dyn ChainSink>) -> Result<Self> {
        let path = root.join("threads.sqlite");
        let conn = Connection::open(path)?;
        Self::setup_connection(&conn)?;
        Self::migrate(&conn)?;
        Ok(Self {
            conn: Mutex::new(conn),
            sink,
        })
    }

    pub fn open_read_only(root: &Path) -> Result<Self> {
        let path = root.join("threads.sqlite");
        let conn = Connection::open_with_flags(
            path,
            rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY | rusqlite::OpenFlags::SQLITE_OPEN_NO_MUTEX,
        )?;
        Self::setup_connection(&conn)?;
        Ok(Self {
            conn: Mutex::new(conn),
            sink: Arc::new(NoopChainSink),
        })
    }

    /// Swap the chain sink in place. Returns the previous sink.
    pub fn set_chain_sink(&mut self, sink: Arc<dyn ChainSink>) -> Arc<dyn ChainSink> {
        std::mem::replace(&mut self.sink, sink)
    }

    fn setup_connection(conn: &Connection) -> Result<()> {
        conn.execute_batch(
            "PRAGMA journal_mode = WAL;
             PRAGMA busy_timeout = 5000;
             PRAGMA synchronous = NORMAL;
             PRAGMA foreign_keys = ON;",
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
CREATE TABLE IF NOT EXISTS threads (
    handle TEXT PRIMARY KEY,
    tension TEXT NOT NULL DEFAULT 'active',
    familiarity INTEGER NOT NULL DEFAULT 0,
    last_touched_at TEXT NOT NULL,
    anchor_chunk_id TEXT,
    fold_override TEXT,
    created_at TEXT NOT NULL,
    created_scope_key TEXT,
    privacy_tier TEXT NOT NULL DEFAULT 't1_project'
);

CREATE INDEX IF NOT EXISTS idx_threads_tension ON threads(tension);
CREATE INDEX IF NOT EXISTS idx_threads_familiarity ON threads(familiarity);

CREATE TABLE IF NOT EXISTS evidence_links (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    thread_handle TEXT NOT NULL REFERENCES threads(handle) ON DELETE CASCADE,
    original_path TEXT NOT NULL,
    current_path TEXT,
    content_hash TEXT,
    last_resolved_chunk_id TEXT,
    relation_state TEXT NOT NULL DEFAULT 'active',
    association_type TEXT NOT NULL,
    category TEXT NOT NULL,
    similarity REAL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE(thread_handle, original_path, category)
);

CREATE INDEX IF NOT EXISTS idx_evidence_thread ON evidence_links(thread_handle);
CREATE INDEX IF NOT EXISTS idx_evidence_state ON evidence_links(relation_state);
CREATE INDEX IF NOT EXISTS idx_evidence_current_path ON evidence_links(current_path);

CREATE TABLE IF NOT EXISTS threads_proposed (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    proposed_handle TEXT NOT NULL UNIQUE,
    chunk_ids TEXT NOT NULL,
    centroid_vec BLOB NOT NULL,
    cohesion REAL NOT NULL,
    created_at TEXT NOT NULL,
    promoted_to TEXT
);

CREATE INDEX IF NOT EXISTS idx_threads_proposed_promoted ON threads_proposed(promoted_to);
CREATE INDEX IF NOT EXISTS idx_threads_proposed_created ON threads_proposed(created_at);
",
        )?;
        Ok(())
    }

    // ---------- threads ----------

    /// Insert-or-replace a thread row.
    ///
    /// Emits `ChainEvent::ThreadCreate` only when the row is newly
    /// created; replacement edits do not re-chain a creation.
    pub fn upsert_thread(&self, record: &ThreadRecord) -> Result<()> {
        let is_new = {
            let conn = self.lock();
            let exists = conn
                .prepare("SELECT 1 FROM threads WHERE handle = ? LIMIT 1")?
                .exists(params![record.handle.as_str()])?;
            conn.execute(
                "INSERT OR REPLACE INTO threads
                 (handle, tension, familiarity, last_touched_at, anchor_chunk_id,
                  fold_override, created_at, created_scope_key, privacy_tier)
                 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                params![
                    record.handle.as_str(),
                    record.tension.as_str(),
                    record.familiarity,
                    record.last_touched_at.to_rfc3339(),
                    record.anchor_chunk_id,
                    record.fold_override.map(fold_to_str),
                    record.created_at.to_rfc3339(),
                    record.created_scope_key,
                    privacy_to_str(record.privacy_tier),
                ],
            )?;
            !exists
        };
        if is_new {
            self.sink.append(&ChainEvent::ThreadCreate {
                handle: record.handle.clone(),
                ts: record.created_at,
            })?;
        }
        Ok(())
    }

    pub fn get_thread(&self, handle: &ThreadHandle) -> Result<Option<ThreadRecord>> {
        let conn = self.lock();
        let mut stmt = conn.prepare(
            "SELECT handle, tension, familiarity, last_touched_at, anchor_chunk_id,
                    fold_override, created_at, created_scope_key, privacy_tier
             FROM threads WHERE handle = ? LIMIT 1",
        )?;
        let res = stmt.query_row(params![handle.as_str()], row_to_thread);
        match res {
            Ok(v) => Ok(Some(v)),
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
            Err(e) => Err(e.into()),
        }
    }

    /// All threads, ordered by `last_touched_at DESC`. Optional tension filter.
    pub fn list_threads(&self, tension: Option<TensionState>) -> Result<Vec<ThreadRecord>> {
        let conn = self.lock();
        let rows = match tension {
            Some(t) => {
                let mut stmt = conn.prepare(
                    "SELECT handle, tension, familiarity, last_touched_at, anchor_chunk_id,
                            fold_override, created_at, created_scope_key, privacy_tier
                     FROM threads WHERE tension = ? ORDER BY last_touched_at DESC",
                )?;
                stmt.query_map(params![t.as_str()], row_to_thread)?
                    .collect::<std::result::Result<Vec<_>, rusqlite::Error>>()?
            }
            None => {
                let mut stmt = conn.prepare(
                    "SELECT handle, tension, familiarity, last_touched_at, anchor_chunk_id,
                            fold_override, created_at, created_scope_key, privacy_tier
                     FROM threads ORDER BY last_touched_at DESC",
                )?;
                stmt.query_map([], row_to_thread)?
                    .collect::<std::result::Result<Vec<_>, rusqlite::Error>>()?
            }
        };
        Ok(rows)
    }

    pub fn delete_thread(&self, handle: &ThreadHandle) -> Result<()> {
        let n = {
            let conn = self.lock();
            conn.execute(
                "DELETE FROM threads WHERE handle = ?",
                params![handle.as_str()],
            )?
        };
        if n > 0 {
            self.sink.append(&ChainEvent::ThreadDelete {
                handle: handle.clone(),
                ts: Utc::now(),
            })?;
        }
        Ok(())
    }

    /// Increment familiarity and return the post-increment value.
    ///
    /// Does NOT chain — the turn observer batches familiarity ticks
    /// through [`ChainEvent::FamiliarityBatch`] (see chain-policy doc).
    pub fn increment_familiarity(&self, handle: &ThreadHandle) -> Result<u32> {
        let conn = self.lock();
        let n = conn.execute(
            "UPDATE threads SET familiarity = familiarity + 1 WHERE handle = ?",
            params![handle.as_str()],
        )?;
        if n == 0 {
            return Err(missing_thread(handle));
        }
        let new_value: i64 = conn.query_row(
            "SELECT familiarity FROM threads WHERE handle = ?",
            params![handle.as_str()],
            |r| r.get(0),
        )?;
        u32::try_from(new_value).map_err(|_| {
            StoreError::Lance(lancedb::Error::Other {
                message: format!("familiarity overflow for thread {handle}: {new_value}"),
                source: None,
            })
        })
    }

    /// Emit a batched familiarity chain row covering the per-turn
    /// increments. Callers stage the totals `(handle, post-batch value)`
    /// across the turn and flush once at turn-end.
    pub fn record_familiarity_batch(
        &self,
        entries: Vec<(ThreadHandle, u32)>,
        turn_seq: u64,
    ) -> Result<()> {
        if entries.is_empty() {
            return Ok(());
        }
        self.sink.append(&ChainEvent::FamiliarityBatch {
            entries,
            turn_seq,
            ts: Utc::now(),
        })
    }

    pub fn touch_thread(&self, handle: &ThreadHandle, now: DateTime<Utc>) -> Result<()> {
        let conn = self.lock();
        let n = conn.execute(
            "UPDATE threads SET last_touched_at = ? WHERE handle = ?",
            params![now.to_rfc3339(), handle.as_str()],
        )?;
        if n == 0 {
            return Err(missing_thread(handle));
        }
        Ok(())
    }

    pub fn set_tension(&self, handle: &ThreadHandle, tension: TensionState) -> Result<()> {
        let from = {
            let conn = self.lock();
            let prev: Option<String> = conn
                .query_row(
                    "SELECT tension FROM threads WHERE handle = ?",
                    params![handle.as_str()],
                    |r| r.get(0),
                )
                .optional()?;
            let Some(prev) = prev else {
                return Err(missing_thread(handle));
            };
            let from = TensionState::parse(&prev)?;
            if from == tension {
                return Ok(());
            }
            conn.execute(
                "UPDATE threads SET tension = ? WHERE handle = ?",
                params![tension.as_str(), handle.as_str()],
            )?;
            from
        };
        self.sink.append(&ChainEvent::TensionTransition {
            handle: handle.clone(),
            from,
            to: tension,
            ts: Utc::now(),
        })?;
        Ok(())
    }

    /// Rename a thread.
    ///
    /// Chains a [`ChainEvent::ThreadRename`] event; preserves evidence
    /// rows (FK is on `thread_handle`, repointed in the same
    /// transaction with `PRAGMA defer_foreign_keys = ON` so `SQLite`
    /// re-checks only at `COMMIT`).
    pub fn rename_thread(&self, old: &ThreadHandle, new: &ThreadHandle) -> Result<()> {
        if old == new {
            return Ok(());
        }
        {
            let mut conn = self.lock();
            let tx = conn.transaction()?;
            tx.execute_batch("PRAGMA defer_foreign_keys = ON")?;
            let n = tx.execute(
                "UPDATE threads SET handle = ? WHERE handle = ?",
                params![new.as_str(), old.as_str()],
            )?;
            if n == 0 {
                return Err(missing_thread(old));
            }
            tx.execute(
                "UPDATE evidence_links SET thread_handle = ? WHERE thread_handle = ?",
                params![new.as_str(), old.as_str()],
            )?;
            tx.commit()?;
        }
        self.sink.append(&ChainEvent::ThreadRename {
            old: old.clone(),
            new: new.clone(),
            ts: Utc::now(),
        })?;
        Ok(())
    }

    // ---------- evidence ----------

    /// Insert an evidence row and return the new `id`.
    ///
    /// Honors the `(thread_handle, original_path, category)`
    /// uniqueness — duplicate inserts surface as a [`StoreError`].
    pub fn add_evidence_link(&self, link: &EvidenceLink) -> Result<i64> {
        let id = {
            let conn = self.lock();
            conn.execute(
                "INSERT INTO evidence_links
                 (thread_handle, original_path, current_path, content_hash,
                  last_resolved_chunk_id, relation_state, association_type,
                  category, similarity, created_at, updated_at)
                 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                params![
                    link.thread_handle.as_str(),
                    path_to_str(&link.original_path),
                    link.current_path.as_deref().map(path_to_str),
                    link.content_hash,
                    link.last_resolved_chunk_id,
                    link.relation_state.as_str(),
                    link.association_type.as_str(),
                    link.category,
                    link.similarity,
                    link.created_at.to_rfc3339(),
                    link.updated_at.to_rfc3339(),
                ],
            )?;
            conn.last_insert_rowid()
        };
        self.sink.append(&ChainEvent::EvidenceAdd {
            thread: link.thread_handle.clone(),
            path: link.original_path.clone(),
            association: link.association_type,
            ts: link.created_at,
        })?;
        Ok(id)
    }

    /// All evidence rows for a thread, ordered by `created_at ASC`.
    pub fn list_evidence(&self, handle: &ThreadHandle) -> Result<Vec<EvidenceLink>> {
        let conn = self.lock();
        let mut stmt = conn.prepare(
            "SELECT id, thread_handle, original_path, current_path, content_hash,
                    last_resolved_chunk_id, relation_state, association_type,
                    category, similarity, created_at, updated_at
             FROM evidence_links
             WHERE thread_handle = ?
             ORDER BY created_at ASC, id ASC",
        )?;
        let rows = stmt
            .query_map(params![handle.as_str()], row_to_evidence)?
            .collect::<std::result::Result<Vec<_>, rusqlite::Error>>()?;
        Ok(rows)
    }

    /// Transition an evidence row to `broken_reference` (chain-emitting).
    pub fn mark_evidence_broken(&self, id: i64) -> Result<()> {
        self.transition_evidence(id, RelationState::BrokenReference, None)
    }

    /// Transition an evidence row to `moved`, updating `current_path`.
    pub fn mark_evidence_moved(&self, id: i64, new_path: &Path) -> Result<()> {
        self.transition_evidence(id, RelationState::Moved, Some(new_path))
    }

    fn transition_evidence(
        &self,
        id: i64,
        to: RelationState,
        new_path: Option<&Path>,
    ) -> Result<()> {
        let now = Utc::now();
        let from = {
            let conn = self.lock();
            let (handle_s, from_s): (String, String) = conn
                .query_row(
                    "SELECT thread_handle, relation_state FROM evidence_links WHERE id = ?",
                    params![id],
                    |r| Ok((r.get(0)?, r.get(1)?)),
                )
                .map_err(|e| match e {
                    rusqlite::Error::QueryReturnedNoRows => missing_evidence(id),
                    other => other.into(),
                })?;
            // Round-trip the stored handle so a corrupted DB surfaces
            // cleanly rather than getting silently typed away later.
            ThreadHandle::new(handle_s).map_err(|e| {
                StoreError::Lance(lancedb::Error::Other {
                    message: format!("threads ledger: stored handle invalid: {e}"),
                    source: None,
                })
            })?;
            let from = RelationState::parse(&from_s)?;
            if let Some(p) = new_path {
                conn.execute(
                    "UPDATE evidence_links
                     SET relation_state = ?, current_path = ?, updated_at = ?
                     WHERE id = ?",
                    params![to.as_str(), path_to_str(p), now.to_rfc3339(), id],
                )?;
            } else {
                conn.execute(
                    "UPDATE evidence_links
                     SET relation_state = ?, updated_at = ?
                     WHERE id = ?",
                    params![to.as_str(), now.to_rfc3339(), id],
                )?;
            }
            from
        };
        if from != to {
            self.sink.append(&ChainEvent::EvidenceStateChange {
                evidence_id: id,
                from,
                to,
                ts: now,
            })?;
        }
        Ok(())
    }

    /// Drop an evidence row entirely.
    ///
    /// Chains [`ChainEvent::EvidenceRemove`] with the originating
    /// thread handle.
    pub fn remove_evidence(&self, id: i64) -> Result<()> {
        let handle = {
            let conn = self.lock();
            let handle_s: Option<String> = conn
                .query_row(
                    "SELECT thread_handle FROM evidence_links WHERE id = ?",
                    params![id],
                    |r| r.get(0),
                )
                .optional()?;
            let Some(handle_s) = handle_s else {
                return Err(missing_evidence(id));
            };
            let handle = ThreadHandle::new(handle_s).map_err(|e| {
                StoreError::Lance(lancedb::Error::Other {
                    message: format!("threads ledger: stored handle invalid: {e}"),
                    source: None,
                })
            })?;
            conn.execute("DELETE FROM evidence_links WHERE id = ?", params![id])?;
            handle
        };
        self.sink.append(&ChainEvent::EvidenceRemove {
            thread: handle,
            evidence_id: id,
            ts: Utc::now(),
        })?;
        Ok(())
    }

    pub fn evidence_count(&self) -> Result<u64> {
        let conn = self.lock();
        let n: i64 = conn.query_row("SELECT COUNT(*) FROM evidence_links", [], |r| r.get(0))?;
        u64::try_from(n).map_err(|_| {
            StoreError::Lance(lancedb::Error::Other {
                message: format!("evidence_count returned negative value: {n}"),
                source: None,
            })
        })
    }

    // ---------- proposed threads ----------

    /// Insert a proposed-thread row. Caller owns the `proposed_handle`
    /// (must be unique). Returns the assigned `id`. Does not chain —
    /// proposals are pre-substrate until an operator promotes one via
    /// `thread create`.
    pub fn insert_proposed_thread(&self, rec: &ProposedThreadRecord) -> Result<i64> {
        let chunk_ids_json = serde_json::to_string(&rec.chunk_ids).map_err(|e| {
            StoreError::Lance(lancedb::Error::Other {
                message: format!("threads ledger: serialize chunk_ids: {e}"),
                source: None,
            })
        })?;
        let centroid_bytes = f32_vec_to_bytes(&rec.centroid_vec);
        let conn = self.lock();
        conn.execute(
            "INSERT INTO threads_proposed
             (proposed_handle, chunk_ids, centroid_vec, cohesion, created_at, promoted_to)
             VALUES (?, ?, ?, ?, ?, ?)",
            params![
                rec.proposed_handle,
                chunk_ids_json,
                centroid_bytes,
                f64::from(rec.cohesion),
                rec.created_at.to_rfc3339(),
                rec.promoted_to.as_deref(),
            ],
        )?;
        Ok(conn.last_insert_rowid())
    }

    /// All proposed threads, newest first.
    pub fn list_proposed_threads(&self) -> Result<Vec<ProposedThreadRecord>> {
        let conn = self.lock();
        let mut stmt = conn.prepare(
            "SELECT id, proposed_handle, chunk_ids, centroid_vec, cohesion,
                    created_at, promoted_to
             FROM threads_proposed
             ORDER BY created_at DESC, id DESC",
        )?;
        let rows = stmt
            .query_map([], row_to_proposed_thread)?
            .collect::<std::result::Result<Vec<_>, rusqlite::Error>>()?;
        Ok(rows)
    }

    pub fn proposed_thread_count(&self) -> Result<u64> {
        let conn = self.lock();
        let n: i64 =
            conn.query_row("SELECT COUNT(*) FROM threads_proposed", [], |r| r.get(0))?;
        u64::try_from(n).map_err(|_| {
            StoreError::Lance(lancedb::Error::Other {
                message: format!("proposed_thread_count returned negative value: {n}"),
                source: None,
            })
        })
    }
}

// ---------- helpers ----------

fn missing_thread(handle: &ThreadHandle) -> StoreError {
    StoreError::Lance(lancedb::Error::Other {
        message: format!("threads ledger: no row for handle {handle}"),
        source: None,
    })
}

fn missing_evidence(id: i64) -> StoreError {
    StoreError::Lance(lancedb::Error::Other {
        message: format!("threads ledger: no evidence row id={id}"),
        source: None,
    })
}

fn path_to_str(p: &Path) -> String {
    p.to_string_lossy().into_owned()
}

const fn fold_to_str(d: FoldDepth) -> &'static str {
    match d {
        FoldDepth::Folded => "folded",
        FoldDepth::Half => "half",
        FoldDepth::Full => "full",
    }
}

fn fold_parse(s: &str) -> Result<FoldDepth> {
    match s {
        "folded" => Ok(FoldDepth::Folded),
        "half" => Ok(FoldDepth::Half),
        "full" => Ok(FoldDepth::Full),
        other => Err(bad_enum("fold_override", other)),
    }
}

const fn privacy_to_str(p: PrivacyTier) -> &'static str {
    match p {
        PrivacyTier::T0Private => "t0_private",
        PrivacyTier::T1Project => "t1_project",
        PrivacyTier::T2Trusted => "t2_trusted",
        PrivacyTier::T3Public => "t3_public",
    }
}

fn privacy_parse(s: &str) -> Result<PrivacyTier> {
    match s {
        "t0_private" => Ok(PrivacyTier::T0Private),
        "t1_project" => Ok(PrivacyTier::T1Project),
        "t2_trusted" => Ok(PrivacyTier::T2Trusted),
        "t3_public" => Ok(PrivacyTier::T3Public),
        other => Err(bad_enum("privacy_tier", other)),
    }
}

fn parse_ts(s: &str) -> Result<DateTime<Utc>> {
    DateTime::parse_from_rfc3339(s)
        .map(|dt| dt.with_timezone(&Utc))
        .map_err(|e| {
            StoreError::Lance(lancedb::Error::Other {
                message: format!("threads ledger: invalid RFC3339 timestamp {s:?}: {e}"),
                source: None,
            })
        })
}

fn row_to_thread(r: &rusqlite::Row<'_>) -> rusqlite::Result<ThreadRecord> {
    let handle_s: String = r.get(0)?;
    let tension_s: String = r.get(1)?;
    let familiarity_i: i64 = r.get(2)?;
    let last_touched_s: String = r.get(3)?;
    let anchor: Option<String> = r.get(4)?;
    let fold_s: Option<String> = r.get(5)?;
    let created_s: String = r.get(6)?;
    let scope_key: Option<String> = r.get(7)?;
    let privacy_s: String = r.get(8)?;

    let handle = ThreadHandle::new(handle_s).map_err(|e| {
        rusqlite::Error::FromSqlConversionFailure(0, rusqlite::types::Type::Text, Box::new(e))
    })?;
    let tension = TensionState::parse(&tension_s).map_err(to_sql_err)?;
    let last_touched_at = parse_ts(&last_touched_s).map_err(to_sql_err)?;
    let fold_override = fold_s
        .as_deref()
        .map(fold_parse)
        .transpose()
        .map_err(to_sql_err)?;
    let created_at = parse_ts(&created_s).map_err(to_sql_err)?;
    let privacy_tier = privacy_parse(&privacy_s).map_err(to_sql_err)?;
    let familiarity = u32::try_from(familiarity_i)
        .map_err(|_| rusqlite::Error::IntegralValueOutOfRange(2, familiarity_i))?;
    Ok(ThreadRecord {
        handle,
        tension,
        familiarity,
        last_touched_at,
        anchor_chunk_id: anchor,
        fold_override,
        created_at,
        created_scope_key: scope_key,
        privacy_tier,
    })
}

fn row_to_evidence(r: &rusqlite::Row<'_>) -> rusqlite::Result<EvidenceLink> {
    let id: i64 = r.get(0)?;
    let handle_s: String = r.get(1)?;
    let original: String = r.get(2)?;
    let current: Option<String> = r.get(3)?;
    let content_hash: Option<String> = r.get(4)?;
    let last_resolved: Option<String> = r.get(5)?;
    let relation_s: String = r.get(6)?;
    let association_s: String = r.get(7)?;
    let category: String = r.get(8)?;
    let similarity: Option<f64> = r.get(9)?;
    let created_s: String = r.get(10)?;
    let updated_s: String = r.get(11)?;

    let handle = ThreadHandle::new(handle_s).map_err(|e| {
        rusqlite::Error::FromSqlConversionFailure(1, rusqlite::types::Type::Text, Box::new(e))
    })?;
    let relation_state = RelationState::parse(&relation_s).map_err(to_sql_err)?;
    let association_type = AssociationType::parse(&association_s).map_err(to_sql_err)?;
    let created_at = parse_ts(&created_s).map_err(to_sql_err)?;
    let updated_at = parse_ts(&updated_s).map_err(to_sql_err)?;
    // SQLite REAL is f64; on-the-wire similarity is f32. The cast
    // truncates mantissa bits but cannot overflow — the value was an
    // f32 at insert time.
    #[allow(clippy::cast_possible_truncation)]
    let similarity = similarity.map(|f| f as f32);
    Ok(EvidenceLink {
        id,
        thread_handle: handle,
        original_path: PathBuf::from(original),
        current_path: current.map(PathBuf::from),
        content_hash,
        last_resolved_chunk_id: last_resolved,
        relation_state,
        association_type,
        category,
        similarity,
        created_at,
        updated_at,
    })
}

fn to_sql_err(e: StoreError) -> rusqlite::Error {
    rusqlite::Error::FromSqlConversionFailure(0, rusqlite::types::Type::Text, Box::new(e))
}

fn f32_vec_to_bytes(v: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(v.len() * 4);
    for f in v {
        out.extend_from_slice(&f.to_le_bytes());
    }
    out
}

fn bytes_to_f32_vec(b: &[u8]) -> Result<Vec<f32>> {
    if b.len() % 4 != 0 {
        return Err(StoreError::Lance(lancedb::Error::Other {
            message: format!("centroid_vec blob length {} not multiple of 4", b.len()),
            source: None,
        }));
    }
    let mut out = Vec::with_capacity(b.len() / 4);
    for chunk in b.chunks_exact(4) {
        let arr: [u8; 4] = chunk.try_into().expect("chunks_exact(4) yields [u8; 4]");
        out.push(f32::from_le_bytes(arr));
    }
    Ok(out)
}

fn row_to_proposed_thread(r: &rusqlite::Row<'_>) -> rusqlite::Result<ProposedThreadRecord> {
    let id: i64 = r.get(0)?;
    let proposed_handle: String = r.get(1)?;
    let chunk_ids_s: String = r.get(2)?;
    let centroid_bytes: Vec<u8> = r.get(3)?;
    let cohesion_f: f64 = r.get(4)?;
    let created_s: String = r.get(5)?;
    let promoted_to: Option<String> = r.get(6)?;

    let chunk_ids: Vec<String> = serde_json::from_str(&chunk_ids_s).map_err(|e| {
        rusqlite::Error::FromSqlConversionFailure(2, rusqlite::types::Type::Text, Box::new(e))
    })?;
    let centroid_vec = bytes_to_f32_vec(&centroid_bytes).map_err(to_sql_err)?;
    let created_at = parse_ts(&created_s).map_err(to_sql_err)?;
    // SQLite REAL is f64; cohesion is an f32 (cosine value in [-1, 1]) so
    // no overflow; mantissa truncation is acceptable for a display metric.
    #[allow(clippy::cast_possible_truncation)]
    let cohesion = cohesion_f as f32;
    Ok(ProposedThreadRecord {
        id,
        proposed_handle,
        chunk_ids,
        centroid_vec,
        cohesion,
        created_at,
        promoted_to,
    })
}

// ---------- tests ----------

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex as StdMutex;
    use tempfile::TempDir;

    #[derive(Default)]
    struct RecordingSink {
        events: StdMutex<Vec<ChainEvent>>,
    }

    impl RecordingSink {
        fn take(&self) -> Vec<ChainEvent> {
            std::mem::take(&mut self.events.lock().unwrap())
        }
    }

    impl ChainSink for RecordingSink {
        fn append(&self, event: &ChainEvent) -> Result<()> {
            self.events.lock().unwrap().push(event.clone());
            Ok(())
        }
    }

    fn handle(s: &str) -> ThreadHandle {
        ThreadHandle::new(s).unwrap()
    }

    fn sample_thread(s: &str) -> ThreadRecord {
        let now = Utc::now();
        ThreadRecord {
            handle: handle(s),
            tension: TensionState::Active,
            familiarity: 0,
            last_touched_at: now,
            anchor_chunk_id: None,
            fold_override: None,
            created_at: now,
            created_scope_key: None,
            privacy_tier: PrivacyTier::T1Project,
        }
    }

    fn sample_evidence(thread: &str, original: &str, category: &str) -> EvidenceLink {
        let now = Utc::now();
        EvidenceLink {
            id: 0,
            thread_handle: handle(thread),
            original_path: PathBuf::from(original),
            current_path: None,
            content_hash: Some("sha-abc".into()),
            last_resolved_chunk_id: None,
            relation_state: RelationState::Active,
            association_type: AssociationType::Curated,
            category: category.into(),
            similarity: None,
            created_at: now,
            updated_at: now,
        }
    }

    #[test]
    fn migration_runs_against_pre_existing_unrelated_db() {
        // Simulate a corpus whose threads.sqlite already exists with
        // tables that pre-date this ledger. Migration must add the
        // new tables alongside, not error.
        let tmp = TempDir::new().unwrap();
        let pre = Connection::open(tmp.path().join("threads.sqlite")).unwrap();
        pre.execute_batch("CREATE TABLE legacy_unrelated (x INTEGER);")
            .unwrap();
        drop(pre);
        let db = ThreadsDb::open(tmp.path()).unwrap();
        // Both new tables exist alongside the legacy one.
        let conn = db.lock();
        let n: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM sqlite_master WHERE type='table' \
                 AND name IN ('legacy_unrelated','threads','evidence_links')",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(n, 3);
    }

    #[test]
    fn schema_migration_idempotent() {
        let tmp = TempDir::new().unwrap();
        let _ = ThreadsDb::open(tmp.path()).unwrap();
        // Re-open: must not error, must not duplicate tables.
        let db = ThreadsDb::open(tmp.path()).unwrap();
        let conn = db.lock();
        let n: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM sqlite_master \
                 WHERE type='table' AND name IN ('threads','evidence_links')",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(n, 2);
    }

    #[test]
    fn thread_crud_round_trip() {
        let tmp = TempDir::new().unwrap();
        let db = ThreadsDb::open(tmp.path()).unwrap();
        let rec = sample_thread("alpha-thread");
        db.upsert_thread(&rec).unwrap();
        let back = db.get_thread(&handle("alpha-thread")).unwrap().unwrap();
        assert_eq!(back.handle, rec.handle);
        assert_eq!(back.tension, TensionState::Active);
        assert_eq!(back.privacy_tier, PrivacyTier::T1Project);

        // Update via replace.
        let mut updated = back;
        updated.anchor_chunk_id = Some("chunk-1".into());
        updated.fold_override = Some(FoldDepth::Half);
        db.upsert_thread(&updated).unwrap();
        let after = db.get_thread(&handle("alpha-thread")).unwrap().unwrap();
        assert_eq!(after.anchor_chunk_id.as_deref(), Some("chunk-1"));
        assert_eq!(after.fold_override, Some(FoldDepth::Half));

        db.delete_thread(&handle("alpha-thread")).unwrap();
        assert!(db.get_thread(&handle("alpha-thread")).unwrap().is_none());
    }

    #[test]
    fn familiarity_inc_returns_new_value() {
        let tmp = TempDir::new().unwrap();
        let db = ThreadsDb::open(tmp.path()).unwrap();
        db.upsert_thread(&sample_thread("fam")).unwrap();
        let v1 = db.increment_familiarity(&handle("fam")).unwrap();
        let v2 = db.increment_familiarity(&handle("fam")).unwrap();
        assert_eq!(v1, 1);
        assert_eq!(v2, 2);
        let row = db.get_thread(&handle("fam")).unwrap().unwrap();
        assert_eq!(row.familiarity, 2);
    }

    #[test]
    fn evidence_link_unique_constraint() {
        let tmp = TempDir::new().unwrap();
        let db = ThreadsDb::open(tmp.path()).unwrap();
        db.upsert_thread(&sample_thread("u")).unwrap();
        let link = sample_evidence("u", "src/foo.rs", "code");
        db.add_evidence_link(&link).unwrap();
        let err = db.add_evidence_link(&link).unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.to_lowercase().contains("unique") || msg.to_lowercase().contains("constraint"),
            "expected unique-constraint failure, got: {msg}"
        );
    }

    #[test]
    fn evidence_broken_reference_preserved() {
        let tmp = TempDir::new().unwrap();
        let db = ThreadsDb::open(tmp.path()).unwrap();
        db.upsert_thread(&sample_thread("br")).unwrap();
        let id = db
            .add_evidence_link(&sample_evidence("br", "src/gone.rs", "code"))
            .unwrap();
        db.mark_evidence_broken(id).unwrap();
        let list = db.list_evidence(&handle("br")).unwrap();
        assert_eq!(list.len(), 1);
        assert_eq!(list[0].relation_state, RelationState::BrokenReference);
        assert_eq!(list[0].original_path, PathBuf::from("src/gone.rs"));
    }

    #[test]
    fn chain_event_emitted_on_mutation() {
        let tmp = TempDir::new().unwrap();
        let sink = Arc::new(RecordingSink::default());
        let db = ThreadsDb::open_with_sink(tmp.path(), sink.clone()).unwrap();
        db.upsert_thread(&sample_thread("ch")).unwrap();
        let id = db
            .add_evidence_link(&sample_evidence("ch", "src/c.rs", "code"))
            .unwrap();
        db.mark_evidence_moved(id, Path::new("src/c2.rs")).unwrap();
        db.set_tension(&handle("ch"), TensionState::Slack).unwrap();
        db.delete_thread(&handle("ch")).unwrap();

        let events = sink.take();
        // Expect: ThreadCreate, EvidenceAdd, EvidenceStateChange,
        // TensionTransition, ThreadDelete — in that order.
        assert!(matches!(events[0], ChainEvent::ThreadCreate { .. }));
        assert!(matches!(events[1], ChainEvent::EvidenceAdd { .. }));
        assert!(matches!(
            events[2],
            ChainEvent::EvidenceStateChange {
                from: RelationState::Active,
                to: RelationState::Moved,
                ..
            }
        ));
        assert!(matches!(
            events[3],
            ChainEvent::TensionTransition {
                from: TensionState::Active,
                to: TensionState::Slack,
                ..
            }
        ));
        assert!(matches!(events[4], ChainEvent::ThreadDelete { .. }));
    }

    #[test]
    fn delete_thread_cascades_evidence() {
        let tmp = TempDir::new().unwrap();
        let db = ThreadsDb::open(tmp.path()).unwrap();
        db.upsert_thread(&sample_thread("cascade")).unwrap();
        db.add_evidence_link(&sample_evidence("cascade", "a", "code"))
            .unwrap();
        db.add_evidence_link(&sample_evidence("cascade", "b", "doc"))
            .unwrap();
        assert_eq!(db.evidence_count().unwrap(), 2);
        db.delete_thread(&handle("cascade")).unwrap();
        assert_eq!(db.evidence_count().unwrap(), 0);
    }

    #[test]
    fn list_threads_filtered_by_tension() {
        let tmp = TempDir::new().unwrap();
        let db = ThreadsDb::open(tmp.path()).unwrap();
        let mut a = sample_thread("a");
        let mut b = sample_thread("b");
        let mut c = sample_thread("c");
        a.tension = TensionState::Active;
        b.tension = TensionState::Slack;
        c.tension = TensionState::Slack;
        db.upsert_thread(&a).unwrap();
        db.upsert_thread(&b).unwrap();
        db.upsert_thread(&c).unwrap();

        let slack = db.list_threads(Some(TensionState::Slack)).unwrap();
        assert_eq!(slack.len(), 2);
        assert!(slack.iter().all(|t| t.tension == TensionState::Slack));

        let all = db.list_threads(None).unwrap();
        assert_eq!(all.len(), 3);
    }

    #[test]
    fn rename_thread_moves_evidence_and_chains() {
        let tmp = TempDir::new().unwrap();
        let sink = Arc::new(RecordingSink::default());
        let db = ThreadsDb::open_with_sink(tmp.path(), sink.clone()).unwrap();
        db.upsert_thread(&sample_thread("old")).unwrap();
        db.add_evidence_link(&sample_evidence("old", "p", "code"))
            .unwrap();
        let _ = sink.take();

        db.rename_thread(&handle("old"), &handle("new")).unwrap();

        assert!(db.get_thread(&handle("old")).unwrap().is_none());
        assert!(db.get_thread(&handle("new")).unwrap().is_some());
        assert_eq!(db.list_evidence(&handle("new")).unwrap().len(), 1);
        let ev = sink.take();
        assert_eq!(ev.len(), 1);
        assert!(matches!(ev[0], ChainEvent::ThreadRename { .. }));
    }

    #[test]
    fn upsert_replacement_does_not_re_chain_create() {
        let tmp = TempDir::new().unwrap();
        let sink = Arc::new(RecordingSink::default());
        let db = ThreadsDb::open_with_sink(tmp.path(), sink.clone()).unwrap();
        db.upsert_thread(&sample_thread("r")).unwrap();
        db.upsert_thread(&sample_thread("r")).unwrap();
        let ev = sink.take();
        assert_eq!(ev.len(), 1, "second upsert must not chain a second create");
    }

    #[test]
    fn familiarity_batch_chain_event_is_emitted() {
        let tmp = TempDir::new().unwrap();
        let sink = Arc::new(RecordingSink::default());
        let db = ThreadsDb::open_with_sink(tmp.path(), sink.clone()).unwrap();
        db.upsert_thread(&sample_thread("fb")).unwrap();
        let _ = sink.take();
        db.record_familiarity_batch(vec![(handle("fb"), 3)], 17)
            .unwrap();
        let ev = sink.take();
        assert_eq!(ev.len(), 1);
        match &ev[0] {
            ChainEvent::FamiliarityBatch {
                entries, turn_seq, ..
            } => {
                assert_eq!(*turn_seq, 17);
                assert_eq!(entries.len(), 1);
                assert_eq!(entries[0].0, handle("fb"));
                assert_eq!(entries[0].1, 3);
            }
            other => panic!("unexpected event: {other:?}"),
        }
    }
}
