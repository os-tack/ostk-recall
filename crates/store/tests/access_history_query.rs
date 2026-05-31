//! P7b gate: `ChainLogReader::access_history` returns the correct
//! per-chunk access events for a fixture, filters by `since`, maps
//! chain-kinds to `AccessKind`, excludes non-access events, and omits
//! chunks/ids with no qualifying events.

use std::sync::Arc;

use chrono::{Duration, Utc};
use ostk_recall_store::{
    AccessKind, ChainEvent, ChainLogReader, ChainSink, SqliteChainSink, ThreadsDb,
};
use tempfile::TempDir;

#[test]
fn access_history_filters_by_since_kind_and_id() {
    let tmp = TempDir::new().unwrap();
    let sink: Arc<dyn ChainSink> = Arc::new(SqliteChainSink::open(tmp.path()).unwrap());
    let db = ThreadsDb::open_with_sink(tmp.path(), Arc::clone(&sink)).unwrap();

    let now = Utc::now();
    let since = now - Duration::hours(24);
    let t_old = now - Duration::hours(48); // before `since` → excluded
    let t1 = now - Duration::hours(10);
    let t2 = now - Duration::hours(5);
    let t3 = now - Duration::hours(1);

    // A: two in-window accesses of different kinds.
    sink.append(&ChainEvent::ExplicitRecall {
        chunk_id: "A".into(),
        query_hash: "h".into(),
        ts: t1,
    })
    .unwrap();
    sink.append(&ChainEvent::LensIncluded {
        chunk_id: "A".into(),
        slot: "attention".into(),
        ts: t2,
    })
    .unwrap();
    // B: one in-window access.
    sink.append(&ChainEvent::RecallFault {
        chunk_id: "B".into(),
        ts: t3,
    })
    .unwrap();
    // C: only an access OLDER than `since` → must be excluded.
    sink.append(&ChainEvent::ExplicitRecall {
        chunk_id: "C".into(),
        query_hash: "h".into(),
        ts: t_old,
    })
    .unwrap();
    // A non-access event in window → must be excluded by kind.
    sink.append(&ChainEvent::LensIncluded {
        chunk_id: "A".into(),
        slot: "attention".into(),
        ts: t3,
    })
    .unwrap();

    let hist = db
        .access_history(
            &["A".into(), "B".into(), "C".into(), "Z".into()],
            since,
        )
        .unwrap();

    // A: ExplicitRecall + 2× LensIncluded (the t2 and t3 ones), all in window.
    let a = hist.get("A").expect("A has in-window accesses");
    assert_eq!(a.len(), 3, "A: 1 ExplicitRecall + 2 LensIncluded");
    assert!(a.iter().any(|(k, _)| *k == AccessKind::ExplicitRecall));
    assert_eq!(
        a.iter().filter(|(k, _)| *k == AccessKind::LensIncluded).count(),
        2
    );

    // B: one RecallFault.
    let b = hist.get("B").expect("B has an in-window access");
    assert_eq!(b.len(), 1);
    assert_eq!(b[0].0, AccessKind::RecallFault);

    // C: its only access predates `since` → absent.
    assert!(!hist.contains_key("C"), "C filtered out by since");
    // Z: never in the chain → absent.
    assert!(!hist.contains_key("Z"));
}

#[test]
fn access_history_empty_inputs_and_kinds() {
    let tmp = TempDir::new().unwrap();
    let sink: Arc<dyn ChainSink> = Arc::new(SqliteChainSink::open(tmp.path()).unwrap());
    let db = ThreadsDb::open_with_sink(tmp.path(), Arc::clone(&sink)).unwrap();

    // Empty id set → empty map, no query.
    assert!(db.access_history(&[], Utc::now()).unwrap().is_empty());

    // A non-access event (ThreadCreate-ish via a recall on an unknown id)
    // never appears: append an OperatorSelected for X, query for Y.
    sink.append(&ChainEvent::OperatorSelected {
        chunk_id: "X".into(),
        ts: Utc::now(),
    })
    .unwrap();
    let hist = db
        .access_history(&["Y".into()], Utc::now() - Duration::hours(1))
        .unwrap();
    assert!(hist.is_empty(), "querying a different id returns nothing");
}
