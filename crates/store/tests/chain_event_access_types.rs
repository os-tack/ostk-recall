//! P7b gate: the three new access-ledger `ChainEvent` variants serialize,
//! round-trip through the manual `to_payload()`/`from_row()` codec, carry
//! stable `kind_str` wire strings (renaming one silently orphans existing
//! rows), and persist + re-read through `SqliteChainSink`/`iter_chain`.

use std::sync::Arc;

use chrono::Utc;
use ostk_recall_store::{ChainEvent, ChainSink, SqliteChainSink, ThreadsDb};
use tempfile::TempDir;

fn access_events() -> Vec<ChainEvent> {
    let now = Utc::now();
    vec![
        ChainEvent::ExplicitRecall {
            chunk_id: "chunk-a".into(),
            query_hash: "deadbeefcafef00d".into(),
            ts: now,
        },
        ChainEvent::RecallFault {
            chunk_id: "chunk-b".into(),
            ts: now,
        },
        ChainEvent::OperatorSelected {
            chunk_id: "chunk-c".into(),
            ts: now,
        },
    ]
}

#[test]
fn access_event_kind_strings_are_stable() {
    // These strings are durable wire identifiers in the chain_log.kind
    // column; a rename orphans every existing row of that kind.
    let evs = access_events();
    assert_eq!(evs[0].kind_str(), "explicit_recall");
    assert_eq!(evs[1].kind_str(), "recall_fault");
    assert_eq!(evs[2].kind_str(), "operator_selected");
}

#[test]
fn access_events_round_trip_through_payload() {
    // `from_row` synthesizes ts = Utc::now() and discards the row ts, so
    // assert only on the durable fields (chunk_id, query_hash) — never ts.
    for ev in access_events() {
        let payload = ev.to_payload().expect("to_payload");
        let back = ChainEvent::from_row(ev.kind_str(), &payload).expect("from_row");
        match (&ev, &back) {
            (
                ChainEvent::ExplicitRecall {
                    chunk_id: a,
                    query_hash: qa,
                    ..
                },
                ChainEvent::ExplicitRecall {
                    chunk_id: b,
                    query_hash: qb,
                    ..
                },
            ) => {
                assert_eq!(a, b);
                assert_eq!(qa, qb);
            }
            (
                ChainEvent::RecallFault { chunk_id: a, .. },
                ChainEvent::RecallFault { chunk_id: b, .. },
            )
            | (
                ChainEvent::OperatorSelected { chunk_id: a, .. },
                ChainEvent::OperatorSelected { chunk_id: b, .. },
            ) => assert_eq!(a, b),
            _ => panic!("variant changed across round-trip: {ev:?} -> {back:?}"),
        }
    }
}

#[test]
fn access_events_persist_and_iter_chain_round_trips() {
    let tmp = TempDir::new().unwrap();
    let sink: Arc<dyn ChainSink> = Arc::new(SqliteChainSink::open(tmp.path()).unwrap());
    let db = ThreadsDb::open_with_sink(tmp.path(), Arc::clone(&sink)).unwrap();
    for ev in access_events() {
        sink.append(&ev).unwrap();
    }
    let kinds: Vec<&str> = db.iter_chain().unwrap().iter().map(ChainEvent::kind_str).collect();
    assert_eq!(
        kinds,
        vec!["explicit_recall", "recall_fault", "operator_selected"],
        "all three access events persist and replay in seq order"
    );
}
