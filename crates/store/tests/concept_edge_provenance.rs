//! Relational-substrate slice 1: the `ConceptConnected` chain event carries
//! edge provenance (`source`/`by`), round-trips through the manual
//! `to_payload()`/`from_row()` codec, keeps its stable `kind_str` wire string
//! (a rename orphans existing rows), and — critically — a *legacy* payload
//! written before provenance existed still decodes (source defaults to
//! `observed`, `by` to `None`). Never assert on `ts`: `from_row` synthesizes
//! `Utc::now()` and discards the row stamp.

use std::sync::Arc;

use chrono::Utc;
use ostk_recall_store::{ChainEvent, ChainSink, SqliteChainSink, ThreadsDb};
use tempfile::TempDir;

fn connected(source: &str, by: Option<&str>) -> ChainEvent {
    ChainEvent::ConceptConnected {
        project: String::new(),
        from: "ostk-recall".into(),
        relation: "memory_layer_for".into(),
        to: "ostk".into(),
        source: source.into(),
        by: by.map(ToString::to_string),
        ts: Utc::now(),
    }
}

#[test]
fn concept_connected_kind_string_is_stable() {
    // Durable wire identifier in chain_log.kind — renaming orphans rows.
    assert_eq!(
        connected("authored", Some("claude")).kind_str(),
        "concept_connected"
    );
}

#[test]
fn concept_connected_round_trips_with_provenance() {
    let ev = connected("authored", Some("claude"));
    let payload = ev.to_payload().expect("to_payload");
    let back = ChainEvent::from_row(ev.kind_str(), &payload).expect("from_row");
    match back {
        ChainEvent::ConceptConnected {
            from,
            relation,
            to,
            source,
            by,
            ..
        } => {
            assert_eq!(from, "ostk-recall");
            assert_eq!(relation, "memory_layer_for");
            assert_eq!(to, "ostk");
            assert_eq!(source, "authored");
            assert_eq!(by.as_deref(), Some("claude"));
        }
        other => panic!("variant changed across round-trip: {other:?}"),
    }
}

#[test]
fn legacy_concept_connected_payload_decodes_as_observed() {
    // A payload written before provenance existed: no `source`/`by` keys.
    // It must decode (never `?`-error) with source=observed, by=None.
    let legacy = serde_json::json!({
        "project": "",
        "from": "a",
        "relation": "pairs_with",
        "to": "b",
    })
    .to_string();
    let back = ChainEvent::from_row("concept_connected", &legacy).expect("legacy decodes");
    match back {
        ChainEvent::ConceptConnected { source, by, .. } => {
            assert_eq!(source, "observed", "legacy edge defaults to observed");
            assert_eq!(by, None, "legacy edge has no known author");
        }
        other => panic!("unexpected variant: {other:?}"),
    }
}

#[test]
fn concept_connected_persists_and_iter_chain_round_trips() {
    let tmp = TempDir::new().unwrap();
    let sink: Arc<dyn ChainSink> = Arc::new(SqliteChainSink::open(tmp.path()).unwrap());
    let db = ThreadsDb::open_with_sink(tmp.path(), Arc::clone(&sink)).unwrap();
    sink.append(&connected("authored", Some("claude"))).unwrap();
    let kinds: Vec<&str> = db
        .iter_chain()
        .unwrap()
        .iter()
        .map(ChainEvent::kind_str)
        .collect();
    assert_eq!(kinds, vec!["concept_connected"]);
}
