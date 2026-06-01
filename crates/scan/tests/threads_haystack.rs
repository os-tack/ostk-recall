//! End-to-end sanity: parse a real haystack thread fixture and dump
//! the resulting `chunk.extra` JSON. This is gated behind `OSTK_HAYSTACK`
//! so CI environments without the fixture skip cleanly.

use std::path::PathBuf;

use ostk_recall_core::{Scanner, SourceConfig, SourceKind};
use ostk_recall_scan::threads::ThreadScanner;

#[test]
fn dumps_extra_for_haystack_thread() {
    let Ok(haystack) = std::env::var("OSTK_HAYSTACK") else {
        eprintln!("OSTK_HAYSTACK unset — skipping haystack sanity check");
        return;
    };
    let root = PathBuf::from(haystack).join(".ostk/threads");
    if !root.exists() {
        eprintln!("{} missing — skipping", root.display());
        return;
    }

    let cfg = SourceConfig {
        kind: SourceKind::Thread,
        project: Some("haystack".into()),
        paths: vec![root.to_string_lossy().into_owned()],
        ignore: vec![],
        extensions: vec![],
        entity_type: None,
        edges: Vec::new(),
        id: None,
        source_config_id: "test-cfg".to_string(),
        facets: Default::default(),
    };
    let scanner = ThreadScanner;
    let items: Vec<_> = scanner.discover(&cfg).filter_map(Result::ok).collect();
    assert!(!items.is_empty(), "expected at least one thread file");

    let target = items
        .iter()
        .find(|it| it.source_id == "hoberman-thread-primitive")
        .expect("hoberman-thread-primitive.md present");
    let chunks = scanner.parse(target.clone()).unwrap();
    assert_eq!(chunks.len(), 1);
    eprintln!(
        "SAMPLE chunk.extra for {}: {}",
        target.source_id,
        serde_json::to_string_pretty(&chunks[0].extra).unwrap()
    );
}
