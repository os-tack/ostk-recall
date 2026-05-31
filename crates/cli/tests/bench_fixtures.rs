//! P5: the rank-bench fixtures are committed, parse, and carry the schema the
//! harness (`crates/cli/examples/rank_bench.rs`) depends on. Cheap + fully
//! deterministic — the heavy end-to-end check is the example itself, run via
//! `cargo run --example rank_bench -- --fixture` (the example is compiled by
//! `cargo test`, so a build break here is caught in CI regardless).

use std::path::PathBuf;

fn fixtures_dir() -> PathBuf {
    // crates/cli -> repo root -> tests/fixtures/bench
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/bench")
}

#[test]
fn queries_fixture_parses_with_schema() {
    let path = fixtures_dir().join("queries.json");
    let txt = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("queries.json must be committed at {}: {e}", path.display()));
    let v: serde_json::Value = serde_json::from_str(&txt).expect("queries.json is valid JSON");
    let queries = v
        .get("queries")
        .and_then(|q| q.as_array())
        .expect("queries.json has a `queries` array");
    assert!(!queries.is_empty(), "query set must not be empty");
    for q in queries {
        assert!(
            q.get("id").and_then(serde_json::Value::as_str).is_some(),
            "each query needs a string `id`"
        );
        assert!(
            q.get("query").and_then(serde_json::Value::as_str).is_some(),
            "each query needs a `query` string"
        );
        let rel = q.get("relevant").expect("each query needs `relevant`");
        // The relevance rule keys off these two arrays (source_id primary).
        assert!(
            rel.get("source_ids")
                .map(serde_json::Value::is_array)
                .unwrap_or(false),
            "relevant.source_ids must be an array"
        );
        assert!(
            rel.get("chunk_ids")
                .map(serde_json::Value::is_array)
                .unwrap_or(false),
            "relevant.chunk_ids must be an array"
        );
    }
}

#[test]
fn lens_turns_fixture_parses() {
    let path = fixtures_dir().join("lens_turns.jsonl");
    let txt = std::fs::read_to_string(&path).unwrap_or_else(|e| {
        panic!(
            "lens_turns.jsonl must be committed at {}: {e}",
            path.display()
        )
    });
    let mut turns = 0usize;
    for line in txt.lines().filter(|l| !l.trim().is_empty()) {
        let v: serde_json::Value =
            serde_json::from_str(line).expect("each lens_turns line is valid JSON");
        assert!(
            v.get("focus_text")
                .and_then(serde_json::Value::as_str)
                .is_some(),
            "each lens turn needs a `focus_text` string"
        );
        turns += 1;
    }
    assert!(
        turns >= 4,
        "expected several lens turns for a rotation signal, got {turns}"
    );
}
