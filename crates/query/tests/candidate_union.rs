//! P3B gate: `build_candidates` unions the BM25 + dense lanes by
//! chunk_id and stamps the exact RRF score `Σ 1/(K_RRF + rank)` over the
//! lanes a chunk appears in. Drives the public lane builder with
//! constructed lane evidence (no Lance), so the RRF math is checked in
//! isolation from retrieval.

use std::collections::HashMap;

use ostk_recall_core::{Chunk, FacetSet, Links, Source};
use ostk_recall_query::lanes::{K_RRF, build_candidates};

fn chunk(id: &str) -> Chunk {
    Chunk {
        chunk_id: id.to_string(),
        source: Source::Markdown,
        project: None,
        source_id: format!("src/{id}"),
        source_config_id: "test:cfg".to_string(),
        chunk_index: 0,
        ts: None,
        role: None,
        text: format!("text {id}"),
        sha256: format!("sha-{id}"),
        links: Links::default(),
        facets: FacetSet::default(),
        embedding_input_sha256: format!("emb-{id}"),
        extra: serde_json::Value::Null,
    }
}

#[test]
fn build_candidates_stamps_exact_rrf_over_union() {
    // "a" appears in BOTH lanes at rank 0; "b" only in BM25 (rank 1);
    // "c" only in dense (rank 2).
    let bm25 = vec![("a".to_string(), 5.0, 0), ("b".to_string(), 4.0, 1)];
    let dense = vec![("a".to_string(), 0.1, 0), ("c".to_string(), 0.2, 2)];
    let mut chunks = HashMap::new();
    for id in ["a", "b", "c"] {
        chunks.insert(id.to_string(), chunk(id));
    }

    let candidates = build_candidates(&bm25, &dense, chunks);
    let by_id: HashMap<&str, _> = candidates
        .iter()
        .map(|c| (c.chunk.chunk_id.as_str(), c))
        .collect();
    assert_eq!(by_id.len(), 3, "union covers every lane id");

    let both = 1.0 / (K_RRF + 0.0) + 1.0 / (K_RRF + 0.0);
    let bm25_only = 1.0 / (K_RRF + 1.0);
    let dense_only = 1.0 / (K_RRF + 2.0);

    let eps = 1e-6;
    assert!((by_id["a"].rrf_score.unwrap() - both).abs() < eps);
    assert!((by_id["b"].rrf_score.unwrap() - bm25_only).abs() < eps);
    assert!((by_id["c"].rrf_score.unwrap() - dense_only).abs() < eps);

    // Per-lane evidence is stamped: "a" has both, "b" bm25-only, "c" dense-only.
    assert!(by_id["a"].has_bm25_evidence() && by_id["a"].has_dense_evidence());
    assert!(by_id["b"].has_bm25_evidence() && !by_id["b"].has_dense_evidence());
    assert!(!by_id["c"].has_bm25_evidence() && by_id["c"].has_dense_evidence());

    // The both-lanes chunk scores strictly higher than either single-lane chunk.
    assert!(by_id["a"].rrf_score.unwrap() > by_id["b"].rrf_score.unwrap());
    assert!(by_id["a"].rrf_score.unwrap() > by_id["c"].rrf_score.unwrap());
}
