//! P1 gate: facet header composition must be order-independent.
//!
//! Two FacetSets with the same key→values map but built in different
//! insertion orders MUST produce the same `compose_header` output, and
//! therefore the same `embedding_input_sha256`. Without this property
//! re-ingest after a multi-value facet edit would non-deterministically
//! re-embed.

use std::collections::{BTreeMap, BTreeSet};

use ostk_recall_core::{Chunk, FacetSet, compose_header, filter_to_allowlist};

fn fset(items: &[(&str, &[&str])]) -> FacetSet {
    let mut out: FacetSet = BTreeMap::new();
    for (k, vs) in items {
        let s: BTreeSet<String> = vs.iter().map(|v| (*v).to_string()).collect();
        out.insert((*k).to_string(), s);
    }
    out
}

#[test]
fn header_stable_across_value_insertion_order() {
    let a = fset(&[("project", &["auth", "billing"])]);
    let b = fset(&[("project", &["billing", "auth"])]);
    assert_eq!(compose_header(&a), compose_header(&b));
}

#[test]
fn header_stable_across_key_insertion_order() {
    let mut a: FacetSet = BTreeMap::new();
    a.insert("project".into(), ["auth".to_string()].into_iter().collect());
    a.insert("lang".into(), ["rust".to_string()].into_iter().collect());

    let mut b: FacetSet = BTreeMap::new();
    b.insert("lang".into(), ["rust".to_string()].into_iter().collect());
    b.insert("project".into(), ["auth".to_string()].into_iter().collect());

    assert_eq!(compose_header(&a), compose_header(&b));
}

#[test]
fn embedding_input_hash_stable_across_orders() {
    let f1 = fset(&[("project", &["auth", "billing"]), ("lang", &["rust"])]);
    let f2 = fset(&[("lang", &["rust"]), ("project", &["billing", "auth"])]);
    let h1 = Chunk::embedding_input_hash(
        "model-x",
        &compose_header(&filter_to_allowlist(&f1)),
        "body",
    );
    let h2 = Chunk::embedding_input_hash(
        "model-x",
        &compose_header(&filter_to_allowlist(&f2)),
        "body",
    );
    assert_eq!(h1, h2, "deterministic header → deterministic hash");
}

#[test]
fn non_allowlisted_facets_dont_change_hash() {
    let mut f1 = fset(&[("project", &["auth"])]);
    let mut f2 = f1.clone();
    f2.insert(
        "session_id".into(),
        ["abc123".to_string()].into_iter().collect(),
    );
    f2.insert(
        "era".into(),
        ["2026-W22-Q2".to_string()].into_iter().collect(),
    );

    let h1 = Chunk::embedding_input_hash("m", &compose_header(&filter_to_allowlist(&f1)), "body");
    let h2 = Chunk::embedding_input_hash("m", &compose_header(&filter_to_allowlist(&f2)), "body");
    assert_eq!(h1, h2, "non-allowlisted facets must not affect the hash");

    // Sanity: an allowlisted change DOES affect it.
    f1.entry("project".into())
        .or_default()
        .insert("billing".into());
    let h3 = Chunk::embedding_input_hash("m", &compose_header(&filter_to_allowlist(&f1)), "body");
    assert_ne!(h1, h3, "allowlisted change must change the hash");
}

#[test]
fn embedder_model_id_participates_in_hash() {
    let h1 = Chunk::embedding_input_hash("model-a", "project:auth", "body");
    let h2 = Chunk::embedding_input_hash("model-b", "project:auth", "body");
    assert_ne!(h1, h2);
}
