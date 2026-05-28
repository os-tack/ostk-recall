//! P1 gate: operator-override merge semantics.
//!
//! - `single` cardinality: override replaces scanner-emitted value.
//! - `multi` cardinality: override unions, except empty-list sentinel
//!   which clears.
//! - Facet escaping: values containing `:`, spaces, `/`, `~` round-trip.
//! - List<Utf8> ↔ FacetSet round-trip preserves multi-value sets.

use std::collections::{BTreeMap, BTreeSet};

use ostk_recall_core::{FacetSet, cardinality_of, from_list, merge_override, to_list, Cardinality};

#[test]
fn single_override_replaces_scanner_value() {
    let mut f: FacetSet = BTreeMap::new();
    f.insert("lang".into(), ["rust".to_string()].into_iter().collect());
    merge_override(&mut f, "lang", vec!["python".into()]);
    assert_eq!(f["lang"], ["python".to_string()].into_iter().collect::<BTreeSet<_>>());
}

#[test]
fn multi_override_unions() {
    let mut f: FacetSet = BTreeMap::new();
    f.insert("project".into(), ["auth".to_string()].into_iter().collect());
    merge_override(&mut f, "project", vec!["billing".into(), "ml".into()]);
    let expected: BTreeSet<String> = ["auth", "billing", "ml"]
        .into_iter()
        .map(String::from)
        .collect();
    assert_eq!(f["project"], expected);
}

#[test]
fn multi_empty_list_clears_scanner_set() {
    let mut f: FacetSet = BTreeMap::new();
    f.insert(
        "tags".into(),
        ["scanner-emitted".to_string()].into_iter().collect(),
    );
    merge_override(&mut f, "tags", vec![]);
    assert!(f.get("tags").is_none(), "empty list = explicit clear");
}

#[test]
fn cardinality_default_for_unknown_key_is_single() {
    assert_eq!(cardinality_of("custom_xyz"), Cardinality::Single);
}

#[test]
fn facet_values_with_colon_round_trip() {
    let mut f: FacetSet = BTreeMap::new();
    f.insert(
        "agent".into(),
        ["claude-3:opus".to_string()].into_iter().collect(),
    );
    let list = to_list(&f);
    assert!(list.iter().any(|s| s == "agent:claude-3:opus"));
    let parsed = from_list(&list);
    assert_eq!(parsed["agent"], ["claude-3:opus".to_string()].into_iter().collect::<BTreeSet<_>>());
}

#[test]
fn facet_values_with_spaces_paths_tilde_round_trip() {
    let mut f: FacetSet = BTreeMap::new();
    f.insert(
        "path_prefix".into(),
        ["~/projects/ostk recall".to_string()].into_iter().collect(),
    );
    f.insert("project".into(), ["foo / bar".to_string()].into_iter().collect());
    let list = to_list(&f);
    let parsed = from_list(&list);
    assert_eq!(parsed, f);
}

#[test]
fn list_serialization_is_sorted() {
    let mut f: FacetSet = BTreeMap::new();
    f.insert(
        "project".into(),
        ["zeta", "alpha", "beta"]
            .into_iter()
            .map(String::from)
            .collect(),
    );
    let list = to_list(&f);
    let mut sorted = list.clone();
    sorted.sort();
    assert_eq!(list, sorted, "list output must be lexicographically sorted");
}
