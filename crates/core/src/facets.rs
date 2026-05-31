//! Facets: per-chunk semantic overlay (P1).
//!
//! Facets are interpretive tags assigned by scanners and operator overrides.
//! Identity (chunk_id, source_config_id) is *physical* — facets DO NOT
//! change identity. Changing a facet that participates in the embedding
//! input triggers re-embed via `embedding_input_sha256` mismatch
//! detection.
//!
//! Shape: `FacetSet = BTreeMap<String, BTreeSet<String>>`. Most keys hold
//! a singleton; `project`, `tags`, `status` may hold multiple values per
//! chunk. Cardinality is declared per-key in the global [`FacetRegistry`].
//!
//! Serialization on Lance is a single `List<Utf8>` column with prefix
//! convention `"key:value"` — sorted for determinism.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

/// Per-chunk facet payload. Keys lowercase ASCII (`[a-z][a-z0-9_]*`),
/// values arbitrary unicode. The first `:` in `key:value` is the
/// delimiter — values may contain further colons.
pub type FacetSet = BTreeMap<String, BTreeSet<String>>;

/// Per-key cardinality declaration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Cardinality {
    /// One value per chunk by construction. Operator overrides
    /// *replace* the scanner-emitted value.
    Single,
    /// Set of values per chunk. Operator overrides *union* with
    /// scanner-emitted set; empty-list sentinel clears.
    Multi,
}

/// Facet-key allowlist for embedding-input composition (P1).
///
/// Only these keys participate in `embedding_input_sha256` — changing a
/// non-allowlisted facet (e.g. `session_id`, `era`) MUST NOT trigger
/// re-embed. `era` is intentionally excluded so weekly bucket rollover
/// doesn't churn the whole corpus.
pub const EMBED_FACET_ALLOWLIST: &[&str] = &["project", "lang", "agent", "status", "record_kind"];

/// Schema-versioning constants for the embedding-input hash. Bumping
/// either invalidates every `embedding_input_sha256` and forces a
/// corpus-wide re-embed on next scan.
pub const HEADER_FORMAT_VERSION: u32 = 1;
pub const ALLOWLIST_VERSION: u32 = 1;

/// Returns the canonical cardinality for a facet key, defaulting to
/// `Single` for unknown keys (scanner-emitted custom keys behave as
/// single-value).
#[must_use]
pub fn cardinality_of(key: &str) -> Cardinality {
    match key {
        "project" | "tags" | "status" => Cardinality::Multi,
        _ => Cardinality::Single,
    }
}

/// Validate a facet key (`[a-z][a-z0-9_]*`). Returns false for empty
/// keys, keys with non-ASCII, keys with uppercase, etc. Used at parse to
/// reject malformed scanner emissions and operator overrides.
#[must_use]
pub fn is_valid_facet_key(k: &str) -> bool {
    let mut bytes = k.bytes();
    match bytes.next() {
        Some(b) if b.is_ascii_lowercase() => {}
        _ => return false,
    }
    bytes.all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'_')
}

/// Merge an operator override into the scanner-emitted facets.
///
/// Single-cardinality: override replaces scanner value entirely.
/// Multi-cardinality: override unions with scanner set, EXCEPT empty-list
/// sentinel which removes the key (operator explicit-clear).
///
/// `override_values` should be the operator's per-key value list as a
/// `Vec<String>` (TOML round-trip shape).
pub fn merge_override(facets: &mut FacetSet, key: &str, override_values: Vec<String>) {
    match cardinality_of(key) {
        Cardinality::Single => {
            let mut set = BTreeSet::new();
            if let Some(v) = override_values.into_iter().next() {
                set.insert(v);
            }
            if set.is_empty() {
                facets.remove(key);
            } else {
                facets.insert(key.to_string(), set);
            }
        }
        Cardinality::Multi => {
            if override_values.is_empty() {
                // Empty-list sentinel = explicit clear (removes scanner-emitted set).
                facets.remove(key);
            } else {
                facets
                    .entry(key.to_string())
                    .or_default()
                    .extend(override_values);
            }
        }
    }
}

/// Filter `facets` down to the embedding-allowlisted subset.
#[must_use]
pub fn filter_to_allowlist(facets: &FacetSet) -> FacetSet {
    facets
        .iter()
        .filter(|(k, _)| EMBED_FACET_ALLOWLIST.contains(&k.as_str()))
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect()
}

/// Compose a deterministic header from an allowlisted facet subset.
///
/// Keys sorted lexicographically (BTreeMap iteration order); within each
/// key, values sorted lexicographically (BTreeSet iteration order).
/// Joined as `key:value` segments separated by `|`. Empty facets yield
/// an empty string.
#[must_use]
pub fn compose_header(allowlisted: &FacetSet) -> String {
    let mut parts: Vec<String> = Vec::new();
    for (k, values) in allowlisted {
        for v in values {
            parts.push(format!("{k}:{v}"));
        }
    }
    parts.join("|")
}

/// Serialize a FacetSet to a sorted `Vec<String>` for Lance's
/// `List<Utf8>` column. `array_contains(facets, 'project:auth')` then
/// works as the single-value filter primitive.
#[must_use]
pub fn to_list(facets: &FacetSet) -> Vec<String> {
    let mut out = Vec::new();
    for (k, values) in facets {
        for v in values {
            out.push(format!("{k}:{v}"));
        }
    }
    // Already in BTree order, but explicit sort keeps the contract obvious.
    out.sort();
    out
}

/// Hash the per-source overlay state for Tier-1 cache invalidation.
///
/// Includes legacy `project` + the operator `facets` override (P1), plus an
/// `extra_digest` (P12) that carries the per-source-kind record-rule digest
/// and the scanner's `parse_version`. When the operator edits any facet, edits
/// a record rule that can match this source, or a scanner bumps its parse
/// version, the hash flips and the pipeline forces a re-parse for the affected
/// sources. Re-parsing then recomputes `embedding_input_sha256`; the Tier-2
/// check decides whether to actually re-embed. Passing `""` for `extra_digest`
/// reproduces the pre-P12 P1-only hash.
#[must_use]
pub fn cfg_overlay_hash(
    legacy_project: Option<&str>,
    facets_override: &BTreeMap<String, Vec<String>>,
    extra_digest: &str,
) -> String {
    use sha2::{Digest, Sha256};
    let mut h = Sha256::new();
    h.update(b"legacy_project:");
    h.update(legacy_project.unwrap_or("").as_bytes());
    h.update(b"|facets:");
    // BTreeMap iteration order is sorted; sort each value list too.
    for (k, vs) in facets_override {
        h.update(k.as_bytes());
        h.update(b"=");
        let mut sorted = vs.clone();
        sorted.sort();
        for v in &sorted {
            h.update(v.as_bytes());
            h.update(b",");
        }
        h.update(b"|");
    }
    // IMPORTANT: only mix in the extra section when non-empty, so a source
    // with no applicable record rules + parse_version 0 hashes
    // byte-identically to the pre-P12 (P1-only) value and keeps Tier-1
    // skipping — otherwise adding P12 would force a full-corpus re-parse.
    if !extra_digest.is_empty() {
        h.update(b"|extra:");
        h.update(extra_digest.as_bytes());
    }
    hex::encode(&h.finalize()[..16])
}

/// Parse a `Vec<String>` (Lance `List<Utf8>` row) back into a FacetSet.
/// First `:` is the delimiter (values may contain further colons).
#[must_use]
pub fn from_list(list: &[String]) -> FacetSet {
    let mut out: FacetSet = BTreeMap::new();
    for entry in list {
        if let Some((k, v)) = entry.split_once(':') {
            if is_valid_facet_key(k) {
                out.entry(k.to_string()).or_default().insert(v.to_string());
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_override_replaces() {
        let mut f: FacetSet = BTreeMap::new();
        f.insert("lang".into(), [String::from("rust")].into_iter().collect());
        merge_override(&mut f, "lang", vec!["python".into()]);
        assert_eq!(f["lang"], [String::from("python")].into_iter().collect());
    }

    #[test]
    fn multi_override_unions() {
        let mut f: FacetSet = BTreeMap::new();
        f.insert(
            "project".into(),
            [String::from("auth")].into_iter().collect(),
        );
        merge_override(&mut f, "project", vec!["billing".into()]);
        assert_eq!(
            f["project"],
            [String::from("auth"), String::from("billing")]
                .into_iter()
                .collect()
        );
    }

    #[test]
    fn multi_empty_list_clears() {
        let mut f: FacetSet = BTreeMap::new();
        f.insert(
            "project".into(),
            [String::from("auth")].into_iter().collect(),
        );
        merge_override(&mut f, "project", vec![]);
        assert!(f.get("project").is_none(), "empty-list sentinel clears");
    }

    #[test]
    fn header_deterministic_across_insertion_orders() {
        let mut a: FacetSet = BTreeMap::new();
        a.insert(
            "project".into(),
            [String::from("auth"), String::from("billing")]
                .into_iter()
                .collect(),
        );
        a.insert("lang".into(), [String::from("rust")].into_iter().collect());

        let mut b: FacetSet = BTreeMap::new();
        // Insert in opposite order.
        b.insert("lang".into(), [String::from("rust")].into_iter().collect());
        b.insert(
            "project".into(),
            [String::from("billing"), String::from("auth")]
                .into_iter()
                .collect(),
        );

        assert_eq!(compose_header(&a), compose_header(&b));
    }

    #[test]
    fn header_only_includes_allowlist() {
        let mut f: FacetSet = BTreeMap::new();
        f.insert(
            "project".into(),
            [String::from("auth")].into_iter().collect(),
        );
        f.insert(
            "era".into(),
            [String::from("2026-W22-Q2")].into_iter().collect(),
        );
        f.insert(
            "session_id".into(),
            [String::from("abc")].into_iter().collect(),
        );
        let allow = filter_to_allowlist(&f);
        let header = compose_header(&allow);
        assert!(header.contains("project:auth"));
        assert!(!header.contains("era:"));
        assert!(!header.contains("session_id:"));
    }

    #[test]
    fn list_round_trip_preserves_set() {
        let mut f: FacetSet = BTreeMap::new();
        f.insert(
            "project".into(),
            [String::from("auth"), String::from("billing")]
                .into_iter()
                .collect(),
        );
        f.insert("lang".into(), [String::from("rust")].into_iter().collect());
        let list = to_list(&f);
        let parsed = from_list(&list);
        assert_eq!(parsed, f);
    }

    #[test]
    fn from_list_handles_colon_in_value() {
        let list = vec!["agent:claude-3:opus".to_string()];
        let f = from_list(&list);
        assert_eq!(
            f["agent"],
            [String::from("claude-3:opus")].into_iter().collect()
        );
    }

    #[test]
    fn invalid_facet_keys_rejected() {
        assert!(!is_valid_facet_key(""));
        assert!(!is_valid_facet_key("Project"));
        assert!(!is_valid_facet_key("123tag"));
        assert!(!is_valid_facet_key("project!"));
        assert!(is_valid_facet_key("project"));
        assert!(is_valid_facet_key("path_prefix"));
        assert!(is_valid_facet_key("ostk_root"));
    }

    #[test]
    fn cardinality_defaults() {
        assert_eq!(cardinality_of("project"), Cardinality::Multi);
        assert_eq!(cardinality_of("tags"), Cardinality::Multi);
        assert_eq!(cardinality_of("status"), Cardinality::Multi);
        assert_eq!(cardinality_of("lang"), Cardinality::Single);
        assert_eq!(cardinality_of("agent"), Cardinality::Single);
        assert_eq!(cardinality_of("custom_key"), Cardinality::Single);
    }
}
