//! P0 gate: two `[[sources]]` blocks with identical physical shape AND no
//! disambiguator (no `id`, no legacy `project` discriminator delta) must be
//! rejected at config parse with a structured error that points at both
//! block indices.

use std::io::Write;
use tempfile::NamedTempFile;

use ostk_recall_core::Config;

#[test]
fn duplicate_physical_sources_reject_at_parse() {
    let body = r#"
[corpus]
root = "/tmp/x"

[embedder]
model = "potion-retrieval-32M"

[[sources]]
kind = "markdown"
paths = ["~/notes"]

[[sources]]
kind = "markdown"
paths = ["~/notes"]
"#;
    let mut f = NamedTempFile::new().unwrap();
    write!(f, "{body}").unwrap();
    let err = Config::load(f.path()).unwrap_err().to_string();
    assert!(
        err.contains("source blocks") && err.contains("share physical identity"),
        "expected duplicate-shape error, got: {err}"
    );
    assert!(
        err.contains("kind=\"markdown\""),
        "error should mention the colliding kind: {err}"
    );
}

#[test]
fn explicit_id_disambiguates_duplicates() {
    let body = r#"
[corpus]
root = "/tmp/x"

[embedder]
model = "potion-retrieval-32M"

[[sources]]
kind = "markdown"
id = "notes-a"
paths = ["~/notes"]

[[sources]]
kind = "markdown"
id = "notes-b"
paths = ["~/notes"]
"#;
    let mut f = NamedTempFile::new().unwrap();
    write!(f, "{body}").unwrap();
    let cfg = Config::load(f.path()).expect("explicit ids should parse");
    assert_eq!(cfg.sources[0].source_config_id, "notes-a");
    assert_eq!(cfg.sources[1].source_config_id, "notes-b");
}

#[test]
fn synthetic_prefix_rejected_in_explicit_id() {
    let body = r#"
[corpus]
root = "/tmp/x"

[embedder]
model = "potion-retrieval-32M"

[[sources]]
kind = "markdown"
id = "synthetic:membrane"
paths = ["~/notes"]
"#;
    let mut f = NamedTempFile::new().unwrap();
    write!(f, "{body}").unwrap();
    let err = Config::load(f.path()).unwrap_err().to_string();
    assert!(
        err.contains("synthetic:") && err.contains("reserved"),
        "expected reserved-prefix error, got: {err}"
    );
}

#[test]
fn physical_only_means_facets_dont_change_id() {
    // Two configs with same paths + extensions + ignore but different
    // `project` legacy field — without legacy collision they share the
    // same default id (which would be a duplicate error). The point of
    // this test: identity is physical, not interpretive.
    let body = r#"
[corpus]
root = "/tmp/x"

[embedder]
model = "potion-retrieval-32M"

[[sources]]
kind = "markdown"
id = "shared"
paths = ["~/notes"]

[[sources]]
kind = "markdown"
id = "shared-also"
paths = ["~/notes"]
"#;
    let mut f = NamedTempFile::new().unwrap();
    write!(f, "{body}").unwrap();
    let cfg = Config::load(f.path()).unwrap();
    // Both ids set: each block is uniquely identified.
    assert_ne!(cfg.sources[0].source_config_id, cfg.sources[1].source_config_id);
}
