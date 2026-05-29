//! P0 gate: v0.5 → v0.6 upgrade path. When two `[[sources]]` blocks share
//! physical scan shape AND each carries a different legacy `project = "..."`
//! field, the discriminator differentiates them — operators don't need to
//! immediately set explicit `id = "..."` to upgrade.
//!
//! Removed at v0.7 when the legacy `project` scalar is fully deprecated.

use std::io::Write;
use tempfile::NamedTempFile;

use ostk_recall_core::Config;

#[test]
fn legacy_project_discriminates_same_physical_shape() {
    let body = r#"
[corpus]
root = "/tmp/x"

[embedder]
model = "potion-retrieval-32M"

[[sources]]
kind = "markdown"
project = "alpha"
paths = ["~/notes"]

[[sources]]
kind = "markdown"
project = "beta"
paths = ["~/notes"]
"#;
    let mut f = NamedTempFile::new().unwrap();
    write!(f, "{body}").unwrap();
    let cfg = Config::load(f.path()).expect("legacy discriminator should accept");
    assert_ne!(
        cfg.sources[0].source_config_id, cfg.sources[1].source_config_id,
        "v0.5 upgrade path: differing `project` differentiates same-shape blocks"
    );
}

#[test]
fn single_block_with_legacy_project_is_stable() {
    let body = r#"
[corpus]
root = "/tmp/x"

[embedder]
model = "potion-retrieval-32M"

[[sources]]
kind = "markdown"
project = "notes"
paths = ["~/notes"]
"#;
    let mut f = NamedTempFile::new().unwrap();
    write!(f, "{body}").unwrap();
    let cfg = Config::load(f.path()).unwrap();
    let id_first = cfg.sources[0].source_config_id.clone();
    // Reload the same config; id must be stable.
    let cfg2 = Config::load(f.path()).unwrap();
    assert_eq!(id_first, cfg2.sources[0].source_config_id);
    assert!(!id_first.is_empty());
}
