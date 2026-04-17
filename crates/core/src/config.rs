use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};
use crate::source::SourceKind;

/// Top-level configuration loaded from `config.toml`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    pub corpus: CorpusConfig,
    pub embedder: EmbedderConfig,
    #[serde(default, rename = "sources")]
    pub sources: Vec<SourceConfig>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CorpusConfig {
    /// Corpus root. Will be shell-expanded (`~` and `$VAR`).
    pub root: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EmbedderConfig {
    /// model2vec-rs model id, e.g. `potion-retrieval-32M`.
    pub model: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SourceConfig {
    pub kind: SourceKind,
    #[serde(default)]
    pub project: Option<String>,
    #[serde(default)]
    pub paths: Vec<String>,
    #[serde(default)]
    pub extensions: Vec<String>,
}

impl Config {
    /// Load and validate config from disk.
    pub fn load(path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path)?;
        let cfg: Self = toml::from_str(&text).map_err(|e| Error::Config(e.to_string()))?;
        cfg.validate()?;
        Ok(cfg)
    }

    pub fn validate(&self) -> Result<()> {
        for (i, s) in self.sources.iter().enumerate() {
            if s.paths.is_empty() {
                return Err(Error::Config(format!(
                    "sources[{i}] (kind={}) has no paths",
                    s.kind.as_str()
                )));
            }
        }
        Ok(())
    }

    /// Expand the corpus root (`~` → home, `$VAR` → env).
    pub fn expanded_root(&self) -> Result<PathBuf> {
        expand_path(&self.corpus.root)
    }
}

impl SourceConfig {
    /// Expand every declared path. Order preserved.
    pub fn expanded_paths(&self) -> Result<Vec<PathBuf>> {
        self.paths.iter().map(|p| expand_path(p)).collect()
    }
}

pub fn expand_path(raw: &str) -> Result<PathBuf> {
    let expanded = shellexpand::full(raw).map_err(|e| Error::PathExpand(e.to_string()))?;
    Ok(PathBuf::from(expanded.into_owned()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;

    const SAMPLE: &str = r#"
[corpus]
root = "~/.local/share/ostk-recall"

[embedder]
model = "potion-retrieval-32M"

[[sources]]
kind = "markdown"
project = "notes"
paths = ["~/notes"]
"#;

    #[test]
    fn loads_valid_config() {
        use std::io::Write;
        let mut f = NamedTempFile::new().unwrap();
        write!(f, "{SAMPLE}").unwrap();
        let cfg = Config::load(f.path()).unwrap();
        assert_eq!(cfg.sources.len(), 1);
        assert_eq!(cfg.embedder.model, "potion-retrieval-32M");
    }

    #[test]
    fn rejects_unknown_fields() {
        let bad = r#"
[corpus]
root = "/tmp"
mystery = true

[embedder]
model = "x"
"#;
        let err = toml::from_str::<Config>(bad).unwrap_err().to_string();
        assert!(err.contains("mystery") || err.contains("unknown field"));
    }

    #[test]
    fn rejects_empty_source_paths() {
        let bad = r#"
[corpus]
root = "/tmp"

[embedder]
model = "x"

[[sources]]
kind = "markdown"
paths = []
"#;
        let cfg: Config = toml::from_str(bad).unwrap();
        let err = cfg.validate().unwrap_err();
        assert!(err.to_string().contains("no paths"));
    }

    #[test]
    fn expands_tilde() {
        let p = expand_path("~/foo").unwrap();
        assert!(!p.to_string_lossy().starts_with('~'));
    }
}
