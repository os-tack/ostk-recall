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
    /// Optional cross-encoder reranker. Omit to keep the default
    /// (enabled, default model). Set `enabled = false` to skip the
    /// rerank pass entirely.
    #[serde(default)]
    pub reranker: Option<RerankerConfig>,
    /// Optional file-watcher tuning. Omit to disable. The watcher
    /// itself is always opt-in (`enabled = true`); when omitted or
    /// disabled, `ostk-recall watch` exits at startup.
    #[serde(default)]
    pub watch: Option<WatchConfig>,
    /// Optional runtime resource caps. Omit to accept the polite
    /// defaults (small thread pool sized for a background substrate).
    /// Power users running a one-shot CLI can raise these.
    #[serde(default)]
    pub runtime: Option<RuntimeConfig>,
}

/// Runtime resource caps. Each field is the upper bound on the
/// matching subsystem's worker pool; absent fields fall back to the
/// `OSTK_RECALL_WORKERS` env var (if set), otherwise [`default_worker_threads`].
///
/// All three pools (tokio runtime, DataFusion query parallelism, rayon
/// compute) get set to the *same* effective value at process startup
/// — no need to tune them independently in normal use, and keeping
/// them aligned avoids over-subscription when lance internally
/// pipelines work across all three.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RuntimeConfig {
    /// Upper bound on concurrent worker threads (tokio +
    /// DataFusion + rayon). The substrate is designed to run in the
    /// background while a human or agent uses the same machine; the
    /// default is intentionally low. Setting this to a value
    /// approaching `num_cpus()` will pin the machine during scans.
    #[serde(default)]
    pub worker_threads: Option<usize>,
}

/// Default cap when neither env var nor config specifies one.
/// Sized to leave plenty of cores for the human/agent on a typical
/// laptop, while still parallelizing the substrate's own work.
#[must_use]
pub const fn default_worker_threads() -> usize {
    4
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

/// Cross-encoder reranker configuration. The reranker is opt-out:
/// omitting the `[reranker]` block means "enabled, default model".
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RerankerConfig {
    /// When false, hybrid recall returns RRF-fused order without a
    /// second cross-encoder pass.
    #[serde(default = "default_reranker_enabled")]
    pub enabled: bool,
    /// fastembed reranker model identifier. Accepted aliases:
    /// `jina-reranker-v1-turbo-en` (default), `jina-reranker-v2-base-multilingual`,
    /// `bge-reranker-base`, `bge-reranker-v2-m3`. The legacy
    /// `ms-marco-MiniLM-L-6-v2` alias maps to the JINA Turbo English
    /// model (closest equivalent in fastembed v5).
    #[serde(default = "default_reranker_model")]
    pub model: String,
}

const fn default_reranker_enabled() -> bool {
    true
}

fn default_reranker_model() -> String {
    "jina-reranker-v1-turbo-en".to_string()
}

impl Default for RerankerConfig {
    fn default() -> Self {
        Self {
            enabled: default_reranker_enabled(),
            model: default_reranker_model(),
        }
    }
}

impl RerankerConfig {
    /// Resolve the effective reranker config: returns the explicit
    /// block when present, otherwise the default (enabled).
    #[must_use]
    pub fn resolve(slot: Option<&Self>) -> Self {
        slot.cloned().unwrap_or_default()
    }
}

/// File-watcher tuning.
///
/// The watcher reuses each `[[sources]].paths` (and `extensions`) — it does
/// not declare its own paths. When a debounced batch contains any event
/// under a watched source path, the watcher pokes the scan-trigger socket
/// once. The scan does the real filtering; the watcher's only job is "did
/// anything we care about change recently?".
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WatchConfig {
    /// Opt-in switch. When false (or `[watch]` omitted), `ostk-recall
    /// watch` exits at startup.
    #[serde(default)]
    pub enabled: bool,
    /// Coalescing window in milliseconds. Longer windows trade trigger
    /// latency for fewer redundant scans during bursty edits.
    #[serde(default = "default_debounce_ms")]
    pub debounce_ms: u64,
    /// Optional override for the trigger socket path. Defaults to
    /// `corpus.root/recall.sock` — same path `serve` binds to.
    #[serde(default)]
    pub socket: Option<String>,
    /// Optional `[[sources]].project` allowlist. When empty, every
    /// source is watched. When set, only sources whose `project` matches
    /// drive the watcher (lets you skip noisy sources like `zip_export`
    /// or `claude_code` that scan often anyway).
    #[serde(default)]
    pub projects: Vec<String>,
    /// Trigger wire mode. `legacy` (default this release) sends
    /// connect-and-close pokes — server runs a full scan. `incremental`
    /// sends the debounced changed paths over the socket as line-delimited
    /// UTF-8; the server scans only those paths via `Pipeline::scan_paths`.
    #[serde(default)]
    pub mode: WatchMode,
}

/// Wire-format selector for the watcher → serve scan-trigger socket.
///
/// Default is [`WatchMode::Legacy`] for the first release that ships path
/// frames so users opt in explicitly. The default flips to `Incremental`
/// in a follow-up release once the per-path scan path bakes.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase", deny_unknown_fields)]
pub enum WatchMode {
    /// Connect-and-close. Server treats this as "scan all sources" — the
    /// behavior shipped before path-aware triggers existed. Reading the
    /// stream on the server side observes empty body and falls back to
    /// the legacy full-pipeline scan.
    #[default]
    Legacy,
    /// Send the debounced changed paths over the socket as line-delimited
    /// UTF-8, terminated by EOF. The server reads the frame and dispatches
    /// to `Pipeline::scan_paths` for a per-path scan.
    Incremental,
}

/// Per-platform debounce defaults. The OS event backends differ enough
/// in granularity and latency that a single number is wrong somewhere:
///
/// - **Linux (inotify)** — sub-ms per-event, 5–10 events per editor save.
///   Debounce is doing real coalescing; 800 ms catches IntelliJ-style
///   "save the world" bursts without making interactive workflows feel
///   sluggish.
/// - **macOS (`FSEvents`)** — the kernel already coalesces at ~30 ms;
///   notify-debouncer-full receives mostly-batched events. Lower windows
///   have no upside (`FSEvents` floor dominates) and risk split bursts on
///   slow disks. 1500 ms is the safe middle.
/// - **Windows (`ReadDirectoryChangesW`)** — per-event with IOCP batching
///   at the OS layer; AV filter drivers (Defender et al.) can stretch
///   delivery another 100–200 ms. 1200 ms absorbs that without going
///   macOS-conservative.
///
/// Other targets fall back to the macOS value (the most conservative).
/// Users always override via `[watch].debounce_ms` in config.
const fn default_debounce_ms() -> u64 {
    if cfg!(target_os = "linux") {
        800
    } else if cfg!(target_os = "windows") {
        1200
    } else {
        // macos + everything else: 1500 ms.
        1500
    }
}

impl Default for WatchConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            debounce_ms: default_debounce_ms(),
            socket: None,
            projects: Vec::new(),
            mode: WatchMode::default(),
        }
    }
}

impl WatchConfig {
    /// True when `[watch]` is present AND `enabled = true`. Used by the
    /// watcher subcommand entry point to bail early on misconfiguration
    /// rather than silently doing nothing.
    #[must_use]
    pub const fn is_active(&self) -> bool {
        self.enabled
    }

    /// Resolve the trigger socket path. Falls back to
    /// `corpus_root/recall.sock` (the same default `serve` binds to)
    /// when no override is set.
    pub fn resolve_socket(&self, corpus_root: &Path) -> Result<PathBuf> {
        self.socket
            .as_ref()
            .map_or_else(|| Ok(corpus_root.join("recall.sock")), |s| expand_path(s))
    }

    /// True when this source's `project` should be watched given the
    /// configured `projects` allowlist. Empty allowlist = watch all.
    #[must_use]
    pub fn watches_project(&self, project: Option<&str>) -> bool {
        if self.projects.is_empty() {
            return true;
        }
        project.is_some_and(|p| self.projects.iter().any(|s| s == p))
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SourceConfig {
    pub kind: SourceKind,
    #[serde(default)]
    pub project: Option<String>,
    #[serde(default)]
    pub paths: Vec<String>,
    /// Per-source ignore patterns layered on top of standard `.gitignore`,
    /// `.ignore`, and `.ostk-recall-ignore` handling. Globs follow the same
    /// syntax as `.gitignore` (`vendor/**`, `**/__pycache__/**`, etc.) and
    /// are anchored at each path root. Underlying engine is the `ignore`
    /// crate (the same one ripgrep uses).
    #[serde(default)]
    pub ignore: Vec<String>,
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

    #[test]
    fn parses_ignore_patterns() {
        let body = r#"
[corpus]
root = "/tmp"

[embedder]
model = "x"

[[sources]]
kind = "code"
project = "p"
paths = ["~/projects/foo"]
ignore = ["vendor/**", "fixtures/"]
extensions = ["rs"]
"#;
        let cfg: Config = toml::from_str(body).unwrap();
        assert_eq!(cfg.sources.len(), 1);
        assert_eq!(cfg.sources[0].ignore, vec!["vendor/**", "fixtures/"]);
    }

    #[test]
    fn reranker_omitted_means_default_enabled() {
        let cfg: Config = toml::from_str(SAMPLE).unwrap();
        let r = RerankerConfig::resolve(cfg.reranker.as_ref());
        assert!(r.enabled);
        assert_eq!(r.model, "jina-reranker-v1-turbo-en");
    }

    #[test]
    fn reranker_explicit_disabled() {
        let body = r#"
[corpus]
root = "/tmp"

[embedder]
model = "x"

[reranker]
enabled = false
model = "bge-reranker-base"
"#;
        let cfg: Config = toml::from_str(body).unwrap();
        let r = cfg.reranker.unwrap();
        assert!(!r.enabled);
        assert_eq!(r.model, "bge-reranker-base");
    }

    #[test]
    fn reranker_block_partial_uses_defaults() {
        let body = r#"
[corpus]
root = "/tmp"

[embedder]
model = "x"

[reranker]
enabled = false
"#;
        let cfg: Config = toml::from_str(body).unwrap();
        let r = cfg.reranker.unwrap();
        assert!(!r.enabled);
        // model field omitted → default kicks in via serde default fn.
        assert_eq!(r.model, "jina-reranker-v1-turbo-en");
    }

    #[test]
    fn watch_omitted_means_disabled() {
        let cfg: Config = toml::from_str(SAMPLE).unwrap();
        assert!(cfg.watch.is_none());
        let resolved = cfg.watch.unwrap_or_default();
        assert!(!resolved.is_active());
        assert_eq!(resolved.debounce_ms, default_debounce_ms());
    }

    #[test]
    fn debounce_default_per_platform() {
        // Pin the per-platform defaults so a tuning change is a deliberate
        // diff with this test, not a silent drift.
        let got = default_debounce_ms();
        let expected = if cfg!(target_os = "linux") {
            800
        } else if cfg!(target_os = "windows") {
            1200
        } else {
            1500
        };
        assert_eq!(got, expected);
    }

    #[test]
    fn watch_block_round_trip() {
        let body = r#"
[corpus]
root = "/tmp"

[embedder]
model = "x"

[watch]
enabled = true
debounce_ms = 800
projects = ["notes", "haystack"]
"#;
        let cfg: Config = toml::from_str(body).unwrap();
        let w = cfg.watch.unwrap();
        assert!(w.is_active());
        assert_eq!(w.debounce_ms, 800);
        assert_eq!(w.projects, vec!["notes", "haystack"]);
        assert!(w.socket.is_none());
    }

    #[test]
    fn watch_socket_override_expands() {
        let body = r#"
[corpus]
root = "/tmp"

[embedder]
model = "x"

[watch]
enabled = true
socket = "~/.local/share/ostk-recall/custom.sock"
"#;
        let cfg: Config = toml::from_str(body).unwrap();
        let w = cfg.watch.unwrap();
        let p = w.resolve_socket(Path::new("/tmp")).unwrap();
        assert!(!p.to_string_lossy().starts_with('~'));
        assert!(p.to_string_lossy().ends_with("custom.sock"));
    }

    #[test]
    fn watch_socket_default_under_corpus_root() {
        let w = WatchConfig {
            enabled: true,
            ..Default::default()
        };
        let p = w.resolve_socket(Path::new("/var/data/recall")).unwrap();
        assert_eq!(p, PathBuf::from("/var/data/recall/recall.sock"));
    }

    #[test]
    fn watch_project_filter_matches() {
        let w = WatchConfig {
            enabled: true,
            projects: vec!["notes".to_string()],
            ..Default::default()
        };
        assert!(w.watches_project(Some("notes")));
        assert!(!w.watches_project(Some("haystack")));
        assert!(!w.watches_project(None));
    }

    #[test]
    fn watch_empty_filter_watches_all() {
        let w = WatchConfig {
            enabled: true,
            ..Default::default()
        };
        assert!(w.watches_project(Some("anything")));
        assert!(w.watches_project(None));
    }

    #[test]
    fn watch_rejects_unknown_field() {
        let body = r#"
[corpus]
root = "/tmp"

[embedder]
model = "x"

[watch]
enabled = true
mystery = 7
"#;
        let err = toml::from_str::<Config>(body).unwrap_err().to_string();
        assert!(
            err.contains("mystery") || err.contains("unknown field"),
            "got: {err}"
        );
    }

    #[test]
    fn ignore_defaults_to_empty() {
        let body = r#"
[corpus]
root = "/tmp"

[embedder]
model = "x"

[[sources]]
kind = "code"
project = "p"
paths = ["~/projects/foo"]
extensions = ["rs"]
"#;
        let cfg: Config = toml::from_str(body).unwrap();
        assert!(cfg.sources[0].ignore.is_empty());
    }

    #[test]
    fn watch_mode_defaults_to_legacy() {
        let body = r#"
[corpus]
root = "/tmp"

[embedder]
model = "x"

[watch]
enabled = true
"#;
        let cfg: Config = toml::from_str(body).unwrap();
        let w = cfg.watch.unwrap();
        assert_eq!(w.mode, WatchMode::Legacy);
    }

    #[test]
    fn watch_mode_legacy_round_trip() {
        let body = r#"
[corpus]
root = "/tmp"

[embedder]
model = "x"

[watch]
enabled = true
mode = "legacy"
"#;
        let cfg: Config = toml::from_str(body).unwrap();
        assert_eq!(cfg.watch.unwrap().mode, WatchMode::Legacy);
    }

    #[test]
    fn watch_mode_incremental_round_trip() {
        let body = r#"
[corpus]
root = "/tmp"

[embedder]
model = "x"

[watch]
enabled = true
mode = "incremental"
"#;
        let cfg: Config = toml::from_str(body).unwrap();
        assert_eq!(cfg.watch.unwrap().mode, WatchMode::Incremental);
    }

    #[test]
    fn watch_mode_unknown_value_rejected() {
        let body = r#"
[corpus]
root = "/tmp"

[embedder]
model = "x"

[watch]
enabled = true
mode = "burst"
"#;
        let err = toml::from_str::<Config>(body).unwrap_err().to_string();
        assert!(
            err.contains("burst") || err.contains("unknown") || err.contains("variant"),
            "got: {err}"
        );
    }
}
