use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};
use crate::record_rules::{RecordRule, default_record_rules};
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
    /// Optional memory-lens (p9b) tuning. Omit the `[lens]` block to
    /// accept the daemon defaults. `ostk-recall serve` maps this onto
    /// the daemon's runtime `LensConfig`.
    #[serde(default)]
    pub lens: Option<LensSettings>,
    /// Optional config-driven interpretation rules (P12). `[[record_rule]]`
    /// blocks each `pattern → action` (drop / tag `record_kind`), applied once
    /// in the pipeline overlay stage for all scanners. **Omit** the section to
    /// use the built-in [`default_record_rules`] (the apparatus filters);
    /// **present** (even an empty list) replaces the defaults entirely. See
    /// [`crate::record_rules`].
    #[serde(default, rename = "record_rule")]
    pub record_rules: Option<Vec<RecordRule>>,
    /// Optional weaver tuning. Omit to accept defaults (attenuate only
    /// `record_kind:harness_orchestration` from weaving).
    #[serde(default)]
    pub weaver: Option<WeaverSettings>,
    /// Optional rank-engine feature weights (P5), keyed by retrieval
    /// profile. Omit the `[ranking]` block to accept the compiled-in
    /// defaults (explicit recall: `rrf = 1.0`; ambient/lens:
    /// `attention_affinity = 1.0`) — these match shipped v0.6 behavior.
    /// Present (even partial) overlays the compiled-in defaults; see
    /// [`ProfileWeights::effective`].
    #[serde(default)]
    pub ranking: Option<RankingConfig>,
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

/// Memory-lens (p9b) tuning, surfaced as the optional `[lens]` block.
///
/// Mirrors the daemon's runtime `LensConfig` (in `ostk-recall-query`)
/// field-for-field. `core` can't depend on `query`, so the daemon
/// maps this onto its own type at `serve` startup; a guard test in
/// the CLI crate keeps these defaults in lock-step. Every field has a
/// serde default, so a `[lens]` block may set only the knobs it cares
/// about.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LensSettings {
    /// Token cap on the rendered lens. Excerpts truncate to fit.
    #[serde(default = "default_lens_token_budget")]
    pub token_budget: usize,
    /// Floor below which a slot is dropped rather than truncated.
    #[serde(default = "default_lens_min_excerpt_tokens")]
    pub min_excerpt_tokens: usize,
    /// Cosine *distance* threshold that triggers a refresh. `0.15` ≈
    /// 0.987 cosine similarity.
    #[serde(default = "default_lens_drift_threshold")]
    pub drift_threshold: f32,
    /// How often the background loop wakes up, in seconds.
    #[serde(default = "default_lens_poll_interval_secs")]
    pub poll_interval_secs: u64,
    /// `key:value` facet entries that exclude a chunk from the lens
    /// (privacy / status denylist), e.g. `["status:archived"]`. Defaults
    /// to attenuating operational telemetry (`record_kind:audit_significant`)
    /// from ambient surfacing — it stays fully recall-able. Set an explicit
    /// (possibly empty) list to override.
    #[serde(default = "default_lens_exclude_facets")]
    pub exclude_facets: Vec<String>,
    /// Per-lane candidate cap (the dense lane in p9b-min).
    #[serde(default = "default_lens_candidate_k_per_lane")]
    pub candidate_k_per_lane: usize,
    /// Minimum share of total score a feature must contribute to own a
    /// slot, as a fraction.
    #[serde(default = "default_lens_dominance_threshold")]
    pub dominance_threshold: f32,
    /// Refractory decay time-constant in seconds (P9b-full). A chunk
    /// included in a recent lens is penalized by
    /// `refractory_weight * exp(-Δt / refractory_tau_secs)`; ~1h default
    /// means a just-surfaced chunk is strongly suppressed and the penalty
    /// fades over a few hours, so the lens doesn't repeat itself.
    #[serde(default = "default_lens_refractory_tau_secs")]
    pub refractory_tau_secs: u64,
    /// Peak refractory penalty (subtracted from `total_score` for a chunk
    /// surfaced just now). P9b-full.
    #[serde(default = "default_lens_refractory_weight")]
    pub refractory_weight: f32,
}

// Defaults below MUST match `ostk_recall_query::lens::LensConfig::default()`.
// The `lens_settings_default_matches_query_default` test in the CLI crate
// fails loudly if they drift.
const fn default_lens_token_budget() -> usize {
    4000
}
const fn default_lens_min_excerpt_tokens() -> usize {
    200
}
fn default_lens_drift_threshold() -> f32 {
    0.15
}
const fn default_lens_poll_interval_secs() -> u64 {
    5
}
const fn default_lens_candidate_k_per_lane() -> usize {
    32
}
fn default_lens_dominance_threshold() -> f32 {
    0.30
}
const fn default_lens_refractory_tau_secs() -> u64 {
    3600
}
fn default_lens_refractory_weight() -> f32 {
    0.5
}
/// Operational telemetry is attenuated from ambient surfacing by default
/// (still fully recall-able). Keep in sync with `LensConfig::default()`.
fn default_lens_exclude_facets() -> Vec<String> {
    vec![
        "record_kind:audit_significant".to_string(),
        // RT-7: Claude Code multi-agent `<teammate-message>` orchestration
        // envelopes — apparatus, attenuated from ambient surfacing but kept
        // recall-able.
        "record_kind:harness_orchestration".to_string(),
    ]
}

impl Default for LensSettings {
    fn default() -> Self {
        Self {
            token_budget: default_lens_token_budget(),
            min_excerpt_tokens: default_lens_min_excerpt_tokens(),
            drift_threshold: default_lens_drift_threshold(),
            poll_interval_secs: default_lens_poll_interval_secs(),
            exclude_facets: default_lens_exclude_facets(),
            candidate_k_per_lane: default_lens_candidate_k_per_lane(),
            dominance_threshold: default_lens_dominance_threshold(),
            refractory_tau_secs: default_lens_refractory_tau_secs(),
            refractory_weight: default_lens_refractory_weight(),
        }
    }
}

/// Weaver tuning, surfaced as the optional `[weaver]` block (P12).
///
/// `exclude_facets` lists `key:value` facet entries whose presence makes the
/// weaver treat a chunk as apparatus — it is skipped for anchor-matching and
/// proposal seeding (generalizes the former hardcoded
/// `is_harness_apparatus` check). The default preserves pre-P12 behavior:
/// only `record_kind:harness_orchestration` is excluded. Excluding
/// `record_kind:audit_significant` from *weaving* would be a deliberate
/// thread-graph behavior change (that exclusion is established for the lens
/// only) and must be opted into explicitly.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WeaverSettings {
    #[serde(default = "default_weaver_exclude_facets")]
    pub exclude_facets: Vec<String>,
}

fn default_weaver_exclude_facets() -> Vec<String> {
    vec!["record_kind:harness_orchestration".to_string()]
}

impl Default for WeaverSettings {
    fn default() -> Self {
        Self {
            exclude_facets: default_weaver_exclude_facets(),
        }
    }
}

impl WeaverSettings {
    /// Resolve from an optional `[weaver]` block, falling back to defaults.
    #[must_use]
    pub fn resolve(slot: Option<&Self>) -> Self {
        slot.cloned().unwrap_or_default()
    }
}

/// Retrieval profile a weight map applies to (P5). `Explicit` is the
/// user-text `recall` path (lanes + rerank); `Ambient` is the
/// attention-driven recall path; `Lens` is the proactive memory-lens
/// portfolio path (P9b-full). Ambient and Lens both run without a
/// reranker — the rank features ARE the ranking — but Lens carries the
/// salience portfolio (freshness now; entity/concept later) while Ambient
/// stays attention-only, so they are distinct profiles.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RankProfile {
    Explicit,
    Ambient,
    Lens,
}

/// Compiled-in default rank-engine weights for a profile. These define
/// shipped v0.6 behavior when no `[ranking]` block is present:
/// explicit recall fuses on RRF only; ambient ranks on attention affinity.
/// P5's empirical bench tunes these; ratified weights are committed here.
#[must_use]
pub fn default_profile_weights(profile: RankProfile) -> std::collections::BTreeMap<String, f32> {
    let mut m = std::collections::BTreeMap::new();
    match profile {
        RankProfile::Explicit => {
            m.insert("rrf".to_string(), 1.0);
        }
        RankProfile::Ambient => {
            m.insert("attention_affinity".to_string(), 1.0);
        }
        // P9b-full: the lens portfolio ranks on attention affinity plus the
        // ACT-R freshness feature (P7b). freshness=0.5 vs attention=1.0 lets a
        // fresh chunk's freshness contribution clear the 0.30 slot-dominance
        // bar with headroom while attention still leads overall. Tunable via
        // `[ranking.weights.lens]`; not yet bench-validated (no ambient/lens
        // corpus run — honest seam, mirrors P5's NEUTRAL explicit-path gate).
        RankProfile::Lens => {
            m.insert("attention_affinity".to_string(), 1.0);
            m.insert("freshness".to_string(), 0.5);
        }
    }
    m
}

/// Rank-engine feature weights (P5), surfaced as the optional `[ranking]`
/// block. Known feature ids: `rrf`, `bm25`, `attention_affinity`,
/// `freshness`. Unknown ids are accepted (forward-compat for features that
/// land later) and ignored by an engine builder that doesn't know them.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RankingConfig {
    #[serde(default)]
    pub weights: ProfileWeights,
}

/// Per-profile weight maps with fall-through. The effective map for a
/// profile is the compiled-in [`default_profile_weights`] overlaid by
/// `[ranking.weights.default]`, then by the profile-specific map (profile
/// entries win; keys absent from the profile fall through).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProfileWeights {
    /// Fall-through weights applied to every profile.
    #[serde(default)]
    pub default: std::collections::BTreeMap<String, f32>,
    /// Explicit-recall overrides.
    #[serde(default)]
    pub explicit: std::collections::BTreeMap<String, f32>,
    /// Ambient-recall overrides (attention-driven, non-lens).
    #[serde(default)]
    pub ambient: std::collections::BTreeMap<String, f32>,
    /// Memory-lens portfolio overrides (P9b-full). The lens is the sole
    /// consumer of this profile today; kept separate from `ambient` so the
    /// attention-only ambient default stays pure.
    #[serde(default)]
    pub lens: std::collections::BTreeMap<String, f32>,
}

impl ProfileWeights {
    /// The effective weight map for `profile`: compiled-in defaults,
    /// overlaid by `default`, overlaid by the profile-specific map.
    #[must_use]
    pub fn effective(&self, profile: RankProfile) -> std::collections::BTreeMap<String, f32> {
        let mut m = default_profile_weights(profile);
        for (k, v) in &self.default {
            m.insert(k.clone(), *v);
        }
        let profile_map = match profile {
            RankProfile::Explicit => &self.explicit,
            RankProfile::Ambient => &self.ambient,
            RankProfile::Lens => &self.lens,
        };
        for (k, v) in profile_map {
            m.insert(k.clone(), *v);
        }
        m
    }
}

impl RankingConfig {
    /// Effective weight map for `profile`, resolving an optional
    /// `[ranking]` block against the compiled-in defaults.
    #[must_use]
    pub fn effective_weights(
        slot: Option<&Self>,
        profile: RankProfile,
    ) -> std::collections::BTreeMap<String, f32> {
        slot.map_or_else(
            || default_profile_weights(profile),
            |r| r.weights.effective(profile),
        )
    }
}

impl Config {
    /// Effective rank-engine weight map for a retrieval profile (P5).
    /// Falls back to [`default_profile_weights`] when `[ranking]` is omitted.
    #[must_use]
    pub fn effective_ranking_weights(
        &self,
        profile: RankProfile,
    ) -> std::collections::BTreeMap<String, f32> {
        RankingConfig::effective_weights(self.ranking.as_ref(), profile)
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
    /// Operator-supplied facet overrides (P1). Merged into each emitted
    /// chunk's `facets` via per-key cardinality (single replaces, multi
    /// unions). Empty-list sentinel `key = []` clears a multi-cardinality
    /// scanner-emitted set.
    ///
    /// Facet keys participate in `embedding_input_sha256` only if they
    /// are in `EMBED_FACET_ALLOWLIST` — other keys are filter-only and
    /// don't trigger re-embed.
    #[serde(default)]
    pub facets: std::collections::BTreeMap<String, Vec<String>>,
    /// Operator-supplied physical-identity discriminator. When two
    /// `[[sources]]` blocks share the same `(kind, paths, extensions,
    /// ignore)` shape, the default `source_config_id` would collide; set
    /// `id = "..."` on each to disambiguate. Reserved prefix `synthetic:`
    /// is rejected at parse (it routes to `Pipeline::ingest_synthetic`).
    #[serde(default)]
    pub id: Option<String>,
    /// Computed at config parse from `(kind, paths, extensions, ignore,
    /// optional legacy project)` via [`compute_source_config_id`]. Always
    /// non-empty after `Config::load` / `Config::validate` returns. Never
    /// read from the TOML.
    #[serde(skip)]
    pub source_config_id: String,
}

/// Reserved prefix for synthetic-ingest `source_config_id` values.
/// User `[[sources]].id` starting with this is rejected at parse so
/// scanner-driven and `Pipeline::ingest_synthetic` chunks never collide.
pub const SYNTHETIC_SOURCE_CONFIG_ID_PREFIX: &str = "synthetic:";

impl Config {
    /// Load and validate config from disk. Mutates each `[[sources]]` block
    /// to populate `source_config_id` (always non-empty when this returns
    /// Ok).
    pub fn load(path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path)?;
        let mut cfg: Self = toml::from_str(&text).map_err(|e| Error::Config(e.to_string()))?;
        cfg.validate_and_seal()?;
        Ok(cfg)
    }

    /// Validate the parsed config AND finalize each source's
    /// `source_config_id`. Use this from any caller that builds a `Config`
    /// programmatically (tests, in-process synthesis) instead of calling
    /// `validate` directly.
    pub fn validate_and_seal(&mut self) -> Result<()> {
        // First, basic per-source checks + explicit-id sanity.
        for (i, s) in self.sources.iter().enumerate() {
            if s.paths.is_empty() {
                return Err(Error::Config(format!(
                    "sources[{i}] (kind={}) has no paths",
                    s.kind.as_str()
                )));
            }
            if let Some(explicit) = &s.id {
                if explicit.is_empty() {
                    return Err(Error::Config(format!(
                        "sources[{i}] (kind={}) has empty `id`",
                        s.kind.as_str()
                    )));
                }
                if explicit.starts_with(SYNTHETIC_SOURCE_CONFIG_ID_PREFIX) {
                    return Err(Error::Config(format!(
                        "sources[{i}] (kind={}) has reserved `id` prefix `{}` — \
                         that prefix is reserved for `Pipeline::ingest_synthetic` chunks",
                        s.kind.as_str(),
                        SYNTHETIC_SOURCE_CONFIG_ID_PREFIX,
                    )));
                }
            }
        }

        // Detect physical-shape duplicates BEFORE legacy-discriminator
        // disambiguation, so the v0.5 legacy-project upgrade path is the
        // ONLY thing that lets two same-shape blocks coexist by default.
        // Operators wanting two blocks over the same physical scan shape
        // (without a legacy `project`) must set `id = "..."` explicitly.
        let has_legacy_collision = legacy_project_collision(&self.sources);

        for s in &mut self.sources {
            s.source_config_id = compute_source_config_id(s, has_legacy_collision);
            debug_assert!(
                !s.source_config_id.is_empty(),
                "source_config_id must be non-empty after parse"
            );
        }

        // After computing, refuse default-id collisions: same id from two
        // blocks WITHOUT operator-supplied disambiguation is a bug.
        let mut seen: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
        for (i, s) in self.sources.iter().enumerate() {
            if let Some(prev) = seen.insert(s.source_config_id.clone(), i) {
                let a = &self.sources[prev];
                return Err(Error::Config(format!(
                    "config error: source blocks {prev} and {i} share physical identity\n  \
                     (kind=\"{}\", paths={:?}, extensions={:?}, ignore={:?})\n  \
                     Either merge them, distinguish by `id = \"...\"` on each block,\n  \
                     or move their differing facets to a single block via the facets override.",
                    a.kind.as_str(),
                    a.paths,
                    a.extensions,
                    a.ignore,
                )));
            }
        }
        Ok(())
    }

    /// Back-compat alias retained for callers that only want syntactic
    /// validation (no mutation). Prefer [`Config::validate_and_seal`] for
    /// any new caller — it leaves `source_config_id` populated so the
    /// chunk_id formula has its discriminator.
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

    /// The effective record-rule set (P12): the operator's `[[record_rule]]`
    /// list verbatim when present (even an empty list = "explicitly no rules"),
    /// else the built-in [`default_record_rules`]. Pipeline construction is fed
    /// this — there is no silent no-op fallback on the production path.
    #[must_use]
    pub fn effective_record_rules(&self) -> Vec<RecordRule> {
        self.record_rules
            .clone()
            .unwrap_or_else(default_record_rules)
    }
}

/// Compute a physical-identity discriminator for a single source block.
///
/// `id` (when set) wins. Otherwise the formula is:
/// `blake3-16(kind || '|' || paths || '|' || extensions || '|' || ignore || '|'
/// || legacy_project)` where every sequence is sorted for stability.
///
/// The legacy project segment is empty UNLESS the operator has two blocks
/// with identical physical shape that differ only by `[[sources]].project`
/// (v0.5 transitional discriminator — removed at v0.7). The
/// `has_legacy_collision` flag drives this: when true, any block carrying
/// a legacy `project` participates with its project as the last segment.
#[must_use]
pub fn compute_source_config_id(cfg: &SourceConfig, has_legacy_collision: bool) -> String {
    if let Some(explicit) = &cfg.id {
        return explicit.clone();
    }
    let mut h = blake3::Hasher::new();
    h.update(cfg.kind.as_str().as_bytes());
    h.update(b"|");
    let mut paths = cfg.paths.clone();
    paths.sort();
    for p in &paths {
        h.update(p.as_bytes());
        h.update(b",");
    }
    h.update(b"|");
    let mut exts = cfg.extensions.clone();
    exts.sort();
    for e in &exts {
        h.update(e.as_bytes());
        h.update(b",");
    }
    h.update(b"|");
    let mut ignores = cfg.ignore.clone();
    ignores.sort();
    for ig in &ignores {
        h.update(ig.as_bytes());
        h.update(b",");
    }
    h.update(b"|");
    let legacy_proj = if has_legacy_collision {
        cfg.project.as_deref().unwrap_or("")
    } else {
        ""
    };
    h.update(legacy_proj.as_bytes());
    let hash = h.finalize();
    hex::encode(&hash.as_bytes()[..16])
}

/// Detect whether two or more `[[sources]]` blocks share physical shape
/// AND differ only by their legacy `project` field. Used to opt those
/// blocks into the v0.5 → v0.6 transitional discriminator so existing
/// configs upgrade without forced explicit-id edits.
fn legacy_project_collision(sources: &[SourceConfig]) -> bool {
    // Hash each block WITHOUT the legacy project segment.
    let mut seen: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
    for (i, s) in sources.iter().enumerate() {
        if s.project.is_none() || s.id.is_some() {
            continue;
        }
        // Recompute the no-project shape hash for comparison.
        let mut tmp = s.clone();
        tmp.project = None;
        let shape_id = compute_source_config_id(&tmp, false);
        if let Some(prev) = seen.insert(shape_id, i) {
            let _ = prev;
            return true;
        }
    }
    false
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

    #[test]
    fn ranking_omitted_uses_compiled_defaults() {
        let cfg: Config = toml::from_str(SAMPLE).unwrap();
        assert!(cfg.ranking.is_none());
        let explicit = cfg.effective_ranking_weights(RankProfile::Explicit);
        assert_eq!(explicit.get("rrf").copied(), Some(1.0));
        assert_eq!(explicit.len(), 1, "explicit default is rrf only");
        let ambient = cfg.effective_ranking_weights(RankProfile::Ambient);
        assert_eq!(ambient.get("attention_affinity").copied(), Some(1.0));
        assert_eq!(ambient.len(), 1, "ambient default is attention only");
        // P9b-full: the lens profile carries the salience portfolio
        // (attention + freshness) while ambient stays attention-only.
        let lens = cfg.effective_ranking_weights(RankProfile::Lens);
        assert_eq!(lens.get("attention_affinity").copied(), Some(1.0));
        assert_eq!(lens.get("freshness").copied(), Some(0.5));
        assert_eq!(lens.len(), 2, "lens default is attention + freshness");
    }

    #[test]
    fn ranking_profile_overlays_default_and_compiled() {
        let body = r#"
[corpus]
root = "/tmp"

[embedder]
model = "x"

[ranking.weights.default]
freshness = 0.5

[ranking.weights.explicit]
bm25 = 2.0
rrf = 0.25
"#;
        let cfg: Config = toml::from_str(body).unwrap();
        let explicit = cfg.effective_ranking_weights(RankProfile::Explicit);
        // compiled rrf=1.0 overridden by explicit rrf=0.25
        assert_eq!(explicit.get("rrf").copied(), Some(0.25));
        // default freshness=0.5 falls through (no explicit override)
        assert_eq!(explicit.get("freshness").copied(), Some(0.5));
        // explicit-only bm25
        assert_eq!(explicit.get("bm25").copied(), Some(2.0));

        // Ambient sees the default overlay but not the explicit map.
        let ambient = cfg.effective_ranking_weights(RankProfile::Ambient);
        assert_eq!(ambient.get("attention_affinity").copied(), Some(1.0));
        assert_eq!(ambient.get("freshness").copied(), Some(0.5));
        assert!(
            !ambient.contains_key("bm25"),
            "explicit-only weight must not leak into ambient"
        );
    }

    #[test]
    fn ranking_lens_profile_overlays_and_isolates() {
        let body = r#"
[corpus]
root = "/tmp"

[embedder]
model = "x"

[ranking.weights.lens]
freshness = 0.8
attention_affinity = 0.6

[ranking.weights.ambient]
attention_affinity = 0.2
"#;
        let cfg: Config = toml::from_str(body).unwrap();
        let lens = cfg.effective_ranking_weights(RankProfile::Lens);
        // compiled lens defaults (attention=1.0, freshness=0.5) overridden by
        // the [ranking.weights.lens] map.
        assert_eq!(lens.get("attention_affinity").copied(), Some(0.6));
        assert_eq!(lens.get("freshness").copied(), Some(0.8));
        // The ambient override must NOT leak into the lens profile (and the
        // lens override must not leak into ambient): they are isolated.
        let ambient = cfg.effective_ranking_weights(RankProfile::Ambient);
        assert_eq!(ambient.get("attention_affinity").copied(), Some(0.2));
        assert!(
            !ambient.contains_key("freshness"),
            "lens-only freshness override must not leak into ambient"
        );
    }

    #[test]
    fn ranking_block_with_lens_round_trips() {
        let body = r#"
[corpus]
root = "/tmp"

[embedder]
model = "x"

[ranking.weights.lens]
freshness = 0.7
"#;
        let cfg: Config = toml::from_str(body).unwrap();
        let reser = toml::to_string(&cfg).unwrap();
        let cfg2: Config = toml::from_str(&reser).unwrap();
        assert_eq!(
            cfg2.effective_ranking_weights(RankProfile::Lens),
            cfg.effective_ranking_weights(RankProfile::Lens),
            "lens weight map must survive a serialize/deserialize round-trip"
        );
        assert_eq!(
            cfg2.ranking.unwrap().weights.lens.get("freshness").copied(),
            Some(0.7)
        );
    }

    #[test]
    fn ranking_block_round_trips() {
        let body = r#"
[corpus]
root = "/tmp"

[embedder]
model = "x"

[ranking.weights.ambient]
attention_affinity = 0.8
freshness = 0.3
"#;
        let cfg: Config = toml::from_str(body).unwrap();
        let r = cfg.ranking.as_ref().unwrap();
        assert_eq!(r.weights.ambient.get("attention_affinity").copied(), Some(0.8));
        assert_eq!(r.weights.ambient.get("freshness").copied(), Some(0.3));
        // Re-serialize and re-parse: stable.
        let toml_out = toml::to_string(&cfg).unwrap();
        let cfg2: Config = toml::from_str(&toml_out).unwrap();
        assert_eq!(
            cfg2.effective_ranking_weights(RankProfile::Ambient),
            cfg.effective_ranking_weights(RankProfile::Ambient)
        );
    }
}
