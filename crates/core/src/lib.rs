//! Shared types for ostk-recall: `Chunk`, `Source`, `Scanner`, configuration.
//!
//! Every other crate in the workspace depends on this one and nothing else
//! from the workspace. Kept small and stable.

use serde::{Deserialize, Serialize};

pub mod attention;
pub mod chunk;
pub mod config;
pub mod error;
pub mod facets;
pub mod scanner;
pub mod source;
pub mod types;

pub use attention::{
    AttentionPage, AttentionScope, FoldDepth, IngestEvent, PrivacyTier, ScoreAttribution,
    ThreadHandle, ThreadHandleError,
};
pub use chunk::{Chunk, Links};
pub use facets::{
    ALLOWLIST_VERSION, Cardinality, EMBED_FACET_ALLOWLIST, FacetSet, HEADER_FORMAT_VERSION,
    cardinality_of, cfg_overlay_hash, compose_header, filter_to_allowlist, from_list,
    is_valid_facet_key, merge_override, to_list,
};
pub use config::{
    Config, CorpusConfig, EmbedderConfig, RerankerConfig, RuntimeConfig, SourceConfig,
    SYNTHETIC_SOURCE_CONFIG_ID_PREFIX, WatchConfig, WatchMode, compute_source_config_id,
    default_worker_threads,
};
pub use error::{Error, Result};
pub use scanner::{Scanner, SourceItem};
pub use source::{RetentionPolicy, Source, SourceKind};
pub use types::{
    AttentionBiasParams, AuditResult, RecallHit, RecallLinkResult, RecallParams, RecallStats,
    RerankerStats,
    SourceCount, SynthesizedPage,
};

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum RecallIntent {
    /// Prioritizes definitions, `SourceKind::Code`, and symbol-bounded chunks.
    Symbol,
    /// Prioritizes \"Why\" logic, `SourceKind::Markdown`, and project specs.
    Narrative,
    /// Prioritizes execution evidence, `SourceKind::Probe`, and error logs.
    Trace,
    /// The default balanced hybrid weight.
    #[default]
    General,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ContextRole {
    /// The \"Current Truth.\" Usually a fresh code definition or a ratified spec.
    Primary,
    /// The \"Lineage.\" A stale version of the primary hit showing what changed.
    Evolution,
    /// The \"Evidence.\" Transcript mentions or probes showing how the primary hit behaves in the wild.
    Usage,
}
