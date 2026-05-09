//! Shared types for ostk-recall: `Chunk`, `Source`, `Scanner`, configuration.
//!
//! Every other crate in the workspace depends on this one and nothing else
//! from the workspace. Kept small and stable.

use serde::{Deserialize, Serialize};

pub mod chunk;
pub mod config;
pub mod error;
pub mod scanner;
pub mod source;
pub mod types;

pub use chunk::{Chunk, Links};
pub use config::{Config, CorpusConfig, EmbedderConfig, RerankerConfig, SourceConfig, WatchConfig};
pub use error::{Error, Result};
pub use scanner::{Scanner, SourceItem};
pub use source::{RetentionPolicy, Source, SourceKind};
pub use types::{
    AuditResult, RecallHit, RecallLinkResult, RecallParams, RecallStats, RerankerStats,
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
