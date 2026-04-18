//! Shared types for ostk-recall: `Chunk`, `Source`, `Scanner`, configuration.
//!
//! Every other crate in the workspace depends on this one and nothing else
//! from the workspace. Kept small and stable.

pub mod chunk;
pub mod config;
pub mod error;
pub mod scanner;
pub mod source;

pub use chunk::{Chunk, Links};
pub use config::{Config, CorpusConfig, EmbedderConfig, RerankerConfig, SourceConfig};
pub use error::{Error, Result};
pub use scanner::{Scanner, SourceItem};
pub use source::{Source, SourceKind};
