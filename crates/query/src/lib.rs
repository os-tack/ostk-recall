//! ostk-recall-query — vector + lexical retrieval against the corpus store.

pub mod audit;
pub mod error;
pub mod hybrid;
pub mod link;
pub mod rerank;
mod row;
pub mod stats;
pub mod synthesis;
pub mod types;

pub use hybrid::recall;
pub use ostk_recall_core::{RecallIntent, Chunk, Source, Links};
pub use ostk_recall_store::CorpusStore;
pub use ostk_recall_pipeline::ChunkEmbedder;
pub use error::{QueryError, Result};
pub use rerank::{Reranker, RerankerError, RerankerLike};
pub use synthesis::{SynthesizedPage, Synthesizer};
pub use types::{
    AuditResult, RecallHit, RecallLinkResult, RecallParams, RecallStats, RerankerStats, SourceCount,
};
