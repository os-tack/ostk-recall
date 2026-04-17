use std::path::PathBuf;

use crate::chunk::Chunk;
use crate::config::SourceConfig;
use crate::error::Result;
use crate::source::SourceKind;

/// A discovered artifact, ready to be parsed into chunks.
#[derive(Debug, Clone)]
pub struct SourceItem {
    pub source_id: String,
    pub path: Option<PathBuf>,
    pub project: Option<String>,
    pub bytes: Option<Vec<u8>>,
}

/// Scanners are synchronous producers.
///
/// `discover` yields items. `parse` turns one item into zero or more chunks.
/// Scanners do not call the embedder or the store — the pipeline does. This
/// keeps scanners cheap to unit-test with in-memory fixtures.
pub trait Scanner: Send + Sync {
    fn kind(&self) -> SourceKind;

    fn discover<'a>(
        &'a self,
        cfg: &'a SourceConfig,
    ) -> Box<dyn Iterator<Item = Result<SourceItem>> + 'a>;

    fn parse(&self, item: SourceItem) -> Result<Vec<Chunk>>;
}
