//! Daemon shared state: the loaded CorpusStore + Embedder.
//!
//! Constructed lazily during `initialize`. Subsequent `recall.fault`
//! calls reuse the same handles. Wrapped in `Arc<Mutex<Option<...>>>`
//! so the daemon can be re-initialized (e.g. corpus path changed)
//! without restarting the process.

use std::path::PathBuf;
use std::sync::Arc;

use ostk_recall_embed::Embedder;
use ostk_recall_store::CorpusStore;

/// Loaded daemon resources. `None` until `initialize` succeeds.
#[allow(dead_code)] // corpus_root + embedder_model are for future status/log surfaces
pub struct State {
    pub corpus_root: PathBuf,
    pub store: Arc<CorpusStore>,
    pub embedder: Arc<Embedder>,
    pub embedder_model: String,
}

impl State {
    #[allow(dead_code)]
    pub fn dim(&self) -> usize {
        self.embedder.dim()
    }
}
