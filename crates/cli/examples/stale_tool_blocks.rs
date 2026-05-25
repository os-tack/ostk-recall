// One-shot maintenance: flip stale=true on every active corpus chunk whose
// extra_json marks it as a tool_use or tool_result block. The data stays on
// disk in case we want to index it separately later (tool-usage analytics).
//
// Run with:
//   cargo run -p ostk-recall --example stale_tool_blocks --release
//
// Reads the corpus root from the same config the binary uses (XDG or
// ~/.config/ostk-recall/config.toml). Embedding dim is irrelevant here
// because the table already exists; any value works.

use anyhow::{Context, Result, anyhow};
use ostk_recall_cli::commands;
use ostk_recall_core::Config;
use ostk_recall_store::CorpusStore;

#[tokio::main]
async fn main() -> Result<()> {
    let cfg_path = commands::default_config_path()?;
    let cfg = Config::load(&cfg_path)
        .with_context(|| format!("loading config from {}", cfg_path.display()))?;
    let root = cfg.expanded_root()?;
    println!("corpus root: {}", root.display());

    let store = CorpusStore::open_or_create(&root, 1)
        .await
        .map_err(|e| anyhow!("open corpus store: {e}"))?;

    let flipped = store
        .mark_tool_blocks_stale()
        .await
        .map_err(|e| anyhow!("mark stale: {e}"))?;
    println!("flipped {flipped} chunks stale=true");
    Ok(())
}
