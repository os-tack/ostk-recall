// One-shot maintenance: run Lance's optimize pass on the corpus table.
// Merges small fragments into larger ones and reindexes new data into
// existing indices. Run after a bulk mutation (e.g. the stale_tool_blocks
// sweep) to drop serve back to idle and reclaim disk.
//
// Run with:
//   cargo run -p ostk-recall --example compact_corpus --release

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

    let started = std::time::Instant::now();
    println!("running OptimizeAction::All (compact + prune + index)…");
    store
        .optimize_all()
        .await
        .map_err(|e| anyhow!("optimize: {e}"))?;
    println!("done in {:.1}s", started.elapsed().as_secs_f64());
    Ok(())
}
