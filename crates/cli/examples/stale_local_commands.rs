// One-shot maintenance: flip stale=true on active corpus chunks whose
// text starts with Claude Code slash-command scaffolding (`<local-command-…>`,
// `<command-name>`, etc.). New ingest already skips these via
// `drop_local_command_wrappers`; this catches the historical backlog.
//
// Run with:
//   cargo run -p ostk-recall --example stale_local_commands --release

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
        .mark_local_command_wrappers_stale()
        .await
        .map_err(|e| anyhow!("mark stale: {e}"))?;
    println!("flipped {flipped} chunks stale=true");
    Ok(())
}
