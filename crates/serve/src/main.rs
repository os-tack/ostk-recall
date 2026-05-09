//! Standalone `ostk-recall-serve` binary entry point.
//!
//! Thin wrapper around [`ostk_recall_serve::run_daemon`]. The same
//! function is also called by `ostk-recall serve --stdio` from the CLI
//! crate, so both entry points share one implementation.

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    ostk_recall_serve::run_daemon().await
}
