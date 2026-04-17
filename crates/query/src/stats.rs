//! `recall_stats` — totals, per-source breakdown, model, last scan.

use std::sync::Arc;

use ostk_recall_store::{CorpusStore, IngestDb};

use crate::error::Result;
use crate::types::{RecallStats, SourceCount};

pub async fn recall_stats(
    store: &Arc<CorpusStore>,
    ingest: &IngestDb,
    model: &str,
) -> Result<RecallStats> {
    let total = store.row_count().await?;
    let by_source_raw = ingest.count_by_source()?;
    let by_source = by_source_raw
        .into_iter()
        .map(|(source, count)| SourceCount { source, count })
        .collect();
    let last_scan_at = ingest.latest_upserted_at()?;
    Ok(RecallStats {
        total,
        by_source,
        model: model.to_string(),
        dim: store.dim(),
        last_scan_at,
    })
}
