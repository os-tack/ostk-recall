//! `recall_stats` — totals, per-source breakdown, model, last scan.

use std::sync::Arc;

use ostk_recall_store::{CorpusStore, EventsDb, IngestDb};

use crate::error::Result;
use crate::rerank::RerankerLike;
use crate::types::{AuditFreshness, RecallStats, RerankerStats, SourceCount};

pub async fn recall_stats(
    store: &Arc<CorpusStore>,
    ingest: &IngestDb,
    events: Option<&EventsDb>,
    model: &str,
    reranker: Option<&dyn RerankerLike>,
    corpus_root: Option<&std::path::Path>,
) -> Result<RecallStats> {
    let total = store.row_count().await?;
    let by_source_raw = ingest.count_by_source()?;
    let by_source = by_source_raw
        .into_iter()
        .map(|(source, count)| SourceCount { source, count })
        .collect();
    let last_scan_at = ingest.latest_upserted_at()?;
    let reranker_stats = reranker.map(|r| RerankerStats {
        model: r.model_id().to_string(),
        enabled: true,
    });
    // →1947 freshness guard: surface newest audit row per project so a
    // frozen events ingest is visible here instead of silently serving
    // stale recall_audit analyses. Best-effort: a read error degrades to
    // omission, never fails stats.
    let audit_newest_ts: Option<Vec<AuditFreshness>> = events.and_then(|db| {
        db.newest_ts_by_project().ok().map(|rows| {
            rows.into_iter()
                .map(|(project, newest_ts)| AuditFreshness { project, newest_ts })
                .collect()
        })
    });
    // →1957 watcher observability: pass the watcher's snapshot through
    // verbatim (drop counters + last-kick stamps). Best-effort — absent
    // or unparsable file degrades to omission.
    let watch: Option<serde_json::Value> = corpus_root
        .map(|root| root.join("watch_status.json"))
        .and_then(|p| std::fs::read(p).ok())
        .and_then(|bytes| serde_json::from_slice(&bytes).ok());
    Ok(RecallStats {
        total,
        by_source,
        model: model.to_string(),
        dim: store.dim(),
        last_scan_at,
        reranker: reranker_stats,
        audit_newest_ts,
        watch,
    })
}
