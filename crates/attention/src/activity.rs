//! Activity-burst attention surface.
//!
//! Robust alternative to the embedding-density clustering in
//! [`crate::emergent`]: ranks "what we're paying attention to" purely
//! by `(project, source_id)` ingest volume and recency. Survives the
//! "thoughts are unique" problem because it never asks chunks to
//! cluster — it asks the timestamps and source-ids directly.
//!
//! Score formula:
//!
//!   `score = count * exp(-(now - max_ts) / decay_hours)`
//!
//! - `count`: chunks ingested into this `(project, source_id)` in the
//!   recency window.
//! - `now - max_ts`: hours since the most recent chunk in the group.
//! - `decay_hours`: half-life-style decay; default 6 hours so a burst
//!   from this morning still ranks high but yesterday's work fades.
//!
//! The output is a `Vec<AttentionBurstReport>` sorted by score desc.

use std::sync::Arc;

use chrono::{DateTime, Utc};
use ostk_recall_store::ActivityBurst;
use ostk_recall_store::CorpusStore;
use ostk_recall_store::corpus::StoreError as CorpusError;
use thiserror::Error;

/// Default look-back window when no value is supplied by the caller.
pub const DEFAULT_SINCE_HOURS: i64 = 24;

/// Default upper bound on bursts returned.
pub const DEFAULT_LIMIT: usize = 10;

/// Default samples (recent chunk snippets) included per burst.
pub const DEFAULT_SAMPLES_PER_BURST: usize = 3;

/// Default half-life-style decay constant in hours. With 6h, a burst
/// from 6h ago scores at ~`count * 0.37`, from 12h ago at ~`count *
/// 0.14`, from 24h ago at ~`count * 0.02`.
pub const DEFAULT_DECAY_HOURS: f32 = 6.0;

/// Maximum characters retained per sample snippet.
pub const SAMPLE_CHAR_LIMIT: usize = 200;

#[derive(Debug, Error)]
pub enum AttentionBurstError {
    #[error("corpus error: {0}")]
    Corpus(#[from] CorpusError),
}

/// Single surfaced burst — one `(project, source_id)` with score,
/// counts, time bounds, and a few sample snippets.
///
/// `samples` is the truncated human-readable view. `chunk_ids` is the
/// full burst membership (sorted lexicographically) and is **internal
/// to the attention crate** — used by `thread_query`'s v0.4.1+
/// cross-axis backfill, which needs exact membership. MCP handlers
/// must not echo `chunk_ids` directly to the wire: a single
/// `(project, source_id)` can contain thousands of chunks and the
/// list is unbounded by design. See
/// `attention_mcp::handlers::thread_attention` for the contract.
#[derive(Debug, Clone)]
pub struct AttentionBurstReport {
    pub project: String,
    pub source_id: String,
    pub count: usize,
    pub score: f32,
    pub max_ts: DateTime<Utc>,
    pub min_ts: DateTime<Utc>,
    pub chunk_ids: Vec<String>,
    pub samples: Vec<String>,
}

/// Run the activity-burst surface against the corpus.
///
/// Returns up to `limit` bursts sorted by `score` descending. Empty
/// vector if no chunks are active in the window.
pub async fn surface_attention(
    corpus: &Arc<CorpusStore>,
    since: DateTime<Utc>,
    limit: usize,
    samples_per_burst: usize,
    decay_hours: f32,
) -> Result<Vec<AttentionBurstReport>, AttentionBurstError> {
    let bursts = corpus.activity_bursts(since, samples_per_burst).await?;
    if bursts.is_empty() {
        return Ok(Vec::new());
    }

    let now = Utc::now();
    let mut scored: Vec<AttentionBurstReport> = bursts
        .into_iter()
        .map(|b| score_burst(b, now, decay_hours))
        .collect();
    scored.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    scored.truncate(limit);
    Ok(scored)
}

#[allow(clippy::cast_precision_loss)]
fn score_burst(b: ActivityBurst, now: DateTime<Utc>, decay_hours: f32) -> AttentionBurstReport {
    let dt_hours = (now - b.max_ts).num_seconds().max(0) as f32 / 3600.0;
    let decay = (-dt_hours / decay_hours.max(0.01)).exp();
    let score = b.count as f32 * decay;
    let samples: Vec<String> = b
        .samples
        .into_iter()
        .map(|(_, _, text)| snippet(&text, SAMPLE_CHAR_LIMIT))
        .collect();
    AttentionBurstReport {
        project: b.project,
        source_id: b.source_id,
        count: b.count,
        score,
        max_ts: b.max_ts,
        min_ts: b.min_ts,
        chunk_ids: b.chunk_ids,
        samples,
    }
}

/// Trim text to the first `max_chars` chars at a UTF-8 boundary,
/// collapsing inner whitespace.
fn snippet(text: &str, max_chars: usize) -> String {
    let collapsed: String = text.split_whitespace().collect::<Vec<_>>().join(" ");
    if collapsed.chars().count() <= max_chars {
        return collapsed;
    }
    let cut: String = collapsed.chars().take(max_chars).collect();
    format!("{cut}…")
}

/// Convenience wrapper using all defaults.
pub async fn surface_default(
    corpus: &Arc<CorpusStore>,
    since: DateTime<Utc>,
) -> Result<Vec<AttentionBurstReport>, AttentionBurstError> {
    surface_attention(
        corpus,
        since,
        DEFAULT_LIMIT,
        DEFAULT_SAMPLES_PER_BURST,
        DEFAULT_DECAY_HOURS,
    )
    .await
}
