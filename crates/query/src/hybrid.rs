//! Hybrid dense + BM25 retrieval over the corpus table.

use std::sync::Arc;

use chrono::{DateTime, Utc};
use futures::TryStreamExt;
use lancedb::Connection;
use lancedb::index::scalar::FullTextSearchQuery;
use lancedb::query::{ExecutableQuery, QueryBase};
use lancedb::rerankers::rrf::RRFReranker;
use ostk_recall_pipeline::ChunkEmbedder;
use ostk_recall_store::CORPUS_TABLE;

use crate::error::Result;
use crate::row::{batch_to_hits, sql_escape};
use crate::types::{RecallHit, RecallParams};

/// Execute a hybrid recall against the corpus table.
pub async fn recall(
    conn: &Connection,
    embedder: &dyn ChunkEmbedder,
    params: &RecallParams,
) -> Result<Vec<RecallHit>> {
    let query_text = params.query.trim();
    if query_text.is_empty() {
        return Ok(Vec::new());
    }

    let limit = params.limit.unwrap_or(10).max(1);
    let vec = embedder
        .encode_batch(&[query_text])
        .into_iter()
        .next()
        .unwrap_or_default();

    let table = conn.open_table(CORPUS_TABLE).execute().await?;

    let filter = build_filter(
        params.project.as_deref(),
        params.source.as_deref(),
        params.since,
    );

    let mut q = table
        .query()
        .nearest_to(vec)?
        .full_text_search(FullTextSearchQuery::new(query_text.to_string()))
        .rerank(Arc::new(RRFReranker::default()))
        .limit(limit);

    if let Some(f) = &filter {
        q = q.only_if(f);
    }

    let stream = q.execute().await?;
    let batches: Vec<_> = stream.try_collect().await?;

    let mut hits = Vec::new();
    for b in &batches {
        hits.extend(batch_to_hits(b)?);
    }
    if hits.len() > limit {
        hits.truncate(limit);
    }
    Ok(hits)
}

fn build_filter(
    project: Option<&str>,
    source: Option<&str>,
    since: Option<DateTime<Utc>>,
) -> Option<String> {
    let mut clauses: Vec<String> = Vec::new();
    if let Some(p) = project {
        clauses.push(format!("project = '{}'", sql_escape(p)));
    }
    if let Some(s) = source {
        clauses.push(format!("source = '{}'", sql_escape(s)));
    }
    if let Some(t) = since {
        clauses.push(format!("ts >= TIMESTAMP '{}'", t.to_rfc3339()));
    }
    if clauses.is_empty() {
        None
    } else {
        Some(clauses.join(" AND "))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_filter_empty() {
        assert!(build_filter(None, None, None).is_none());
    }

    #[test]
    fn build_filter_project_and_source() {
        let f = build_filter(Some("foo"), Some("markdown"), None).unwrap();
        assert!(f.contains("project = 'foo'"));
        assert!(f.contains("source = 'markdown'"));
        assert!(f.contains("AND"));
    }

    #[test]
    fn build_filter_escapes_quotes() {
        let f = build_filter(Some("a'b"), None, None).unwrap();
        assert_eq!(f, "project = 'a''b'");
    }

    #[test]
    fn build_filter_since_is_iso() {
        let t = DateTime::parse_from_rfc3339("2026-04-17T10:00:00Z")
            .unwrap()
            .with_timezone(&Utc);
        let f = build_filter(None, None, Some(t)).unwrap();
        assert!(f.contains("TIMESTAMP '2026-04-17T10:00:00+00:00'"));
    }
}
