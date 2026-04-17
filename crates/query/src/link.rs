//! `recall_link` — fetch a chunk by id plus its parent chain.

use futures::TryStreamExt;
use lancedb::Connection;
use lancedb::query::{ExecutableQuery, QueryBase};
use ostk_recall_store::CORPUS_TABLE;

use crate::error::{QueryError, Result};
use crate::row::{batch_to_hits, sql_escape};
use crate::types::RecallLinkResult;

pub async fn recall_link(conn: &Connection, chunk_id: &str) -> Result<RecallLinkResult> {
    let table = conn.open_table(CORPUS_TABLE).execute().await?;
    let filter = format!("chunk_id = '{}'", sql_escape(chunk_id));

    let stream = table.query().only_if(filter).limit(1).execute().await?;
    let batches: Vec<_> = stream.try_collect().await?;
    let mut hits = Vec::new();
    for b in &batches {
        hits.extend(batch_to_hits(b)?);
    }
    let chunk = hits
        .into_iter()
        .next()
        .ok_or_else(|| QueryError::NotFound(chunk_id.to_string()))?;

    let parent_ids = chunk.links.parent_ids.clone();
    let parents = if parent_ids.is_empty() {
        Vec::new()
    } else {
        let in_list = parent_ids
            .iter()
            .map(|p| format!("'{}'", sql_escape(p)))
            .collect::<Vec<_>>()
            .join(", ");
        let pf = format!("chunk_id IN ({in_list})");
        let pstream = table
            .query()
            .only_if(pf)
            .limit(parent_ids.len().max(1))
            .execute()
            .await?;
        let pbatches: Vec<_> = pstream.try_collect().await?;
        let mut phits = Vec::new();
        for b in &pbatches {
            phits.extend(batch_to_hits(b)?);
        }
        let mut ordered = Vec::with_capacity(phits.len());
        for pid in &parent_ids {
            if let Some(pos) = phits.iter().position(|h| &h.chunk_id == pid) {
                ordered.push(phits.swap_remove(pos));
            }
        }
        ordered.extend(phits);
        ordered
    };

    Ok(RecallLinkResult { chunk, parents })
}
