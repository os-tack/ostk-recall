//! Decode Arrow `RecordBatch` rows from the corpus table into `RecallHit`s.

use arrow::array::{
    Array, Float32Array, Int64Array, RecordBatch, StringArray, TimestampMicrosecondArray,
};
use chrono::{DateTime, TimeZone, Utc};
use ostk_recall_core::Links;

use crate::error::{QueryError, Result};
use crate::types::RecallHit;

pub const SNIPPET_CHARS: usize = 400;

/// Decode each row of a batch into a `RecallHit`. Score is taken from
/// `_relevance_score` / `_score` / `_distance` — whichever is present, in
/// that order. Unknown → 0.0.
pub fn batch_to_hits(batch: &RecordBatch) -> Result<Vec<RecallHit>> {
    let schema = batch.schema();
    let n = batch.num_rows();
    if n == 0 {
        return Ok(Vec::new());
    }

    let chunk_id = col_str(batch, "chunk_id")?;
    let source = col_str(batch, "source")?;
    let source_id = col_str(batch, "source_id")?;
    let text = col_str(batch, "text")?;

    let project = col_str_opt(batch, "project");
    let links_json = col_str_opt(batch, "links_json");
    let extra_json = col_str_opt(batch, "extra_json");
    let ts = col_ts_opt(batch, "ts");

    // Pick a score column. `_relevance_score` is the post-rerank column;
    // otherwise fall back to raw FTS `_score` or vector `_distance`.
    let score_idx = ["_relevance_score", "_score", "_distance"]
        .into_iter()
        .find_map(|name| schema.index_of(name).ok());
    let score = score_idx.and_then(|i| {
        batch
            .column(i)
            .as_any()
            .downcast_ref::<Float32Array>()
            .cloned()
    });

    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let links: Links = links_json.map_or_else(Links::default, |arr| {
            if arr.is_null(i) {
                Links::default()
            } else {
                serde_json::from_str(arr.value(i)).unwrap_or_default()
            }
        });

        // Decode `extra_json` if the column exists. Empty/null/unparseable
        // strings degrade to `Value::Null` so a corrupt row doesn't fail
        // the whole batch — the field is advisory metadata for the UI.
        let extra: serde_json::Value = extra_json.map_or(serde_json::Value::Null, |arr| {
            if arr.is_null(i) {
                serde_json::Value::Null
            } else {
                let raw = arr.value(i);
                if raw.is_empty() {
                    serde_json::Value::Null
                } else {
                    serde_json::from_str(raw).unwrap_or(serde_json::Value::Null)
                }
            }
        });

        let snippet = snippet_of(text.value(i));
        let score_value = score
            .as_ref()
            .map_or(0.0, |a| if a.is_null(i) { 0.0 } else { a.value(i) });

        out.push(RecallHit {
            chunk_id: chunk_id.value(i).to_string(),
            project: project.and_then(|a| {
                if a.is_null(i) {
                    None
                } else {
                    Some(a.value(i).to_string())
                }
            }),
            source: source.value(i).to_string(),
            source_id: source_id.value(i).to_string(),
            ts: ts.as_ref().and_then(|t| t[i]),
            snippet,
            score: score_value,
            links,
            extra,
        });
    }

    Ok(out)
}

fn col_str<'a>(batch: &'a RecordBatch, name: &str) -> Result<&'a StringArray> {
    let idx = batch
        .schema()
        .index_of(name)
        .map_err(|e| QueryError::Decode(format!("column {name}: {e}")))?;
    batch
        .column(idx)
        .as_any()
        .downcast_ref::<StringArray>()
        .ok_or_else(|| QueryError::Decode(format!("column {name} is not Utf8")))
}

fn col_str_opt<'a>(batch: &'a RecordBatch, name: &str) -> Option<&'a StringArray> {
    batch
        .schema()
        .index_of(name)
        .ok()
        .and_then(|i| batch.column(i).as_any().downcast_ref::<StringArray>())
}

#[allow(clippy::needless_pass_by_value)]
fn col_ts_opt(batch: &RecordBatch, name: &str) -> Option<Vec<Option<DateTime<Utc>>>> {
    let idx = batch.schema().index_of(name).ok()?;
    let arr = batch
        .column(idx)
        .as_any()
        .downcast_ref::<TimestampMicrosecondArray>();
    if let Some(a) = arr {
        return Some(
            (0..a.len())
                .map(|i| {
                    if a.is_null(i) {
                        None
                    } else {
                        Utc.timestamp_micros(a.value(i)).single()
                    }
                })
                .collect(),
        );
    }
    // Some DuckDB-style paths come through as Int64 microseconds. Handle gracefully.
    let arr = batch.column(idx).as_any().downcast_ref::<Int64Array>()?;
    Some(
        (0..arr.len())
            .map(|i| {
                if arr.is_null(i) {
                    None
                } else {
                    Utc.timestamp_micros(arr.value(i)).single()
                }
            })
            .collect(),
    )
}

pub fn snippet_of(text: &str) -> String {
    if text.chars().count() <= SNIPPET_CHARS {
        return text.to_string();
    }
    let mut out = String::with_capacity(SNIPPET_CHARS + 1);
    for (i, c) in text.chars().enumerate() {
        if i >= SNIPPET_CHARS {
            break;
        }
        out.push(c);
    }
    out
}

/// Escape a single-quoted SQL literal. `LanceDB`'s `.only_if` takes SQL
/// strings verbatim, so we double single quotes. Not a full SQL sanitizer —
/// don't trust it with arbitrary identifiers.
pub fn sql_escape(value: &str) -> String {
    value.replace('\'', "''")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn snippet_short_returns_full() {
        assert_eq!(snippet_of("hi"), "hi");
    }

    #[test]
    fn snippet_long_truncates_to_400_chars() {
        let s: String = "a".repeat(500);
        let out = snippet_of(&s);
        assert_eq!(out.chars().count(), 400);
    }

    #[test]
    fn escape_sql_quotes() {
        assert_eq!(sql_escape("o'brien"), "o''brien");
        assert_eq!(sql_escape("plain"), "plain");
    }
}
