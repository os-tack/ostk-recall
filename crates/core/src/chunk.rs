use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::source::Source;

/// A unit of content ready to be embedded and stored.
///
/// Scanners emit chunks. The pipeline computes content hashes, runs embeddings,
/// and writes rows into the `LanceDB` table.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Chunk {
    pub chunk_id: String,
    pub source: Source,
    pub project: Option<String>,
    pub source_id: String,
    pub chunk_index: u32,
    pub ts: Option<DateTime<Utc>>,
    pub role: Option<String>,
    pub text: String,
    pub sha256: String,
    pub links: Links,
    #[serde(default)]
    pub extra: serde_json::Value,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Links {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub duckdb_row_key: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub file_path: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub parent_ids: Vec<String>,
}

impl Chunk {
    /// Compute a deterministic chunk id.
    ///
    /// `chunk_id = sha256(source || ':' || source_id || ':' || chunk_index)`.
    /// Re-running a scanner with identical inputs produces identical ids, so
    /// upsert by `chunk_id` is idempotent.
    pub fn make_id(source: Source, source_id: &str, chunk_index: u32) -> String {
        let mut h = Sha256::new();
        h.update(source.as_str().as_bytes());
        h.update(b":");
        h.update(source_id.as_bytes());
        h.update(b":");
        h.update(chunk_index.to_le_bytes());
        hex::encode(h.finalize())
    }

    /// Content hash for the `text` field. Used by the pipeline's dedupe step:
    /// if `sha256(text)` already present in `ingest_chunks`, skip embedding.
    pub fn content_hash(text: &str) -> String {
        hex::encode(Sha256::digest(text.as_bytes()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chunk_id_is_deterministic() {
        let a = Chunk::make_id(Source::Markdown, "notes/foo.md", 0);
        let b = Chunk::make_id(Source::Markdown, "notes/foo.md", 0);
        assert_eq!(a, b);
    }

    #[test]
    fn chunk_id_differs_by_index() {
        let a = Chunk::make_id(Source::Markdown, "notes/foo.md", 0);
        let b = Chunk::make_id(Source::Markdown, "notes/foo.md", 1);
        assert_ne!(a, b);
    }

    #[test]
    fn chunk_id_differs_by_source() {
        let a = Chunk::make_id(Source::Markdown, "x", 0);
        let b = Chunk::make_id(Source::Code, "x", 0);
        assert_ne!(a, b);
    }

    #[test]
    fn content_hash_stable() {
        assert_eq!(Chunk::content_hash("hello"), Chunk::content_hash("hello"));
        assert_ne!(Chunk::content_hash("hello"), Chunk::content_hash("world"));
    }
}
