use thiserror::Error;

#[derive(Debug, Error)]
pub enum QueryError {
    #[error("lancedb: {0}")]
    Lance(#[from] lancedb::Error),

    #[error("arrow: {0}")]
    Arrow(#[from] arrow::error::ArrowError),

    #[error("store: {0}")]
    Store(#[from] ostk_recall_store::StoreError),

    #[error("json: {0}")]
    Json(#[from] serde_json::Error),

    #[error("forbidden: {0}")]
    Forbidden(String),

    #[error("events store not configured")]
    NoEventsStore,

    #[error("not found: {0}")]
    NotFound(String),

    #[error("decode: {0}")]
    Decode(String),
}

pub type Result<T> = std::result::Result<T, QueryError>;
