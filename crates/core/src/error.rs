use thiserror::Error;

pub type Result<T, E = Error> = std::result::Result<T, E>;

#[derive(Debug, Error)]
pub enum Error {
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),

    #[error("config parse: {0}")]
    Config(String),

    #[error("scan: {0}")]
    Scan(String),

    #[error("parse: {0}")]
    Parse(String),

    #[error("path expand: {0}")]
    PathExpand(String),

    #[error("json: {0}")]
    Json(#[from] serde_json::Error),
}
