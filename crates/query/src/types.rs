//! Result/parameter types — moved to `ostk_recall_core::types` in v0.1.5.
//!
//! This module is kept as a backward-compatible re-export shim so existing
//! consumers using `ostk_recall_query::types::*` or
//! `ostk_recall_query::{RecallHit, RecallParams, ...}` keep compiling.
//!
//! New callers should import directly from `ostk_recall_core` instead;
//! cut #3 (→1848) ships ostk-recall-serve as a peer-process daemon, and
//! haystack consumes types via `ostk_recall_core` so it can drop the
//! ostk-recall-query dep entirely.

pub use ostk_recall_core::{
    AuditResult, RecallHit, RecallLinkResult, RecallParams, RecallStats, RerankerStats, SourceCount,
};
