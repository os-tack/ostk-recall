//! Attention-substrate schema types (Phase 1).
//!
//! Pure serde shapes that cross process boundaries between the recall
//! daemon (ostk-recall-serve), the haystack kernel, and any future MCP
//! consumer of the attention ABI. No lance, lancedb, sqlite, arrow, or
//! tokio dependencies live here — this crate is intentionally
//! schema-only (see `crates/core/src/types.rs` for the same invariant
//! applied to the recall surface).
//!
//! Companion threads:
//! - `abi-as-sovereign-boundary` — why the ABI is the portable contract
//! - `three-time-scales` — chain / graph / score separation
//!
//! Runtime (scoring math, decay, push channels) lives in the
//! forthcoming `ostk-recall-attention` crate; broadcast plumbing for
//! `IngestEvent` lives in the `pipeline` crate. This module defines
//! only the wire shapes.

use crate::source::SourceKind;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::fmt;

/// Hoberman fold-depth tier for a surfaced thread.
///
/// Ordered: `Folded < Half < Full`. Comparison and ordering let callers
/// pick the deepest tier they can afford against a token budget without
/// branching on string names.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FoldDepth {
    Folded,
    Half,
    Full,
}

/// Privacy tier on an `AttentionScope`. Mirrors the T0/T1/T2/T3
/// sovereign-OS theme.
///
/// `T1Project` is the default — surfaces within a project but never
/// federates without explicit promotion.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PrivacyTier {
    /// Operator-only; never surfaces outside this scope.
    T0Private,
    /// Surfaces within project; not federated. Default.
    #[default]
    T1Project,
    /// Federable to T0/T1 operators only.
    T2Trusted,
    /// Surfaceable to any operator.
    T3Public,
}

/// Per-axis attribution for an `AttentionPage` score.
///
/// Every surfaced page carries one of these so callers can argue with
/// the math, not the vibe (see `abi-as-sovereign-boundary`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScoreAttribution {
    /// Active concern weight in `[0, 1]`.
    pub tension: f32,
    /// Cosine similarity against the current attention vector.
    pub resonance: f32,
    /// Turns containing this handle within the attention window.
    pub familiarity: u32,
    /// Bonus for the low-tension + high-resonance quadrant.
    pub off_diagonal_lift: f32,
    /// Seconds since the thread was last touched.
    pub time_since_touch_secs: u64,
}

/// One row returned by `surface()`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttentionPage {
    pub handle: String,
    pub depth: FoldDepth,
    pub score: f32,
    pub why: ScoreAttribution,
}

/// Newtype wrapping a validated kebab-case thread handle.
///
/// Validation rules (enforced by `ThreadHandle::new`):
/// - lowercase ASCII letters, digits, or `-`
/// - no leading or trailing `-`
/// - at most 4 hyphens
/// - 1..=64 characters
///
/// Stored as `String` on the wire; deserialization re-runs validation
/// so untrusted JSON cannot bypass it.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ThreadHandle(String);

impl ThreadHandle {
    pub const MAX_LEN: usize = 64;
    pub const MAX_HYPHENS: usize = 4;

    /// Validate and wrap a candidate handle.
    pub fn new(s: impl Into<String>) -> Result<Self, ThreadHandleError> {
        let s = s.into();
        Self::validate(&s)?;
        Ok(Self(s))
    }

    /// Borrow the underlying string slice.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// Consume the wrapper and return the owned string.
    #[must_use]
    pub fn into_inner(self) -> String {
        self.0
    }

    fn validate(s: &str) -> Result<(), ThreadHandleError> {
        if s.is_empty() {
            return Err(ThreadHandleError::Empty);
        }
        if s.len() > Self::MAX_LEN {
            return Err(ThreadHandleError::TooLong(s.len()));
        }
        if s.starts_with('-') || s.ends_with('-') {
            return Err(ThreadHandleError::EdgeHyphen);
        }
        let mut hyphens = 0usize;
        for b in s.bytes() {
            match b {
                b'a'..=b'z' | b'0'..=b'9' => {}
                b'-' => hyphens += 1,
                _ => return Err(ThreadHandleError::InvalidChar(b as char)),
            }
        }
        if hyphens > Self::MAX_HYPHENS {
            return Err(ThreadHandleError::TooManyHyphens(hyphens));
        }
        Ok(())
    }
}

impl fmt::Display for ThreadHandle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl Serialize for ThreadHandle {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        self.0.serialize(s)
    }
}

impl<'de> Deserialize<'de> for ThreadHandle {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let s = String::deserialize(d)?;
        Self::new(s).map_err(serde::de::Error::custom)
    }
}

/// Validation failure shapes for `ThreadHandle::new`.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ThreadHandleError {
    #[error("thread handle must not be empty")]
    Empty,
    #[error("thread handle exceeds {max} chars (got {0})", max = ThreadHandle::MAX_LEN)]
    TooLong(usize),
    #[error("thread handle must not start or end with '-'")]
    EdgeHyphen,
    #[error("thread handle has {0} hyphens (max {max})", max = ThreadHandle::MAX_HYPHENS)]
    TooManyHyphens(usize),
    #[error("thread handle contains invalid char {0:?} (kebab-case lowercase ASCII only)")]
    InvalidChar(char),
}

/// Mandatory scope on every attention surface.
///
/// Without scope, projects share an attention vector and the surfacer
/// leaks across boundaries — both a correctness and a privacy
/// regression. `privacy_tier` defaults to `T1Project`.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AttentionScope {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub project: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub session_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub agent: Option<String>,
    #[serde(default)]
    pub privacy_tier: PrivacyTier,
}

/// Event broadcast after a pipeline batch completes.
///
/// Event shape only — the publisher (`Pipeline::subscribe_ingest`) and
/// the broadcast channel live in the `pipeline` crate so this crate
/// stays free of `tokio` / runtime concerns.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IngestEvent {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub project: Option<String>,
    pub source: SourceKind,
    pub source_ids: Vec<String>,
    pub chunk_ids_upserted: Vec<String>,
    pub chunks_upserted: usize,
    pub chunks_stale: usize,
    pub ts: DateTime<Utc>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{RecallHit, RecallParams, SynthesizedPage};

    fn round_trip<T>(value: &T) -> T
    where
        T: Serialize + for<'de> Deserialize<'de>,
    {
        let json = serde_json::to_string(value).expect("serialize");
        serde_json::from_str(&json).expect("deserialize")
    }

    #[test]
    fn fold_depth_serde_round_trip() {
        for d in [FoldDepth::Folded, FoldDepth::Half, FoldDepth::Full] {
            let v: FoldDepth = round_trip(&d);
            assert_eq!(v, d);
        }
        // snake_case wire form
        assert_eq!(serde_json::to_string(&FoldDepth::Half).unwrap(), "\"half\"");
    }

    #[test]
    fn fold_depth_is_ordered() {
        assert!(FoldDepth::Folded < FoldDepth::Half);
        assert!(FoldDepth::Half < FoldDepth::Full);
    }

    #[test]
    fn privacy_tier_default_is_t1_project() {
        assert_eq!(PrivacyTier::default(), PrivacyTier::T1Project);
    }

    #[test]
    fn privacy_tier_serde_round_trip() {
        for t in [
            PrivacyTier::T0Private,
            PrivacyTier::T1Project,
            PrivacyTier::T2Trusted,
            PrivacyTier::T3Public,
        ] {
            let v: PrivacyTier = round_trip(&t);
            assert_eq!(v, t);
        }
        assert_eq!(
            serde_json::to_string(&PrivacyTier::T0Private).unwrap(),
            "\"t0_private\""
        );
    }

    // Compare floats by exact bit pattern. Serde round-trips preserve
    // bit-level equality of finite, non-NaN f32 values, so this is the
    // right assertion (and avoids the `clippy::float_cmp` lint that
    // fires on `==` over floats).
    fn f32_eq(a: f32, b: f32) -> bool {
        a.to_bits() == b.to_bits()
    }

    #[test]
    fn score_attribution_round_trip() {
        let s = ScoreAttribution {
            tension: 0.42,
            resonance: 0.71,
            familiarity: 47,
            off_diagonal_lift: 0.03,
            time_since_touch_secs: 11 * 86_400,
        };
        let v: ScoreAttribution = round_trip(&s);
        assert!(f32_eq(v.tension, s.tension));
        assert!(f32_eq(v.resonance, s.resonance));
        assert_eq!(v.familiarity, s.familiarity);
        assert!(f32_eq(v.off_diagonal_lift, s.off_diagonal_lift));
        assert_eq!(v.time_since_touch_secs, s.time_since_touch_secs);
    }

    #[test]
    fn attention_page_round_trip() {
        let p = AttentionPage {
            handle: "hoberman-thread-primitive".to_string(),
            depth: FoldDepth::Half,
            score: 0.74,
            why: ScoreAttribution {
                tension: 0.0,
                resonance: 0.71,
                familiarity: 47,
                off_diagonal_lift: 0.03,
                time_since_touch_secs: 950_400,
            },
        };
        let v: AttentionPage = round_trip(&p);
        assert_eq!(v.handle, p.handle);
        assert_eq!(v.depth, p.depth);
        assert!(f32_eq(v.score, p.score));
        assert_eq!(v.why.familiarity, 47);
    }

    #[test]
    fn attention_scope_round_trip_full() {
        let s = AttentionScope {
            project: Some("haystack".to_string()),
            session_id: Some("sess-abc".to_string()),
            agent: Some("claude".to_string()),
            privacy_tier: PrivacyTier::T2Trusted,
        };
        let v: AttentionScope = round_trip(&s);
        assert_eq!(v.project.as_deref(), Some("haystack"));
        assert_eq!(v.session_id.as_deref(), Some("sess-abc"));
        assert_eq!(v.agent.as_deref(), Some("claude"));
        assert_eq!(v.privacy_tier, PrivacyTier::T2Trusted);
    }

    #[test]
    fn attention_scope_round_trip_defaults() {
        let s = AttentionScope::default();
        let json = serde_json::to_string(&s).unwrap();
        // Optional fields skipped, privacy_tier present.
        assert_eq!(json, "{\"privacy_tier\":\"t1_project\"}");
        let v: AttentionScope = round_trip(&s);
        assert!(v.project.is_none());
        assert_eq!(v.privacy_tier, PrivacyTier::T1Project);
    }

    #[test]
    fn attention_scope_accepts_empty_object() {
        let v: AttentionScope = serde_json::from_str("{}").unwrap();
        assert!(v.project.is_none());
        assert!(v.session_id.is_none());
        assert!(v.agent.is_none());
        assert_eq!(v.privacy_tier, PrivacyTier::T1Project);
    }

    #[test]
    fn ingest_event_round_trip() {
        let e = IngestEvent {
            project: Some("haystack".to_string()),
            source: SourceKind::Markdown,
            source_ids: vec!["docs/a.md".to_string(), "docs/b.md".to_string()],
            chunk_ids_upserted: vec!["c1".to_string(), "c2".to_string()],
            chunks_upserted: 2,
            chunks_stale: 0,
            ts: Utc::now(),
        };
        let v: IngestEvent = round_trip(&e);
        assert_eq!(v.source_ids, e.source_ids);
        assert_eq!(v.chunk_ids_upserted, e.chunk_ids_upserted);
        assert_eq!(v.chunks_upserted, 2);
        assert_eq!(v.chunks_stale, 0);
    }

    #[test]
    fn thread_handle_accepts_valid() {
        let h = ThreadHandle::new("hoberman-thread-primitive").unwrap();
        assert_eq!(h.as_str(), "hoberman-thread-primitive");
        assert_eq!(h.to_string(), "hoberman-thread-primitive");
        ThreadHandle::new("a").unwrap();
        ThreadHandle::new("abc123").unwrap();
        ThreadHandle::new("a-b-c-d-e").unwrap(); // exactly 4 hyphens
    }

    #[test]
    fn thread_handle_rejects_invalid() {
        // "Hello World" fails on the first invalid byte ('H', uppercase).
        // Either the uppercase or the space would be a legitimate
        // rejection; the validator reports the first one it sees.
        assert_eq!(
            ThreadHandle::new("Hello World").unwrap_err(),
            ThreadHandleError::InvalidChar('H')
        );
        assert_eq!(
            ThreadHandle::new("ok mid").unwrap_err(),
            ThreadHandleError::InvalidChar(' ')
        );
        assert_eq!(
            ThreadHandle::new("_underscore").unwrap_err(),
            ThreadHandleError::InvalidChar('_')
        );
        assert_eq!(
            ThreadHandle::new("trailing-").unwrap_err(),
            ThreadHandleError::EdgeHyphen
        );
        assert_eq!(
            ThreadHandle::new("-leading").unwrap_err(),
            ThreadHandleError::EdgeHyphen
        );
        assert_eq!(ThreadHandle::new("").unwrap_err(), ThreadHandleError::Empty);
        // > 4 hyphens
        assert_eq!(
            ThreadHandle::new("a-b-c-d-e-f").unwrap_err(),
            ThreadHandleError::TooManyHyphens(5)
        );
        // > 64 chars
        let long = "a".repeat(65);
        assert_eq!(
            ThreadHandle::new(long).unwrap_err(),
            ThreadHandleError::TooLong(65)
        );
        // 64 is allowed
        ThreadHandle::new("a".repeat(64)).unwrap();
    }

    #[test]
    fn thread_handle_serde_round_trip() {
        let h = ThreadHandle::new("attention-substrate").unwrap();
        let json = serde_json::to_string(&h).unwrap();
        assert_eq!(json, "\"attention-substrate\"");
        let v: ThreadHandle = serde_json::from_str(&json).unwrap();
        assert_eq!(v, h);
    }

    #[test]
    fn thread_handle_serde_rejects_invalid() {
        let err = serde_json::from_str::<ThreadHandle>("\"Hello World\"").unwrap_err();
        assert!(
            err.to_string().contains("invalid char"),
            "unexpected serde error: {err}"
        );
    }

    /// Adding the `attention` module must not break deserialization of
    /// existing recall wire shapes that pre-date these fields. Fixtures
    /// here mirror what older daemons / kernels emit before they learn
    /// about attention.
    #[test]
    fn legacy_recall_params_deserializes() {
        let json = r#"{"query":"alloc_page"}"#;
        let v: RecallParams = serde_json::from_str(json).unwrap();
        assert_eq!(v.query, "alloc_page");
        assert!(v.project.is_none());
    }

    #[test]
    fn legacy_recall_hit_deserializes() {
        let json = r#"{
            "chunk_id":"c1",
            "project":null,
            "source":"markdown",
            "source_id":"docs/a.md",
            "ts":null,
            "snippet":"hello",
            "score":0.5,
            "links":{"parents":[],"children":[]}
        }"#;
        let v: RecallHit = serde_json::from_str(json).unwrap();
        assert_eq!(v.chunk_id, "c1");
        assert!(f32_eq(v.score, 0.5));
        assert!(!v.stale);
        assert!(v.role.is_none());
    }

    #[test]
    fn legacy_synthesized_page_deserializes() {
        let json = r#"{
            "title":"Symbol: alloc_page",
            "head":{
                "chunk_id":"c1",
                "project":null,
                "source":"code",
                "source_id":"src/main.rs",
                "ts":null,
                "snippet":"fn alloc_page()",
                "score":0.9,
                "links":{"parents":[],"children":[]}
            },
            "lineage":[],
            "evidence":[],
            "total_lineage":0,
            "total_evidence":0,
            "summary":"primary def"
        }"#;
        let v: SynthesizedPage = serde_json::from_str(json).unwrap();
        assert_eq!(v.title, "Symbol: alloc_page");
        assert!(v.lineage.is_empty());
    }
}
