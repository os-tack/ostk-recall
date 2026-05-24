//! Attention substrate MCP + CLI verb surface (Phase 9).
//!
//! Exposes [`AttentionForwardStore`] methods and [`ThreadsDb`] CRUD
//! through MCP tool schemas + a parallel set of pure-Rust dispatch
//! handlers. The CLI binary (`ostk-recall`) wraps the same dispatch
//! layer so the wire format and the human-readable surface stay in
//! lock-step.
//!
//! Companion threads:
//! - `abi-as-sovereign-boundary` — every surfaced page carries
//!   `ScoreAttribution`, and every tool validates `PrivacyTier`.
//! - `chain-as-cognition-history` — thread/evidence mutations flow
//!   through the existing `ChainSink` on [`ThreadsDb`].
//!
//! # Render decision (V1 deferral)
//!
//! The original plan called for an `attention_render` MCP tool that
//! would re-materialise a thread at a chosen depth. The Phase 2
//! `AttentionForwardStore` trait does not yet expose a `render` method —
//! it was flagged as a V1.1 gap during the corrections pass. Rather
//! than ship a half-wired tool that returns the same data
//! `attention_surface` already carries on every page (`AttentionPage`
//! already includes `depth`), Phase 9 ships:
//!
//! - `attention_surface` returning `AttentionPage` rows with `depth`
//!   already populated — callers that just want a depth-bound view can
//!   filter the response.
//! - A `TODO_RENDER` constant + module-level note as the V1.1 handle so
//!   the work is discoverable when the trait grows `render(handle, depth)`.
//!
//! No silent regression: a caller asking "give me thread X at depth Y"
//! today walks `attention_surface` + filter on `handle` + read `depth`
//! from the returned `AttentionPage`. The dedicated verb lands when the
//! trait grows the method.

pub mod cli;
pub mod handlers;
pub mod tools;

pub use handlers::{
    AttentionDispatch, AttentionHandlersError, DefaultAttentionHandlers, ThreadCreateInput,
    ThreadLinkInput, ThreadPromoteInput, attend, decay, default_scope, familiarize, fold, surface,
    thread_create, thread_link, thread_list, thread_promote, thread_unlink, validate_privacy_tier,
};
pub use tools::{ATTENTION_TOOL_NAMES, THREAD_TOOL_NAMES, attention_tools, thread_tools};

/// V1.1 marker for the `attention_render` verb. See module docs.
pub const TODO_RENDER: &str = "attention_render: deferred to V1.1 — \
    needs AttentionForwardStore::render(handle, depth). Until then, \
    use attention_surface and filter the AttentionPage by handle + depth.";
