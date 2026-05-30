//! P9b-min lens — first active memory lens.
//!
//! Builds a small, structured markdown payload that the daemon
//! injects into the LLM's context as the `memory-lens` MCP resource.
//! This module owns:
//!
//! - Type surface: [`Lens`], [`LensEntry`], [`LensConfig`].
//! - Ambient candidate generation via [`crate::lanes::ambient_candidates`].
//! - In-engine ranking against the attention-affinity feature.
//! - Slot allocation. P9b-min ships only the attention slot; the
//!   full portfolio (freshness / entity / concept / diversity_jump)
//!   lands in P9b-full when the underlying features ship.
//! - Token-budget enforcement with excerpt truncation. Slots that
//!   can't fit the configured floor are skipped, not truncated past
//!   it.
//! - Privacy denylist filtering.
//! - Markdown rendering. The rendered bytes are what the daemon
//!   loop fingerprints for the unchanged-content skip.
//!
//! Layered shape: this module is pure — given an `AttentionContext`
//! and a `CorpusStore`, it returns a `Lens`. The daemon loop in
//! `crates/cli/src/lens_loop.rs` owns the drift / pin-fingerprint /
//! content-fingerprint state machine and calls into here. That
//! split lets us unit-test the builder without spinning up a tokio
//! daemon, and lets the loop reason about state transitions without
//! reaching into Lance.

use std::collections::{BTreeMap, HashSet};

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use ostk_recall_core::{FacetSet, facets};
use ostk_recall_store::corpus::CorpusStore;

use crate::context::{AttentionContext, QueryContext};
use crate::error::Result;
use crate::lanes::ambient_candidates;
use crate::rank::{FeatureAttribution, RankEngine, RankedHit, attention_affinity_score};

// ---------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------

/// Tunable parameters for the lens loop. Defaults match
/// `p9b-lens-portfolio.md` "Privacy + control" section so production
/// behavior is config-table-equivalent when no overrides are set.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LensConfig {
    /// Token cap on the rendered lens. The allocator truncates
    /// excerpts to stay within budget; slots whose floor wouldn't
    /// fit are skipped instead of being padded with a fragment.
    pub token_budget: usize,
    /// Floor below which a slot is dropped rather than truncated.
    /// Default 200 tokens — much below that and the excerpt loses
    /// the structural sentences readers rely on.
    pub min_excerpt_tokens: usize,
    /// Cosine **DISTANCE** threshold for triggering a refresh.
    /// `0.15` corresponds to ~0.987 cosine similarity. Stored here
    /// so the lens loop and the lens builder both read the same
    /// value; the builder itself never consults drift (the loop
    /// gates the call).
    pub drift_threshold: f32,
    /// How often the background loop wakes up. Sub-poll changes are
    /// invisible to the lens.
    pub poll_interval_secs: u64,
    /// `key:value` facet entries that exclude a chunk from the lens
    /// outright (e.g. `["status:archived", "privacy:t0"]`). Matching
    /// is escape-safe because we go through `facets::to_list` which
    /// roundtrips values containing `:`.
    pub exclude_facets: Vec<String>,
    /// Per-lane candidate cap. The total candidate pool is at most
    /// `candidate_k_per_lane * (lane count)`; P9b-min has one lane
    /// (dense) so the pool is bounded by this value.
    pub candidate_k_per_lane: usize,
    /// Dominance threshold for slot assignment, expressed as a
    /// fraction of total score. A feature with weighted contribution
    /// below this share is not considered dominant in its slot.
    pub dominance_threshold: f32,
}

impl Default for LensConfig {
    fn default() -> Self {
        Self {
            token_budget: 4000,
            min_excerpt_tokens: 200,
            drift_threshold: 0.15,
            poll_interval_secs: 5,
            // Attenuate operational telemetry from ambient surfacing by
            // default (still fully recall-able). Keep in sync with
            // `ostk_recall_core::config::default_lens_exclude_facets`.
            // (RT-7 added `harness_orchestration` to the core default but left
            // this copy behind — the guard test caught the drift.)
            exclude_facets: vec![
                "record_kind:audit_significant".to_string(),
                "record_kind:harness_orchestration".to_string(),
            ],
            candidate_k_per_lane: 32,
            dominance_threshold: 0.30,
        }
    }
}

// ---------------------------------------------------------------------
// Lens types
// ---------------------------------------------------------------------

/// A rendered lens — what the daemon publishes as the `memory-lens`
/// MCP resource and what the unchanged-content fingerprint hashes.
#[derive(Debug, Clone)]
pub struct Lens {
    pub entries: Vec<LensEntry>,
    pub generated_at: DateTime<Utc>,
    /// `"rolling"` in P9b-min — the only drift basis. Future phases
    /// may switch to `"entity"` or `"concept"` once those slots ship.
    pub drift_basis: String,
    /// `true` when an operator pin is driving ranking. Surfaced into
    /// the markdown so the operator can tell the lens is pinned.
    pub pinned: bool,
}

/// One entry in the portfolio. Carries enough provenance for the
/// `LensIncluded` chain event (chunk_id, slot) and for `lens show`
/// debugging (slot_reason, feature_breakdown, truncated).
#[derive(Debug, Clone)]
pub struct LensEntry {
    pub chunk_id: String,
    pub source_kind: String,
    pub source_id: String,
    pub slot_name: &'static str,
    pub slot_reason: String,
    pub text_excerpt: String,
    pub feature_breakdown: BTreeMap<&'static str, FeatureAttribution>,
    pub total_score: f32,
    pub truncated: bool,
}

// ---------------------------------------------------------------------
// Builder
// ---------------------------------------------------------------------

/// Build a lens for the current `AttentionContext`.
///
/// P9b-min path:
///   1. Generate ambient candidates from the corpus against
///      `attn_ctx.scope_vector` (dense lane only — BM25 is off in
///      ambient mode by invariant).
///   2. Rank with an internally-constructed engine whose only
///      registered feature is `attention_affinity` at weight 1.0.
///   3. Filter the denylist facets.
///   4. Allocate the attention slot (up to 2 entries) honoring the
///      dominance check and the token budget.
///
/// Returns an empty lens when there are no candidates — the loop
/// will still call `emit_resource_updated`, but the rendered
/// markdown will say "no entries". Callers that want to suppress
/// empty lenses should check `entries.is_empty()`.
pub async fn build_lens(
    attn_ctx: &AttentionContext,
    corpus: &CorpusStore,
    config: &LensConfig,
) -> Result<Lens> {
    let candidates =
        ambient_candidates(corpus, attn_ctx, None, config.candidate_k_per_lane).await?;

    // P9b-min owns its own engine — there is exactly one feature in
    // play (attention_affinity) and no async prepare(). P9b-full
    // swaps to an externally-constructed engine that carries the
    // freshness / entity / concept features registered at boot.
    let engine = lens_engine();
    let ranked = engine
        .rank(candidates, &QueryContext::Ambient, attn_ctx)
        .await?;

    let filtered: Vec<RankedHit> = ranked
        .into_iter()
        .filter(|h| !has_denylist_facet(&h.candidate.chunk.facets, &config.exclude_facets))
        .collect();

    let entries = allocate_attention_portfolio(&filtered, config);

    Ok(Lens {
        entries,
        generated_at: Utc::now(),
        drift_basis: "rolling".into(),
        pinned: attn_ctx.pinned(),
    })
}

/// The P9b-min rank engine: a single attention-affinity feature at
/// weight 1.0. Built fresh per call because `RankEngine` is `!Clone`
/// — but it's cheap and stateless, so cost is irrelevant.
fn lens_engine() -> RankEngine {
    RankEngine::new().with_fn_feature("attention_affinity", 1.0, |c, _q, a| {
        attention_affinity_score(c, a)
    })
}

// ---------------------------------------------------------------------
// Allocator (P9b-min: attention slot only)
// ---------------------------------------------------------------------

/// Allocate up to 2 attention-dominant entries, honoring the token
/// budget. P9b-min is intentionally narrow: no freshness/entity/
/// concept/diversity slots, no refractory.
fn allocate_attention_portfolio(ranked: &[RankedHit], config: &LensConfig) -> Vec<LensEntry> {
    const SLOT: &str = "attention";
    const FEATURE: &str = "attention_affinity";
    const COUNT: usize = 2;

    let mut entries: Vec<LensEntry> = Vec::new();
    let mut tokens_used = 0_usize;
    let mut selected: HashSet<String> = HashSet::new();
    // Dedup by *content* (sha256), not only chunk_id: two distinct
    // observations of byte-identical text must not both take a slot — the
    // lens reflects distinct information, not repeated strikes of the same.
    let mut selected_content: HashSet<String> = HashSet::new();

    for _seat in 0..COUNT {
        // Pick the highest-scoring unselected hit where attention is
        // dominant. Without the dominance gate the slot would fill
        // with random high-RRF hits whose attention contribution was
        // accidental.
        let pick = ranked
            .iter()
            .filter(|h| !selected.contains(&h.candidate.chunk.chunk_id))
            .filter(|h| !selected_content.contains(&h.candidate.chunk.sha256))
            .filter(|h| is_attention_dominant(h, FEATURE, config.dominance_threshold))
            .max_by(|a, b| {
                let aa = attention_contribution(a, FEATURE);
                let bb = attention_contribution(b, FEATURE);
                aa.partial_cmp(&bb).unwrap_or(std::cmp::Ordering::Equal)
            });
        let Some(hit) = pick else {
            break;
        };

        let remaining = config.token_budget.saturating_sub(tokens_used);
        if remaining < config.min_excerpt_tokens {
            break;
        }
        let target = approximate_token_count(&hit.candidate.chunk.text).min(remaining);
        let excerpt = excerpt_centered_on_best_sentence(&hit.candidate.chunk.text, target);
        let actual = approximate_token_count(&excerpt);
        tokens_used += actual;

        let reason = if let Some(attr) = hit.features.get(FEATURE) {
            format!(
                "aligned with current focus (attention raw={:.2}, contribution={:.2})",
                attr.raw, attr.contribution
            )
        } else {
            "aligned with current focus".to_string()
        };

        let truncated = actual < approximate_token_count(&hit.candidate.chunk.text);

        entries.push(LensEntry {
            chunk_id: hit.candidate.chunk.chunk_id.clone(),
            source_kind: hit.candidate.chunk.source.as_str().to_string(),
            source_id: hit.candidate.chunk.source_id.clone(),
            slot_name: SLOT,
            slot_reason: reason,
            text_excerpt: excerpt,
            feature_breakdown: hit.features.clone(),
            total_score: hit.total,
            truncated,
        });
        selected.insert(hit.candidate.chunk.chunk_id.clone());
        selected_content.insert(hit.candidate.chunk.sha256.clone());
    }

    entries
}

fn is_attention_dominant(h: &RankedHit, feature: &str, threshold: f32) -> bool {
    if h.total <= f32::EPSILON {
        return false;
    }
    let Some(attr) = h.features.get(feature) else {
        return false;
    };
    (attr.contribution / h.total) >= threshold
}

fn attention_contribution(h: &RankedHit, feature: &str) -> f32 {
    h.features
        .get(feature)
        .map(|a| a.contribution)
        .unwrap_or(0.0)
}

// ---------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------

/// Rough whitespace-delimited token count. Good enough for budget
/// shaping; the LLM tokenizer will diverge by ~20% but the budget
/// is a soft cap.
pub(crate) fn approximate_token_count(text: &str) -> usize {
    text.split_whitespace().count()
}

/// Window the chunk text to approximately `target_tokens` tokens.
///
/// P9b-min v1: take the first `target_tokens` whitespace-delimited
/// tokens. The "centered on the highest-similarity sentence"
/// heuristic from the spec lands in P9b-full when entity / concept
/// signal exists to anchor on — for an attention-only lens, the
/// chunk's leading text already has the strongest topical signal
/// because that's how the embedder chunks them.
pub(crate) fn excerpt_centered_on_best_sentence(text: &str, target_tokens: usize) -> String {
    if target_tokens == 0 {
        return String::new();
    }
    let mut tokens: Vec<&str> = text.split_whitespace().take(target_tokens).collect();
    if tokens.len() < text.split_whitespace().count() {
        tokens.push("…");
    }
    tokens.join(" ")
}

/// True when any chunk facet matches a `key:value` entry in
/// `denylist`. Roundtrips through `facets::to_list` so values
/// containing `:` match correctly (per P1's escape contract).
pub(crate) fn has_denylist_facet(facets: &FacetSet, denylist: &[String]) -> bool {
    if denylist.is_empty() {
        return false;
    }
    let listed = facets::to_list(facets);
    let set: HashSet<&str> = listed.iter().map(String::as_str).collect();
    denylist.iter().any(|d| set.contains(d.as_str()))
}

// ---------------------------------------------------------------------
// Markdown rendering
// ---------------------------------------------------------------------

impl Lens {
    /// Render the lens to the markdown format spec'd in
    /// `p9b-lens-portfolio.md` "Lens markdown format". This is the
    /// payload the loop ships to the registry, and what gets
    /// fingerprinted for the unchanged-content skip — every byte
    /// matters.
    #[must_use]
    pub fn to_markdown(&self) -> String {
        let mut out = String::new();
        out.push_str("# Memory lens\n");
        let total_tokens: usize = self
            .entries
            .iter()
            .map(|e| approximate_token_count(&e.text_excerpt))
            .sum();
        out.push_str(&format!(
            "Refreshed at {ts}; pinned={pinned}; drift_basis={basis}.\n",
            ts = self.generated_at.format("%Y-%m-%dT%H:%MZ"),
            pinned = self.pinned,
            basis = self.drift_basis,
        ));
        out.push_str(&format!(
            "{n} entries, {used} tokens used.\n",
            n = self.entries.len(),
            used = total_tokens,
        ));
        if self.entries.is_empty() {
            out.push_str("\n_No attended content currently dominates the lens._\n");
            return out;
        }
        for entry in &self.entries {
            out.push_str("\n## [");
            out.push_str(entry.slot_name);
            out.push_str("] ");
            out.push_str(&entry.source_id);
            out.push_str(" — ");
            out.push_str(&entry.slot_reason);
            if entry.truncated {
                out.push_str(" (truncated)");
            }
            out.push('\n');
            for line in entry.text_excerpt.lines() {
                out.push_str("> ");
                out.push_str(line);
                out.push('\n');
            }
        }
        out
    }
}

// ---------------------------------------------------------------------
// AttentionContext convenience
// ---------------------------------------------------------------------

impl AttentionContext {
    /// Convenience: did the upstream lens loop see a pinned focus?
    /// The lens module wants a single bool for the rendered metadata;
    /// the loop owns the actual `effective_vec()` resolution.
    ///
    /// P9b-min approximation: any time `scope_vector` differs from
    /// `rolling_vec` (or rolling_vec is None while scope_vector is
    /// Some), a pin is driving ranking. P9b-full replaces this with
    /// an explicit `pinned` field carried in the snapshot.
    #[must_use]
    pub fn pinned(&self) -> bool {
        match (&self.scope_vector, &self.rolling_vec) {
            (Some(sv), Some(rv)) => sv != rv,
            (Some(_), None) => true,
            _ => false,
        }
    }
}

// ---------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::candidate::Candidate;
    use ostk_recall_core::{Chunk, FacetSet, Links, Source};
    use std::collections::BTreeSet;

    fn chunk(id: &str, text: &str, facets: FacetSet) -> Chunk {
        Chunk {
            chunk_id: id.into(),
            source: Source::Markdown,
            project: Some("test".into()),
            source_id: format!("{id}.md"),
            source_config_id: "test:cfg".into(),
            chunk_index: 0,
            ts: None,
            role: None,
            text: text.into(),
            sha256: format!("sha-{id}"),
            links: Links::default(),
            facets,
            embedding_input_sha256: format!("emb-{id}"),
            extra: serde_json::Value::Null,
        }
    }

    fn cand_with_embedding(id: &str, text: &str, embedding: Vec<f32>) -> Candidate {
        let mut c = Candidate::for_chunk(chunk(id, text, FacetSet::default()));
        c.dense_embedding = Some(embedding);
        c
    }

    #[test]
    fn approximate_token_count_counts_words() {
        assert_eq!(approximate_token_count(""), 0);
        assert_eq!(approximate_token_count("one two three"), 3);
    }

    #[test]
    fn excerpt_truncates_and_marks() {
        let text = "alpha bravo charlie delta echo foxtrot";
        let excerpt = excerpt_centered_on_best_sentence(text, 3);
        assert!(excerpt.starts_with("alpha bravo charlie"));
        assert!(
            excerpt.contains('…'),
            "truncated excerpts must include an ellipsis"
        );
    }

    #[test]
    fn excerpt_zero_target_returns_empty() {
        assert_eq!(excerpt_centered_on_best_sentence("alpha", 0), "");
    }

    #[test]
    fn excerpt_under_budget_returns_unchanged() {
        let text = "short text";
        let excerpt = excerpt_centered_on_best_sentence(text, 50);
        assert_eq!(excerpt, "short text");
    }

    #[test]
    fn denylist_matches_exact_key_value() {
        let mut facets: FacetSet = FacetSet::default();
        facets
            .entry("status".into())
            .or_default()
            .insert("archived".into());
        assert!(has_denylist_facet(&facets, &["status:archived".into()]));
        assert!(!has_denylist_facet(&facets, &["status:active".into()]));
    }

    #[test]
    fn denylist_handles_colon_in_value() {
        // Facets module roundtrips colons in values. The denylist
        // matcher must too.
        let mut facets: FacetSet = FacetSet::default();
        let mut vals = BTreeSet::new();
        vals.insert("ostk://memory-lens".into());
        facets.insert("path".into(), vals);
        // Encoded form per facets::to_list — values with `:` get
        // wrapped in quotes there.
        let encoded = facets::to_list(&facets);
        assert!(
            !encoded.is_empty(),
            "facets::to_list must produce an encoded entry for the denylist check"
        );
        // The denylist entry must use the same encoded form.
        let denylist = vec![encoded[0].clone()];
        assert!(has_denylist_facet(&facets, &denylist));
    }

    #[test]
    fn denylist_empty_does_not_match() {
        let facets = FacetSet::default();
        assert!(!has_denylist_facet(&facets, &[]));
    }

    #[test]
    fn attention_dominance_threshold_filters_low_contribution() {
        use crate::rank::{FeatureAttribution, RankedHit};
        let mut features: BTreeMap<&'static str, FeatureAttribution> = BTreeMap::new();
        // raw=0.5, weight=0.4 → contribution=0.2; total=1.0 → 20%
        // share. Default threshold (0.30) rejects.
        features.insert("attention_affinity", FeatureAttribution::new(0.5, 0.4));
        let h = RankedHit {
            candidate: cand_with_embedding("c1", "text", vec![1.0, 0.0]),
            total: 1.0,
            features,
        };
        assert!(!is_attention_dominant(&h, "attention_affinity", 0.30));
    }

    #[test]
    fn attention_dominance_threshold_accepts_high_contribution() {
        use crate::rank::{FeatureAttribution, RankedHit};
        let mut features: BTreeMap<&'static str, FeatureAttribution> = BTreeMap::new();
        features.insert("attention_affinity", FeatureAttribution::new(0.9, 1.0));
        let h = RankedHit {
            candidate: cand_with_embedding("c1", "text", vec![1.0, 0.0]),
            total: 0.9,
            features,
        };
        assert!(is_attention_dominant(&h, "attention_affinity", 0.30));
    }

    #[test]
    fn attention_dominance_rejects_zero_total() {
        // Defensive: a zero/negative total must never let a slot
        // claim dominance. Otherwise a feature returning 0 paired
        // with `total = 0` would pass the `>= 0.30` check trivially.
        use crate::rank::{FeatureAttribution, RankedHit};
        let mut features: BTreeMap<&'static str, FeatureAttribution> = BTreeMap::new();
        features.insert("attention_affinity", FeatureAttribution::new(0.0, 1.0));
        let h = RankedHit {
            candidate: cand_with_embedding("c1", "text", vec![1.0, 0.0]),
            total: 0.0,
            features,
        };
        assert!(!is_attention_dominant(&h, "attention_affinity", 0.30));
    }

    fn ranked_hit(id: &str, text: &str, raw: f32) -> RankedHit {
        use crate::rank::FeatureAttribution;
        let mut features: BTreeMap<&'static str, FeatureAttribution> = BTreeMap::new();
        let attr = FeatureAttribution::new(raw, 1.0);
        let total = attr.contribution;
        features.insert("attention_affinity", attr);
        RankedHit {
            candidate: cand_with_embedding(id, text, vec![1.0, 0.0]),
            total,
            features,
        }
    }

    #[test]
    fn allocator_picks_up_to_two_attention_entries_in_score_order() {
        let hits = vec![
            ranked_hit("c1", "alpha content one", 0.5),
            ranked_hit("c2", "bravo content two", 0.9),
            ranked_hit("c3", "charlie content three", 0.7),
        ];
        let entries = allocate_attention_portfolio(&hits, &LensConfig::default());
        let ids: Vec<&str> = entries.iter().map(|e| e.chunk_id.as_str()).collect();
        assert_eq!(ids, vec!["c2", "c3"], "highest two by contribution");
        for entry in &entries {
            assert_eq!(entry.slot_name, "attention");
            assert!(entry.slot_reason.contains("aligned"));
        }
    }

    #[test]
    fn allocator_dedups_byte_identical_content() {
        // Two distinct observations (different chunk_id) of byte-identical
        // text (same sha256). The lens reflects distinct information, so the
        // second must not take a slot — this is the two-identical-entries bug.
        let mut h1 = ranked_hit("c1", "same body text", 0.9);
        let mut h2 = ranked_hit("c2", "same body text", 0.8);
        h1.candidate.chunk.sha256 = "dup-sha".to_string();
        h2.candidate.chunk.sha256 = "dup-sha".to_string();
        let entries = allocate_attention_portfolio(&[h1, h2], &LensConfig::default());
        assert_eq!(entries.len(), 1, "second byte-identical candidate dropped");
        assert_eq!(entries[0].chunk_id, "c1", "highest-scoring kept");
    }

    #[test]
    fn allocator_respects_token_budget_floor() {
        // budget=300, min=200, two ~200-token chunks. First fits
        // (uses 200, leaves 100); second's floor (200) doesn't fit
        // → slot skipped. Result: one entry only.
        let big_text: String = (0..200)
            .map(|i| format!("w{i}"))
            .collect::<Vec<_>>()
            .join(" ");
        let hits = vec![
            ranked_hit("c1", &big_text, 0.9),
            ranked_hit("c2", &big_text, 0.8),
        ];
        let config = LensConfig {
            token_budget: 300,
            min_excerpt_tokens: 200,
            ..LensConfig::default()
        };
        let entries = allocate_attention_portfolio(&hits, &config);
        assert_eq!(entries.len(), 1, "second slot skipped under budget floor");
    }

    #[test]
    fn render_markdown_includes_metadata_and_entries() {
        let entries = vec![ranked_hit("c1", "alpha bravo charlie", 0.9)];
        let allocated = allocate_attention_portfolio(&entries, &LensConfig::default());
        let lens = Lens {
            entries: allocated,
            generated_at: Utc::now(),
            drift_basis: "rolling".into(),
            pinned: false,
        };
        let md = lens.to_markdown();
        assert!(md.starts_with("# Memory lens\n"));
        assert!(md.contains("drift_basis=rolling"));
        assert!(md.contains("pinned=false"));
        assert!(md.contains("## [attention]"));
        assert!(md.contains("> alpha bravo charlie"));
    }

    #[test]
    fn render_markdown_empty_state_message() {
        let lens = Lens {
            entries: Vec::new(),
            generated_at: Utc::now(),
            drift_basis: "rolling".into(),
            pinned: false,
        };
        let md = lens.to_markdown();
        assert!(md.contains("No attended content"));
    }

    #[test]
    fn pinned_helper_detects_divergence() {
        let mut ctx = AttentionContext::empty();
        ctx.scope_vector = Some(vec![1.0, 0.0]);
        ctx.rolling_vec = Some(vec![0.0, 1.0]);
        assert!(ctx.pinned(), "scope ≠ rolling → pin is driving");

        ctx.rolling_vec = Some(vec![1.0, 0.0]);
        assert!(!ctx.pinned(), "scope == rolling → no pin");

        ctx.rolling_vec = None;
        assert!(ctx.pinned(), "scope set, rolling None → pin newly placed");

        ctx.scope_vector = None;
        assert!(!ctx.pinned(), "empty mind → no pin");
    }
}
