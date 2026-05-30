//! P7b — ACT-R base-activation "Freshness" rank feature.
//!
//! Turns chunk recency into *cognitive* base activation: a chunk is
//! fresher the more — and more recently — it has been retrieved, not
//! merely created. `B_i = ln(1 + Σ_k weight_k · age_hours_k^{-d})` over
//! every access event in the window (plus a synthetic `Creation` access
//! from `chunk.ts`), with `d = 0.5` and `age_hours = max(hours, 1)`.
//! Min-max normalized across the candidate pool so `score()` is `[0, 1]`.
//!
//! The first hand-written stateful [`RankFeatureInstance`] (P3B): the
//! per-query `access_history` I/O happens once in async `prepare()`;
//! `score()` is pure lookup. Degrades to `chunk.ts`-only (creation
//! recency) when no [`ChainLogReader`] is wired into the
//! [`AttentionContext`] (explicit recall today; P9b-full wires the lens).
//!
//! **Not yet registered in any live engine** — the freshness portfolio
//! slot is P9b-full. P7b ships the tested mechanism + the ledger.

use std::collections::HashMap;

use async_trait::async_trait;
use chrono::{Duration, Utc};
use ostk_recall_store::{AccessKind, AccessWeights};

use crate::candidate::Candidate;
use crate::context::{AttentionContext, QueryContext};
use crate::error::Result;
use crate::rank::{RankFeatureFactory, RankFeatureInstance};

/// ACT-R decay exponent. Higher = older accesses fade faster.
pub const DEFAULT_DECAY_D: f32 = 0.5;
/// How far back the access ledger is consulted.
pub const DEFAULT_WINDOW_DAYS: i64 = 30;

/// Immutable config for the Freshness feature; cheap to share via `Arc`.
#[derive(Debug, Clone)]
pub struct FreshnessFactory {
    weights: AccessWeights,
    decay_d: f32,
    window: Duration,
}

impl FreshnessFactory {
    #[must_use]
    pub const fn new(weights: AccessWeights, decay_d: f32, window: Duration) -> Self {
        Self {
            weights,
            decay_d,
            window,
        }
    }
}

impl Default for FreshnessFactory {
    fn default() -> Self {
        Self {
            weights: AccessWeights::default(),
            decay_d: DEFAULT_DECAY_D,
            window: Duration::days(DEFAULT_WINDOW_DAYS),
        }
    }
}

impl RankFeatureFactory for FreshnessFactory {
    fn name(&self) -> &'static str {
        "freshness"
    }
    fn build_instance(&self) -> Box<dyn RankFeatureInstance> {
        Box::new(FreshnessInstance {
            weights: self.weights,
            decay_d: self.decay_d,
            window: self.window,
            pool_min: 0.0,
            pool_max: 0.0,
            raw: HashMap::new(),
        })
    }
}

/// Per-query instance. Scratch state (`raw` base activations + pool
/// min/max) lives here and is dropped when `rank()` returns.
pub struct FreshnessInstance {
    weights: AccessWeights,
    decay_d: f32,
    window: Duration,
    pool_min: f32,
    pool_max: f32,
    raw: HashMap<String, f32>,
}

#[async_trait]
impl RankFeatureInstance for FreshnessInstance {
    fn name(&self) -> &'static str {
        "freshness"
    }

    #[allow(clippy::cast_precision_loss)]
    async fn prepare(
        &mut self,
        candidates: &mut [Candidate],
        _query: &QueryContext,
        attn: &AttentionContext,
    ) -> Result<()> {
        let now = Utc::now();
        let since = now - self.window;
        let ids: Vec<String> = candidates.iter().map(|c| c.chunk.chunk_id.clone()).collect();

        // Pull retrieval history once for the whole pool. Absent reader
        // (explicit recall today) → empty history → creation-recency only.
        let mut history = match attn.chain_log.as_ref() {
            Some(reader) => reader.access_history(&ids, since)?,
            None => HashMap::new(),
        };

        let mut raw = HashMap::with_capacity(candidates.len());
        for c in candidates.iter() {
            let mut events = history.remove(&c.chunk.chunk_id).unwrap_or_default();
            // Creation counts as a (creation-weighted) access — this is the
            // mtime-recency baseline the ledger enriches, not replaces.
            if let Some(ts) = c.chunk.ts {
                events.push((AccessKind::Creation, ts));
            }
            let b_i = if events.is_empty() {
                0.0
            } else {
                let sum: f32 = events
                    .iter()
                    .map(|(kind, ts)| {
                        // Floor at 1h so a just-created/just-accessed chunk
                        // doesn't blow age^-d up, and a future ts (clock skew)
                        // can't go negative.
                        let age_hours = (now - *ts).num_hours().max(1) as f32;
                        self.weights.weight_of(*kind) * age_hours.powf(-self.decay_d)
                    })
                    .sum();
                // ln(1 + sum); ln_1p is the accurate form for small sums.
                sum.ln_1p()
            };
            raw.insert(c.chunk.chunk_id.clone(), b_i);
        }

        self.pool_min = raw.values().copied().fold(f32::INFINITY, f32::min);
        self.pool_max = raw.values().copied().fold(f32::NEG_INFINITY, f32::max);
        self.raw = raw;
        Ok(())
    }

    fn score(&self, candidate: &Candidate, _query: &QueryContext, _attn: &AttentionContext) -> f32 {
        let raw = self
            .raw
            .get(&candidate.chunk.chunk_id)
            .copied()
            .unwrap_or(0.0);
        // Flat pool (all equal, or a single candidate) carries no
        // discriminating information → neutral 0.5.
        if self.pool_max - self.pool_min < 1e-6 {
            return 0.5;
        }
        (raw - self.pool_min) / (self.pool_max - self.pool_min)
    }
}
