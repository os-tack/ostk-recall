//! `thread_query` — the multi-signal verb (v0.4.1).
//!
//! Runs the three signal primitives (density, activity, novelty) against
//! the same recency window and joins their results into a unified list
//! of [`ThreadQueryReport`] rows. Each row carries:
//!
//! - per-axis scores (only the surfacing primitive populates a non-`None`
//!   value; cross-axis backfill is deferred to a follow-up release)
//! - a `composite_score` formed by the weighted sum of the populated
//!   axes, using caller-supplied `composite_weights` (default uniform)
//! - a [`ThreadQueryAttribution`] that breaks the composite down per
//!   axis (weight, score, contribution) so any aggregate can be argued
//!   with against the math, never the vibe
//!
//! ## Why a single verb
//!
//! The three legacy verbs (`thread_emergent`, `thread_attention`,
//! `thread_novelty`) each made a hidden bet about which axis matters.
//! The "bet IS sentiment" (see `.ostk/threads/post-v0.3.0.md`). Lifting
//! the rank-by axis from the verb name to a tool argument keeps the
//! substrate neutral and lets federated callers (Claude, Gemini, …)
//! ship their own weights without forking the substrate.
//!
//! ## Sentiment-trap discipline
//!
//! - Default `composite_weights` is **uniform** (1/3, 1/3, 1/3). Any
//!   non-uniform default would re-introduce the sentiment trap one
//!   layer up — the only honest default for a combiner is the
//!   uninformative one.
//! - Per-axis floors (`min_density`, `min_activity`, `min_novelty`)
//!   default to `0.0`. No baked filters.
//! - `signals` selects which axes to **compute**; default is all three.
//!   `signals: ["density"]` means "only the density primitive runs."
//!
//! ## Cluster identity
//!
//! Each row carries a `cluster_id` — a stable string the caller can use
//! to correlate the same group across queries. Derived from the
//! surfacing primitive's identity:
//!
//! - emergent → the `proposed-<8-hex>` handle (deterministic from
//!   member `chunk_ids` + date)
//! - activity → `burst:<project>:<source_id>`
//! - novelty → `novelty:<8-hex>` of a stable shape

use std::collections::HashMap;
use std::sync::Arc;

use chrono::{DateTime, Utc};
use ostk_recall_store::corpus::StoreError as CorpusError;
use ostk_recall_store::{CorpusStore, ThreadsDb};
use sha2::{Digest, Sha256};
use thiserror::Error;

use crate::activity::{
    AttentionBurstError, AttentionBurstReport, DEFAULT_DECAY_HOURS, DEFAULT_SAMPLES_PER_BURST,
    surface_attention,
};
use crate::cluster::{EMERGENT_THRESHOLD, MIN_NEIGHBOURS_IN_CLUSTER, mean_pairwise_cosine};
use crate::cosine_similarity;
use crate::emergent::{DEFAULT_MIN_CLUSTER_SIZE as EMERGENT_DEFAULT_MIN_CLUSTER, EmergentError,
    EmergentReport, discover_and_surface};
use crate::novelty::{
    DEFAULT_BASELINE_DAYS, DEFAULT_MIN_CLUSTER as NOVELTY_DEFAULT_MIN_CLUSTER, NoveltyError,
    NoveltyReport, surface_novelty,
};

/// Axis of a multi-signal cluster score. Stable string repr matches the
/// JSON-facing names in the MCP schema.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Axis {
    Density,
    Activity,
    Novelty,
    /// Cosine of the cluster's centroid against the scope's pinned
    /// focus vector. Opt-in only — `Axis::all()` (the default
    /// `signals` array) excludes it so callers that don't care
    /// about a focus pin keep getting v0.4.x behaviour. When opted
    /// in but no pin is set, every cluster's resonance_score stays
    /// `None` and contributes 0 to the composite (decomposable
    /// attribution discipline).
    Resonance,
}

impl Axis {
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Density => "density",
            Self::Activity => "activity",
            Self::Novelty => "novelty",
            Self::Resonance => "resonance",
        }
    }

    #[must_use]
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "density" => Some(Self::Density),
            "activity" => Some(Self::Activity),
            "novelty" => Some(Self::Novelty),
            "resonance" => Some(Self::Resonance),
            _ => None,
        }
    }

    /// All three v0.4.x axes, in canonical order. Used to seed the
    /// default `signals` array. Resonance is excluded by design —
    /// callers opt in by listing it in `signals` so the substrate
    /// never silently applies a focus-driven re-rank to a query
    /// that didn't ask for one.
    #[must_use]
    pub fn all() -> [Self; 3] {
        [Self::Density, Self::Activity, Self::Novelty]
    }

    /// Every known axis (including resonance). Used by
    /// `refresh_attribution` / `finalize` to enumerate the
    /// attribution rows so a Resonance opt-in shows up in the
    /// per-cluster attribution even when the surfacing primitive
    /// didn't supply it.
    #[must_use]
    pub fn all_known() -> [Self; 4] {
        [
            Self::Density,
            Self::Activity,
            Self::Novelty,
            Self::Resonance,
        ]
    }
}

/// Weight map over the four axes. Construction enforces non-negativity
/// and a non-zero sum (across the v0.4.x triad — resonance defaults to
/// 0 and doesn't participate in the fallback test); mismatched or
/// all-zero inputs fall back to the uniform (1/3, 1/3, 1/3, 0) default
/// so existing callers see no change.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CompositeWeights {
    pub density: f32,
    pub activity: f32,
    pub novelty: f32,
    /// Weight on the resonance axis (Phase E). Default 0.0 — opt-in
    /// so v0.4.x callers see unchanged composite scoring.
    pub resonance: f32,
}

impl Default for CompositeWeights {
    /// Uniform over the v0.4.x triad with resonance=0. Existing
    /// callers see no change in composite scoring; opting into
    /// resonance requires setting both `signals` and the resonance
    /// weight explicitly.
    fn default() -> Self {
        let third = 1.0_f32 / 3.0;
        Self {
            density: third,
            activity: third,
            novelty: third,
            resonance: 0.0,
        }
    }
}

impl CompositeWeights {
    /// v0.4.x constructor — resonance defaults to 0.0.
    #[must_use]
    pub fn new(density: f32, activity: f32, novelty: f32) -> Self {
        Self::new_with_resonance(density, activity, novelty, 0.0)
    }

    /// v0.5.x constructor — caller specifies all four weights.
    /// Used by `attention-mcp::handlers::thread_query` when the
    /// request carries `composite_weights.resonance`.
    #[must_use]
    pub fn new_with_resonance(
        density: f32,
        activity: f32,
        novelty: f32,
        resonance: f32,
    ) -> Self {
        let safe = |w: f32| if w.is_finite() && w >= 0.0 { w } else { 0.0 };
        let w = Self {
            density: safe(density),
            activity: safe(activity),
            novelty: safe(novelty),
            resonance: safe(resonance),
        };
        // Same fallback test as before — purely on the v0.4.x triad
        // so a caller that explicitly zeroes them while setting
        // resonance > 0 still falls back to uniform. That's a
        // judgement call; the alternative ("resonance can be the
        // sole axis") felt too far from the v0.4.x discipline of
        // never returning silently zero composites.
        if w.density + w.activity + w.novelty <= 0.0 {
            Self::default()
        } else {
            w
        }
    }

    #[must_use]
    pub fn for_axis(&self, axis: Axis) -> f32 {
        match axis {
            Axis::Density => self.density,
            Axis::Activity => self.activity,
            Axis::Novelty => self.novelty,
            Axis::Resonance => self.resonance,
        }
    }
}

/// Ranking choice. Drives which score determines the sort order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RankBy {
    Composite,
    Axis(Axis),
}

impl RankBy {
    #[must_use]
    pub fn parse(s: &str) -> Option<Self> {
        if s == "composite" {
            Some(Self::Composite)
        } else {
            Axis::parse(s).map(Self::Axis)
        }
    }

    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Composite => "composite",
            Self::Axis(a) => a.as_str(),
        }
    }
}

/// Per-axis attribution row. `score` is `None` when the axis was not
/// populated for this cluster (the surfacing primitive didn't cover it
/// and cross-axis backfill is deferred). `contribution = weight * score`
/// when both are `Some`, else `0.0` — so contributions are additive and
/// `composite_score` is exactly their sum.
#[derive(Debug, Clone, PartialEq)]
pub struct AxisAttribution {
    pub axis: Axis,
    pub weight: f32,
    pub score: Option<f32>,
    pub contribution: f32,
}

/// Composite-score breakdown attached to every [`ThreadQueryReport`].
/// Mirrors the spirit of `ScoreAttribution` on `AttentionPage`: any
/// aggregate score the substrate returns must come with its component
/// axes visible.
#[derive(Debug, Clone, PartialEq)]
pub struct ThreadQueryAttribution {
    pub axes: Vec<AxisAttribution>,
    pub composite: f32,
}

/// One cluster row returned by [`run_query`].
#[derive(Debug, Clone)]
pub struct ThreadQueryReport {
    pub cluster_id: String,
    pub origin: Axis,
    pub project: String,
    pub members: usize,
    /// Full cluster membership (sorted lexicographically). **Internal
    /// to the attention crate** — feeds the cross-axis backfill pass.
    /// Available to Rust callers that want to correlate the cluster
    /// across queries by exact chunk identity, but unbounded in size;
    /// MCP handlers must not echo it to the wire.
    pub chunk_ids: Vec<String>,
    pub density_score: Option<f32>,
    pub activity_score: Option<f32>,
    pub novelty_score: Option<f32>,
    /// Resonance with the operator's pinned focus (Phase E). `None`
    /// when the resonance axis wasn't requested OR no pin is set
    /// OR the cluster has no usable embeddings to derive a centroid
    /// from. Contributes 0 to the composite when `None`.
    pub resonance_score: Option<f32>,
    pub composite_score: f32,
    pub samples: Vec<String>,
    pub attribution: ThreadQueryAttribution,
}

/// Verb args — kept as a plain struct (not the JSON `Value`) so the
/// CLI surface and any future Rust caller can drive the same code path
/// the MCP handler uses.
#[derive(Debug, Clone)]
pub struct ThreadQueryParams {
    pub since_hours: i64,
    pub baseline_days: i64,
    pub signals: Vec<Axis>,
    pub rank_by: RankBy,
    pub composite_weights: CompositeWeights,
    pub min_density: f32,
    pub min_activity: f32,
    pub min_novelty: f32,
    pub min_resonance: f32,
    pub min_cluster_size: usize,
    pub limit: usize,
    pub samples_per_cluster: usize,
    /// Focus vector for the resonance axis. `Some(vec)` when the
    /// caller has a pinned focus (typically pulled from
    /// `attention.focus_status(scope).pinned.vec`); `None`
    /// otherwise. Set by the MCP handler — the query engine itself
    /// has no opinion about where this vector comes from.
    pub resonance_focus_vec: Option<Vec<f32>>,
}

impl Default for ThreadQueryParams {
    fn default() -> Self {
        Self {
            since_hours: 24,
            baseline_days: DEFAULT_BASELINE_DAYS,
            signals: Axis::all().to_vec(),
            rank_by: RankBy::Composite,
            composite_weights: CompositeWeights::default(),
            min_density: 0.0,
            min_activity: 0.0,
            min_novelty: 0.0,
            min_resonance: 0.0,
            min_cluster_size: 3,
            limit: 10,
            samples_per_cluster: DEFAULT_SAMPLES_PER_BURST,
            resonance_focus_vec: None,
        }
    }
}

#[derive(Debug, Error)]
pub enum ThreadQueryError {
    #[error("density (emergent) signal failed: {0}")]
    Emergent(#[from] EmergentError),
    #[error("activity signal failed: {0}")]
    Activity(#[from] AttentionBurstError),
    #[error("novelty signal failed: {0}")]
    Novelty(#[from] NoveltyError),
    #[error("corpus error during backfill: {0}")]
    Corpus(#[from] CorpusError),
}

/// Run a multi-signal query over the corpus + threads ledger.
///
/// Each enabled axis is computed by its primitive; results are mapped
/// into `ThreadQueryReport` rows, decorated with composite_score +
/// attribution, optionally floor-filtered per axis, sorted by `rank_by`,
/// and truncated to `limit`.
pub async fn run_query(
    corpus: &Arc<CorpusStore>,
    threads: &Arc<ThreadsDb>,
    params: ThreadQueryParams,
) -> Result<Vec<ThreadQueryReport>, ThreadQueryError> {
    let since: DateTime<Utc> = Utc::now() - chrono::Duration::hours(params.since_hours);
    let enabled = |a: Axis| params.signals.iter().any(|x| *x == a);

    let mut rows: Vec<ThreadQueryReport> = Vec::new();

    if enabled(Axis::Density) {
        let emergent_min = params.min_cluster_size.max(EMERGENT_DEFAULT_MIN_CLUSTER);
        let reports = discover_and_surface(
            corpus,
            threads,
            since,
            500,
            emergent_min,
            EMERGENT_THRESHOLD,
            MIN_NEIGHBOURS_IN_CLUSTER,
            false, // never persist from thread_query — discovery only
        )
        .await?;
        for r in reports {
            rows.push(from_emergent(r, &params));
        }
    }

    if enabled(Axis::Activity) {
        let reports = surface_attention(
            corpus,
            since,
            params.limit.saturating_mul(2).max(params.limit),
            params.samples_per_cluster,
            DEFAULT_DECAY_HOURS,
        )
        .await?;
        for r in reports {
            rows.push(from_activity(r, &params));
        }
    }

    if enabled(Axis::Novelty) {
        let novelty_min = params.min_cluster_size.max(NOVELTY_DEFAULT_MIN_CLUSTER);
        let reports = surface_novelty(
            corpus,
            since,
            params.baseline_days,
            params.limit.saturating_mul(2).max(params.limit),
            novelty_min,
            EMERGENT_THRESHOLD,
            0.0,
        )
        .await?;
        for r in reports {
            rows.push(from_novelty(r, &params));
        }
    }

    // Cross-axis backfill — every cluster gets every enabled axis
    // scored against its own membership, so composite_score is honest
    // for "activity ∩ novelty"-style questions rather than degenerating
    // to the single surfacing axis. Skipped for clusters with no
    // chunk_ids (defensive — should not happen post-v0.4.2 since every
    // primitive carries them).
    backfill_cross_axis(corpus, &mut rows, since, &params).await?;

    rows.retain(|r| {
        let d_ok = r
            .density_score
            .map_or(true, |s| s >= params.min_density);
        let a_ok = r
            .activity_score
            .map_or(true, |s| s >= params.min_activity);
        let n_ok = r
            .novelty_score
            .map_or(true, |s| s >= params.min_novelty);
        let res_ok = r
            .resonance_score
            .map_or(true, |s| s >= params.min_resonance);
        d_ok && a_ok && n_ok && res_ok
    });

    rows.sort_by(|a, b| {
        let key = |r: &ThreadQueryReport| -> f32 {
            match params.rank_by {
                RankBy::Composite => r.composite_score,
                RankBy::Axis(Axis::Density) => r.density_score.unwrap_or(0.0),
                RankBy::Axis(Axis::Activity) => r.activity_score.unwrap_or(0.0),
                RankBy::Axis(Axis::Novelty) => r.novelty_score.unwrap_or(0.0),
                RankBy::Axis(Axis::Resonance) => r.resonance_score.unwrap_or(0.0),
            }
        };
        key(b)
            .partial_cmp(&key(a))
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    rows.truncate(params.limit);
    Ok(rows)
}

// --- cross-axis backfill ----------------------------------------------

/// For each cluster, compute the axis scores the surfacing primitive
/// didn't supply. Activity needs per-chunk timestamps; density needs
/// embeddings; novelty needs embeddings + the per-project baseline.
/// Embeddings and timestamps are fetched in two corpus queries each
/// (one batched call per kind), so the cost is bounded by the total
/// number of cluster members, not by the number of clusters.
async fn backfill_cross_axis(
    corpus: &Arc<CorpusStore>,
    rows: &mut [ThreadQueryReport],
    since: DateTime<Utc>,
    params: &ThreadQueryParams,
) -> Result<(), ThreadQueryError> {
    // Collect every chunk_id we might need, deduplicated. Skip clusters
    // that already have all enabled axes populated (no work to do).
    // Resonance always triggers a fetch when enabled + a focus vec is
    // present, since no surfacing primitive supplies it.
    let resonance_active = params.signals.contains(&Axis::Resonance)
        && params.resonance_focus_vec.is_some();
    let mut all_ids: Vec<String> = Vec::new();
    for r in rows.iter() {
        let needs_any = (params.signals.contains(&Axis::Density) && r.density_score.is_none())
            || (params.signals.contains(&Axis::Activity) && r.activity_score.is_none())
            || (params.signals.contains(&Axis::Novelty) && r.novelty_score.is_none())
            || (resonance_active && r.resonance_score.is_none());
        if needs_any {
            all_ids.extend(r.chunk_ids.iter().cloned());
        }
    }
    if all_ids.is_empty() {
        return Ok(());
    }
    all_ids.sort();
    all_ids.dedup();

    // One shot each for the two corpus-side lookups.
    let embeddings: HashMap<String, Vec<f32>> = corpus.fetch_embeddings(&all_ids).await?;
    let timestamps: HashMap<String, DateTime<Utc>> = corpus.fetch_timestamps(&all_ids).await?;

    // Baselines for novelty backfill — keyed by Option<project>, with a
    // None fallback for clusters whose project is the empty string.
    let mut baselines: HashMap<Option<String>, Option<Vec<f32>>> = HashMap::new();
    if params.signals.contains(&Axis::Novelty) {
        baselines.insert(None, corpus.project_baseline_mean(None, params.baseline_days).await?);
        let mut projects: Vec<String> = rows
            .iter()
            .filter(|r| !r.project.is_empty())
            .map(|r| r.project.clone())
            .collect();
        projects.sort();
        projects.dedup();
        for p in projects {
            let b = corpus.project_baseline_mean(Some(&p), params.baseline_days).await?;
            baselines.insert(Some(p), b);
        }
    }

    let now = Utc::now();
    #[allow(clippy::cast_precision_loss)]
    let decay_hours_recip = 1.0_f32 / DEFAULT_DECAY_HOURS.max(0.01);

    for r in rows.iter_mut() {
        if r.chunk_ids.is_empty() {
            continue;
        }
        // Density: average pairwise cosine over the cluster's embeddings.
        if params.signals.contains(&Axis::Density) && r.density_score.is_none() {
            let vecs: Vec<Vec<f32>> = r
                .chunk_ids
                .iter()
                .filter_map(|id| embeddings.get(id).cloned())
                .collect();
            if vecs.len() >= 2 {
                r.density_score = Some(mean_pairwise_cosine(&vecs));
            }
        }
        // Activity: count_in_window * exp(-(now - max_ts) / decay_hours).
        if params.signals.contains(&Axis::Activity) && r.activity_score.is_none() {
            let mut in_window: usize = 0;
            let mut max_ts: Option<DateTime<Utc>> = None;
            for id in &r.chunk_ids {
                if let Some(&ts) = timestamps.get(id) {
                    if ts >= since {
                        in_window += 1;
                    }
                    max_ts = Some(max_ts.map_or(ts, |cur| cur.max(ts)));
                }
            }
            if in_window > 0 {
                #[allow(clippy::cast_precision_loss)]
                let count = in_window as f32;
                let decay = max_ts.map_or(1.0, |t| {
                    let dt_hours = ((now - t).num_seconds().max(0) as f32) / 3600.0;
                    (-dt_hours * decay_hours_recip).exp()
                });
                r.activity_score = Some(count * decay);
            }
        }
        // Novelty: mean of (1 - cos(emb, baseline)) over the cluster.
        // Falls back to the global baseline if the project's is None.
        if params.signals.contains(&Axis::Novelty) && r.novelty_score.is_none() {
            let baseline = lookup_baseline(&baselines, &r.project);
            if let Some(b) = baseline {
                let mut total = 0.0_f32;
                let mut n: u32 = 0;
                for id in &r.chunk_ids {
                    if let Some(v) = embeddings.get(id) {
                        total += 1.0 - cosine_similarity(v, b);
                        n += 1;
                    }
                }
                if n > 0 {
                    r.novelty_score = Some(total / f32::from(u16::try_from(n).unwrap_or(u16::MAX)));
                }
            }
        }
        // Resonance: cos(cluster_centroid, focus_vec). Clamped to
        // [0, 1] so the axis composes with the others under the
        // same `[0, 1]` contract. Skipped when the focus vec is
        // absent (no pin) — resonance_score stays None and
        // contributes 0 to the composite.
        if resonance_active && r.resonance_score.is_none() {
            if let Some(focus) = params.resonance_focus_vec.as_ref() {
                let centroid = mean_vec(&r.chunk_ids, &embeddings);
                if let Some(c) = centroid {
                    r.resonance_score = Some(cosine_similarity(&c, focus).clamp(0.0, 1.0));
                }
            }
        }
        // Re-finalize composite_score + attribution to reflect newly
        // populated axes. Cheap; runs once per cluster.
        refresh_attribution(r, params);
    }

    Ok(())
}

/// Mean vector over the embeddings present in `embeddings` for the
/// given chunk_ids. Returns `None` when no chunk_id resolves —
/// callers treat that as "centroid undefined, skip the axis."
fn mean_vec(
    chunk_ids: &[String],
    embeddings: &HashMap<String, Vec<f32>>,
) -> Option<Vec<f32>> {
    let mut sum: Option<Vec<f32>> = None;
    let mut n: u32 = 0;
    for id in chunk_ids {
        if let Some(v) = embeddings.get(id) {
            if v.is_empty() {
                continue;
            }
            match &mut sum {
                Some(acc) if acc.len() == v.len() => {
                    for (a, b) in acc.iter_mut().zip(v.iter()) {
                        *a += b;
                    }
                }
                Some(_) => {
                    // Dim mismatch within the cluster — shouldn't
                    // happen but better to abandon than corrupt.
                    return None;
                }
                None => sum = Some(v.clone()),
            }
            n = n.saturating_add(1);
        }
    }
    let mut acc = sum?;
    if n == 0 {
        return None;
    }
    #[allow(clippy::cast_precision_loss)]
    let inv = 1.0_f32 / n as f32;
    for x in &mut acc {
        *x *= inv;
    }
    Some(acc)
}

fn lookup_baseline<'a>(
    cache: &'a HashMap<Option<String>, Option<Vec<f32>>>,
    project: &str,
) -> Option<&'a Vec<f32>> {
    if !project.is_empty() {
        if let Some(Some(v)) = cache.get(&Some(project.to_string())) {
            return Some(v);
        }
    }
    if let Some(Some(v)) = cache.get(&None) {
        return Some(v);
    }
    None
}

fn refresh_attribution(row: &mut ThreadQueryReport, p: &ThreadQueryParams) {
    let weights = p.composite_weights;
    let mut axes: Vec<AxisAttribution> = Vec::with_capacity(4);
    let mut composite = 0.0_f32;
    for axis in Axis::all_known() {
        let score = match axis {
            Axis::Density => row.density_score,
            Axis::Activity => row.activity_score,
            Axis::Novelty => row.novelty_score,
            Axis::Resonance => row.resonance_score,
        };
        let weight = weights.for_axis(axis);
        let contribution = score.map_or(0.0, |s| weight * s);
        composite += contribution;
        axes.push(AxisAttribution {
            axis,
            weight,
            score,
            contribution,
        });
    }
    row.composite_score = composite;
    row.attribution = ThreadQueryAttribution { axes, composite };
}

// --- per-primitive adapters -------------------------------------------

fn from_emergent(r: EmergentReport, p: &ThreadQueryParams) -> ThreadQueryReport {
    let mut row = base_row(
        r.handle.clone(),
        Axis::Density,
        String::new(),
        r.members,
        r.chunk_ids,
        r.samples,
    );
    row.density_score = Some(r.cohesion);
    finalize(row, p)
}

fn from_activity(r: AttentionBurstReport, p: &ThreadQueryParams) -> ThreadQueryReport {
    let cluster_id = format!("burst:{}:{}", r.project, r.source_id);
    let mut row = base_row(
        cluster_id,
        Axis::Activity,
        r.project,
        r.count,
        r.chunk_ids,
        r.samples,
    );
    row.activity_score = Some(r.score);
    finalize(row, p)
}

fn from_novelty(r: NoveltyReport, p: &ThreadQueryParams) -> ThreadQueryReport {
    // chunk_ids are sorted lexicographically by the primitive, so the
    // hash is stable across runs over the same corpus.
    let mut h = Sha256::new();
    h.update(r.project.as_bytes());
    h.update(b":");
    h.update(r.members.to_le_bytes());
    for id in &r.chunk_ids {
        h.update(b"|");
        h.update(id.as_bytes());
    }
    let cluster_id = format!("novelty:{}", &hex_short(&h.finalize()));
    let mut row = base_row(
        cluster_id,
        Axis::Novelty,
        r.project,
        r.members,
        r.chunk_ids,
        r.samples,
    );
    row.novelty_score = Some(r.mean_novelty);
    finalize(row, p)
}

fn base_row(
    cluster_id: String,
    origin: Axis,
    project: String,
    members: usize,
    chunk_ids: Vec<String>,
    samples: Vec<String>,
) -> ThreadQueryReport {
    ThreadQueryReport {
        cluster_id,
        origin,
        project,
        members,
        chunk_ids,
        density_score: None,
        activity_score: None,
        novelty_score: None,
        resonance_score: None,
        composite_score: 0.0,
        samples,
        attribution: ThreadQueryAttribution {
            axes: Vec::new(),
            composite: 0.0,
        },
    }
}

fn finalize(mut row: ThreadQueryReport, p: &ThreadQueryParams) -> ThreadQueryReport {
    refresh_attribution(&mut row, p);
    row
}

fn hex_short(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(8);
    for b in bytes.iter().take(4) {
        s.push_str(&format!("{b:02x}"));
    }
    s
}

#[cfg(test)]
#[allow(clippy::float_cmp)]
mod tests {
    use super::*;

    #[test]
    fn composite_weights_default_uniform() {
        let w = CompositeWeights::default();
        let third = 1.0_f32 / 3.0;
        assert!((w.density - third).abs() < 1e-6);
        assert!((w.activity - third).abs() < 1e-6);
        assert!((w.novelty - third).abs() < 1e-6);
    }

    #[test]
    fn composite_weights_reject_negative_and_zero_sum() {
        let w = CompositeWeights::new(-1.0, 0.0, 0.0);
        assert_eq!(w, CompositeWeights::default(), "all-zero falls back to uniform");
        let w2 = CompositeWeights::new(2.0, 1.0, 1.0);
        assert!((w2.density - 2.0).abs() < 1e-6);
        assert!((w2.activity - 1.0).abs() < 1e-6);
    }

    #[test]
    fn axis_parse_round_trip() {
        for a in Axis::all() {
            assert_eq!(Axis::parse(a.as_str()), Some(a));
        }
        assert_eq!(Axis::parse("bogus"), None);
    }

    #[test]
    fn rank_by_parse() {
        assert_eq!(RankBy::parse("composite"), Some(RankBy::Composite));
        assert_eq!(
            RankBy::parse("density"),
            Some(RankBy::Axis(Axis::Density))
        );
        assert_eq!(RankBy::parse("nope"), None);
    }

    #[test]
    fn attribution_is_decomposable() {
        // A single-axis-populated row's composite is exactly
        // weight[axis] * score, every other contribution zero.
        let p = ThreadQueryParams::default();
        let r = NoveltyReport {
            project: "p".into(),
            members: 4,
            mean_novelty: 0.9,
            max_novelty: 1.0,
            chunk_ids: vec!["c1".into(), "c2".into(), "c3".into(), "c4".into()],
            samples: vec!["s".into()],
        };
        let row = from_novelty(r, &p);
        let third = 1.0_f32 / 3.0;
        assert_eq!(row.novelty_score, Some(0.9));
        assert!((row.composite_score - third * 0.9).abs() < 1e-6);
        let by_axis: std::collections::HashMap<Axis, &AxisAttribution> =
            row.attribution.axes.iter().map(|a| (a.axis, a)).collect();
        assert_eq!(by_axis[&Axis::Density].score, None);
        assert_eq!(by_axis[&Axis::Density].contribution, 0.0);
        assert_eq!(by_axis[&Axis::Novelty].score, Some(0.9));
        assert!((by_axis[&Axis::Novelty].contribution - third * 0.9).abs() < 1e-6);
        // Contributions sum to composite — the substrate-level promise
        // that aggregates decompose cleanly.
        let sum: f32 = row.attribution.axes.iter().map(|a| a.contribution).sum();
        assert!((sum - row.composite_score).abs() < 1e-6);
    }

    #[test]
    fn rank_by_axis_uses_axis_score() {
        // Two rows: one with a high novelty, one with low; ranking
        // by novelty must put the high one first regardless of
        // composite weights or other axes' presence.
        let p = ThreadQueryParams {
            rank_by: RankBy::Axis(Axis::Novelty),
            ..ThreadQueryParams::default()
        };
        let high = from_novelty(
            NoveltyReport {
                project: "p".into(),
                members: 3,
                mean_novelty: 1.5,
                max_novelty: 1.8,
                chunk_ids: vec!["h1".into(), "h2".into(), "h3".into()],
                samples: vec!["h".into()],
            },
            &p,
        );
        let low = from_novelty(
            NoveltyReport {
                project: "p".into(),
                members: 3,
                mean_novelty: 0.2,
                max_novelty: 0.3,
                chunk_ids: vec!["l1".into(), "l2".into(), "l3".into()],
                samples: vec!["l".into()],
            },
            &p,
        );
        let mut rows = vec![low.clone(), high.clone()];
        rows.sort_by(|a, b| {
            let key = |r: &ThreadQueryReport| r.novelty_score.unwrap_or(0.0);
            key(b)
                .partial_cmp(&key(a))
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        assert_eq!(rows[0].cluster_id, high.cluster_id);
    }
}
