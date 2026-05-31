//! Hybrid dense + BM25 retrieval over the corpus table.
//!
//! P3A: this module now drives per-lane candidate generation via
//! `crates/query/src/lanes.rs` and explicit RRF fusion via the new
//! `RankEngine`. Lance's internal `RRFReranker` is no longer called
//! from this path — per-lane rank + score is captured on each
//! `Candidate` so downstream features (P9b lens portfolio, P3B/P4/P7
//! enrichment) can score with full attribution.
//!
//! Backward-compat output: `recall` still returns `Vec<RecallHit>`.
//! Internally, the pipeline is:
//!
//! ```text
//! lane_bm25 ╮
//!           ├── build_candidates → Vec<Candidate>
//! lane_dense ╯           ↓
//!                  RankEngine{Rrf=1.0}.rank → Vec<RankedHit>
//!                                          ↓
//!                                Convert → Vec<RecallHit>
//!                                          ↓
//!                  cross-encoder rerank → identifier boost
//!                                       → source-id diversify → truncate
//! ```
//!
//! The four existing v0.5 heuristics are preserved:
//! - Stratified code prefetch: an additional `lane_dense` filtered to
//!   `source = 'code'` when the caller hasn't bound `source` itself.
//! - Cross-encoder rerank (jina-turbo) via `RerankerLike`.
//! - Identifier code boost (additive +3.0 on snippet-matching code rows).
//! - Per-`source_id` diversification.

use std::collections::HashMap;

use chrono::{DateTime, Utc};
use ostk_recall_core::{Chunk, ContextRole, Source};
use ostk_recall_pipeline::ChunkEmbedder;
use ostk_recall_store::{CORPUS_TABLE, CorpusStore};

use crate::candidate::Candidate;
use crate::context::{AttentionContext, QueryContext};
use crate::error::Result;
use crate::lanes::{LaneEntry, build_candidates, lane_bm25, lane_dense};
use crate::rank::RankedHit;
use crate::rerank::RerankerLike;
use crate::row::{snippet_of, sql_escape};
use crate::types::{RecallHit, RecallParams};

/// Default cap on hits per `source_id` after RRF rerank. One chatty session
/// could otherwise dominate top-K via shared tokens.
pub const DEFAULT_MAX_PER_SOURCE_ID: usize = 3;

/// How many candidates to fetch from `LanceDB` relative to the requested
/// `limit`.
///
/// Leaves room after the per-`source_id` post-filter and the cross-encoder
/// rerank pass. Bumped from 4× → 6× because the cross-encoder needs a
/// wider candidate pool to find precise matches that BM25/dense scored
/// mid-pack.
pub const PREFETCH_MULTIPLIER: usize = 6;

/// Extra candidates pulled from `source = 'code'` when the caller hasn't
/// filtered by source.
///
/// The 442 k-chunk corpus is ~1 % code by volume — without this stratified
/// prefetch, a top-`PREFETCH_MULTIPLIER * limit` unfiltered query
/// statistically misses every code candidate. This boost guarantees the
/// cross-encoder reranker sees code rows whenever they exist for a query.
pub const STRATIFIED_CODE_PREFETCH: usize = 12;

/// Additive boost applied to `source = "code"` post-rerank scores when
/// [`is_identifier_query`] flags the query as identifier-shaped (a
/// `snake_case` / `CamelCase` symbol name or a single short token).
///
/// Tuning: 1.5 wasn't enough to overcome cross-encoder's strong
/// preference for conversation transcripts that contain the literal
/// identifier many times. 3.0 ensures code chunks win for identifier
/// queries while staying low enough that irrelevant code (other files
/// with similar tokens) stays below prose answers for non-identifier
/// queries.
///
/// Set to `0.0` to disable the heuristic without removing the call site.
pub const IDENTIFIER_CODE_BOOST: f32 = 3.0;

/// Soft-sigmoid normalization constant for the `Bm25` rank feature
/// (`raw_bm25 / (raw_bm25 + K_BM25)` → bounded `(0, 1)`).
///
/// **Inert in alpha.1**: the `Bm25` feature ships at weight `0.0`,
/// so this constant has no effect on retrieval ordering today. P5
/// sweeps the weight + this constant together. Exposed as a public
/// constant + per-call override on `RankingOverrides::k_bm25` so
/// the eventual tuner can vary it without code edits.
pub const K_BM25: f32 = 10.0;

/// Full-strength penalty subtracted from a freshly self-authored chunk
/// in the post-rerank salience layer (`dampen_self_reference_recency`).
///
/// The membrane captures the present by design — but its own
/// freshly-written narration then dominates an explicit `recall` of the
/// very topic it describes (the live "this session's own output tops
/// recall of its own topic" regression). The cross-encoder owns
/// *relevance*; this is the *salience* axis layered on top: self-authored
/// narration of the present is anti-salient when retrieving the corpus it
/// narrates. On the cross-encoder's score scale (≈ identifier boost's
/// +3.0); `0.0` disables via `RankingOverrides::self_reference_penalty`.
///
/// **Forward-compat (P7b):** this is the explicit-path debut of the
/// salience axis. When the P7b access ledger lands, the crude
/// `source==membrane` + `ts`-recency proxy should become a principled
/// creation-vs-access term — penalize creation-only-recency, but never a
/// chunk with `ExplicitRecall`/`OperatorSelected` history (proven-useful
/// memory). The application point (post-rerank, additive, attributed)
/// stays; only the signal source upgrades.
pub const SELF_REFERENCE_PENALTY: f32 = 2.0;

/// Hours over which the self-reference penalty decays linearly to zero.
/// A just-created membrane chunk takes the full penalty; one older than
/// this window is untouched (legitimate historical recognition). Keyed
/// on `chunk.ts` (creation) until the P7b ledger provides true access
/// history.
pub const SELF_REFERENCE_RECENCY_WINDOW_HOURS: f32 = 24.0;

/// Execute a hybrid recall against the corpus table.
///
/// Pipeline:
/// 1. Dense + BM25 retrieval, fused by RRF in `LanceDB` → ~`limit *
///    PREFETCH_MULTIPLIER` candidates.
/// 2. Soft-stratified augmentation: when the caller hasn't filtered by
///    source, run a second targeted query with `source = 'code'` to pull
///    [`STRATIFIED_CODE_PREFETCH`] additional candidates. The 442k-chunk
///    corpus is ~1 % code; without this, code candidates rarely reach
///    the reranker. Results are merged (dedupe by `chunk_id`).
/// 3. Optional cross-encoder rerank: if `reranker` is `Some`, score each
///    candidate's full text against `query` and reorder by the new score.
///    Without a reranker, the RRF-fused order is preserved.
/// 4. Per-`source_id` diversity filter, truncated to `limit`.
pub async fn recall(
    store: &CorpusStore,
    embedder: &dyn ChunkEmbedder,
    reranker: Option<&dyn RerankerLike>,
    params: &RecallParams,
) -> Result<Vec<RecallHit>> {
    let query_text = params.query.trim();
    if query_text.is_empty() {
        return Ok(Vec::new());
    }

    let limit = params.limit.unwrap_or(10).max(1);
    let max_per_source_id = params
        .max_per_source_id
        .unwrap_or(DEFAULT_MAX_PER_SOURCE_ID);
    // Pre-fetch more candidates than `limit` so the diversity filter and
    // the cross-encoder rerank both have room to operate.
    let fetch_limit = limit.saturating_mul(PREFETCH_MULTIPLIER).max(limit);

    // Resolve P3A per-call overrides. `None` field → compiled-in default.
    // P5 / file-config will populate `ranking_overrides` from
    // `[ranking.stages]` / `[ranking.weights]` and pass through unchanged.
    let overrides = params.ranking_overrides.clone().unwrap_or_default();
    let stratified_prefetch = overrides
        .stratified_code_prefetch
        .unwrap_or(STRATIFIED_CODE_PREFETCH);
    let identifier_boost = overrides
        .identifier_code_boost
        .unwrap_or(IDENTIFIER_CODE_BOOST);
    let _k_bm25 = overrides.k_bm25.unwrap_or(K_BM25); // wired when Bm25 feature lands
    let self_reference_penalty = overrides
        .self_reference_penalty
        .unwrap_or(SELF_REFERENCE_PENALTY);

    let vec = embedder
        .encode_batch(&[query_text])
        .into_iter()
        .next()
        .unwrap_or_default();

    let conn = store.connection();
    let table = conn.open_table(CORPUS_TABLE).execute().await?;

    let primary_filter = build_filter(
        params.project.as_deref(),
        params.source.as_deref(),
        params.since,
        params.before,
    );

    // Per-lane queries. Lance executes these sequentially against the
    // same table; running them via join_all gives a small win on cold
    // index pages but adds complexity — keep sequential for P3A.
    let bm25 = lane_bm25(&table, query_text, primary_filter.as_deref(), fetch_limit).await?;
    let dense = lane_dense(&table, &vec, primary_filter.as_deref(), fetch_limit).await?;

    // Stratified code prefetch: when the caller hasn't filtered by
    // source, top up the dense lane with a code-only dense pass so the
    // reranker always sees code candidates. Mirrors v0.5 behavior
    // (`STRATIFIED_CODE_PREFETCH = 12`). On failure we log and proceed
    // with primary lanes only.
    let dense = if params.source.is_none() {
        let code_filter = build_filter(
            params.project.as_deref(),
            Some(Source::Code.as_str()),
            params.since,
            params.before,
        );
        match lane_dense(&table, &vec, code_filter.as_deref(), stratified_prefetch).await {
            Ok(extras) => {
                tracing::debug!(
                    primary = dense.len(),
                    code_extras = extras.len(),
                    "stratified prefetch"
                );
                merge_dense_lanes(dense, extras)
            }
            Err(e) => {
                tracing::warn!(error = %e, "stratified code prefetch failed; continuing with primary lanes");
                dense
            }
        }
    } else {
        dense
    };

    // Union ids → batch-fetch full chunks + dense embeddings → build
    // Candidates with lane evidence + RRF.
    let mut union_ids: Vec<String> = Vec::with_capacity(bm25.len() + dense.len());
    union_ids.extend(bm25.iter().map(|(id, _, _)| id.clone()));
    union_ids.extend(dense.iter().map(|(id, _, _)| id.clone()));
    union_ids.sort();
    union_ids.dedup();
    let fetched = store.fetch_chunks_by_ids(&union_ids).await?;
    let mut chunks: HashMap<String, Chunk> = HashMap::with_capacity(fetched.len());
    let mut embeddings: HashMap<String, Vec<f32>> = HashMap::new();
    for (id, (chunk, emb)) in fetched {
        if let Some(e) = emb {
            embeddings.insert(id.clone(), e);
        }
        chunks.insert(id, chunk);
    }
    let mut candidates = build_candidates(&bm25, &dense, chunks);
    // Stamp dense embeddings onto candidates for downstream P9b /
    // AttentionAffinity scoring. Cheap — clone of Vec<f32>.
    for c in &mut candidates {
        if let Some(e) = embeddings.remove(&c.chunk.chunk_id) {
            c.dense_embedding = Some(e);
        }
    }

    // Rank with Rrf=1.0 (alpha.1 default). Other features ship at
    // weight 0.0 until P5 measures (and per-phase as P6/P7/P7b/P8
    // land). Engine takes &self → no lock; an Arc<RankEngine> here
    // would be needed once the lens loop shares one with explicit
    // recall, but the explicit path can keep building per-call.
    // Build the rank engine from the effective explicit-profile feature
    // weights (P5). `overrides.weights == None` resolves to the compiled
    // default `{rrf: 1.0}` — numerically identical to the pre-P5 hardcoded
    // rrf-only engine, so shipped behavior is unchanged until config / an
    // MCP arg supplies tuned weights. Under the cross-encoder reranker the
    // engine score is replaced downstream, so non-rrf weights reshape only
    // the candidate pool on the explicit path; they are decisive with the
    // reranker off or in the ambient/lens path.
    let weights = overrides.weights.clone().unwrap_or_else(|| {
        ostk_recall_core::default_profile_weights(ostk_recall_core::RankProfile::Explicit)
    });
    let engine = crate::rank::build_engine_from_weights(&weights);
    let query_ctx = QueryContext::explicit(query_text, vec.clone());
    let ranked: Vec<RankedHit> = engine
        .rank(candidates, &query_ctx, &AttentionContext::empty())
        .await?;

    // Convert to RecallHit for the existing post-rank stages. The
    // top-N for rerank is `fetch_limit`; rerank truncates further.
    let mut hits: Vec<RecallHit> = ranked
        .into_iter()
        .take(fetch_limit)
        .map(ranked_to_recall_hit)
        .collect();

    // Cross-encoder pass. The candidate text we score is the snippet
    // already capped at `SNIPPET_CHARS` — same shape as v0.5.
    if let Some(reranker) = reranker {
        hits = rerank_candidates(reranker, query_text, hits)?;
    }

    // Identifier-mode boost: when the query reads like a symbol name,
    // bias actual code definitions above conversation transcripts.
    // Bumps post-rerank scores in place and re-sorts. Future work
    // (P3B AC): emit a `FeatureAttribution { name: "identifier_boost",
    // raw: 1.0, weight: IDENTIFIER_CODE_BOOST, contribution: 3.0 }`
    // row on the MCP response so the boost is auditable.
    let hits = boost_code_for_identifier_queries(query_text, hits, identifier_boost);

    // Post-rerank salience layer: demote freshly self-authored narration
    // (membrane + recent creation `ts`) so the substrate's own present-tense
    // output doesn't crowd out the corpus it narrates. Additive + attributed
    // on the cross-encoder scale; re-sorts. Runs before diversify so a
    // demoted self-ref chunk can fall out of the truncated top-K.
    let hits = dampen_self_reference_recency(hits, self_reference_penalty, Utc::now());

    Ok(diversify_by_source_id(hits, limit, max_per_source_id))
}

/// Recency weight for a self-authored chunk: `1.0` at creation, decaying
/// linearly to `0.0` at [`SELF_REFERENCE_RECENCY_WINDOW_HOURS`]. A `None`
/// `ts` yields `0.0` — without a creation time we can't tell if it's fresh,
/// and "no penalty" is the safe default.
fn self_reference_recency_weight(ts: Option<DateTime<Utc>>, now: DateTime<Utc>) -> f32 {
    let Some(ts) = ts else { return 0.0 };
    #[allow(clippy::cast_precision_loss)]
    let age_hours = (now - ts).num_seconds().max(0) as f32 / 3600.0;
    (1.0 - age_hours / SELF_REFERENCE_RECENCY_WINDOW_HOURS).clamp(0.0, 1.0)
}

/// Post-rerank salience stage: subtract a recency-scaled penalty from
/// freshly self-authored chunks (currently `source == "membrane"`), then
/// re-sort by descending score. The penalty is `penalty * recency_weight(ts)`,
/// recorded as a `self_reference` match-feature row with a **negative**
/// contribution so the `score = Σ contribution` invariant holds and the
/// demotion is auditable in the MCP response. `penalty == 0.0` is a no-op
/// (override-disabled).
///
/// Scoped narrowly to `membrane` — the substrate's own narration — so it
/// structurally cannot demote real corpus content; it only stops the
/// membrane quoting its own fresh output back over the corpus it narrates.
/// See [`SELF_REFERENCE_PENALTY`] for the P7b forward-compat plan (swap the
/// `ts`-recency proxy for the access ledger's creation-vs-access signal).
fn dampen_self_reference_recency(
    mut hits: Vec<RecallHit>,
    penalty: f32,
    now: DateTime<Utc>,
) -> Vec<RecallHit> {
    if penalty == 0.0 {
        return hits;
    }
    let membrane = Source::Membrane.as_str();
    let mut touched = false;
    for hit in &mut hits {
        if hit.source != membrane {
            continue;
        }
        let recency = self_reference_recency_weight(hit.ts, now);
        if recency <= 0.0 {
            continue;
        }
        let contribution = -(penalty * recency);
        hit.score += contribution;
        hit.match_features.insert(
            "self_reference".to_string(),
            ostk_recall_core::MatchFeature {
                raw: recency,
                weight: -penalty,
                contribution,
            },
        );
        touched = true;
    }
    if touched {
        hits.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.chunk_id.cmp(&b.chunk_id))
        });
    }
    hits
}

/// Convert a `RankedHit` to the public `RecallHit` shape.
///
/// `score` is filled from `total` (sum of weighted feature
/// contributions); after the cross-encoder rerank stage runs, this
/// score is overwritten with the cross-encoder score, matching v0.5.
fn ranked_to_recall_hit(ranked: RankedHit) -> RecallHit {
    let Candidate { chunk, .. } = ranked.candidate;
    let Chunk {
        chunk_id,
        source,
        project,
        source_id,
        ts,
        role,
        text,
        links,
        extra,
        ..
    } = chunk;

    let snippet = snippet_of(&text);
    // Chunk stores `role: Option<String>` (wire form); RecallHit
    // surfaces the typed `Option<ContextRole>`. Unknown strings are
    // simply dropped — UI clients that care can re-derive from
    // `extra` if needed.
    let role = role.as_deref().and_then(parse_context_role);

    // Carry rank-engine attribution forward to the wire. Per P3A:
    // `total_score = Σ contribution`. Post-rank stages
    // (identifier_boost) may add their own entries downstream so the
    // boost is auditable.
    let match_features: std::collections::BTreeMap<String, ostk_recall_core::MatchFeature> = ranked
        .features
        .into_iter()
        .map(|(name, attr)| {
            (
                name.to_string(),
                ostk_recall_core::MatchFeature {
                    raw: attr.raw,
                    weight: attr.weight,
                    contribution: attr.contribution,
                },
            )
        })
        .collect();

    RecallHit {
        chunk_id,
        project,
        source: source.as_str().to_string(),
        source_id,
        ts,
        snippet,
        score: ranked.total,
        links,
        extra,
        stale: false,
        role,
        base_score: None,
        thread_score: None,
        embedding_score: None,
        thread_weight: None,
        embedding_weight: None,
        attention_score: None,
        attention_weight: None,
        match_features,
    }
}

/// Parse the snake_case wire form of `ContextRole`. Returns `None` for
/// unknown strings; mirrors the serde rename used on the enum itself.
fn parse_context_role(s: &str) -> Option<ContextRole> {
    match s {
        "primary" => Some(ContextRole::Primary),
        "evolution" => Some(ContextRole::Evolution),
        "usage" => Some(ContextRole::Usage),
        _ => None,
    }
}

/// Merge two dense lane outputs by chunk_id, keeping the BEST (lowest)
/// rank seen for each id. The `extras` lane's ranks are kept as-is —
/// they're scored within their own subset (code-only) and may legitimately
/// be rank 0 inside the subset while the primary dense lane has them at
/// rank 50; we keep the better evidence either way.
fn merge_dense_lanes(primary: Vec<LaneEntry>, extras: Vec<LaneEntry>) -> Vec<LaneEntry> {
    if extras.is_empty() {
        return primary;
    }
    let mut by_id: HashMap<String, (f32, u32)> =
        HashMap::with_capacity(primary.len() + extras.len());
    for (id, score, rank) in primary.into_iter().chain(extras.into_iter()) {
        by_id
            .entry(id)
            .and_modify(|cur| {
                if rank < cur.1 {
                    *cur = (score, rank);
                }
            })
            .or_insert((score, rank));
    }
    let mut out: Vec<LaneEntry> = by_id
        .into_iter()
        .map(|(id, (score, rank))| (id, score, rank))
        .collect();
    // Restore rank order; tie-break on chunk_id so equal-rank rows
    // appear in a deterministic sequence (HashMap iteration above is
    // unordered).
    out.sort_by(|a, b| a.2.cmp(&b.2).then_with(|| a.0.cmp(&b.0)));
    out
}

/// Returns true when the query reads like a code identifier.
///
/// Identifier-shaped means a single short token, `snake_case`, or
/// `CamelCase`. For these queries, the user almost certainly wants the
/// actual definition over conversation transcripts that mention the
/// symbol.
///
/// Heuristic, in order:
/// * Empty / >3 tokens → `false` (likely natural language).
/// * Any token containing `_` → `true` (snake_case-ish).
/// * Any token with a non-leading uppercase char → `true` (CamelCase-ish).
/// * Single token, all alphanumeric/underscore → `true` (bare symbol).
/// * Otherwise → `false`.
///
/// Tune by widening the token cap (currently 3) or by adjusting
/// [`IDENTIFIER_CODE_BOOST`].
#[must_use]
pub fn is_identifier_query(q: &str) -> bool {
    let q = q.trim();
    if q.is_empty() {
        return false;
    }
    let tokens: Vec<&str> = q.split_whitespace().collect();
    if tokens.len() > 3 {
        return false;
    }
    // Any token with underscore -> snake_case-like
    if tokens.iter().any(|t| t.contains('_')) {
        return true;
    }
    // Any token with mid-word uppercase -> CamelCase-like
    if tokens.iter().any(|t| {
        t.chars()
            .enumerate()
            .any(|(i, c)| i > 0 && c.is_ascii_uppercase())
    }) {
        return true;
    }
    // Single token, all alphanumeric/underscore — likely a bare symbol
    // name like "alloc_page" or "memcpy".
    if tokens.len() == 1 && tokens[0].chars().all(|c| c.is_alphanumeric() || c == '_') {
        return true;
    }
    false
}

/// If `query` looks like an identifier, add [`IDENTIFIER_CODE_BOOST`] to
/// every `source == "code"` candidate **whose snippet actually contains
/// the query token** and re-sort by descending score. Otherwise returns
/// `candidates` unchanged.
///
/// The substring check is case-insensitive and only applied to code-source
/// candidates. It exists so that `snake_case` queries with no code match in
/// the corpus (e.g. a markdown-only term that happens to look like an
/// identifier) do not crowd out the genuine non-code matches with an
/// undeserved +3.0. Code candidates that *do* contain the identifier still
/// get the full boost. Pre-fix, an identifier query would lift every code
/// candidate equally, which broke the `markdown/file_glob` branches of the
/// verification panel.
fn boost_code_for_identifier_queries(
    query: &str,
    mut candidates: Vec<RecallHit>,
    boost: f32,
) -> Vec<RecallHit> {
    if !is_identifier_query(query) || boost == 0.0 {
        return candidates;
    }
    let needle = query.trim().to_lowercase();
    for hit in &mut candidates {
        if hit.source == Source::Code.as_str() && hit.snippet.to_lowercase().contains(&needle) {
            hit.score += boost;
            // P3A AC: post-rank boost stages emit a match_features
            // entry so the boost is auditable in the MCP response and
            // the `score = Σ contribution` invariant on
            // `core::types::MatchFeature` is preserved.
            hit.match_features.insert(
                "identifier_boost".to_string(),
                ostk_recall_core::MatchFeature::new(1.0, boost),
            );
        }
    }
    candidates.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.chunk_id.cmp(&b.chunk_id))
    });
    candidates
}

/// Apply the cross-encoder reranker to a candidate pool. Drops candidates
/// whose snippet is empty (the reranker would score them at noise floor
/// anyway, and they're not useful answers).
fn rerank_candidates(
    reranker: &dyn RerankerLike,
    query: &str,
    candidates: Vec<RecallHit>,
) -> Result<Vec<RecallHit>> {
    if candidates.is_empty() {
        return Ok(candidates);
    }
    // Build the doc list parallel to candidates. Empty-snippet rows would
    // still be re-attached by index but produce noisy scores; we keep them
    // in place but feed the reranker a single space so indices stay aligned.
    let docs: Vec<String> = candidates
        .iter()
        .map(|h| {
            if h.snippet.trim().is_empty() {
                " ".to_string()
            } else {
                h.snippet.clone()
            }
        })
        .collect();
    let take = candidates.len();
    let ranked = reranker
        .rerank(query, &docs, take)
        .map_err(|e| crate::error::QueryError::Decode(format!("rerank: {e}")))?;

    // Reassemble candidates in the new order. The cross-encoder REPLACES
    // the engine score (it isn't an additive feature — see
    // p3-rank-evidence.md "Reranker is a post-rank stage, not a
    // feature"), so to keep the `score = Σ contribution` invariant on
    // the MatchFeature doc honest, we reset `match_features` to a
    // single `rerank` entry whose contribution equals the new score.
    // The rank-engine attribution (rrf, etc.) is dropped — operators
    // who want pre-rerank attribution should use the explicit
    // `recall_audit` path or run with the reranker disabled.
    let mut by_idx: Vec<Option<RecallHit>> = candidates.into_iter().map(Some).collect();
    let mut out = Vec::with_capacity(ranked.len());
    for r in ranked {
        if let Some(slot) = by_idx.get_mut(r.idx).and_then(Option::take) {
            let mut hit = slot;
            hit.score = r.score;
            hit.match_features.clear();
            hit.match_features.insert(
                "rerank".to_string(),
                ostk_recall_core::MatchFeature {
                    raw: r.score,
                    weight: 1.0,
                    contribution: r.score,
                },
            );
            out.push(hit);
        }
    }
    // Append any candidates the reranker dropped (shouldn't happen with
    // top_k = take, but defensive).
    for slot in by_idx.into_iter().flatten() {
        out.push(slot);
    }
    Ok(out)
}

/// Post-filter ranked hits so no single `source_id` exceeds
/// `max_per_source_id`. `max_per_source_id == 0` disables the filter
/// (returns the first `limit` hits unchanged). Stops short once `limit`
/// hits are collected or candidates are exhausted.
fn diversify_by_source_id(
    candidates: Vec<RecallHit>,
    limit: usize,
    max_per_source_id: usize,
) -> Vec<RecallHit> {
    if max_per_source_id == 0 {
        let mut out = candidates;
        if out.len() > limit {
            out.truncate(limit);
        }
        return out;
    }
    let mut counts: HashMap<String, usize> = HashMap::new();
    let mut out = Vec::with_capacity(limit);
    for hit in candidates {
        if out.len() >= limit {
            break;
        }
        let count = counts.entry(hit.source_id.clone()).or_insert(0);
        if *count >= max_per_source_id {
            continue;
        }
        *count += 1;
        out.push(hit);
    }
    out
}

fn build_filter(
    project: Option<&str>,
    source: Option<&str>,
    since: Option<DateTime<Utc>>,
    before: Option<DateTime<Utc>>,
) -> Option<String> {
    let mut clauses: Vec<String> = Vec::new();
    if let Some(p) = project {
        clauses.push(format!("project = '{}'", sql_escape(p)));
    }
    if let Some(s) = source {
        clauses.push(format!("source = '{}'", sql_escape(s)));
    }
    if let Some(t) = since {
        clauses.push(format!("ts >= TIMESTAMP '{}'", t.to_rfc3339()));
    }
    if let Some(t) = before {
        // Half-open: combined with `since`, this yields [since, before).
        clauses.push(format!("ts < TIMESTAMP '{}'", t.to_rfc3339()));
    }
    if clauses.is_empty() {
        None
    } else {
        Some(clauses.join(" AND "))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rerank::{RerankedHit, RerankerLike};
    use ostk_recall_core::Links;

    #[test]
    fn build_filter_empty() {
        assert!(build_filter(None, None, None, None).is_none());
    }

    #[test]
    fn build_filter_project_and_source() {
        let f = build_filter(Some("foo"), Some("markdown"), None, None).unwrap();
        assert!(f.contains("project = 'foo'"));
        assert!(f.contains("source = 'markdown'"));
        assert!(f.contains("AND"));
    }

    #[test]
    fn build_filter_escapes_quotes() {
        let f = build_filter(Some("a'b"), None, None, None).unwrap();
        assert_eq!(f, "project = 'a''b'");
    }

    #[test]
    fn build_filter_since_is_iso() {
        let t = DateTime::parse_from_rfc3339("2026-04-17T10:00:00Z")
            .unwrap()
            .with_timezone(&Utc);
        let f = build_filter(None, None, Some(t), None).unwrap();
        assert!(f.contains("TIMESTAMP '2026-04-17T10:00:00+00:00'"));
    }

    fn fake_hit(chunk_id: &str, source_id: &str) -> RecallHit {
        RecallHit {
            chunk_id: chunk_id.to_string(),
            project: None,
            source: "markdown".to_string(),
            source_id: source_id.to_string(),
            ts: None,
            snippet: String::new(),
            score: 1.0,
            links: Links::default(),
            extra: serde_json::Value::Null,
            stale: false,
            role: None,
            base_score: None,
            thread_score: None,
            embedding_score: None,
            thread_weight: None,
            embedding_weight: None,
            attention_score: None,
            attention_weight: None,
            match_features: Default::default(),
        }
    }

    #[test]
    fn diversify_caps_per_source_id() {
        // 5 from "X" interleaved at the top; 5 from unique sources
        // following — emulates a chatty session monopolizing top-K.
        let mut candidates = Vec::new();
        for i in 0..5 {
            candidates.push(fake_hit(&format!("x{i}"), "X"));
        }
        for i in 0..5 {
            candidates.push(fake_hit(&format!("u{i}"), &format!("U{i}")));
        }
        let out = diversify_by_source_id(candidates, 5, 2);
        assert_eq!(out.len(), 5);
        let x_count = out.iter().filter(|h| h.source_id == "X").count();
        assert!(
            x_count <= 2,
            "expected ≤2 hits from source_id X, got {x_count}"
        );
    }

    #[test]
    fn diversify_zero_means_unlimited() {
        let candidates: Vec<RecallHit> = (0..7).map(|i| fake_hit(&format!("x{i}"), "X")).collect();
        let out = diversify_by_source_id(candidates, 5, 0);
        assert_eq!(out.len(), 5);
        assert!(out.iter().all(|h| h.source_id == "X"));
    }

    #[test]
    fn diversify_returns_short_when_pool_exhausted() {
        // Only one source_id and a strict cap — fewer hits than limit
        // is acceptable.
        let candidates: Vec<RecallHit> = (0..10).map(|i| fake_hit(&format!("x{i}"), "X")).collect();
        let out = diversify_by_source_id(candidates, 5, 2);
        assert_eq!(out.len(), 2);
    }

    /// Fake reranker that scores documents by how many query terms they
    /// contain (case-insensitive whitespace tokenization). Lets us prove
    /// the rerank wire-up changes order without pulling ONNX.
    struct TokenOverlapReranker;

    impl RerankerLike for TokenOverlapReranker {
        fn rerank(
            &self,
            query: &str,
            docs: &[String],
            top_k: usize,
        ) -> crate::rerank::Result<Vec<RerankedHit>> {
            let q_terms: Vec<String> = query.split_whitespace().map(str::to_lowercase).collect();
            let mut scored: Vec<(usize, f32)> = docs
                .iter()
                .enumerate()
                .map(|(i, d)| {
                    let lower = d.to_lowercase();
                    #[allow(clippy::cast_precision_loss)]
                    let s = q_terms
                        .iter()
                        .filter(|t| lower.contains(t.as_str()))
                        .count() as f32;
                    (i, s)
                })
                .collect();
            scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            Ok(scored
                .into_iter()
                .take(top_k)
                .map(|(idx, score)| RerankedHit { idx, score })
                .collect())
        }
        fn model_id(&self) -> &'static str {
            "token-overlap-fake"
        }
    }

    fn hit_with_text(chunk_id: &str, source_id: &str, text: &str) -> RecallHit {
        RecallHit {
            chunk_id: chunk_id.to_string(),
            project: None,
            source: "markdown".to_string(),
            source_id: source_id.to_string(),
            ts: None,
            snippet: text.to_string(),
            score: 0.0,
            links: Links::default(),
            extra: serde_json::Value::Null,
            stale: false,
            role: None,
            base_score: None,
            thread_score: None,
            embedding_score: None,
            thread_weight: None,
            embedding_weight: None,
            attention_score: None,
            attention_weight: None,
            match_features: Default::default(),
        }
    }

    #[test]
    fn rerank_candidates_promotes_more_relevant() {
        // RRF orders the noisy doc first; reranker should push the
        // term-rich doc to the top.
        let candidates = vec![
            hit_with_text("noise", "n", "lorem ipsum dolor sit amet"),
            hit_with_text("good", "g", "the answer is rust async runtime"),
        ];
        let out =
            rerank_candidates(&TokenOverlapReranker, "rust async runtime", candidates).unwrap();
        assert_eq!(out[0].chunk_id, "good", "expected term-rich doc on top");
    }

    #[test]
    fn rerank_candidates_handles_empty_input() {
        let out = rerank_candidates(&TokenOverlapReranker, "anything", Vec::new()).unwrap();
        assert!(out.is_empty());
    }

    #[test]
    fn rerank_candidates_attaches_score() {
        let candidates = vec![
            hit_with_text("a", "a", "alpha alpha alpha"),
            hit_with_text("b", "b", "beta"),
        ];
        let out = rerank_candidates(&TokenOverlapReranker, "alpha", candidates).unwrap();
        assert!(out[0].score >= out[1].score);
        assert_eq!(out[0].chunk_id, "a");
    }

    #[test]
    fn merge_dense_lanes_keeps_better_rank_on_duplicate() {
        // Chunk "a" ranks 5 in the primary dense lane but rank 0 in the
        // stratified code prefetch — keep the rank-0 evidence so RRF
        // reflects the better signal.
        let primary = vec![("a".into(), 0.5, 5), ("b".into(), 0.4, 1)];
        let extras = vec![("a".into(), 0.1, 0), ("c".into(), 0.2, 2)];
        let merged = merge_dense_lanes(primary, extras);
        let by_id: std::collections::HashMap<_, _> = merged
            .iter()
            .map(|(id, score, rank)| (id.as_str(), (*score, *rank)))
            .collect();
        assert_eq!(by_id["a"].1, 0);
        assert!(by_id.contains_key("b"));
        assert!(by_id.contains_key("c"));
        // Output is sorted by rank ascending.
        let ranks: Vec<u32> = merged.iter().map(|(_, _, r)| *r).collect();
        assert!(ranks.windows(2).all(|w| w[0] <= w[1]));
    }

    #[test]
    fn merge_dense_lanes_empty_extras_returns_primary() {
        let primary = vec![("a".into(), 0.5, 0), ("b".into(), 0.4, 1)];
        let merged = merge_dense_lanes(primary.clone(), Vec::new());
        assert_eq!(merged.len(), 2);
        // identity on (id, rank) is enough; merge sorts by rank.
        let ids: Vec<_> = merged.iter().map(|(id, _, _)| id.as_str()).collect();
        assert_eq!(ids, vec!["a", "b"]);
    }

    #[test]
    fn parse_context_role_known_values() {
        assert_eq!(parse_context_role("primary"), Some(ContextRole::Primary));
        assert_eq!(
            parse_context_role("evolution"),
            Some(ContextRole::Evolution)
        );
        assert_eq!(parse_context_role("usage"), Some(ContextRole::Usage));
        assert_eq!(parse_context_role("nonsense"), None);
        assert_eq!(parse_context_role(""), None);
    }

    #[test]
    fn is_identifier_snake_case() {
        assert!(is_identifier_query("tier2_line_rebase"));
        assert!(is_identifier_query("alloc_page"));
    }

    #[test]
    fn is_identifier_camel_case() {
        assert!(is_identifier_query("MemoryRegion"));
    }

    #[test]
    fn is_identifier_single_short_token() {
        assert!(is_identifier_query("memcpy"));
    }

    #[test]
    fn is_identifier_natural_language_false() {
        assert!(!is_identifier_query("fleet heartbeat"));
        assert!(!is_identifier_query("what is X"));
        assert!(!is_identifier_query(""));
        assert!(!is_identifier_query(
            "how do we wire the reranker into recall"
        ));
    }

    #[test]
    fn boost_promotes_code_when_identifier_query() {
        let candidates = vec![
            RecallHit {
                chunk_id: "conv1".into(),
                project: None,
                source: "anthropic_session".into(),
                source_id: "s1".into(),
                ts: None,
                snippet: "we discussed alloc_page".into(),
                score: 5.0,
                links: Links::default(),
                extra: serde_json::Value::Null,
                stale: false,
                role: None,
                base_score: None,
                thread_score: None,
                embedding_score: None,
                thread_weight: None,
                embedding_weight: None,
                attention_score: None,
                attention_weight: None,
                match_features: Default::default(),
            },
            RecallHit {
                chunk_id: "code1".into(),
                project: None,
                source: "code".into(),
                source_id: "src/mm.rs".into(),
                ts: None,
                snippet: "fn alloc_page() {}".into(),
                score: 4.0,
                links: Links::default(),
                extra: serde_json::Value::Null,
                stale: false,
                role: None,
                base_score: None,
                thread_score: None,
                embedding_score: None,
                thread_weight: None,
                embedding_weight: None,
                attention_score: None,
                attention_weight: None,
                match_features: Default::default(),
            },
        ];
        let out =
            boost_code_for_identifier_queries("alloc_page", candidates, IDENTIFIER_CODE_BOOST);
        assert_eq!(out[0].chunk_id, "code1", "code hit should win after boost");
        // Boost lifted 4.0 by IDENTIFIER_CODE_BOOST (3.0) → 7.0, comfortably
        // above the 5.0 conversation row.
        assert!((out[0].score - (4.0 + IDENTIFIER_CODE_BOOST)).abs() < f32::EPSILON);
    }

    #[test]
    fn boost_noop_for_natural_language_query() {
        let candidates = vec![
            RecallHit {
                chunk_id: "conv1".into(),
                project: None,
                source: "anthropic_session".into(),
                source_id: "s1".into(),
                ts: None,
                snippet: "answer text".into(),
                score: 5.0,
                links: Links::default(),
                extra: serde_json::Value::Null,
                stale: false,
                role: None,
                base_score: None,
                thread_score: None,
                embedding_score: None,
                thread_weight: None,
                embedding_weight: None,
                attention_score: None,
                attention_weight: None,
                match_features: Default::default(),
            },
            RecallHit {
                chunk_id: "code1".into(),
                project: None,
                source: "code".into(),
                source_id: "src/mm.rs".into(),
                ts: None,
                snippet: "fn x() {}".into(),
                score: 4.0,
                links: Links::default(),
                extra: serde_json::Value::Null,
                stale: false,
                role: None,
                base_score: None,
                thread_score: None,
                embedding_score: None,
                thread_weight: None,
                embedding_weight: None,
                attention_score: None,
                attention_weight: None,
                match_features: Default::default(),
            },
        ];
        let out = boost_code_for_identifier_queries(
            "how do we wire the reranker",
            candidates,
            IDENTIFIER_CODE_BOOST,
        );
        // Order untouched, scores untouched.
        assert_eq!(out[0].chunk_id, "conv1");
        assert!((out[0].score - 5.0).abs() < f32::EPSILON);
        assert!((out[1].score - 4.0).abs() < f32::EPSILON);
    }

    /// Build a post-rerank-shaped hit: a single `rerank` match-feature
    /// whose contribution equals the score (the state `rerank_candidates`
    /// produces), so the `score = Σ contribution` invariant is testable
    /// through the self-reference stage.
    fn hit_full(chunk_id: &str, source: &str, score: f32, ts: Option<DateTime<Utc>>) -> RecallHit {
        let mut mf = std::collections::BTreeMap::new();
        mf.insert(
            "rerank".to_string(),
            ostk_recall_core::MatchFeature {
                raw: score,
                weight: 1.0,
                contribution: score,
            },
        );
        RecallHit {
            chunk_id: chunk_id.to_string(),
            project: None,
            source: source.to_string(),
            source_id: chunk_id.to_string(),
            ts,
            snippet: String::new(),
            score,
            links: Links::default(),
            extra: serde_json::Value::Null,
            stale: false,
            role: None,
            base_score: None,
            thread_score: None,
            embedding_score: None,
            thread_weight: None,
            embedding_weight: None,
            attention_score: None,
            attention_weight: None,
            match_features: mf,
        }
    }

    #[test]
    fn self_reference_recency_weight_ramps() {
        let now = Utc::now();
        assert!((self_reference_recency_weight(Some(now), now) - 1.0).abs() < 1e-6);
        let half = now - chrono::Duration::hours((SELF_REFERENCE_RECENCY_WINDOW_HOURS / 2.0) as i64);
        assert!((self_reference_recency_weight(Some(half), now) - 0.5).abs() < 0.05);
        let old =
            now - chrono::Duration::hours(SELF_REFERENCE_RECENCY_WINDOW_HOURS as i64 + 1);
        assert_eq!(self_reference_recency_weight(Some(old), now), 0.0);
        // No creation time → no penalty (safe default).
        assert_eq!(self_reference_recency_weight(None, now), 0.0);
    }

    #[test]
    fn self_reference_demotes_fresh_membrane() {
        // A fresh membrane chunk out-scoring a corpus chunk pre-stage
        // (the live "membrane narration tops recall of its own topic"
        // regression) must fall below the corpus chunk after the penalty.
        let now = Utc::now();
        let hits = vec![
            hit_full("mem", Source::Membrane.as_str(), 3.0, Some(now)),
            hit_full("corpus", "markdown", 2.0, Some(now)),
        ];
        let out = dampen_self_reference_recency(hits, SELF_REFERENCE_PENALTY, now);
        assert_eq!(
            out[0].chunk_id, "corpus",
            "fresh membrane narration must fall below substantive corpus"
        );
        let mem = out.iter().find(|h| h.chunk_id == "mem").unwrap();
        // Invariant: score == Σ contribution (rerank 3.0 + self_reference −2.0).
        let sum: f32 = mem.match_features.values().map(|m| m.contribution).sum();
        assert!((mem.score - sum).abs() < 1e-5, "score {} != Σ {sum}", mem.score);
        let sr = mem
            .match_features
            .get("self_reference")
            .expect("self_reference attribution row present");
        assert!(sr.contribution < 0.0, "self-reference is a penalty");
    }

    #[test]
    fn self_reference_spares_old_membrane() {
        // A membrane chunk older than the recency window is legitimate
        // historical recognition — untouched.
        let now = Utc::now();
        let old = now - chrono::Duration::hours(SELF_REFERENCE_RECENCY_WINDOW_HOURS as i64 + 5);
        let hits = vec![hit_full("mem", Source::Membrane.as_str(), 3.0, Some(old))];
        let out = dampen_self_reference_recency(hits, SELF_REFERENCE_PENALTY, now);
        assert!((out[0].score - 3.0).abs() < 1e-6, "old membrane untouched");
        assert!(!out[0].match_features.contains_key("self_reference"));
    }

    #[test]
    fn self_reference_ignores_non_membrane_and_zero_disables() {
        let now = Utc::now();
        // Non-membrane fresh chunk: untouched (cannot demote real corpus).
        let hits = vec![hit_full("doc", "markdown", 3.0, Some(now))];
        let out = dampen_self_reference_recency(hits, SELF_REFERENCE_PENALTY, now);
        assert!((out[0].score - 3.0).abs() < 1e-6);
        assert!(!out[0].match_features.contains_key("self_reference"));
        // penalty == 0.0 disables even for a fresh membrane chunk.
        let hits2 = vec![hit_full("mem", Source::Membrane.as_str(), 3.0, Some(now))];
        let out2 = dampen_self_reference_recency(hits2, 0.0, now);
        assert!((out2[0].score - 3.0).abs() < 1e-6);
        assert!(!out2[0].match_features.contains_key("self_reference"));
    }
}
