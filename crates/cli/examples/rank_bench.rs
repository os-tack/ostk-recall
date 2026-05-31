//! P5 empirical rank-bench.
//!
//! Tunes `[ranking.weights]` and reports retrieval quality + latency over a
//! query set with relevance labels. Two modes:
//!
//! - `--fixture` (default): builds a small deterministic in-memory corpus with
//!   a [`FakeEmbedder`] (no model download — CI-safe). Proves the harness runs
//!   end-to-end and emits every column; the numbers are a *plumbing* check, not
//!   a retrieval verdict.
//! - `--corpus <lance_dir> --model <id>`: opens the maintainer's real corpus
//!   with the real model2vec embedder. The operator step that produces the
//!   committed `docs/rank-bench-results.md` quality numbers.
//!
//! ```bash
//! # CI-safe:
//! HF_HUB_OFFLINE=1 cargo run -p ostk-recall-cli --example rank_bench -- --fixture
//! # Real corpus (operator):
//! cargo run --release -p ostk-recall-cli --example rank_bench -- \
//!   --corpus ~/.local/share/ostk-recall --model potion-retrieval-32M \
//!   --queries tests/fixtures/bench/queries.json --out docs/rank-bench-results.md
//! ```
//!
//! Relevance rule: a hit is relevant iff its `source_id` ∈ `relevant.source_ids`
//! OR its `chunk_id` ∈ `relevant.chunk_ids`. Because a real corpus can't
//! enumerate every relevant chunk, the primary quality metrics are **MRR@10**
//! and **Success@10** (hit rate — the honest stand-in for "Recall@10"), with
//! **Recall@10 vs the labeled set** reported as secondary.

use std::collections::{BTreeMap, HashMap};
use std::fmt::Write as _;
use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result};
use ostk_recall_core::{
    Chunk, FacetSet, Links, RankProfile, RankingOverrides, RecallParams, Source,
};
use ostk_recall_embed::Embedder;
use ostk_recall_pipeline::ChunkEmbedder;
use ostk_recall_query::lanes::{build_candidates, lane_bm25, lane_dense};
use ostk_recall_query::{
    AttentionContext, QueryContext, ambient_candidates, build_engine_from_weights, hybrid,
};
use ostk_recall_store::{CORPUS_TABLE, CorpusStore};
use serde::Deserialize;

const FAKE_DIM: usize = 32;
const K: usize = 10;
const SWEEP_GRID: [f32; 5] = [0.0, 0.25, 0.5, 1.0, 2.0];

// ---- embedders ---------------------------------------------------------

/// Deterministic single-hot embedder (length-bucketed). CI-safe; no model
/// download. Dense matches are crude (length-mod buckets) — the BM25/Tantivy
/// lane carries the real lexical signal in fixture mode.
struct FakeEmbedder;

impl ChunkEmbedder for FakeEmbedder {
    fn dim(&self) -> usize {
        FAKE_DIM
    }
    fn encode_batch(&self, texts: &[&str]) -> Vec<Vec<f32>> {
        texts
            .iter()
            .map(|t| {
                let mut v = vec![0.0_f32; FAKE_DIM];
                // Spread a little signal across a few buckets so cosine isn't
                // all-or-nothing: hash word lengths into buckets.
                for w in t.split_whitespace() {
                    v[w.len() % FAKE_DIM] += 1.0;
                }
                if t.is_empty() {
                    v[0] = 1.0;
                }
                v
            })
            .collect()
    }
}

// ---- query set ---------------------------------------------------------

#[derive(Debug, Deserialize)]
struct QuerySet {
    #[serde(default)]
    queries: Vec<BenchQuery>,
}

#[derive(Debug, Deserialize)]
struct BenchQuery {
    id: String,
    query: String,
    #[serde(default)]
    category: String,
    #[serde(default)]
    profile: String,
    #[serde(default)]
    relevant: Relevant,
}

#[derive(Debug, Default, Deserialize)]
struct Relevant {
    #[serde(default)]
    source_ids: Vec<String>,
    #[serde(default)]
    chunk_ids: Vec<String>,
}

impl Relevant {
    fn matches(&self, source_id: &str, chunk_id: &str) -> bool {
        self.source_ids.iter().any(|s| s == source_id)
            || self.chunk_ids.iter().any(|c| c == chunk_id)
    }
    fn labeled_count(&self) -> usize {
        self.source_ids.len() + self.chunk_ids.len()
    }
}

#[derive(Debug, Deserialize)]
struct LensTurn {
    #[serde(default)]
    turn: usize,
    focus_text: String,
}

// ---- synthetic fixture corpus -----------------------------------------

/// `(source, source_id, text)` triples. Authored to match the relevance
/// labels in `tests/fixtures/bench/queries.json`.
fn synthetic_corpus() -> Vec<(Source, &'static str, &'static str)> {
    vec![
        (
            Source::Markdown,
            "notes/auth-redirect.md",
            "The auth redirect flow validates the issuer claim then checks the state cookie matches before completing login.",
        ),
        (
            Source::Markdown,
            "notes/retention.md",
            "Retention policy: the substrate keeps conversation data for ninety days then prunes it on the next sweep.",
        ),
        (
            Source::Code,
            "src/redirect.rs",
            "fn validate_redirect(issuer: &str) -> bool { check_state_cookie() && issuer_allowed(issuer) }",
        ),
        (
            Source::Code,
            "src/alloc.rs",
            "fn alloc_page() -> Page { allocate_one_page_from_the_free_list() }",
        ),
        (
            Source::Markdown,
            "decisions/d1840.md",
            "Decision: ABI rename ratified. auth_redirect_v2 deprecates oauth_redirect across the whole surface.",
        ),
        (
            Source::Markdown,
            "notes/subprocess.md",
            "Discussion about subprocesses and spawning background workers under the daemon supervisor.",
        ),
        (
            Source::Membrane,
            "mem/echo.md",
            "Container up. The machine is humming. Done.",
        ),
        (
            Source::Code,
            "src/cache.rs",
            "fn cache_control_header() { set_max_age_for_the_reviewed_url() }",
        ),
    ]
}

fn build_chunk(idx: u32, source: Source, source_id: &str, text: &str) -> Chunk {
    Chunk {
        chunk_id: format!("{}-{idx}-{source_id}", source.as_str()),
        source,
        project: Some("bench".into()),
        source_id: source_id.into(),
        source_config_id: "bench:cfg".into(),
        chunk_index: idx,
        ts: None,
        role: None,
        text: text.into(),
        sha256: format!("sha-{idx}"),
        links: Links::default(),
        facets: FacetSet::default(),
        embedding_input_sha256: format!("emb-{idx}"),
        extra: serde_json::Value::Null,
    }
}

async fn build_fixture_store(
    embedder: &dyn ChunkEmbedder,
) -> Result<(tempfile::TempDir, CorpusStore)> {
    let tmp = tempfile::TempDir::new()?;
    let store = CorpusStore::open_or_create(tmp.path(), embedder.dim()).await?;
    let chunks: Vec<Chunk> = synthetic_corpus()
        .into_iter()
        .enumerate()
        .map(|(i, (s, sid, txt))| build_chunk(i as u32, s, sid, txt))
        .collect();
    let embeddings =
        embedder.encode_batch(&chunks.iter().map(|c| c.text.as_str()).collect::<Vec<_>>());
    store.upsert(&chunks, &embeddings).await?;
    store.ensure_fts_index().await?;
    Ok((tmp, store))
}

// ---- metrics -----------------------------------------------------------

struct QueryEval {
    mrr: f64,
    success: f64,
    recall: f64,
    distinct_sources: usize,
    max_dup: usize,
}

fn eval_hits(hits: &[ostk_recall_core::RecallHit], rel: &Relevant) -> QueryEval {
    let topk: Vec<&ostk_recall_core::RecallHit> = hits.iter().take(K).collect();
    let mut first_rank: Option<usize> = None;
    let mut relevant_found = 0usize;
    let mut source_counts: HashMap<&str, usize> = HashMap::new();
    for (i, h) in topk.iter().enumerate() {
        if rel.matches(&h.source_id, &h.chunk_id) {
            relevant_found += 1;
            if first_rank.is_none() {
                first_rank = Some(i + 1);
            }
        }
        *source_counts.entry(h.source_id.as_str()).or_insert(0) += 1;
    }
    let labeled = rel.labeled_count().max(1);
    QueryEval {
        mrr: first_rank.map_or(0.0, |r| 1.0 / r as f64),
        success: if first_rank.is_some() { 1.0 } else { 0.0 },
        recall: relevant_found as f64 / labeled as f64,
        distinct_sources: source_counts.len(),
        max_dup: source_counts.values().copied().max().unwrap_or(0),
    }
}

fn pctl(sorted_us: &[f64], p: f64) -> f64 {
    if sorted_us.is_empty() {
        return 0.0;
    }
    let rank = (p / 100.0 * (sorted_us.len() - 1) as f64).round() as usize;
    sorted_us[rank.min(sorted_us.len() - 1)]
}

/// Best-effort peak resident-set size.
///
/// On Linux we read `VmHWM` from `/proc/self/status` (process peak RSS, no
/// `libc`/FFI). Elsewhere (macOS in particular) we return `None` and the
/// report prints "n/a" — the AC asks for best-effort + platform-guarded, and
/// the meaningful memory figure is the operator's `--corpus` run on Linux.
/// Process-wide peak is the honest granularity here (per-`prepare()` RSS would
/// need an allocator shim).
fn peak_rss_bytes() -> Option<u64> {
    #[cfg(target_os = "linux")]
    {
        let status = std::fs::read_to_string("/proc/self/status").ok()?;
        for line in status.lines() {
            if let Some(rest) = line.strip_prefix("VmHWM:") {
                let kb: u64 = rest.split_whitespace().next()?.parse().ok()?;
                return Some(kb * 1024);
            }
        }
        None
    }
    #[cfg(not(target_os = "linux"))]
    {
        None
    }
}

// ---- per-config run ----------------------------------------------------

struct StageLatency {
    candgen: Vec<f64>,
    rank: Vec<f64>,
    e2e: Vec<f64>,
}

impl StageLatency {
    fn new() -> Self {
        Self {
            candgen: Vec::new(),
            rank: Vec::new(),
            e2e: Vec::new(),
        }
    }
    fn postrank_us(&self) -> Vec<f64> {
        // Derived: e2e - (candgen + rank), clamped ≥0. Documented as an
        // approximation in the output (post-rank stages aren't separately
        // instrumented inside hybrid::recall).
        self.e2e
            .iter()
            .zip(&self.candgen)
            .zip(&self.rank)
            .map(|((e, c), r)| (e - c - r).max(0.0))
            .collect()
    }
}

/// Mean quality over the query set + accumulated latency samples for one
/// weight configuration on the explicit path.
struct ConfigResult {
    label: String,
    mrr: f64,
    success: f64,
    recall: f64,
    avg_distinct: f64,
    avg_max_dup: f64,
    lat: StageLatency,
}

#[allow(clippy::too_many_arguments)]
async fn run_explicit_config(
    label: &str,
    store: &CorpusStore,
    embedder: &dyn ChunkEmbedder,
    queries: &[BenchQuery],
    weights: &BTreeMap<String, f32>,
    iters: usize,
) -> Result<ConfigResult> {
    let mut sum = QueryEval {
        mrr: 0.0,
        success: 0.0,
        recall: 0.0,
        distinct_sources: 0,
        max_dup: 0,
    };
    let mut distinct_acc = 0.0;
    let mut dup_acc = 0.0;
    let mut lat = StageLatency::new();
    let n = queries.len().max(1) as f64;

    let conn = store.connection();
    let table = conn.open_table(CORPUS_TABLE).execute().await?;

    for q in queries {
        if !q.profile.is_empty() && q.profile != "explicit" {
            continue;
        }
        let params = RecallParams {
            query: q.query.clone(),
            limit: Some(K),
            ranking_overrides: Some(RankingOverrides {
                weights: Some(weights.clone()),
                ..RankingOverrides::default()
            }),
            ..RecallParams::default()
        };

        // Quality is measured on the real end-to-end pipeline output.
        let hits = hybrid::recall(store, embedder, None, &params).await?;
        let ev = eval_hits(&hits, &q.relevant);
        sum.mrr += ev.mrr;
        sum.success += ev.success;
        sum.recall += ev.recall;
        distinct_acc += ev.distinct_sources as f64;
        dup_acc += ev.max_dup as f64;

        // Latency: repeat each query `iters` times, timing the stages.
        let qvec = embedder
            .encode_batch(&[q.query.as_str()])
            .into_iter()
            .next()
            .unwrap_or_default();
        let fetch_limit = (K * 6).max(K);
        for _ in 0..iters {
            // candidate generation
            let t0 = Instant::now();
            let bm25 = lane_bm25(&table, &q.query, None, fetch_limit).await?;
            let dense = lane_dense(&table, &qvec, None, fetch_limit).await?;
            let mut ids: Vec<String> = bm25
                .iter()
                .map(|(id, _, _)| id.clone())
                .chain(dense.iter().map(|(id, _, _)| id.clone()))
                .collect();
            ids.sort();
            ids.dedup();
            let fetched = store.fetch_chunks_by_ids(&ids).await?;
            let mut chunks: HashMap<String, Chunk> = HashMap::with_capacity(fetched.len());
            let mut embs: HashMap<String, Vec<f32>> = HashMap::new();
            for (id, (chunk, emb)) in fetched {
                if let Some(e) = emb {
                    embs.insert(id.clone(), e);
                }
                chunks.insert(id, chunk);
            }
            let mut cands = build_candidates(&bm25, &dense, chunks);
            for c in &mut cands {
                if let Some(e) = embs.remove(&c.chunk.chunk_id) {
                    c.dense_embedding = Some(e);
                }
            }
            lat.candgen.push(t0.elapsed().as_secs_f64() * 1e6);

            // rank
            let engine = build_engine_from_weights(weights);
            let qctx = QueryContext::explicit(&q.query, qvec.clone());
            let t1 = Instant::now();
            let _ranked = engine
                .rank(cands, &qctx, &AttentionContext::empty())
                .await?;
            lat.rank.push(t1.elapsed().as_secs_f64() * 1e6);

            // end-to-end
            let t2 = Instant::now();
            let _ = hybrid::recall(store, embedder, None, &params).await?;
            lat.e2e.push(t2.elapsed().as_secs_f64() * 1e6);
        }
    }

    Ok(ConfigResult {
        label: label.to_string(),
        mrr: sum.mrr / n,
        success: sum.success / n,
        recall: sum.recall / n,
        avg_distinct: distinct_acc / n,
        avg_max_dup: dup_acc / n,
        lat,
    })
}

// ---- lens rotation -----------------------------------------------------

struct RotationResult {
    turns: usize,
    avg_repeat_rate: f64,
    avg_rotation_rate: f64,
}

async fn lens_rotation(
    store: &CorpusStore,
    embedder: &dyn ChunkEmbedder,
    turns: &[LensTurn],
) -> Result<RotationResult> {
    let ambient_weights = ostk_recall_core::default_profile_weights(RankProfile::Ambient);
    let engine = build_engine_from_weights(&ambient_weights);
    let mut prev: Option<Vec<String>> = None;
    let mut repeat_sum = 0.0;
    let mut rot_sum = 0.0;
    let mut pairs = 0usize;

    for t in turns {
        let scope = embedder
            .encode_batch(&[t.focus_text.as_str()])
            .into_iter()
            .next()
            .unwrap_or_default();
        let attn = AttentionContext::with_scope_vector(scope);
        let cands = ambient_candidates(store, &attn, None, K).await?;
        let ranked = engine.rank(cands, &QueryContext::Ambient, &attn).await?;
        let ids: Vec<String> = ranked
            .iter()
            .take(K)
            .map(|h| h.candidate.chunk.chunk_id.clone())
            .collect();
        if let Some(p) = &prev {
            let prev_set: std::collections::HashSet<&String> = p.iter().collect();
            let repeats = ids.iter().filter(|id| prev_set.contains(id)).count();
            let denom = ids.len().max(1) as f64;
            repeat_sum += repeats as f64 / denom;
            rot_sum += 1.0 - repeats as f64 / denom;
            pairs += 1;
        }
        prev = Some(ids);
    }
    let d = pairs.max(1) as f64;
    Ok(RotationResult {
        turns: turns.len(),
        avg_repeat_rate: repeat_sum / d,
        avg_rotation_rate: rot_sum / d,
    })
}

// ---- markdown report ---------------------------------------------------

fn render_report(
    mode: &str,
    n_queries: usize,
    configs: &[ConfigResult],
    sweep: &[(String, Vec<(f32, f64)>)],
    rotation: &RotationResult,
    rss: Option<u64>,
) -> String {
    let mut s = String::new();
    let _ = writeln!(s, "# Rank-bench results\n");
    let _ = writeln!(
        s,
        "Mode: **{mode}**. Query set: {n_queries} queries. Latency in microseconds (µs).\n"
    );
    if mode == "fixture" {
        let _ = writeln!(
            s,
            "> ⚠️ **Fixture mode** — synthetic in-memory corpus + FakeEmbedder. \
             These numbers prove the harness runs and emits every column; they are \
             NOT a retrieval verdict. The verdict comes from a `--corpus` run on a \
             representative corpus.\n"
        );
    }

    let _ = writeln!(s, "## Quality (explicit profile)\n");
    let _ = writeln!(
        s,
        "| Config | MRR@10 | Success@10 | Recall@10* | avg distinct src | avg max dup |"
    );
    let _ = writeln!(s, "|---|---|---|---|---|---|");
    for c in configs {
        let _ = writeln!(
            s,
            "| {} | {:.3} | {:.3} | {:.3} | {:.2} | {:.2} |",
            c.label, c.mrr, c.success, c.recall, c.avg_distinct, c.avg_max_dup
        );
    }
    let _ = writeln!(
        s,
        "\n*Recall@10 is vs the labeled marker set (no full-corpus relevance \
         enumeration); Success@10 (hit rate) is the honest recall stand-in.\n"
    );

    let _ = writeln!(s, "## Latency (µs, p50 / p95 / p99)\n");
    let _ = writeln!(
        s,
        "| Config | candidate-gen | rank | post-rank† | end-to-end |"
    );
    let _ = writeln!(s, "|---|---|---|---|---|");
    for c in configs {
        let mut cg = c.lat.candgen.clone();
        let mut rk = c.lat.rank.clone();
        let mut pr = c.lat.postrank_us();
        let mut e2 = c.lat.e2e.clone();
        for v in [&mut cg, &mut rk, &mut pr, &mut e2] {
            v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        }
        let col = |v: &[f64]| {
            format!(
                "{:.0}/{:.0}/{:.0}",
                pctl(v, 50.0),
                pctl(v, 95.0),
                pctl(v, 99.0)
            )
        };
        let _ = writeln!(
            s,
            "| {} | {} | {} | {} | {} |",
            c.label,
            col(&cg),
            col(&rk),
            col(&pr),
            col(&e2)
        );
    }
    let _ = writeln!(
        s,
        "\n†post-rank is derived (`e2e − candgen − rank`, clamped ≥0): the \
         rerank / identifier-boost / self-reference / diversify stages inside \
         `hybrid::recall` are not separately instrumented.\n"
    );
    match rss {
        Some(b) => {
            let _ = writeln!(
                s,
                "Peak process RSS (best-effort, getrusage): {} MiB.\n",
                b / (1024 * 1024)
            );
        }
        None => {
            let _ = writeln!(s, "Peak process RSS: n/a on this platform.\n");
        }
    }

    let _ = writeln!(s, "## Sensitivity sweep (MRR@10 vs weight)\n");
    for (feat, points) in sweep {
        let cells: Vec<String> = points.iter().map(|(w, m)| format!("{w}={m:.3}")).collect();
        let _ = writeln!(s, "- **{feat}**: {}", cells.join("  "));
    }

    let _ = writeln!(s, "\n## Lens rotation ({} turns)\n", rotation.turns);
    let _ = writeln!(
        s,
        "- avg repeated-chunk rate (turn→turn): {:.3}\n- avg rotation rate: {:.3}",
        rotation.avg_repeat_rate, rotation.avg_rotation_rate
    );
    let _ = writeln!(
        s,
        "\n> Rotation here is the P9b-min ambient baseline (attention-only, no \
         refractory). The P9b-full refractory penalty is expected to raise the \
         rotation rate; that comparison lands with P9b-full.\n"
    );

    let _ = writeln!(s, "## Skipped configurations\n");
    let _ = writeln!(
        s,
        "- **+rerank (configs 2/7)**: cross-encoder needs a model download — skipped in fixture mode; run `--corpus` with a reranker to measure."
    );
    let _ = writeln!(
        s,
        "- **multivector (configs 6/7)**: `skipped: multivector unavailable` — P4 not started."
    );
    let _ = writeln!(
        s,
        "- **header-format (config 8)**: deferred to its own phase (needs an embed-header change + full corpus re-embed)."
    );

    s
}

// ---- main --------------------------------------------------------------

#[tokio::main]
async fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let mut corpus: Option<PathBuf> = None;
    let mut model: Option<String> = None;
    let mut queries_path = PathBuf::from("tests/fixtures/bench/queries.json");
    let mut lens_path = PathBuf::from("tests/fixtures/bench/lens_turns.jsonl");
    let mut out: Option<PathBuf> = None;
    let mut iters = 20usize;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--fixture" => {}
            "--corpus" => {
                i += 1;
                corpus = args.get(i).map(PathBuf::from);
            }
            "--model" => {
                i += 1;
                model = args.get(i).cloned();
            }
            "--queries" => {
                i += 1;
                if let Some(p) = args.get(i) {
                    queries_path = PathBuf::from(p);
                }
            }
            "--lens-turns" => {
                i += 1;
                if let Some(p) = args.get(i) {
                    lens_path = PathBuf::from(p);
                }
            }
            "--out" => {
                i += 1;
                out = args.get(i).map(PathBuf::from);
            }
            "--iters" => {
                i += 1;
                if let Some(v) = args.get(i).and_then(|v| v.parse().ok()) {
                    iters = v;
                }
            }
            other => eprintln!("warning: ignoring unknown arg {other}"),
        }
        i += 1;
    }

    // Load query set + lens turns.
    let qset: QuerySet = serde_json::from_str(
        &std::fs::read_to_string(&queries_path)
            .with_context(|| format!("reading {}", queries_path.display()))?,
    )
    .context("parsing queries.json")?;
    let turns: Vec<LensTurn> = std::fs::read_to_string(&lens_path)
        .unwrap_or_default()
        .lines()
        .filter(|l| !l.trim().is_empty())
        .filter_map(|l| serde_json::from_str::<LensTurn>(l).ok())
        .collect();

    // Build embedder + store.
    let (mode, _tmp_guard, store, embedder): (
        &str,
        Option<tempfile::TempDir>,
        CorpusStore,
        Box<dyn ChunkEmbedder>,
    ) = match (&corpus, &model) {
        (Some(dir), Some(m)) => {
            let emb = Embedder::load(m).with_context(|| format!("loading model {m}"))?;
            let store = CorpusStore::open_or_create(dir, emb.dim())
                .await
                .with_context(|| format!("opening corpus {}", dir.display()))?;
            ("corpus", None, store, Box::new(emb))
        }
        _ => {
            let emb = FakeEmbedder;
            let (tmp, store) = build_fixture_store(&emb).await?;
            ("fixture", Some(tmp), store, Box::new(emb))
        }
    };

    eprintln!(
        "rank_bench: mode={mode} queries={} lens_turns={} iters={iters}",
        qset.queries.len(),
        turns.len()
    );

    // Configurations (explicit profile). Reranker configs are skipped in
    // fixture mode (no model); multivector/header are out of scope.
    let mut configs = Vec::new();
    configs.push(
        run_explicit_config(
            "C1 rrf=1.0 (no rerank)",
            &store,
            embedder.as_ref(),
            &qset.queries,
            &ostk_recall_core::default_profile_weights(RankProfile::Explicit),
            iters,
        )
        .await?,
    );
    let tuned: BTreeMap<String, f32> = [("rrf".to_string(), 1.0), ("bm25".to_string(), 0.5)].into();
    configs.push(
        run_explicit_config(
            "C4 rrf=1.0+bm25=0.5 (no rerank)",
            &store,
            embedder.as_ref(),
            &qset.queries,
            &tuned,
            iters,
        )
        .await?,
    );

    // Sensitivity sweep: vary bm25 (rrf fixed 1.0), then rrf (bm25 fixed 0).
    let mut sweep = Vec::new();
    for feat in ["bm25", "rrf"] {
        let mut points = Vec::new();
        for &w in &SWEEP_GRID {
            let mut weights: BTreeMap<String, f32> = [("rrf".to_string(), 1.0)].into();
            weights.insert(feat.to_string(), w);
            let r = run_explicit_config(
                &format!("sweep:{feat}={w}"),
                &store,
                embedder.as_ref(),
                &qset.queries,
                &weights,
                1,
            )
            .await?;
            points.push((w, r.mrr));
        }
        sweep.push((feat.to_string(), points));
    }

    let rotation = lens_rotation(&store, embedder.as_ref(), &turns).await?;
    let rss = peak_rss_bytes();

    let report = render_report(mode, qset.queries.len(), &configs, &sweep, &rotation, rss);
    println!("{report}");
    if let Some(path) = out {
        std::fs::write(&path, &report).with_context(|| format!("writing {}", path.display()))?;
        eprintln!("wrote {}", path.display());
    }
    Ok(())
}
