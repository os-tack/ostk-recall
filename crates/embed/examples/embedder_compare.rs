//! Compare the production embedder (`model2vec` / potion-retrieval-32M) against
//! two `fastembed` transformer-based embedders on a synthetic paraphrase
//! benchmark. Reports encode latency, dimension, and recall@1 / recall@3.
//!
//! Run with:
//!
//! ```text
//! cargo run --release --example embedder_compare -p ostk-recall-embed
//! ```
//!
//! Downloads model weights on first run; subsequent runs are fast.

#![allow(clippy::cast_precision_loss)]

use std::time::Instant;

use fastembed::{EmbeddingModel, InitOptions, TextEmbedding};
use ostk_recall_embed::{Embedder, cosine_similarity};

/// Ten paraphrase clusters. Each cluster has one anchor and two queries
/// expressing the same idea. A good embedder should produce vectors where
/// each query's top cosine match (among all 10 anchors) is its own anchor.
const CLUSTERS: &[(&str, &str, &str)] = &[
    (
        "Attention is concentration of focus on a few salient threads.",
        "Focusing attention means narrowing what you care about right now.",
        "Concentration on the important threads is what attention provides.",
    ),
    (
        "Fade is concentration, not deletion — unused threads dim, never vanish.",
        "When threads aren't used they dim down but remain stored.",
        "Cognitive fade means low attention, not erasure from memory.",
    ),
    (
        "The chain is an immutable cognition history written turn by turn.",
        "Each turn appends to a permanent chain of cognitive events.",
        "Chain entries record cognition once and cannot be rewritten.",
    ),
    (
        "Threads form the working-memory graph linking concepts you're thinking about.",
        "The working set is a graph of currently-considered threads.",
        "Threads are graph nodes representing active concepts.",
    ),
    (
        "Score tier is a fast in-memory activation count per thread.",
        "Activation scores live in RAM and rise with thread mentions.",
        "In-memory scoring captures how hot each thread is right now.",
    ),
    (
        "AutoWeaver attaches resonant ingest evidence to existing threads.",
        "When new content matches a thread's anchor, the weaver links it.",
        "The weaver routes ingested chunks to threads they resonate with.",
    ),
    (
        "Privacy tier T0 means private — never crosses scope boundaries.",
        "T0 content is strictly scope-local and cannot bleed between projects.",
        "Private tier zero keeps threads isolated from cross-scope surfaces.",
    ),
    (
        "Idle curator demotes cold threads from Full to Half to Folded depth.",
        "Threads that go quiet slide from full visibility toward folded state.",
        "The curator handles the fade transitions when activation drops.",
    ),
    (
        "Familiarity counter advances each turn the thread handle is mentioned.",
        "Mentioning a known thread increments its familiarity score per turn.",
        "Each turn-end tick raises familiarity for the threads it referenced.",
    ),
    (
        "The substrate is a kernel-level boundary; the ABI is sovereign.",
        "Substrate ABI defines a hard boundary nothing crosses informally.",
        "Kernel sovereignty means the substrate ABI is the canonical contract.",
    ),
];

struct BenchResult {
    name: String,
    dim: usize,
    encode_secs: f64,
    recall_at_1: f32,
    recall_at_3: f32,
}

fn recall_at_k(anchors: &[Vec<f32>], queries: &[Vec<f32>], k: usize) -> f32 {
    let n = anchors.len();
    assert_eq!(queries.len(), 2 * n, "two queries per anchor");
    let mut hits = 0usize;
    for (qi, q) in queries.iter().enumerate() {
        let expected = qi / 2;
        let mut sims: Vec<(usize, f32)> = anchors
            .iter()
            .enumerate()
            .map(|(ai, a)| (ai, cosine_similarity(q, a)))
            .collect();
        sims.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        if sims.iter().take(k).any(|(ai, _)| *ai == expected) {
            hits += 1;
        }
    }
    hits as f32 / queries.len() as f32
}

fn run_model2vec() -> BenchResult {
    let emb = Embedder::load("minishlab/potion-retrieval-32M").expect("load model2vec");
    let dim = emb.dim();

    let mut anchor_texts: Vec<&str> = Vec::with_capacity(CLUSTERS.len());
    let mut query_texts: Vec<&str> = Vec::with_capacity(CLUSTERS.len() * 2);
    for &(a, q1, q2) in CLUSTERS {
        anchor_texts.push(a);
        query_texts.push(q1);
        query_texts.push(q2);
    }
    let all: Vec<&str> = anchor_texts.iter().chain(query_texts.iter()).copied().collect();

    let t = Instant::now();
    let vecs = emb.encode_batch(&all);
    let encode_secs = t.elapsed().as_secs_f64();

    let anchors: Vec<Vec<f32>> = vecs[..anchor_texts.len()].to_vec();
    let queries: Vec<Vec<f32>> = vecs[anchor_texts.len()..].to_vec();

    BenchResult {
        name: "model2vec / potion-retrieval-32M".into(),
        dim,
        encode_secs,
        recall_at_1: recall_at_k(&anchors, &queries, 1),
        recall_at_3: recall_at_k(&anchors, &queries, 3),
    }
}

fn run_fastembed(name: &str, model: EmbeddingModel) -> Option<BenchResult> {
    let init = InitOptions::new(model).with_show_download_progress(false);
    let mut emb = match TextEmbedding::try_new(init) {
        Ok(e) => e,
        Err(err) => {
            eprintln!("skipping {name}: {err}");
            return None;
        }
    };

    let mut anchor_texts: Vec<String> = Vec::with_capacity(CLUSTERS.len());
    let mut query_texts: Vec<String> = Vec::with_capacity(CLUSTERS.len() * 2);
    for &(a, q1, q2) in CLUSTERS {
        anchor_texts.push(a.to_string());
        query_texts.push(q1.to_string());
        query_texts.push(q2.to_string());
    }
    let all: Vec<String> = anchor_texts.iter().chain(query_texts.iter()).cloned().collect();

    let t = Instant::now();
    let vecs = emb.embed(all, None).expect("encode");
    let encode_secs = t.elapsed().as_secs_f64();

    let dim = vecs.first().map_or(0, Vec::len);
    let anchors: Vec<Vec<f32>> = vecs[..anchor_texts.len()].to_vec();
    let queries: Vec<Vec<f32>> = vecs[anchor_texts.len()..].to_vec();

    Some(BenchResult {
        name: name.into(),
        dim,
        encode_secs,
        recall_at_1: recall_at_k(&anchors, &queries, 1),
        recall_at_3: recall_at_k(&anchors, &queries, 3),
    })
}

fn print_table(results: &[BenchResult]) {
    let total_chunks = CLUSTERS.len() * 3;
    println!();
    println!("| Embedder | Dim | Encode {total_chunks}ch (s) | µs/chunk | Recall@1 | Recall@3 |");
    println!("|---|---:|---:|---:|---:|---:|");
    for r in results {
        let per_chunk_us = r.encode_secs * 1_000_000.0 / total_chunks as f64;
        println!(
            "| {} | {} | {:.3} | {:.0} | {:.2} | {:.2} |",
            r.name, r.dim, r.encode_secs, per_chunk_us, r.recall_at_1, r.recall_at_3,
        );
    }
    println!();
}

fn main() {
    println!(
        "Encoding {} chunks ({} anchors + {} queries) across each embedder…",
        CLUSTERS.len() * 3,
        CLUSTERS.len(),
        CLUSTERS.len() * 2,
    );

    let mut results = Vec::new();
    results.push(run_model2vec());
    if let Some(r) = run_fastembed("fastembed / BGE-small-en-v1.5", EmbeddingModel::BGESmallENV15) {
        results.push(r);
    }
    if let Some(r) = run_fastembed("fastembed / MiniLM-L6-v2", EmbeddingModel::AllMiniLML6V2) {
        results.push(r);
    }

    print_table(&results);
}
