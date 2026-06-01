//! P8 contextual-embedder probe — does a small CONTEXTUAL model form topical
//! density basins where static potion did not?
//!
//! Takes the same uniform stride sample of apparatus-excluded chunk TEXT, but
//! re-embeds it with `fastembed` (default: bge-small-en-v1.5, selectable via
//! `MODEL=BGEBaseENV15` / `MODEL=NomicEmbedTextV15` / etc.) instead of reading
//! the stored potion vectors, then runs the identical
//! HDBSCAN harness. Compare noise% / cluster count / largest-cluster snippets
//! to the potion probe (`hdbscan_corpus_probe`):
//!   - potion (static): ~90% noise, clusters = near-dup procedural filler.
//!   - if contextual gives lower noise + topical clusters → P8 is unlocked.
//!
//! First run downloads the selected ONNX model via fastembed.
//! Run: cargo run --release -p ostk-recall-attention --example contextual_embed_probe
//! Env:
//!   MODEL=BGEBaseENV15|NomicEmbedTextV15|BGESmallENV15 (default BGESmallENV15)
//!   SAMPLE_TARGET=2000
//!   HEAVY_N=2000
//!   PREFIX='search_document: '   # optional, useful for Nomic-style models
//!   SOURCE=claude_code       # optional concrete source filter

use std::collections::HashMap;
use std::path::PathBuf;
use std::str::FromStr;
use std::time::Instant;

use fastembed::{EmbeddingModel, InitOptions, TextEmbedding};
use hdbscan::{Hdbscan, HdbscanHyperParams};
use ostk_recall_core::Source;
use ostk_recall_store::CorpusStore;

const STORE_DIM: usize = 512; // potion table dim (for open_or_create; unused here)
const LOAD_CAP: usize = 1_000_000;
const STEPS: &[usize] = &[1_000, 2_000];

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(default)
}

fn selected_model() -> EmbeddingModel {
    std::env::var("MODEL")
        .ok()
        .and_then(|s| EmbeddingModel::from_str(&s).ok())
        .unwrap_or(EmbeddingModel::BGESmallENV15)
}

fn source_filter() -> Option<Source> {
    std::env::var("SOURCE")
        .ok()
        .and_then(|s| Source::parse_str(&s))
}

fn cluster_stats(labels: &[i32]) -> (usize, usize, Vec<(i32, usize)>) {
    let mut counts: HashMap<i32, usize> = HashMap::new();
    for &l in labels {
        *counts.entry(l).or_default() += 1;
    }
    let noise = counts.get(&-1).copied().unwrap_or(0);
    let mut sizes: Vec<(i32, usize)> = counts.into_iter().filter(|(l, _)| *l != -1).collect();
    sizes.sort_by(|a, b| b.1.cmp(&a.1));
    (sizes.len(), noise, sizes)
}

fn run_hdbscan(data: &[Vec<f32>], min_cluster_size: usize) -> (Vec<i32>, f64) {
    let hp = HdbscanHyperParams::builder()
        .min_cluster_size(min_cluster_size)
        .build();
    let model = Hdbscan::new(data, hp);
    let t = Instant::now();
    let labels = model.cluster().expect("clustering should succeed");
    (labels, t.elapsed().as_secs_f64())
}

fn root() -> PathBuf {
    std::env::var("OSTK_RECALL_ROOT")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            PathBuf::from(std::env::var("HOME").expect("HOME")).join(".local/share/ostk-recall")
        })
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let store = CorpusStore::open_or_create(&root(), STORE_DIM).await?;

    let source = source_filter();
    let all = match source {
        Some(source) => {
            eprintln!("source filter: {}", source.as_str());
            store
                .fetch_sample_texts_for_source(LOAD_CAP, true, source)
                .await?
        }
        None => store.fetch_sample_texts(LOAD_CAP, true).await?,
    };
    if all.is_empty() {
        return Err("no texts loaded — wrong root or empty corpus?".into());
    }
    let sample_target = env_usize("SAMPLE_TARGET", 2_000);
    let heavy_n = env_usize("HEAVY_N", 2_000);
    let prefix = std::env::var("PREFIX").unwrap_or_default();
    let stride = (all.len() / sample_target).max(1);
    let sample: Vec<&(String, String)> = all.iter().step_by(stride).take(sample_target).collect();
    let raw_texts: Vec<String> = sample.iter().map(|r| r.1.clone()).collect();
    let texts: Vec<String> = if prefix.is_empty() {
        raw_texts.clone()
    } else {
        raw_texts.iter().map(|t| format!("{prefix}{t}")).collect()
    };
    eprintln!(
        "stride sample: {} texts (every {}th of {})",
        texts.len(),
        stride,
        all.len()
    );

    let model = selected_model();
    eprintln!("loading {model} (fastembed; downloads on first run if uncached)...");
    let mut embedder =
        TextEmbedding::try_new(InitOptions::new(model.clone()).with_show_download_progress(true))
            .expect("fastembed init");
    let t = Instant::now();
    let embeds: Vec<Vec<f32>> = embedder
        .embed(texts.clone(), Some(256))
        .expect("fastembed embed");
    eprintln!(
        "embedded {} chunks ({}-d) in {:.1}s",
        embeds.len(),
        embeds.first().map_or(0, Vec::len),
        t.elapsed().as_secs_f64()
    );

    println!("\n# {model} CONTEXTUAL — stride sample, apparatus-excluded (mcs=5)");
    println!(
        "{:>7} | {:>8} | {:>8} | {:>7} | top-5 cluster sizes",
        "N", "secs", "clusters", "noise%"
    );
    for &n in STEPS {
        if n > embeds.len() {
            break;
        }
        let (labels, secs) = run_hdbscan(&embeds[..n], 5);
        let (nc, noise, sizes) = cluster_stats(&labels);
        let top: Vec<usize> = sizes.iter().take(5).map(|(_, c)| *c).collect();
        println!(
            "{:>7} | {:>8.2} | {:>8} | {:>6.1}% | {:?}",
            n,
            secs,
            nc,
            100.0 * noise as f64 / n as f64,
            top
        );
    }

    let n = embeds.len().min(heavy_n);
    println!("\n# min_cluster_size sweep at N={n}");
    for mcs in [5usize, 10, 20, 40] {
        let (labels, secs) = run_hdbscan(&embeds[..n], mcs);
        let (nc, noise, _) = cluster_stats(&labels);
        println!(
            "  mcs={mcs:>3}: {nc:>5} clusters, {:>5.1}% noise  ({secs:.2}s)",
            100.0 * noise as f64 / n as f64
        );
    }

    // Inspect: are the contextual clusters real topics?
    let (labels, _) = run_hdbscan(&embeds[..n], 5);
    let (_, _, sizes) = cluster_stats(&labels);
    println!("\n# largest 3 clusters at N={n} (mcs=5) — member snippets");
    for (label, size) in sizes.iter().take(3) {
        println!("  -- cluster {label} ({size} chunks) --");
        let members: Vec<usize> = (0..n).filter(|&i| labels[i] == *label).collect();
        for &i in members.iter().take(4) {
            let snip: String = raw_texts[i]
                .split_whitespace()
                .collect::<Vec<_>>()
                .join(" ");
            let snip: String = snip.chars().take(120).collect();
            println!("     {snip}");
        }
    }

    Ok(())
}
