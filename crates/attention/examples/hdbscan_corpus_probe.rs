//! P8 scale + quality probe — `hdbscan` 0.12 over REAL corpus embeddings.
//!
//! v2: clusters a **uniform stride sample** of the apparatus-excluded corpus
//! (not the scan-order head — that was apparatus-light and unrepresentative in
//! v1), and **inspects the largest cluster's member text** to judge whether a
//! "concept" is a real topic or just near-duplicate boilerplate.
//!
//! Run: cargo run --release -p ostk-recall-attention --example hdbscan_corpus_probe
//! Env: OSTK_RECALL_ROOT (default ~/.local/share/ostk-recall)

use std::collections::HashMap;
use std::path::PathBuf;
use std::time::Instant;

use hdbscan::{Hdbscan, HdbscanHyperParams};
use ostk_recall_store::CorpusStore;

const DIM: usize = 512; // potion-retrieval-32M
const LOAD_CAP: usize = 1_000_000; // > corpus size → load all (apparatus-excluded)
const SAMPLE_TARGET: usize = 5_000; // uniform stride sample size
const STEPS: &[usize] = &[2_000];
const HEAVY_N: usize = 2_000; // composition + sweep + inspect run at this N

fn block_kind(extra_json: &str) -> String {
    serde_json::from_str::<serde_json::Value>(extra_json)
        .ok()
        .and_then(|v| {
            v.get("block_kind")
                .and_then(serde_json::Value::as_str)
                .map(String::from)
        })
        .unwrap_or_else(|| "?".into())
}

/// (n_clusters, noise_count, cluster sizes desc) — label -1 is noise.
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
    let root = root();
    eprintln!("opening corpus at {}", root.display());
    let store = CorpusStore::open_or_create(&root, DIM).await?;

    // Load the full apparatus-excluded population, then take a uniform stride
    // sample — avoids the scan-order-head bias of a plain LIMIT.
    let t = Instant::now();
    let all = store.fetch_sample_embeddings(LOAD_CAP, true).await?;
    eprintln!(
        "loaded {} apparatus-excluded chunks in {:.1}s",
        all.len(),
        t.elapsed().as_secs_f64()
    );
    if all.is_empty() {
        return Err("no embeddings loaded — wrong root or empty corpus?".into());
    }

    let stride = (all.len() / SAMPLE_TARGET).max(1);
    let sample: Vec<&(String, Vec<f32>, String)> =
        all.iter().step_by(stride).take(SAMPLE_TARGET).collect();
    eprintln!(
        "uniform stride sample: {} chunks (every {}th of {})",
        sample.len(),
        stride,
        all.len()
    );

    let ids: Vec<String> = sample.iter().map(|r| r.0.clone()).collect();
    let embeds: Vec<Vec<f32>> = sample.iter().map(|r| r.1.clone()).collect();
    let kinds: Vec<String> = sample.iter().map(|r| block_kind(&r.2)).collect();

    println!("\n# HDBSCAN scale curve — STRIDE sample, apparatus-excluded (mcs=5)");
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

    let n = embeds.len().min(HEAVY_N);
    let (labels, _) = run_hdbscan(&embeds[..n], 5);
    let (_, _, sizes) = cluster_stats(&labels);

    println!("\n# block_kind composition of top clusters at N={n} (mcs=5)");
    for (label, size) in sizes.iter().take(6) {
        let mut kc: HashMap<&str, usize> = HashMap::new();
        for (i, &l) in labels.iter().enumerate().take(n) {
            if l == *label {
                *kc.entry(kinds[i].as_str()).or_default() += 1;
            }
        }
        let mut kv: Vec<(&str, usize)> = kc.into_iter().collect();
        kv.sort_by(|a, b| b.1.cmp(&a.1));
        println!("  cluster {label:>4}: {size:>5} chunks  kinds={kv:?}");
    }

    // Inspect: is the largest cluster a real topic or near-dup boilerplate?
    if let Some((top_label, top_size)) = sizes.first() {
        let member_ids: Vec<String> = (0..n)
            .filter(|&i| labels[i] == *top_label)
            .map(|i| ids[i].clone())
            .collect();
        let probe_ids: Vec<String> = member_ids.iter().take(8).cloned().collect();
        let texts = store.fetch_texts(&probe_ids).await?;
        println!("\n# largest cluster (label {top_label}, {top_size} chunks) — member snippets");
        for id in &probe_ids {
            let raw = texts.get(id).cloned().unwrap_or_default();
            let snip: String = raw.split_whitespace().collect::<Vec<_>>().join(" ");
            let snip: String = snip.chars().take(150).collect();
            println!("  - {snip}");
        }
    }

    Ok(())
}
