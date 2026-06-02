//! Calibration probe for `[relational] latent_sim_floor` (relational-substrate
//! slice 2cA). Over the **live corpus**, for each non-terminal concept's anchor
//! (`concept_anchors` — the production codebook candidate set) measures the
//! cosine to its nearest *other-concept* anchor — the exact signal the
//! latent-edge promoter gates on — so the floor is set from
//! `potion-retrieval-32M`'s real distribution, not a guess. The swamping probe
//! (rank of the first concept neighbour in a raw `nearest_to`) is kept as a
//! regression note for why 2cA abandoned ANN-over-corpus for the codebook.
//!
//! Run (read-only; safe while the daemon is up — Lance is versioned, SQLite WAL
//! allows concurrent readers):
//!   cargo run -p ostk-recall --example latent_floor_bench

use std::collections::HashMap;

use ostk_recall_store::corpus::CorpusStore;
use ostk_recall_store::{CORPUS_TABLE, ConceptActivationReader, ThreadsDb};

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let (mut dot, mut na, mut nb) = (0.0f32, 0.0f32, 0.0f32);
    for i in 0..a.len() {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    if na == 0.0 || nb == 0.0 {
        0.0
    } else {
        dot / (na.sqrt() * nb.sqrt())
    }
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> anyhow::Result<()> {
    let root = dirs::home_dir().unwrap().join(".local/share/ostk-recall");
    eprintln!("opening corpus + threads at {}", root.display());
    let threads = ThreadsDb::open(&root)?;
    let corpus = CorpusStore::open_or_create(&root, 512).await?;

    // Exactly the production codebook candidate set: one anchor chunk per
    // non-terminal concept (`concept_anchors`), so the measured distribution
    // matches what the promoter actually scores.
    let anchors = threads.concept_anchors()?;
    let handle: HashMap<i64, String> = anchors
        .iter()
        .map(|a| (a.concept_id, a.handle.clone()))
        .collect();
    let chunk_concept: Vec<(String, i64)> = anchors
        .iter()
        .map(|a| (a.chunk_id.clone(), a.concept_id))
        .collect();
    let ids: Vec<String> = anchors.iter().map(|a| a.chunk_id.clone()).collect();
    eprintln!(
        "{} non-terminal concept anchors (the production codebook candidate set)",
        anchors.len()
    );
    let embs = corpus.fetch_embeddings(&ids).await?;
    eprintln!("{} embeddings resolved from Lance", embs.len());

    // For each anchor (chunk of concept X), the cosine to its nearest chunk of a
    // DIFFERENT concept Y — the best off-diagonal bridge that anchor could form.
    let mut nearest: Vec<(f32, i64, i64)> = Vec::new(); // (cosine, anchor_concept, other_concept)
    for (ca, xa) in &chunk_concept {
        let Some(va) = embs.get(ca) else { continue };
        let mut best = -1.0f32;
        let mut other = -1i64;
        for (cb, xb) in &chunk_concept {
            if xa == xb || ca == cb {
                continue;
            }
            let Some(vb) = embs.get(cb) else { continue };
            let c = cosine(va, vb);
            if c > best {
                best = c;
                other = *xb;
            }
        }
        if best >= 0.0 {
            nearest.push((best, *xa, other));
        }
    }
    nearest.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

    if nearest.is_empty() {
        eprintln!("no cross-concept pairs measurable");
        return Ok(());
    }

    let vals: Vec<f32> = nearest.iter().map(|(c, ..)| *c).collect();
    let pct = |p: f32| -> f32 {
        let idx = ((vals.len() as f32 - 1.0) * p).round() as usize;
        vals[idx]
    };
    println!(
        "\n=== nearest other-concept cosine, per anchor chunk (n={}) ===",
        vals.len()
    );
    println!(
        "max={:.3}  p90={:.3}  p75={:.3}  median={:.3}  p25={:.3}  min={:.3}",
        vals[0],
        pct(0.10),
        pct(0.25),
        pct(0.50),
        pct(0.75),
        vals[vals.len() - 1],
    );
    println!("\n=== how many anchors clear a candidate floor ===");
    for thr in [0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.55] {
        let n = vals.iter().filter(|v| **v >= thr).count();
        println!(
            "  floor {:.2}: {:>4} / {} anchors have a bridge",
            thr,
            n,
            vals.len()
        );
    }
    // Swamping probe: does `nearest_to` even RETURN a concept neighbour within
    // a usable K, or is the top-K all non-concept transcript chunks? For each
    // anchor, find the rank of the first OTHER-concept evidence chunk in the
    // raw ANN ordering over the whole corpus.
    let concept_chunks: HashMap<String, i64> = chunk_concept.iter().cloned().collect();
    let table = corpus
        .connection()
        .open_table(CORPUS_TABLE)
        .execute()
        .await?;
    println!("\n=== rank of first other-concept chunk in nearest_to (K=300) ===");
    for (ca, xa) in &chunk_concept {
        let Some(va) = embs.get(ca) else { continue };
        let near = ostk_recall_query::lanes::lane_dense(&table, va, None, 300).await?;
        let mut found = None;
        for (rank, (cid, ..)) in near.iter().enumerate() {
            if let Some(other) = concept_chunks.get(cid) {
                if other != xa && cid != ca {
                    found = Some(rank + 1);
                    break;
                }
            }
        }
        let ha = handle.get(xa).map(String::as_str).unwrap_or("?");
        match found {
            Some(r) => println!("  {:<28} first concept neighbour at rank {}", ha, r),
            None => println!("  {:<28} NO concept neighbour in top 300", ha),
        }
    }

    println!("\n=== top 12 off-diagonal pairs (the bridges a low floor would admit) ===");
    let mut seen_pair = std::collections::HashSet::new();
    for (c, xa, xb) in nearest.iter().take(60) {
        let key = (xa.min(xb), xa.max(xb));
        if !seen_pair.insert(key) {
            continue;
        }
        let ha = handle.get(xa).map(String::as_str).unwrap_or("?");
        let hb = handle.get(xb).map(String::as_str).unwrap_or("?");
        println!("  {:.3}  {} ~ {}", c, ha, hb);
        if seen_pair.len() >= 12 {
            break;
        }
    }
    Ok(())
}
