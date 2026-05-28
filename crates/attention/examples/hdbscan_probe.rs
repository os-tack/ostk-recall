//! P2 Probe 5: is the Rust `hdbscan` crate deterministic on a fixed
//! seed for a small embedding fixture, and does it correctly recover
//! three well-separated clusters?
//!
//! Generates 200 points in 4-D arranged as three Gaussian-ish blobs
//! around (0,0,0,0), (5,5,5,5), (-5,-5,-5,-5), runs HDBSCAN with
//! `min_cluster_size=5`, and:
//!   1. asserts two consecutive runs over the same input return the
//!      same `Vec<i32>` of labels (determinism).
//!   2. asserts the clustering produces ≥ 3 non-noise clusters and
//!      that the blob-1, blob-2, blob-3 majority labels are distinct.
//!
//! Exit 0 → P8 can use the `hdbscan` crate. Exit 1 → fall back to
//! DBSCAN / k-means / sidecar (maintainer call documented in
//! `docs/hdbscan-probe.md`).

use std::collections::HashMap;

use hdbscan::{Hdbscan, HdbscanHyperParams};

const DIM: usize = 4;
const N_PER_BLOB: usize = 67; // 3 * 67 = 201 ≈ 200

fn synth_dataset() -> Vec<Vec<f32>> {
    // Deterministic pseudo-Gaussian: pick a few small offsets per blob.
    let centers = [
        [0.0f32, 0.0, 0.0, 0.0],
        [5.0, 5.0, 5.0, 5.0],
        [-5.0, -5.0, -5.0, -5.0],
    ];
    // Offsets that fit cleanly inside a ±1.0 ball around each center.
    let offsets: Vec<[f32; DIM]> = (0..N_PER_BLOB)
        .map(|i| {
            let f = (i % 11) as f32 / 11.0 - 0.5; // [-0.5, 0.5)
            let g = ((i * 7) % 13) as f32 / 13.0 - 0.5;
            let h = ((i * 17) % 19) as f32 / 19.0 - 0.5;
            let k = ((i * 23) % 29) as f32 / 29.0 - 0.5;
            [f, g, h, k]
        })
        .collect();

    let mut data: Vec<Vec<f32>> = Vec::with_capacity(N_PER_BLOB * 3);
    for c in &centers {
        for off in &offsets {
            let row: Vec<f32> = (0..DIM).map(|d| c[d] + off[d]).collect();
            data.push(row);
        }
    }
    data
}

fn run_once(data: &[Vec<f32>]) -> Vec<i32> {
    let hp = HdbscanHyperParams::builder()
        .min_cluster_size(5)
        .build();
    let model = Hdbscan::new(data, hp);
    model.cluster().expect("clustering should succeed")
}

fn main() {
    let data = synth_dataset();
    println!("synthesized {} points across 3 blobs in {}-D", data.len(), DIM);

    // Determinism: run twice, compare labels element-wise.
    let labels_1 = run_once(&data);
    let labels_2 = run_once(&data);
    if labels_1 != labels_2 {
        eprintln!("FAIL: hdbscan is non-deterministic on the same input");
        std::process::exit(1);
    }
    println!("PASS: two runs produced identical label vectors");

    // Cluster recovery: per-blob majority labels must be distinct.
    let mut blob_labels: Vec<HashMap<i32, usize>> = vec![HashMap::new(); 3];
    for (i, &lbl) in labels_1.iter().enumerate() {
        let blob_idx = i / N_PER_BLOB;
        *blob_labels[blob_idx].entry(lbl).or_insert(0) += 1;
    }
    let majorities: Vec<i32> = blob_labels
        .iter()
        .map(|m| {
            *m.iter()
                .max_by_key(|&(_, count)| *count)
                .map(|(k, _)| k)
                .unwrap_or(&-1)
        })
        .collect();
    println!("blob majority labels: {majorities:?}");

    // Distinct non-noise majorities. -1 is HDBSCAN's noise label.
    let distinct: std::collections::HashSet<_> =
        majorities.iter().filter(|&&l| l >= 0).collect();
    if distinct.len() < 3 {
        eprintln!(
            "FAIL: expected 3 distinct non-noise majorities, got {distinct:?}"
        );
        std::process::exit(1);
    }
    println!("PASS: 3 distinct non-noise majority labels — clusters recovered");

    println!(
        "RESULT: hdbscan 0.12 is deterministic on the fixture and recovers the planted clusters; \
         P8 can adopt it."
    );
}
