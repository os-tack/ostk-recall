# HDBSCAN probe result (P2 Probe 5)

Date: 2026-05-27.
crate: **`hdbscan = "0.12"`** (pure-Rust HDBSCAN implementation).
Probe: `crates/attention/examples/hdbscan_probe.rs`.

## Question

Is there a usable Rust HDBSCAN crate that produces deterministic
cluster assignments for a fixed seed on a small embedding fixture?

## Method

Synthesize 201 points in 4-D arranged as three Gaussian-ish blobs
around (0,0,0,0), (5,5,5,5), (-5,-5,-5,-5). Run HDBSCAN with
`min_cluster_size = 5`, twice. Assert:

1. The two runs produce identical `Vec<i32>` label vectors
   (determinism).
2. Each blob's majority label is non-noise AND distinct from the
   other two blobs' majorities (recovery).

## Result: PASS

- Two consecutive runs over the same input → identical label vectors.
- Blob majority labels: `[1, 2, 0]` — three distinct non-noise
  clusters, one per planted blob.

The `hdbscan` 0.12 crate is pure Rust (no Python dependency, no C
build step) and produces stable results across runs without an
explicit seed parameter (its hierarchical algorithm is deterministic
given fixed input order).

## Decision

P8 (concept overlay side table) uses `hdbscan = "0.12"` as the
production clustering crate. The greedy-centroid alignment strategy
(per `p8-concepts.md`) layers stable cluster ids on top of HDBSCAN's
otherwise-unstable label numbering across refreshes.

## Caveats

- HDBSCAN is non-deterministic on near-duplicates with identical
  density (label numbering can swap). The greedy-centroid alignment
  in P8 mitigates this by matching new centroids to the closest
  previous-refresh centroid rather than relying on raw label
  numbering.
- The probe uses well-separated synthetic clusters. On real ostk
  embeddings (potion-retrieval-32M; 512-D; semantically related
  conversations form fuzzy blobs), the cluster boundaries will be
  softer; P8's `min_cluster_size` is a tuning knob that defaults to 5
  and may need raising for production corpora.
