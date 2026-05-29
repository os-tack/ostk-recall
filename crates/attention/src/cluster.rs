//! Emergent-thread clustering (Phase 8 seed).
//!
//! After the auto-weaver finishes matching new chunks against existing
//! thread anchors, the leftover chunks — the ones that didn't resonate
//! with anything we already know — are candidates for emergent threads.
//! This module groups those leftovers by dense cosine neighbourhoods so
//! the substrate can propose new threads without operator prompting.
//!
//! V1 is intentionally simple: a density-based scan over a bounded
//! batch (per-scan, N ≤ ~50 chunks). For each unmatched chunk we count
//! its neighbours within `threshold`; a cluster is a connected set of
//! ≥ `MIN_CLUSTER_SIZE` chunks where every member has ≥
//! `MIN_NEIGHBOURS_IN_CLUSTER` neighbours inside the same set. The
//! algorithm is deterministic given a sorted input — handy for tests
//! and for replay.

use std::collections::{HashMap, HashSet};

use crate::cosine_similarity;

/// Cosine similarity at or above which two chunks count as neighbours
/// when clustering emergent threads. Matches the prose anchor threshold.
pub const EMERGENT_THRESHOLD: f32 = 0.82;

/// Minimum number of chunks required to surface a cluster.
pub const MIN_CLUSTER_SIZE: usize = 3;

/// Minimum number of in-cluster neighbours each member must have.
pub const MIN_NEIGHBOURS_IN_CLUSTER: usize = 2;

/// A detected emergent cluster.
///
/// `chunk_ids` are sorted lexicographically so callers get stable
/// ordering across runs; `centroid` is the unnormalized mean of the
/// member embeddings (caller can renormalize if needed); `cohesion`
/// is the average pairwise cosine similarity within the cluster.
#[derive(Debug, Clone)]
pub struct EmergentCluster {
    pub chunk_ids: Vec<String>,
    pub centroid: Vec<f32>,
    pub cohesion: f32,
}

/// Find emergent clusters in a batch of unmatched chunks.
///
/// `chunks` is an unordered slice of `(chunk_id, embedding)`. The
/// function is read-only — it does not mutate the input or call into
/// any store. Returns clusters in deterministic order (sorted by the
/// lexicographically smallest member id).
#[must_use]
pub fn find_clusters(chunks: &[(String, Vec<f32>)], threshold: f32) -> Vec<EmergentCluster> {
    find_clusters_with(
        chunks,
        threshold,
        MIN_CLUSTER_SIZE,
        MIN_NEIGHBOURS_IN_CLUSTER,
    )
}

/// Tunable variant of [`find_clusters`] — kept for tests that want to
/// exercise smaller / larger cluster shapes without crossing the
/// production constants.
#[must_use]
pub fn find_clusters_with(
    chunks: &[(String, Vec<f32>)],
    threshold: f32,
    min_cluster_size: usize,
    min_in_cluster_neighbours: usize,
) -> Vec<EmergentCluster> {
    let n = chunks.len();
    if n < min_cluster_size {
        return Vec::new();
    }

    // Build the adjacency: undirected edges for cosine >= threshold.
    let mut neighbours: Vec<HashSet<usize>> = vec![HashSet::new(); n];
    for i in 0..n {
        for j in (i + 1)..n {
            let sim = cosine_similarity(&chunks[i].1, &chunks[j].1);
            if sim >= threshold {
                neighbours[i].insert(j);
                neighbours[j].insert(i);
            }
        }
    }

    // Connected components over the threshold-graph. Union-find keeps
    // the implementation small and deterministic.
    let mut parent: Vec<usize> = (0..n).collect();
    for (i, ns) in neighbours.iter().enumerate() {
        for &j in ns {
            union(&mut parent, i, j);
        }
    }

    let mut components: HashMap<usize, Vec<usize>> = HashMap::new();
    for i in 0..n {
        let root = find(&mut parent, i);
        components.entry(root).or_default().push(i);
    }

    let mut clusters: Vec<EmergentCluster> = Vec::new();
    for (_, members) in components {
        if members.len() < min_cluster_size {
            continue;
        }
        // Density filter: every member must have at least
        // `min_in_cluster_neighbours` neighbours *inside* the cluster.
        let member_set: HashSet<usize> = members.iter().copied().collect();
        let all_dense = members.iter().all(|&i| {
            neighbours[i]
                .iter()
                .filter(|n| member_set.contains(*n))
                .count()
                >= min_in_cluster_neighbours
        });
        if !all_dense {
            continue;
        }

        let mut ids: Vec<String> = members.iter().map(|&i| chunks[i].0.clone()).collect();
        ids.sort();

        let dim = chunks[members[0]].1.len();
        let mut centroid = vec![0.0_f32; dim];
        for &i in &members {
            let v = &chunks[i].1;
            if v.len() != dim {
                continue;
            }
            for (slot, x) in centroid.iter_mut().zip(v.iter()) {
                *slot += *x;
            }
        }
        // Bounded by the cluster size, which is at most the batch
        // size (~50 in production). Precision loss in the cast is
        // negligible for a display metric.
        #[allow(clippy::cast_precision_loss)]
        let inv = 1.0_f32 / members.len() as f32;
        for slot in &mut centroid {
            *slot *= inv;
        }

        let cohesion = average_pairwise_cosine(chunks, &members);

        clusters.push(EmergentCluster {
            chunk_ids: ids,
            centroid,
            cohesion,
        });
    }

    clusters.sort_by(|a, b| a.chunk_ids[0].cmp(&b.chunk_ids[0]));
    clusters
}

fn find(parent: &mut [usize], mut x: usize) -> usize {
    while parent[x] != x {
        parent[x] = parent[parent[x]];
        x = parent[x];
    }
    x
}

fn union(parent: &mut [usize], a: usize, b: usize) {
    let ra = find(parent, a);
    let rb = find(parent, b);
    if ra != rb {
        parent[ra] = rb;
    }
}

/// Mean pairwise cosine similarity over a flat slice of embeddings.
/// Exposed for `thread_query`'s cross-axis density backfill — same
/// formula as the internal cluster cohesion metric, but takes a plain
/// vector instead of indices into a `(id, vec)` slice.
#[must_use]
pub fn mean_pairwise_cosine(embeddings: &[Vec<f32>]) -> f32 {
    if embeddings.len() < 2 {
        return 0.0;
    }
    let mut total = 0.0_f32;
    let mut pairs = 0_u32;
    for i in 0..embeddings.len() {
        for j in (i + 1)..embeddings.len() {
            total += cosine_similarity(&embeddings[i], &embeddings[j]);
            pairs += 1;
        }
    }
    if pairs == 0 {
        return 0.0;
    }
    total / f32::from(u16::try_from(pairs).unwrap_or(u16::MAX))
}

fn average_pairwise_cosine(chunks: &[(String, Vec<f32>)], members: &[usize]) -> f32 {
    if members.len() < 2 {
        return 0.0;
    }
    let mut total = 0.0_f32;
    let mut pairs = 0_u32;
    for i in 0..members.len() {
        for j in (i + 1)..members.len() {
            total += cosine_similarity(&chunks[members[i]].1, &chunks[members[j]].1);
            pairs += 1;
        }
    }
    if pairs == 0 {
        return 0.0;
    }
    total / f32::from(u16::try_from(pairs).unwrap_or(u16::MAX))
}

// -----------------------------------------------------------------------
// tests
// -----------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn v(id: &str, vec: Vec<f32>) -> (String, Vec<f32>) {
        (id.into(), vec)
    }

    /// A near-axis vector — small perturbations on the chosen axis so
    /// four vectors all cluster tightly (cosine > 0.99) without being
    /// identical.
    fn near_axis(axis: usize, dim: usize, jitter: f32) -> Vec<f32> {
        let mut v = vec![0.0_f32; dim];
        v[axis] = 1.0;
        // Spread `jitter` across the other axes so every vector still
        // has nearly unit norm and is dominated by `axis`.
        let other = dim - 1;
        if other > 0 {
            #[allow(clippy::cast_precision_loss)]
            let share = jitter / other as f32;
            for (i, slot) in v.iter_mut().enumerate() {
                if i != axis {
                    *slot = share;
                }
            }
        }
        v
    }

    #[test]
    fn cluster_emerges_from_dense_neighborhood() {
        // 16-dim vectors: 4 cluster tightly on axis 0; 6 noise vectors
        // each pinned to their own axis so they have no neighbours.
        let dim = 16;
        let jitters = [0.001_f32, 0.002, 0.003, 0.004];
        let mut chunks: Vec<(String, Vec<f32>)> = Vec::new();
        for (i, &j) in jitters.iter().enumerate() {
            chunks.push(v(&format!("hot-{i}"), near_axis(0, dim, j)));
        }
        for i in 0..6 {
            // each noise vector is on its own axis (1..=6) → orthogonal
            // to the cluster and to every other noise vector.
            let mut nv = vec![0.0_f32; dim];
            nv[i + 1] = 1.0;
            chunks.push(v(&format!("noise-{i}"), nv));
        }

        let clusters = find_clusters(&chunks, EMERGENT_THRESHOLD);
        assert_eq!(clusters.len(), 1, "expected exactly one emergent cluster");
        let c = &clusters[0];
        assert_eq!(
            c.chunk_ids.len(),
            4,
            "cluster should contain all 4 hot chunks"
        );
        for id in &c.chunk_ids {
            assert!(id.starts_with("hot-"), "unexpected cluster member: {id}");
        }
        assert!(
            c.cohesion > 0.99,
            "cohesion should be near-1 for tight cluster, got {}",
            c.cohesion
        );
        assert_eq!(c.centroid.len(), dim);
    }

    #[test]
    fn no_clusters_when_batch_below_min_size() {
        let chunks = vec![v("a", near_axis(0, 4, 0.01)), v("b", near_axis(0, 4, 0.01))];
        let clusters = find_clusters(&chunks, EMERGENT_THRESHOLD);
        assert!(clusters.is_empty());
    }

    #[test]
    fn sparse_pair_rejected_by_density_filter() {
        // Three vectors arranged so only a single pair is above
        // threshold — the connected component has 2 members, but the
        // third vector joins via a single edge → density filter
        // requires each member to have >= 2 in-cluster neighbours, so
        // nothing surfaces.
        let chunks = vec![
            v("a", vec![1.0, 0.0, 0.0, 0.0]),
            v("b", vec![1.0, 0.0, 0.0, 0.0]),
            // 'c' is orthogonal — no edges.
            v("c", vec![0.0, 1.0, 0.0, 0.0]),
        ];
        let clusters = find_clusters(&chunks, EMERGENT_THRESHOLD);
        assert!(clusters.is_empty());
    }

    #[test]
    fn deterministic_ordering_by_smallest_member_id() {
        // Two well-separated clusters; the function should always
        // surface them in sorted-id order regardless of input order.
        let dim = 8;
        let jitters = [0.001_f32, 0.002, 0.003];
        let mut chunks: Vec<(String, Vec<f32>)> = Vec::new();
        for (i, &j) in jitters.iter().enumerate() {
            chunks.push(v(&format!("zeta-{i}"), near_axis(0, dim, j)));
        }
        for (i, &j) in jitters.iter().enumerate() {
            chunks.push(v(&format!("alpha-{i}"), near_axis(7, dim, j)));
        }

        let clusters = find_clusters(&chunks, EMERGENT_THRESHOLD);
        assert_eq!(clusters.len(), 2);
        assert!(clusters[0].chunk_ids[0].starts_with("alpha-"));
        assert!(clusters[1].chunk_ids[0].starts_with("zeta-"));
    }
}
