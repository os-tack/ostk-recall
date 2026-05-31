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

        // Resonance is weighted by distinct information, not raw count: a
        // cluster that is mostly the same content repeated (identical
        // embeddings) is redundancy, not an emergent theme. Require enough
        // *distinct* embeddings to clear the size floor before proposing.
        if distinct_embedding_count(chunks, &members) < min_cluster_size {
            continue;
        }

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

        // Order members by similarity to the centroid (nearest first) so
        // `chunk_ids[0]` is the cluster's most-representative chunk — the
        // anchor the weaver promotes. Anchor quality is the binding constraint
        // on whether the off-diagonal bridge can fire: an arbitrary first-chunk
        // anchor (e.g. a bare `struct`) resonates poorly, while the
        // centroid-nearest chunk best represents the idea. Tiebreak lexically
        // for determinism. The proposed handle hashes a *sorted* copy
        // (`generate_proposed_handle`), so this display order never affects
        // RT-5 idempotency.
        let mut scored: Vec<(String, f32)> = members
            .iter()
            .map(|&i| {
                (
                    chunks[i].0.clone(),
                    cosine_similarity(&chunks[i].1, &centroid),
                )
            })
            .collect();
        scored.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.0.cmp(&b.0))
        });
        let ids: Vec<String> = scored.into_iter().map(|(id, _)| id).collect();

        let cohesion = average_pairwise_cosine(chunks, &members);

        clusters.push(EmergentCluster {
            chunk_ids: ids,
            centroid,
            cohesion,
        });
    }

    // Deterministic cluster order by the lexically-smallest member id.
    // `chunk_ids[0]` is now the centroid-nearest representative (not the
    // lex-min), so sort on the canonical min rather than the first element.
    clusters.sort_by(|a, b| a.chunk_ids.iter().min().cmp(&b.chunk_ids.iter().min()));
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
    // `pairs` is N(N-1)/2 and can exceed u16::MAX for large clusters (N>362);
    // the old `u16::try_from(...).unwrap_or(u16::MAX)` capped the divisor and
    // inflated cohesion above 1.0. Divide by the true pair count.
    #[allow(clippy::cast_precision_loss)] // display metric; precision loss negligible
    let denom = pairs as f32;
    total / denom
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
    #[allow(clippy::cast_precision_loss)] // display metric; precision loss negligible
    let denom = pairs as f32;
    total / denom
}

/// Count distinct embeddings among a cluster's members.
///
/// Identical content produces an identical (deterministic) embedding, so
/// this is the resonance-layer proxy for distinct *information*: a cluster
/// of 509 byte-identical chunks has distinct-count 1. We hash the f32 bit
/// patterns; an astronomically-unlikely collision would only undercount,
/// which is the conservative direction (treats more as redundant).
fn distinct_embedding_count(chunks: &[(String, Vec<f32>)], members: &[usize]) -> usize {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    let mut seen: HashSet<u64> = HashSet::new();
    for &i in members {
        let mut h = DefaultHasher::new();
        for x in &chunks[i].1 {
            x.to_bits().hash(&mut h);
        }
        seen.insert(h.finish());
    }
    seen.len()
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
    fn identical_content_cluster_rejected_as_redundancy() {
        // Six byte-identical embeddings: dense and tight, but distinct
        // information is 1 — redundancy, not an emergent theme. The
        // distinct-content gate must reject it (this is the 509× class).
        let dim = 16;
        let dup = near_axis(0, dim, 0.001);
        let chunks: Vec<(String, Vec<f32>)> = (0..6)
            .map(|i| v(&format!("dup-{i}"), dup.clone()))
            .collect();
        let clusters = find_clusters(&chunks, EMERGENT_THRESHOLD);
        assert!(
            clusters.is_empty(),
            "identical-content cluster must be rejected by the distinct-count gate"
        );
    }

    #[test]
    fn cohesion_does_not_overflow_for_large_cluster() {
        // >362 distinct members → pair count exceeds u16::MAX. The old
        // divisor cap inflated cohesion above 1.0; it must stay a valid
        // similarity in [0, 1].
        let dim = 16;
        let chunks: Vec<(String, Vec<f32>)> = (0..400)
            .map(|i| {
                let jitter = (i as f32).mul_add(1e-5, 0.001);
                v(&format!("hot-{i:04}"), near_axis(0, dim, jitter))
            })
            .collect();
        let clusters = find_clusters(&chunks, EMERGENT_THRESHOLD);
        assert_eq!(clusters.len(), 1, "400 near-axis vectors form one cluster");
        assert!(
            clusters[0].cohesion <= 1.0 && clusters[0].cohesion > 0.9,
            "cohesion must be a valid similarity, got {}",
            clusters[0].cohesion
        );
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

    #[test]
    fn chunk_ids_lead_with_centroid_nearest_representative() {
        // Anchor quality (M4): chunk_ids[0] must be the centroid-nearest
        // (most-representative) member, NOT the lexicographically-smallest, so
        // the weaver promotes a representative anchor instead of an arbitrary
        // chunk. The center vector here is lexically LAST, so passing proves
        // ordering by centroid similarity rather than by id.
        let chunks = vec![
            v("z-center", vec![1.0, 0.0, 0.0]),
            v("a-edge", vec![0.98, 0.2, 0.0]),
            v("m-edge", vec![0.98, 0.0, 0.2]),
        ];
        let clusters = find_clusters_with(&chunks, 0.82, 3, 2);
        assert_eq!(
            clusters.len(),
            1,
            "the three tight vectors form one cluster"
        );
        let c = &clusters[0];
        assert_eq!(
            c.chunk_ids[0], "z-center",
            "chunk_ids[0] must be the centroid-nearest member, not the lex-min; got {:?}",
            c.chunk_ids
        );
        assert_eq!(c.chunk_ids.len(), 3);
    }
}
