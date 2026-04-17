//! Simple arithmetic helpers used by the verification panel.
//!
//! The panel searches for the identifier `compute_lagrangian_delta` — a
//! deliberately distinctive name that BM25 can pin without a real embedder.

/// Compute the Lagrangian delta between two state samples.
///
/// This function name is the canonical keyword that the recall verification
/// panel uses to probe the `code` scanner.
pub fn compute_lagrangian_delta(a: f64, b: f64) -> f64 {
    (a - b).abs()
}

/// Dual routine: `invert_hoberman_matrix` is a second unique keyword the
/// panel may use to prove scope-filtered code retrieval.
pub fn invert_hoberman_matrix(m: f64) -> f64 {
    if m.abs() < f64::EPSILON {
        0.0
    } else {
        1.0 / m
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lagrangian_is_non_negative() {
        assert!(compute_lagrangian_delta(1.0, 2.0) >= 0.0);
    }

    #[test]
    fn invert_on_nonzero() {
        assert_eq!(invert_hoberman_matrix(2.0), 0.5);
    }
}
