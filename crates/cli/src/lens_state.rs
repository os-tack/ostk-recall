//! P9b-min — persisted lens state.
//!
//! `LensState` is the small piece of in-process memory the daemon
//! must preserve across restarts so a freshly-spawned `serve`
//! doesn't see a synthetic drift = ∞ on the first poll and spam a
//! gratuitous lens refresh.
//!
//! Persisted as JSON at `{serve_dir}/lens_state.json`. Writes
//! happen after every successful refresh, so corruption of the file
//! is recoverable: the loop will just treat the missing/corrupt
//! state as `Default` and seed naturally on the next genuine drift.

use std::path::Path;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// Filename within the serve directory. Constant so `lens show`
/// and the loop point at the same path without a wiring mismatch.
pub const LENS_STATE_FILE: &str = "lens_state.json";

/// What the daemon remembers between refreshes so subsequent polls
/// can decide whether to skip.
///
/// Each field corresponds directly to a check in
/// `run_lens_loop` (p9b-lens-portfolio.md "Background loop"):
///
/// - `last_rolling_vec` → drift detection against the current
///   `attn_ctx.rolling_vec` via cosine **distance**.
/// - `last_pin_fingerprint` → pin-change trigger; blake3 of
///   `pinned_vec || scope_bytes`. Storing the hash (not the vec)
///   keeps the file small even with high-dim embeddings.
/// - `last_portfolio_chunk_ids` → debug surface for `lens show`,
///   plus telemetry hook for P9b-full's refractory penalty.
/// - `last_content_fp` → blake3 of the rendered markdown bytes;
///   gates the unchanged-content skip.
/// - `last_lens_ts` → wall-clock of the last successful refresh.
///
/// Fingerprints stored as `Vec<u8>` rather than `[u8; 32]` so the
/// JSON shape stays a plain byte array (serde_json's `[u8; N]`
/// support is friendlier through `Vec<u8>`); the loop compares
/// slices, so the fixed-vs-dynamic distinction is invisible.
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct LensState {
    #[serde(default)]
    pub last_rolling_vec: Option<Vec<f32>>,
    #[serde(default)]
    pub last_pin_fingerprint: Option<Vec<u8>>,
    #[serde(default)]
    pub last_portfolio_chunk_ids: Vec<String>,
    #[serde(default)]
    pub last_content_fp: Option<Vec<u8>>,
    #[serde(default)]
    pub last_lens_ts: Option<DateTime<Utc>>,
}

/// Read `{dir}/lens_state.json`. Returns `Ok(None)` when the file
/// is absent (fresh install / first run); `Err` when the file
/// exists but can't be parsed (corruption — surfaced so the caller
/// can log it and fall through to `Default`).
pub fn load_lens_state(dir: &Path) -> std::io::Result<Option<LensState>> {
    let path = dir.join(LENS_STATE_FILE);
    if !path.exists() {
        return Ok(None);
    }
    let contents = std::fs::read_to_string(&path)?;
    let state: LensState = serde_json::from_str(&contents)
        .map_err(|e| std::io::Error::other(format!("lens_state.json: {e}")))?;
    Ok(Some(state))
}

/// Write `{dir}/lens_state.json` (creates the directory if absent).
/// Called after every successful lens refresh; a failure here is
/// not fatal — the loop logs and continues, so a transient disk
/// error means the next restart sees stale state, not a wedged
/// daemon.
pub fn save_lens_state(dir: &Path, state: &LensState) -> std::io::Result<()> {
    std::fs::create_dir_all(dir)?;
    let path = dir.join(LENS_STATE_FILE);
    let json = serde_json::to_string_pretty(state)
        .map_err(|e| std::io::Error::other(format!("lens_state.json: {e}")))?;
    std::fs::write(&path, json)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn round_trips_through_disk() {
        let tmp = TempDir::new().unwrap();
        let state = LensState {
            last_rolling_vec: Some(vec![0.1, 0.2, 0.3]),
            last_pin_fingerprint: Some(vec![42; 32]),
            last_portfolio_chunk_ids: vec!["c1".into(), "c2".into()],
            last_content_fp: Some(vec![1; 32]),
            last_lens_ts: Some(Utc::now()),
        };
        save_lens_state(tmp.path(), &state).unwrap();
        let loaded = load_lens_state(tmp.path()).unwrap().unwrap();
        assert_eq!(loaded.last_rolling_vec, state.last_rolling_vec);
        assert_eq!(loaded.last_pin_fingerprint, state.last_pin_fingerprint);
        assert_eq!(
            loaded.last_portfolio_chunk_ids,
            state.last_portfolio_chunk_ids
        );
        assert_eq!(loaded.last_content_fp, state.last_content_fp);
        assert!(loaded.last_lens_ts.is_some());
    }

    #[test]
    fn load_returns_none_when_absent() {
        let tmp = TempDir::new().unwrap();
        let loaded = load_lens_state(tmp.path()).unwrap();
        assert!(loaded.is_none(), "missing file → Ok(None), not error");
    }

    #[test]
    fn default_state_is_empty() {
        let s = LensState::default();
        assert!(s.last_rolling_vec.is_none());
        assert!(s.last_pin_fingerprint.is_none());
        assert!(s.last_portfolio_chunk_ids.is_empty());
        assert!(s.last_content_fp.is_none());
        assert!(s.last_lens_ts.is_none());
    }

    #[test]
    fn corrupted_file_returns_err() {
        // Truncated/corrupt file surfaces as an error so the caller
        // can log it; the spec'd recovery is to fall through to
        // Default, but that policy lives in the caller, not here.
        let tmp = TempDir::new().unwrap();
        std::fs::write(tmp.path().join(LENS_STATE_FILE), "{not json").unwrap();
        assert!(load_lens_state(tmp.path()).is_err());
    }

    #[test]
    fn save_creates_directory_when_absent() {
        let tmp = TempDir::new().unwrap();
        let sub = tmp.path().join("does/not/exist");
        let state = LensState::default();
        save_lens_state(&sub, &state).unwrap();
        assert!(sub.join(LENS_STATE_FILE).exists());
    }
}
