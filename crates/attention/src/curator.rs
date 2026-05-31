//! Idle curator (Phase 8 of the attention substrate).
//!
//! Timer-driven daemon: every `tick_interval`, recompute each thread's
//! fade score, compare against tension thresholds, and transition any
//! threads that crossed a boundary. Realizes the `fade-is-concentration`
//! thread — the substrate doesn't forget, but the surfacer stops
//! shouting about threads whose score has fallen below the archive
//! line.
//!
//! Hysteresis (a small dead-band around each up-threshold) prevents a
//! thread sitting near a boundary from flipping every tick. Threads
//! touched within the last tick are skipped — their score would be
//! based on stale state until the tick that observed the touch.
//!
//! V1 known limitation: `AttentionForwardStore::decay` (and the new
//! `score_thread`) lack a scope parameter — the curator operates on
//! threads globally, taking the max score across all scopes that hold
//! the handle. Cross-scope leakage at the curator boundary is the
//! trait's known shortcoming flagged for V1.1.

use std::sync::Arc;
use std::time::Duration;

use chrono::Utc;
use ostk_recall_core::ThreadHandle;
use ostk_recall_store::{StoreError, TensionState, ThreadsDb};
use thiserror::Error;
use tokio::time::sleep;
use tokio_util::sync::CancellationToken;

use crate::{AttentionError, AttentionForwardStore};

// --- errors ----------------------------------------------------------------

#[derive(Debug, Error)]
pub enum CuratorError {
    #[error(transparent)]
    Store(#[from] StoreError),
    #[error(transparent)]
    Attention(#[from] AttentionError),
}

// --- config ----------------------------------------------------------------

/// Tunables for the idle curator. All scores are on the same scale as
/// `AttentionPage::score`.
#[derive(Debug, Clone)]
pub struct CuratorConfig {
    /// Interval between fade-score sweeps.
    pub tick_interval: Duration,
    /// score >= this → `Active`.
    pub active_score_threshold: f32,
    /// score >= this (and below active) → `Slack`.
    pub slack_score_threshold: f32,
    /// score < this → `Dormant`.
    pub archive_threshold: f32,
    /// Hysteresis margin: a thread already at the higher tension state
    /// only down-shifts once its score falls below `(up_threshold -
    /// hysteresis)`. Prevents bouncing around the boundary.
    pub hysteresis: f32,
}

impl Default for CuratorConfig {
    fn default() -> Self {
        Self {
            tick_interval: Duration::from_secs(60),
            active_score_threshold: 1.0,
            slack_score_threshold: 0.3,
            archive_threshold: 0.1,
            hysteresis: 0.05,
        }
    }
}

// --- tick output -----------------------------------------------------------

/// Summary of one curator tick. Returned by [`IdleCurator::tick`] and
/// useful for tests and operator-facing diagnostics.
#[derive(Debug, Clone)]
pub struct CuratorTick {
    pub threads_scored: usize,
    pub transitions: Vec<TensionTransition>,
}

/// A single tension-state transition emitted by a tick.
///
/// The curator calls `ThreadsDb::set_tension`, which itself emits
/// `ChainEvent::TensionTransition` through the configured chain sink.
/// This struct is the tick-local mirror of that event, returned in
/// `CuratorTick` so callers don't have to wire up a sink to observe
/// what happened.
#[derive(Debug, Clone)]
pub struct TensionTransition {
    pub handle: ThreadHandle,
    pub from: TensionState,
    pub to: TensionState,
    pub score: f32,
}

// --- curator ---------------------------------------------------------------

pub struct IdleCurator {
    store: Arc<ThreadsDb>,
    attention: Arc<dyn AttentionForwardStore>,
    config: CuratorConfig,
}

impl IdleCurator {
    #[must_use]
    pub fn new(
        store: Arc<ThreadsDb>,
        attention: Arc<dyn AttentionForwardStore>,
        config: CuratorConfig,
    ) -> Self {
        Self {
            store,
            attention,
            config,
        }
    }

    /// Long-running loop. Each iteration ticks, sleeps `tick_interval`,
    /// then checks the cancel token. Exits cleanly on cancel.
    pub async fn run(&self, cancel: CancellationToken) -> Result<(), CuratorError> {
        loop {
            if cancel.is_cancelled() {
                return Ok(());
            }
            self.tick().await?;
            tokio::select! {
                () = sleep(self.config.tick_interval) => {}
                () = cancel.cancelled() => return Ok(()),
            }
        }
    }

    /// Compute current fade score for one thread.
    pub async fn fade_score(&self, handle: &ThreadHandle) -> Result<f32, CuratorError> {
        Ok(self.attention.score_thread(handle).await?)
    }

    /// Single tick. Factored out so tests can drive the loop one step
    /// at a time without juggling timers.
    pub async fn tick(&self) -> Result<CuratorTick, CuratorError> {
        let threads = self.store.list_threads(None)?;
        let now = Utc::now();
        let stale_cutoff = chrono::Duration::from_std(self.config.tick_interval)
            .unwrap_or_else(|_| chrono::Duration::seconds(60));

        let mut transitions = Vec::new();
        let mut scored = 0usize;

        for thread in threads {
            // Skip threads touched within the last tick — their score
            // is based on state the runtime may not have observed yet.
            if now - thread.last_touched_at < stale_cutoff {
                continue;
            }
            scored += 1;

            let score = self.attention.score_thread(&thread.handle).await?;
            let target = self.resolve_target(thread.tension, score);
            if target == thread.tension {
                continue;
            }
            self.store.set_tension(&thread.handle, target)?;
            transitions.push(TensionTransition {
                handle: thread.handle,
                from: thread.tension,
                to: target,
                score,
            });
        }

        Ok(CuratorTick {
            threads_scored: scored,
            transitions,
        })
    }

    /// Map (current tension, raw score) → target tension.
    ///
    /// Up-transitions fire as soon as the score crosses the up
    /// threshold. Down-transitions only fire once the score has fallen
    /// below `(up_threshold - hysteresis)` — the dead-band that
    /// prevents per-tick flapping.
    fn resolve_target(&self, current: TensionState, score: f32) -> TensionState {
        let c = &self.config;

        if score >= c.active_score_threshold {
            return TensionState::Active;
        }
        if score < c.archive_threshold {
            return TensionState::Dormant;
        }

        match current {
            TensionState::Active => {
                if score < c.active_score_threshold - c.hysteresis {
                    if score >= c.slack_score_threshold {
                        TensionState::Slack
                    } else {
                        TensionState::Dormant
                    }
                } else {
                    TensionState::Active
                }
            }
            TensionState::Slack => {
                if score < c.slack_score_threshold - c.hysteresis {
                    TensionState::Dormant
                } else {
                    TensionState::Slack
                }
            }
            TensionState::Dormant => {
                if score >= c.slack_score_threshold {
                    TensionState::Slack
                } else {
                    TensionState::Dormant
                }
            }
        }
    }
}

// --- tests -----------------------------------------------------------------

#[cfg(test)]
#[allow(clippy::float_cmp)]
mod tests {
    use super::*;
    use std::sync::Mutex as StdMutex;

    use chrono::{DateTime, Duration as ChronoDuration};
    use ostk_recall_core::attention::{AttentionScope, FoldDepth, PrivacyTier};
    use ostk_recall_store::{ChainEvent, ChainSink, ThreadRecord};
    use tempfile::TempDir;

    use crate::{InMemoryAttention, stub_embed};

    // --- fake chain sink ---------------------------------------------------

    #[derive(Default)]
    struct RecordingSink {
        events: StdMutex<Vec<ChainEvent>>,
    }

    impl RecordingSink {
        fn transitions(&self) -> Vec<(ThreadHandle, TensionState, TensionState)> {
            self.events
                .lock()
                .unwrap()
                .iter()
                .filter_map(|e| match e {
                    ChainEvent::TensionTransition {
                        handle, from, to, ..
                    } => Some((handle.clone(), *from, *to)),
                    _ => None,
                })
                .collect()
        }
    }

    impl ChainSink for RecordingSink {
        fn append(
            &self,
            event: &ChainEvent,
        ) -> std::result::Result<(), ostk_recall_store::StoreError> {
            self.events.lock().unwrap().push(event.clone());
            Ok(())
        }
    }

    // --- fixtures ----------------------------------------------------------

    struct Fixture {
        _tmp: TempDir,
        sink: Arc<RecordingSink>,
        store: Arc<ThreadsDb>,
        attention: Arc<InMemoryAttention>,
        scope: AttentionScope,
    }

    impl Fixture {
        fn new() -> Self {
            let tmp = TempDir::new().unwrap();
            let sink: Arc<RecordingSink> = Arc::new(RecordingSink::default());
            let store = Arc::new(
                ThreadsDb::open_with_sink(tmp.path(), sink.clone() as Arc<dyn ChainSink>).unwrap(),
            );
            let attention = Arc::new(InMemoryAttention::new());
            let scope = AttentionScope {
                project: Some("haystack".into()),
                session_id: Some("s1".into()),
                agent: Some("claude".into()),
                privacy_tier: PrivacyTier::T1Project,
            };
            Self {
                _tmp: tmp,
                sink,
                store,
                attention,
                scope,
            }
        }

        fn curator(&self, config: CuratorConfig) -> IdleCurator {
            IdleCurator::new(
                self.store.clone(),
                self.attention.clone() as Arc<dyn AttentionForwardStore>,
                config,
            )
        }

        fn insert_thread(
            &self,
            name: &str,
            tension: TensionState,
            last_touched_at: DateTime<Utc>,
            familiarity: u32,
        ) -> ThreadHandle {
            let h = ThreadHandle::new(name).unwrap();
            let rec = ThreadRecord {
                handle: h.clone(),
                tension,
                mentions: familiarity,
                resonance: familiarity,
                last_touched_at,
                anchor_chunk_id: None,
                fold_override: None,
                created_at: last_touched_at,
                created_scope_key: None,
                privacy_tier: PrivacyTier::T1Project,
            };
            self.store.upsert_thread(&rec).unwrap();
            h
        }

        async fn install_attention_thread(
            &self,
            handle: &ThreadHandle,
            tension: f32,
            familiarity: u32,
            last_touched_at: DateTime<Utc>,
            anchor: Vec<f32>,
        ) {
            self.attention
                .__install_thread_for_test(
                    &self.scope,
                    handle.clone(),
                    tension,
                    familiarity,
                    last_touched_at,
                    FoldDepth::Folded,
                    anchor,
                )
                .await;
        }
    }

    fn quick_config() -> CuratorConfig {
        // Short tick_interval so the stale-touch guard doesn't swallow
        // test threads whose timestamps are merely a few seconds old.
        CuratorConfig {
            tick_interval: Duration::from_millis(50),
            ..CuratorConfig::default()
        }
    }

    // --- tests -------------------------------------------------------------

    #[tokio::test]
    async fn thread_without_touch_fades_over_simulated_time() {
        let fx = Fixture::new();
        fx.attention
            .attend(&fx.scope, "membrane of possibility")
            .await
            .unwrap();
        let opposite: Vec<f32> = stub_embed("membrane of possibility")
            .iter()
            .map(|x| -x)
            .collect();
        let stale = Utc::now() - ChronoDuration::days(7);
        let h = fx.insert_thread("fading", TensionState::Active, stale, 0);
        fx.install_attention_thread(&h, 0.9, 0, stale, opposite)
            .await;

        let curator = fx.curator(quick_config());
        let tick = curator.tick().await.unwrap();

        assert_eq!(tick.threads_scored, 1);
        let transition = tick
            .transitions
            .iter()
            .find(|t| t.handle == h)
            .expect("expected transition for fading thread");
        assert_eq!(transition.to, TensionState::Dormant);
        let after = fx.store.get_thread(&h).unwrap().unwrap();
        assert_eq!(after.tension, TensionState::Dormant);
        let chain = fx.sink.transitions();
        assert!(
            chain
                .iter()
                .any(|(hh, _, to)| hh == &h && *to == TensionState::Dormant),
            "chain must record the dormant transition"
        );
    }

    #[tokio::test]
    async fn touching_reanimates_thread() {
        let fx = Fixture::new();
        fx.attention.attend(&fx.scope, "warm topic").await.unwrap();
        let now = Utc::now();
        // 1 second is far enough past the 50ms tick_interval to pass
        // the stale-touch guard, but still reads as "essentially now"
        // so the decay term in the score remains at its floor.
        let touched = now - ChronoDuration::seconds(1);
        let h = fx.insert_thread("reanimated", TensionState::Dormant, touched, 20);
        fx.install_attention_thread(&h, 0.5, 20, touched, stub_embed("warm topic"))
            .await;

        let curator = fx.curator(quick_config());
        let tick = curator.tick().await.unwrap();

        let target = tick
            .transitions
            .iter()
            .find(|t| t.handle == h)
            .expect("dormant thread should transition upward");
        assert_eq!(target.to, TensionState::Active);
    }

    #[tokio::test]
    async fn off_diagonal_lift_keeps_dormant_resonant_alive() {
        let fx = Fixture::new();
        fx.attention
            .attend(&fx.scope, "off diagonal topic")
            .await
            .unwrap();
        let stale = Utc::now() - ChronoDuration::days(5);
        let h = fx.insert_thread("dormant-resonant", TensionState::Dormant, stale, 1);
        fx.install_attention_thread(&h, 0.05, 1, stale, stub_embed("off diagonal topic"))
            .await;

        let curator = fx.curator(quick_config());
        let score = curator.fade_score(&h).await.unwrap();
        assert!(
            score >= curator.config.archive_threshold,
            "off-diagonal lift should keep score above archive: {score}"
        );

        let tick = curator.tick().await.unwrap();
        for t in &tick.transitions {
            if t.handle == h {
                assert_ne!(
                    t.to,
                    TensionState::Dormant,
                    "off-diagonal-lifted thread must not transition to Dormant"
                );
            }
        }
    }

    #[tokio::test]
    async fn archive_threshold_from_config() {
        let fx = Fixture::new();
        fx.attention.attend(&fx.scope, "ctx").await.unwrap();
        let stale = Utc::now() - ChronoDuration::days(2);
        let opposite: Vec<f32> = stub_embed("ctx").iter().map(|x| -x).collect();
        let h = fx.insert_thread("threshold-test", TensionState::Slack, stale, 10);
        fx.install_attention_thread(&h, 0.9, 10, stale, opposite)
            .await;

        let curator = fx.curator(CuratorConfig {
            archive_threshold: 0.95,
            slack_score_threshold: 0.96,
            active_score_threshold: 0.99,
            ..quick_config()
        });
        let score = curator.fade_score(&h).await.unwrap();
        assert!(
            score < 0.95,
            "test precondition: score {score} must be below 0.95"
        );

        let tick = curator.tick().await.unwrap();
        let t = tick.transitions.iter().find(|t| t.handle == h).unwrap();
        assert_eq!(t.to, TensionState::Dormant);
    }

    #[tokio::test]
    async fn hysteresis_prevents_bouncing() {
        let fx = Fixture::new();
        fx.attention.attend(&fx.scope, "boundary").await.unwrap();
        let stale = Utc::now() - ChronoDuration::days(1);

        let h = fx.insert_thread("boundary", TensionState::Active, stale, 10);
        fx.install_attention_thread(&h, 0.5, 10, stale, stub_embed("boundary"))
            .await;

        let curator = fx.curator(CuratorConfig {
            active_score_threshold: 2.0,
            slack_score_threshold: 0.01,
            archive_threshold: 0.0,
            hysteresis: 0.2,
            ..quick_config()
        });

        let score = curator.fade_score(&h).await.unwrap();
        assert!(score > 0.01 && score < 2.0, "got {score}");

        let t1 = curator.tick().await.unwrap();
        let t2 = curator.tick().await.unwrap();
        let t3 = curator.tick().await.unwrap();
        let bounces = t1.transitions.len() + t2.transitions.len() + t3.transitions.len();
        assert!(
            bounces <= 1,
            "expected at most one transition for boundary thread, got {bounces}"
        );
    }

    #[tokio::test]
    async fn tick_with_no_threads_succeeds() {
        let fx = Fixture::new();
        let curator = fx.curator(quick_config());
        let tick = curator.tick().await.unwrap();
        assert_eq!(tick.threads_scored, 0);
        assert!(tick.transitions.is_empty());
    }

    #[tokio::test]
    async fn recently_touched_thread_skipped() {
        let fx = Fixture::new();
        let now = Utc::now();
        let h = fx.insert_thread("fresh", TensionState::Active, now, 0);
        fx.install_attention_thread(&h, 0.5, 0, now, stub_embed("anything"))
            .await;

        let curator = fx.curator(CuratorConfig {
            tick_interval: Duration::from_secs(60),
            ..CuratorConfig::default()
        });
        let tick = curator.tick().await.unwrap();
        assert_eq!(
            tick.threads_scored, 0,
            "recently-touched thread must be skipped"
        );
        assert!(tick.transitions.is_empty());
    }

    #[tokio::test]
    async fn run_exits_on_cancel() {
        let fx = Fixture::new();
        fx.attention.attend(&fx.scope, "run-test").await.unwrap();
        let stale = Utc::now() - ChronoDuration::days(10);
        let opposite: Vec<f32> = stub_embed("run-test").iter().map(|x| -x).collect();
        let h = fx.insert_thread("run-target", TensionState::Active, stale, 0);
        fx.install_attention_thread(&h, 0.9, 0, stale, opposite)
            .await;

        let curator = Arc::new(fx.curator(CuratorConfig {
            tick_interval: Duration::from_millis(10),
            ..CuratorConfig::default()
        }));
        let cancel = CancellationToken::new();
        let runner = {
            let c = curator.clone();
            let token = cancel.clone();
            tokio::spawn(async move { c.run(token).await })
        };
        sleep(Duration::from_millis(50)).await;
        cancel.cancel();
        let result = tokio::time::timeout(Duration::from_secs(2), runner)
            .await
            .expect("run did not exit after cancel")
            .expect("join failed");
        result.expect("curator returned error");

        let chain = fx.sink.transitions();
        assert!(
            chain.iter().any(|(hh, _, _)| hh == &h),
            "expected at least one chained transition for run-target"
        );
    }
}
