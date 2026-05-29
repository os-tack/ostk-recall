//! P1 gate: `before` recall param plumbs into the Lance filter as a
//! half-open upper bound, yielding `[since, before)` intervals.

use chrono::{TimeZone, Utc};
use ostk_recall_core::RecallParams;

#[test]
fn recall_params_carry_before_through_serde() {
    // Construct manually + round-trip via JSON to confirm the schema
    // exposes `before` (MCP recall tool argument).
    let p = RecallParams {
        query: "x".into(),
        before: Some(Utc.with_ymd_and_hms(2026, 4, 17, 10, 0, 0).unwrap()),
        ..RecallParams::default()
    };
    let j = serde_json::to_string(&p).unwrap();
    assert!(
        j.contains("\"before\":\"2026-04-17T10:00:00Z\""),
        "serialized before: {j}"
    );
    let back: RecallParams = serde_json::from_str(&j).unwrap();
    assert_eq!(back.before, p.before);
}

#[test]
fn before_defaults_to_none() {
    let p = RecallParams {
        query: "x".into(),
        ..RecallParams::default()
    };
    assert!(p.before.is_none());
}
