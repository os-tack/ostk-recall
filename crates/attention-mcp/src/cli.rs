//! CLI subcommand surface for the attention substrate.
//!
//! The `ostk-recall` binary defines a thin clap subcommand tree that
//! routes into the structs and `run_*` functions here. Keeping the
//! parsing here (rather than in the CLI crate) lets a future
//! `ostk-recall-attention-cli` consume the same surface without
//! duplicating clap derives.
//!
//! Each `run_*` returns a `serde_json::Value`. The CLI binary chooses
//! human-readable or JSON output via the existing `--json` flag pattern.

use serde_json::{Value, json};

use crate::handlers::{
    AttentionDispatch, AttentionHandlersError, attend, decay, familiarize, fold, surface,
    thread_create, thread_link, thread_list, thread_promote, thread_unlink,
};

/// `attention attend --scope-project P --context <STR>`.
pub async fn run_attend(
    d: &AttentionDispatch,
    scope_project: Option<String>,
    session_id: Option<String>,
    agent: Option<String>,
    privacy_tier: Option<String>,
    context: String,
) -> Result<Value, AttentionHandlersError> {
    let args = json!({
        "scope": build_scope(scope_project, session_id, agent, privacy_tier),
        "context": context,
    });
    attend(d, args).await
}

/// `attention surface --scope-project P --limit N`.
pub async fn run_surface(
    d: &AttentionDispatch,
    scope_project: Option<String>,
    session_id: Option<String>,
    agent: Option<String>,
    privacy_tier: Option<String>,
    limit: Option<u64>,
) -> Result<Value, AttentionHandlersError> {
    let args = json!({
        "scope": build_scope(scope_project, session_id, agent, privacy_tier),
        "limit": limit.unwrap_or(20),
    });
    surface(d, args).await
}

/// `attention fold --handle H --depth D`.
pub async fn run_fold(
    d: &AttentionDispatch,
    scope_project: Option<String>,
    session_id: Option<String>,
    agent: Option<String>,
    privacy_tier: Option<String>,
    handle: String,
    depth: String,
) -> Result<Value, AttentionHandlersError> {
    let args = json!({
        "scope": build_scope(scope_project, session_id, agent, privacy_tier),
        "handle": handle,
        "depth": depth,
    });
    fold(d, args).await
}

/// `attention familiarize --handle H`.
pub async fn run_familiarize(
    d: &AttentionDispatch,
    scope_project: Option<String>,
    session_id: Option<String>,
    agent: Option<String>,
    privacy_tier: Option<String>,
    handle: String,
) -> Result<Value, AttentionHandlersError> {
    let args = json!({
        "scope": build_scope(scope_project, session_id, agent, privacy_tier),
        "handle": handle,
    });
    familiarize(d, args).await
}

/// `attention decay --handle H --factor F`.
pub async fn run_decay(
    d: &AttentionDispatch,
    handle: String,
    factor: f64,
) -> Result<Value, AttentionHandlersError> {
    decay(d, json!({ "handle": handle, "factor": factor })).await
}

/// `thread create --handle H [--body-from-file PATH]`.
pub async fn run_thread_create(
    d: &AttentionDispatch,
    scope_project: Option<String>,
    handle: String,
    body: Option<String>,
    tension: Option<String>,
) -> Result<Value, AttentionHandlersError> {
    let mut args = json!({
        "scope": build_scope(scope_project, None, None, None),
        "handle": handle,
    });
    if let Some(b) = body {
        args["body"] = Value::String(b);
    }
    if let Some(t) = tension {
        args["tension"] = Value::String(t);
    }
    thread_create(d, args).await
}

/// `thread link --handle H --target P --category C`.
pub async fn run_thread_link(
    d: &AttentionDispatch,
    scope_project: Option<String>,
    handle: String,
    target_path: String,
    category: String,
) -> Result<Value, AttentionHandlersError> {
    let args = json!({
        "scope": build_scope(scope_project, None, None, None),
        "handle": handle,
        "target_path": target_path,
        "category": category,
    });
    thread_link(d, args).await
}

/// `thread unlink --evidence-id N`.
pub async fn run_thread_unlink(
    d: &AttentionDispatch,
    evidence_id: i64,
) -> Result<Value, AttentionHandlersError> {
    thread_unlink(d, json!({ "evidence_id": evidence_id })).await
}

/// `thread promote --from PROPOSED_HANDLE --to active|slack`.
pub async fn run_thread_promote(
    d: &AttentionDispatch,
    handle_from_proposed: String,
    target_tier: String,
) -> Result<Value, AttentionHandlersError> {
    thread_promote(
        d,
        json!({
            "handle_from_proposed": handle_from_proposed,
            "target_tier": target_tier,
        }),
    )
    .await
}

/// `thread list [--tension active|slack|dormant]`.
pub async fn run_thread_list(
    d: &AttentionDispatch,
    scope_project: Option<String>,
    tension: Option<String>,
) -> Result<Value, AttentionHandlersError> {
    let mut args = json!({
        "scope": build_scope(scope_project, None, None, None),
    });
    if let Some(t) = tension {
        args["tension"] = Value::String(t);
    }
    thread_list(d, args).await
}

fn build_scope(
    project: Option<String>,
    session_id: Option<String>,
    agent: Option<String>,
    privacy_tier: Option<String>,
) -> Value {
    let mut m = serde_json::Map::new();
    if let Some(p) = project {
        m.insert("project".into(), Value::String(p));
    }
    if let Some(s) = session_id {
        m.insert("session_id".into(), Value::String(s));
    }
    if let Some(a) = agent {
        m.insert("agent".into(), Value::String(a));
    }
    if let Some(t) = privacy_tier {
        m.insert("privacy_tier".into(), Value::String(t));
    }
    Value::Object(m)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handlers::AttentionDispatch;
    use ostk_recall_attention::{AttentionForwardStore, InMemoryAttention};
    use ostk_recall_store::ThreadsDb;
    use std::sync::Arc;
    use tempfile::TempDir;

    fn build_dispatch() -> (TempDir, AttentionDispatch) {
        let tmp = TempDir::new().unwrap();
        let threads = Arc::new(ThreadsDb::open(tmp.path()).unwrap());
        let attention: Arc<dyn AttentionForwardStore> = Arc::new(InMemoryAttention::new());
        (tmp, AttentionDispatch::new(attention, threads))
    }

    #[tokio::test]
    async fn cli_attention_surface_json_output() {
        // Mirror the test in the task brief: invoke the CLI subcommand
        // and assert valid JSON shape comes out.
        let (_tmp, d) = build_dispatch();
        run_attend(
            &d,
            Some("haystack".into()),
            None,
            None,
            None,
            "context body".into(),
        )
        .await
        .unwrap();
        let out = run_surface(&d, Some("haystack".into()), None, None, None, Some(5))
            .await
            .unwrap();
        let s = serde_json::to_string(&out).unwrap();
        // Round-trip through the JSON layer to assert it's valid.
        let back: Value = serde_json::from_str(&s).unwrap();
        assert!(back["pages"].is_array());
    }

    #[tokio::test]
    async fn cli_thread_create_list_round_trip() {
        let (_tmp, d) = build_dispatch();
        let out = run_thread_create(
            &d,
            Some("p".into()),
            "cli-thread".into(),
            None,
            Some("active".into()),
        )
        .await
        .unwrap();
        assert_eq!(out["record"]["handle"], "cli-thread");
        let list = run_thread_list(&d, Some("p".into()), Some("active".into()))
            .await
            .unwrap();
        let recs = list["records"].as_array().unwrap();
        assert!(recs.iter().any(|r| r["handle"] == "cli-thread"));
    }
}
