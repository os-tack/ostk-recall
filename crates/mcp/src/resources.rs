//! MCP resources protocol surface (P9a-min).
//!
//! A [`Resource`] is read-only addressable content keyed by URI; a
//! [`ResourceRegistry`] holds the live set and supports the four
//! protocol methods needed by the first active lens (P9b-min):
//!
//! - `resources/list` — advertise URI / name / description / mime.
//! - `resources/read` — fetch the current content.
//! - `resources/subscribe` — register interest for a URI.
//! - `notifications/resources/updated` — pushed from the server when
//!   the underlying content changes.
//!
//! P9a-min is the singleton-stdio cut: there is exactly one client per
//! process, modeled by [`ClientId::StdioSingleton`]. Unsubscribe and
//! per-client disconnect cleanup are P9a-full.
//!
//! ## Locking
//!
//! Resources and subscribers are held behind `std::sync::RwLock` so
//! dispatch (which runs on the tokio runtime) doesn't pay async
//! overhead for a cheap HashMap lookup. Outbound notifications
//! collect the subscriber list under a read lock, drop the lock,
//! then send into the writer-task channel — exactly the pattern
//! `p9a-mcp-resources.md` calls out in the "race condition between
//! resource update and subscriber notification" mitigation.
//!
//! ## Wire shape
//!
//! Methods that produce JSON-RPC results return `serde_json::Value`
//! pre-shaped to the MCP spec ("resources" array for list,
//! "contents" array for read). The dispatcher hands them straight to
//! `JsonRpcResponse::ok` without re-wrapping.

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex, RwLock};

use serde_json::{Value, json};
use thiserror::Error;
use tokio::sync::mpsc::UnboundedSender;
use tracing::warn;

use crate::protocol::JsonRpcError;

// ---------------------------------------------------------------------
// Resource trait + content type
// ---------------------------------------------------------------------

/// Read-only addressable content. Implementations advertise an
/// immutable identity (`uri` / `name` / `description` / `mime_type`)
/// and a snapshot read.
///
/// `read()` is synchronous because the P6A rolling vector and the
/// P9b-min lens text both live in memory — there is no IO to await.
/// If a later phase needs async (e.g. Lance fetch), the trait can
/// gain an `async fn read_async` without breaking the sync path.
pub trait Resource: Send + Sync {
    /// Stable URI identifying this resource. Returned by `list` and
    /// keys both the registry map and the subscriber map.
    fn uri(&self) -> &str;
    /// Short human-readable name (≤ 60 chars by convention).
    fn name(&self) -> &str;
    /// One-paragraph description shown to clients during discovery.
    fn description(&self) -> &str;
    /// MIME type of the rendered content. Defaults to
    /// `text/markdown` — the first lens renders markdown.
    fn mime_type(&self) -> &str {
        "text/markdown"
    }
    /// Snapshot the current content. Called on `resources/read` and
    /// not otherwise — registry never caches the result.
    fn read(&self) -> Result<ResourceContent, ResourceError>;
}

/// Snapshot returned by [`Resource::read`].
///
/// Exactly one of `text` / `blob` is `Some`; setting both is a
/// programming error and serializes only `text`.
#[derive(Debug, Clone)]
pub struct ResourceContent {
    pub uri: String,
    pub mime_type: String,
    pub text: Option<String>,
    pub blob: Option<Vec<u8>>,
}

impl ResourceContent {
    /// Convenience constructor for text payloads.
    #[must_use]
    pub fn text(
        uri: impl Into<String>,
        mime_type: impl Into<String>,
        body: impl Into<String>,
    ) -> Self {
        Self {
            uri: uri.into(),
            mime_type: mime_type.into(),
            text: Some(body.into()),
            blob: None,
        }
    }

    /// Wire shape for the `contents[]` entry in `resources/read`.
    ///
    /// P9a-min ships text only. The `blob` field stays on the type
    /// so the eventual binary path doesn't break the API shape, but
    /// resources that populate `blob` here are silently dropped —
    /// P9a-full pulls in a `base64` dep and encodes properly.
    fn to_json(&self) -> Value {
        let mut obj = serde_json::Map::new();
        obj.insert("uri".into(), Value::String(self.uri.clone()));
        obj.insert("mimeType".into(), Value::String(self.mime_type.clone()));
        if let Some(t) = &self.text {
            obj.insert("text".into(), Value::String(t.clone()));
        } else if self.blob.is_some() {
            warn!(
                uri = %self.uri,
                "P9a-min: blob resource content not yet supported on the wire (P9a-full adds base64)"
            );
        }
        Value::Object(obj)
    }
}

// ---------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------

#[derive(Debug, Clone, Error)]
pub enum ResourceError {
    #[error("resource not found: {0}")]
    NotFound(String),
    #[error("invalid params: {0}")]
    InvalidParams(String),
    #[error("read failed: {0}")]
    ReadFailed(String),
}

impl ResourceError {
    /// Convert to the JSON-RPC error code the dispatch handler ships
    /// back to the client. `NotFound` maps to invalid params (the
    /// URI parameter referenced an unknown resource); `ReadFailed`
    /// maps to internal so a misbehaving resource impl doesn't look
    /// like client error.
    #[must_use]
    pub fn into_rpc(self) -> JsonRpcError {
        match &self {
            Self::NotFound(_) | Self::InvalidParams(_) => {
                JsonRpcError::invalid_params(self.to_string())
            }
            Self::ReadFailed(_) => JsonRpcError::internal(self.to_string()),
        }
    }
}

// ---------------------------------------------------------------------
// ClientId
// ---------------------------------------------------------------------

/// Placeholder for the eventual multi-transport client identifier.
///
/// P9a-min targets the stdio singleton — exactly one client per
/// process. `StdioSingleton` is the only variant. When the server
/// gains a network transport, this gains a `Network { id: u64 }`
/// variant and the subscriber map starts pruning on disconnect; the
/// existing surface (subscribe / notify_updated) stays.
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub enum ClientId {
    StdioSingleton,
}

impl ClientId {
    /// Construct the canonical stdio client identifier. Used by the
    /// stdio dispatch path so every subscribe lands under the same
    /// key.
    #[must_use]
    pub const fn stdio_singleton() -> Self {
        Self::StdioSingleton
    }
}

// ---------------------------------------------------------------------
// Registry
// ---------------------------------------------------------------------

/// Live set of resources plus subscriber routing.
///
/// Cheap to clone via `Arc`; the inner state is shared. Construct via
/// `Default` (empty) and hand to [`crate::Server`] via
/// `with_resources`. Tests register resources directly with
/// [`Self::register`]; production wiring (P9b-min) registers the
/// `memory-lens` resource at boot.
#[derive(Default)]
pub struct ResourceRegistry {
    resources: RwLock<HashMap<String, Arc<dyn Resource>>>,
    subscribers: RwLock<HashMap<String, HashSet<ClientId>>>,
    /// Outbound channel into the stdio writer task. `None` until
    /// `set_outbound` is called by `run_stdio` (or test fixtures).
    /// `emit_resource_updated` is a no-op when this is unset, so
    /// libraries can construct a registry without wiring a transport.
    outbound: Mutex<Option<UnboundedSender<String>>>,
}

impl ResourceRegistry {
    /// Empty registry.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a resource under its `uri()`. Replaces any prior
    /// registration on the same URI — the new resource owns the
    /// existing subscriber list. (Re-registration is a P9b-full
    /// concern; for P9a-min it's defined so test fixtures can swap
    /// implementations.)
    pub fn register(&self, resource: Arc<dyn Resource>) {
        let uri = resource.uri().to_string();
        if let Ok(mut guard) = self.resources.write() {
            guard.insert(uri, resource);
        }
    }

    /// `resources/list` wire result. Empty registry returns an empty
    /// array (not an error) — that matches MCP spec for "no
    /// resources advertised yet."
    #[must_use]
    pub fn list(&self) -> Value {
        let guard = match self.resources.read() {
            Ok(g) => g,
            Err(p) => p.into_inner(),
        };
        let mut items: Vec<Value> = guard
            .values()
            .map(|r| {
                json!({
                    "uri": r.uri(),
                    "name": r.name(),
                    "description": r.description(),
                    "mimeType": r.mime_type(),
                })
            })
            .collect();
        // Deterministic ordering: clients shouldn't depend on it,
        // but golden tests will, so sort by URI.
        items.sort_by(|a, b| {
            a.get("uri")
                .and_then(Value::as_str)
                .cmp(&b.get("uri").and_then(Value::as_str))
        });
        json!({ "resources": items })
    }

    /// `resources/read` wire result for the given URI. Unknown URI
    /// returns [`ResourceError::NotFound`].
    pub fn read(&self, uri: &str) -> Result<Value, ResourceError> {
        let resource = {
            let guard = match self.resources.read() {
                Ok(g) => g,
                Err(p) => p.into_inner(),
            };
            guard
                .get(uri)
                .cloned()
                .ok_or_else(|| ResourceError::NotFound(uri.to_string()))?
        };
        let content = resource.read()?;
        Ok(json!({ "contents": [content.to_json()] }))
    }

    /// Register `client` as a subscriber to `uri`.
    ///
    /// Returns `NotFound` when the URI is unknown — subscribing to a
    /// resource the server hasn't advertised is a client bug worth
    /// surfacing.
    pub fn subscribe(&self, client: ClientId, uri: &str) -> Result<(), ResourceError> {
        // Existence check under the resources lock; release before
        // grabbing the subscribers lock so the two never deadlock.
        let exists = {
            let guard = match self.resources.read() {
                Ok(g) => g,
                Err(p) => p.into_inner(),
            };
            guard.contains_key(uri)
        };
        if !exists {
            return Err(ResourceError::NotFound(uri.to_string()));
        }
        let mut subs = match self.subscribers.write() {
            Ok(g) => g,
            Err(p) => p.into_inner(),
        };
        subs.entry(uri.to_string()).or_default().insert(client);
        Ok(())
    }

    /// Subscribers currently registered for `uri`. Exposed for the
    /// notification path and for tests that want to assert
    /// subscription bookkeeping without driving a fake transport.
    #[must_use]
    pub fn subscribers_for(&self, uri: &str) -> Vec<ClientId> {
        let guard = match self.subscribers.read() {
            Ok(g) => g,
            Err(p) => p.into_inner(),
        };
        guard
            .get(uri)
            .map(|s| s.iter().cloned().collect())
            .unwrap_or_default()
    }

    /// Install the outbound channel used by
    /// [`Self::emit_resource_updated`].
    pub fn set_outbound(&self, tx: UnboundedSender<String>) {
        if let Ok(mut guard) = self.outbound.lock() {
            *guard = Some(tx);
        }
    }

    /// Drop the held outbound `Sender`, if any.
    ///
    /// Called by [`crate::Server::serve`] during shutdown so the
    /// writer task's receiver sees every sender close and exits
    /// cleanly. Without this, the registry's clone keeps the
    /// channel alive past stdin EOF and the writer task wedges on
    /// `rx.recv().await` forever.
    ///
    /// After `clear_outbound`, `emit_resource_updated` becomes a
    /// silent no-op until a new sender is installed.
    pub fn clear_outbound(&self) {
        if let Ok(mut guard) = self.outbound.lock() {
            *guard = None;
        }
    }

    /// Push `notifications/resources/updated` to every subscriber of
    /// `uri`. Best-effort: if no outbound channel is wired or no
    /// subscribers exist, the call is a silent no-op (so library
    /// users that don't subscribe still see resource changes the
    /// next time they poll `resources/read`).
    ///
    /// Subscribers are collected under a read lock, then released
    /// before the channel send so a slow writer task can't block the
    /// next subscribe.
    pub fn emit_resource_updated(&self, uri: &str) {
        let subscribers = self.subscribers_for(uri);
        if subscribers.is_empty() {
            return;
        }
        let tx = match self.outbound.lock() {
            Ok(g) => g.clone(),
            Err(p) => p.into_inner().clone(),
        };
        let Some(tx) = tx else { return };
        let envelope = json!({
            "jsonrpc": "2.0",
            "method": "notifications/resources/updated",
            "params": { "uri": uri },
        })
        .to_string();
        // The MCP spec treats notifications as fire-and-forget per
        // subscriber, but the singleton stdio transport has exactly
        // one channel — multi-client routing is P9a-full work, so
        // sending once is sufficient even when subscribers.len() > 1
        // (which can't happen in P9a-min).
        if tx.send(envelope).is_err() {
            warn!(uri = %uri, "writer task closed; resource update dropped");
        }
    }
}

// ---------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    struct FakeResource {
        uri: String,
        name: String,
        description: String,
        body: String,
    }

    impl Resource for FakeResource {
        fn uri(&self) -> &str {
            &self.uri
        }
        fn name(&self) -> &str {
            &self.name
        }
        fn description(&self) -> &str {
            &self.description
        }
        fn read(&self) -> Result<ResourceContent, ResourceError> {
            Ok(ResourceContent::text(
                &self.uri,
                self.mime_type(),
                self.body.clone(),
            ))
        }
    }

    fn fake(uri: &str, body: &str) -> Arc<dyn Resource> {
        Arc::new(FakeResource {
            uri: uri.into(),
            name: format!("name-of-{uri}"),
            description: "test resource".into(),
            body: body.into(),
        })
    }

    #[test]
    fn list_returns_registered_resources_sorted() {
        let reg = ResourceRegistry::new();
        reg.register(fake("ostk://b", "B"));
        reg.register(fake("ostk://a", "A"));
        let v = reg.list();
        let arr = v.get("resources").and_then(Value::as_array).unwrap();
        assert_eq!(arr.len(), 2);
        assert_eq!(arr[0].get("uri").and_then(Value::as_str), Some("ostk://a"));
        assert_eq!(arr[1].get("uri").and_then(Value::as_str), Some("ostk://b"));
    }

    #[test]
    fn read_returns_content_for_known_uri() {
        let reg = ResourceRegistry::new();
        reg.register(fake("ostk://x", "hello"));
        let v = reg.read("ostk://x").unwrap();
        let contents = v.get("contents").and_then(Value::as_array).unwrap();
        assert_eq!(contents.len(), 1);
        assert_eq!(
            contents[0].get("text").and_then(Value::as_str),
            Some("hello")
        );
        assert_eq!(
            contents[0].get("mimeType").and_then(Value::as_str),
            Some("text/markdown")
        );
    }

    #[test]
    fn read_unknown_uri_errors_as_not_found() {
        let reg = ResourceRegistry::new();
        let err = reg.read("ostk://missing").unwrap_err();
        assert!(matches!(err, ResourceError::NotFound(_)));
    }

    #[test]
    fn subscribe_rejects_unknown_uri() {
        let reg = ResourceRegistry::new();
        let err = reg
            .subscribe(ClientId::stdio_singleton(), "ostk://missing")
            .unwrap_err();
        assert!(matches!(err, ResourceError::NotFound(_)));
    }

    #[test]
    fn subscribe_records_singleton_subscriber() {
        let reg = ResourceRegistry::new();
        reg.register(fake("ostk://x", "body"));
        reg.subscribe(ClientId::stdio_singleton(), "ostk://x")
            .unwrap();
        let subs = reg.subscribers_for("ostk://x");
        assert_eq!(subs.len(), 1);
        assert!(matches!(subs[0], ClientId::StdioSingleton));
    }

    #[tokio::test]
    async fn emit_pushes_envelope_to_outbound_channel() {
        let reg = ResourceRegistry::new();
        reg.register(fake("ostk://x", "body"));
        reg.subscribe(ClientId::stdio_singleton(), "ostk://x")
            .unwrap();

        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<String>();
        reg.set_outbound(tx);
        reg.emit_resource_updated("ostk://x");

        let envelope = rx.recv().await.expect("envelope must be sent");
        let v: Value = serde_json::from_str(&envelope).unwrap();
        assert_eq!(v.get("jsonrpc").and_then(Value::as_str), Some("2.0"));
        assert_eq!(
            v.get("method").and_then(Value::as_str),
            Some("notifications/resources/updated")
        );
        assert_eq!(
            v.pointer("/params/uri").and_then(Value::as_str),
            Some("ostk://x")
        );
    }

    #[tokio::test]
    async fn emit_without_subscribers_does_not_send() {
        let reg = ResourceRegistry::new();
        reg.register(fake("ostk://x", "body"));
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<String>();
        reg.set_outbound(tx);
        reg.emit_resource_updated("ostk://x");
        // try_recv should be empty.
        assert!(rx.try_recv().is_err());
    }
}
