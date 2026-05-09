# `ostk-recall-serve` driver protocol

Wire protocol between the haystack kernel (client) and `ostk-recall-serve`
(peer-process daemon). Spec for haystack →1846 cut #3 (→1848).

## Status

`v0.1` — draft. First implementation lands with `ostk-recall` v0.2.0.

## Transport

Matches the existing fcp-* driver pattern in haystack:

- The daemon is a long-running subprocess started by the kernel via the
  driver registry (`fcp-recall` slot, currently `Internal { dormant }`,
  promoted to `External { spawn-on-demand }`).
- Communication is **JSON-RPC 2.0** over **stdin/stdout** of the
  subprocess.
- The kernel exposes a Unix-domain socket at
  `.ostk/drivers/fcp-recall.sock` and multiplexes incoming socket
  connections onto the single subprocess stdio stream via the existing
  `kernel::driver_relay` (concurrent relay; see `driver_relay.rs`).
- One frame per line — newline-delimited JSON. The relay buffers reads
  with `BufReader::read_line`.
- Idle-timeout shutdown: 30 minutes by default. Configurable via
  `RelayConfig::idle_timeout`.

The kernel handles spawning, socket exposure, multiplexing, and idle
lifecycle. The daemon only needs to:

1. Read JSON-RPC requests from stdin (newline-framed).
2. Write JSON-RPC responses to stdout (newline-framed).
3. Exit cleanly on `SIGTERM` or stdin EOF.

## Lifecycle

```
kernel                                        ostk-recall-serve
──────                                        ─────────────────
    │  spawn (lazy, on first request)            │
    │ ─────────────────────────────────────────► │
    │                                            │  bind/load model
    │  initialize { ... }                        │
    │ ─────────────────────────────────────────► │
    │                                            │  open CorpusStore
    │       initialize result { ... }            │  load Embedder
    │ ◄───────────────────────────────────────── │
    │                                            │
    │  recall.fault { query, intent, ... }       │
    │ ─────────────────────────────────────────► │
    │                                            │  embed + lance search
    │       recall.fault result { pages }        │  + synthesize
    │ ◄───────────────────────────────────────── │
    │                                            │
    │  ... more requests ...                     │
    │                                            │
    │  (idle 30 min)                             │
    │  SIGTERM                                   │
    │ ─────────────────────────────────────────► │
    │                                            │  flush + exit 0
```

## Methods

### `initialize`

Sent once after spawn, before any operational request.

**Params:**
```json
{
  "ostk_dir": "/abs/path/to/.ostk",
  "embed_dim": 256
}
```

- `ostk_dir` (required): absolute path to the project's `.ostk/` directory.
  The daemon resolves the corpus database location relative to this.
- `embed_dim` (optional, default 256): expected embedding dimension. The
  daemon validates this matches the loaded embedder; if not, returns an
  error (caller should re-initialize with the daemon's actual dim).

**Result:**
```json
{
  "name": "ostk-recall-serve",
  "version": "0.2.0",
  "embedder": { "model": "potion-base-8M", "dim": 256 },
  "corpus_root": "/abs/path/to/.ostk/recall"
}
```

**Errors:**
- `-32602 invalid_params` — missing or non-absolute `ostk_dir`
- `-32000 corpus_not_found` — `ostk_dir/recall/` doesn't exist or isn't
  a valid corpus
- `-32001 embedder_load_failed` — model couldn't be loaded (corruption,
  missing weights, etc.)

### `recall.fault`

Materializes recall hits into virtual memory pages. This is the
operation called from `kernel::recall::vm::fault_recall` in haystack.

**Params:**
```json
{
  "query": "alloc_page",
  "intent": "symbol",
  "limit": 8,
  "max_per_source_id": null
}
```

- `query` (required): the search string.
- `intent` (optional, default `general`): one of
  `symbol | narrative | trace | general`. Maps to
  `ostk_recall_core::RecallIntent`.
- `limit` (optional): cap on hits before synthesis.
- `max_per_source_id` (optional): cap on hits per source-id after RRF.

**Result:**
```json
{
  "pages": [
    {
      "name": "recall:src:kernel:memory.rs",
      "content": "<JSON-encoded SynthesizedPage>"
    },
    ...
  ]
}
```

The daemon performs embedding, lance search, RRF reranking, and
synthesis (`Synthesizer::collapse`). It does NOT write to disk —
`name`/`content` pairs are returned and the kernel calls
`store_page_owned()` itself, since page-table writes are kernel
territory.

`content` is the JSON serialization of `ostk_recall_core::SynthesizedPage`
(the same shape currently written by `kernel::recall::vm::fault_recall`
post-`Synthesizer::collapse`).

**Errors:**
- `-32602 invalid_params` — missing `query`, unknown `intent`, etc.
- `-32002 recall_failed` — lance query, embedding, or synthesis error.
  The error message string carries detail.

### `ping`

Liveness check. No params, returns `{ "ok": true }`. Used by the kernel
to detect zombie daemons before treating an idle process as healthy.

## Error model

Standard JSON-RPC 2.0 error codes plus daemon-specific codes in the
`-32000` to `-32099` range:

| Code     | Name                  | Meaning                                  |
|----------|-----------------------|------------------------------------------|
| `-32700` | `parse_error`         | invalid JSON                             |
| `-32600` | `invalid_request`     | not a valid request object               |
| `-32601` | `method_not_found`    | unknown method                           |
| `-32602` | `invalid_params`      | bad/missing params                       |
| `-32603` | `internal_error`      | uncategorized server-side error          |
| `-32000` | `corpus_not_found`    | `ostk_dir/recall/` invalid               |
| `-32001` | `embedder_load_failed`| model load failed                        |
| `-32002` | `recall_failed`       | lance query / synthesis runtime error    |
| `-32003` | `not_initialized`     | operational request before `initialize`  |

`error.data` MAY include a structured payload (e.g.
`{ "ostk_dir": "...", "missing": "recall/" }` for `corpus_not_found`).
Clients should treat unknown fields as advisory.

## Notifications

Optional. The relay forwards notifications matching MCP shape to any
attached socket client:

- `notifications/progress` (optional): emitted during long
  `recall.fault` operations.
- `notifications/log` (optional): structured log lines.

Daemons MAY skip both. The kernel renderer treats them as advisory.

## Versioning

The protocol is versioned via the daemon's `initialize` result `version`
field. Haystack v6.x speaks v0.1. Breaking changes increment the major
component of the daemon version; additive changes (new optional params,
new error codes) bump the minor component without breaking older
clients.

The kernel checks the `embedder.dim` field against haystack's
`squasher::embeddings::EMBED_DIM` constant; mismatch is a hard error
(no embedding-dim coercion).

## Why this shape

- **One core method (`recall.fault`)** — haystack's only call site is
  `kernel::recall::vm::fault_recall`. The recall-isa verbs (`recall`,
  `recall_search`, `recall_outline`) operate on the kernel substrate
  (journal, decisions, files), not the lance corpus, so they stay
  in-process and are not part of this protocol.
- **Daemon owns embedding** — haystack does NOT pre-embed and pass
  vectors. The daemon embeds the query string itself (using
  `model2vec-rs` via `ostk-recall-embed`), so the haystack binary
  doesn't need to know about embedding at all for the recall path.
  Haystack's `squasher::embeddings` stays in-process for the squasher
  pipeline (semantic dedup, classifier) — independent concern.
- **Pages returned, not written** — the daemon doesn't touch the kernel
  page table directly. It returns serialized pages; the kernel calls
  `store_page_owned()` to commit them. Keeps the trust boundary clean.
- **JSON-RPC over stdio** — matches the existing `fcp-rust` /
  `fcp-web` / `fcp-olleh` pattern. Reuses `kernel::driver_relay` for
  free.

## References

- →1848 (cut #3 needle)
- decision `cut3_recall_driver_architecture_confirmed_2026_05_09`
- haystack `src/kernel/driver_relay.rs` — concurrent JSON-RPC relay
- haystack `src/kernel/drivers.rs` — driver registry + `DriverMode`
- haystack `src/kernel/recall/vm.rs` — current in-process call site
  (replaced by driver dispatch in cut #3 sub-unit #5)
