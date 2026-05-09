# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.1] - 2026-05-08

### Fixed

- **Gemini scanner: complete rewrite for the real on-disk schema.**
  v0.1.0 assumed a `{sessionId, projectHash, messages:[...]}` envelope
  with `content:[{text}]` parts. Real corpora have *two* shapes:
  flat-array `logs.json` files containing user prompts only
  (`{message, messageId, sessionId, timestamp, type:"user"}`), and
  `chats/session-*.json` envelopes whose `content` field is `String`
  for gemini turns and `[{text}]` for user turns — often empty for
  tool-only gemini turns. v0.1.0 produced zero parsed chunks on every
  real file. The rewrite detects shape from the JSON top-level type,
  walks both file-name patterns (previous code only matched
  `session-*.json`), summarizes `toolCalls` so tool-only turns remain
  searchable, and disambiguates `source_id` so two `logs.json` files
  from different projects don't collide. Validated on 108 real files /
  3343 chunks (Scott's machine).

- **Pipeline 0-chunk-per-item bug: dry-runs poisoned the metadata cache.**
  v0.1.0 wrote to `ingest_sources` *before* the dry-run gate, so the
  next scan matched the cached mtime, took the skip path, and emitted
  zero chunks even though the corpus was empty. Fixed two ways:
  dry-runs no longer write source metadata, and the metadata-skip path
  now requires `ingest_chunks` to actually contain rows for the
  source_id (defends against interrupted scans too).

### Added
- Initial workspace with core, embed, store, scan, pipeline, query, mcp, cli crates.
- Six source scanners: markdown, code, claude_code, file_glob, zip_export, ostk_project (composite).
- LanceDB-backed corpus with Tantivy FTS index.
- DuckDB-backed ingest manifest and audit events store.
- MCP stdio server exposing recall, recall_link, recall_stats, recall_audit tools.
- Hybrid dense + BM25 retrieval with RRF fusion.
