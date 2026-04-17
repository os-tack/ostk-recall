# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial workspace with core, embed, store, scan, pipeline, query, mcp, cli crates.
- Six source scanners: markdown, code, claude_code, file_glob, zip_export, ostk_project (composite).
- LanceDB-backed corpus with Tantivy FTS index.
- DuckDB-backed ingest manifest and audit events store.
- MCP stdio server exposing recall, recall_link, recall_stats, recall_audit tools.
- Hybrid dense + BM25 retrieval with RRF fusion.
