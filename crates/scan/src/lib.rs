//! ostk-recall-scan — source scanners.
//!
//! Phase A: stub only.
//! Phase B wires `MarkdownScanner` to `core::Scanner`.
//! Phase C adds the remaining five:
//!
//! * [`claude_code`]  — `~/.claude/projects/*.jsonl` session logs
//! * [`code`]         — source-code trees (line-window chunks)
//! * [`file_glob`]    — generic text files by glob
//! * [`zip_export`]   — Claude.ai `claude-data-export-*.zip`
//! * [`ostk_project`] — composite scanner for haystack-style `.ostk/` roots
//!
//! [`anthropic_session`] is a shared helper used by `claude_code` and the
//! `ostk_project` session sub-scan.

pub mod anthropic_session;
pub mod claude_code;
pub mod code;
pub mod file_glob;
pub mod markdown;
pub mod ostk_project;
pub mod walk;
pub mod zip_export;
