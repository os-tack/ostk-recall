use serde::{Deserialize, Serialize};

/// Kind of source declared in the user's config.toml.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SourceKind {
    Markdown,
    Code,
    ClaudeCode,
    OstkProject,
    FileGlob,
    ZipExport,
}

impl SourceKind {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Markdown => "markdown",
            Self::Code => "code",
            Self::ClaudeCode => "claude_code",
            Self::OstkProject => "ostk_project",
            Self::FileGlob => "file_glob",
            Self::ZipExport => "zip_export",
        }
    }
}

/// Concrete source of a chunk, as stored on a row. Granular subtypes for
/// `ostk_project` (which expands into decisions, needles, audit, ...) are
/// distinguished at row time.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Source {
    Markdown,
    Code,
    ClaudeCode,
    OstkDecision,
    OstkNeedle,
    OstkAuditSignificant,
    OstkConversation,
    OstkSession,
    OstkMemory,
    OstkSpec,
    FileGlob,
    ZipExport,
}

impl Source {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Markdown => "markdown",
            Self::Code => "code",
            Self::ClaudeCode => "claude_code",
            Self::OstkDecision => "ostk_decision",
            Self::OstkNeedle => "ostk_needle",
            Self::OstkAuditSignificant => "ostk_audit_significant",
            Self::OstkConversation => "ostk_conversation",
            Self::OstkSession => "ostk_session",
            Self::OstkMemory => "ostk_memory",
            Self::OstkSpec => "ostk_spec",
            Self::FileGlob => "file_glob",
            Self::ZipExport => "zip_export",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn source_kind_round_trips_through_toml() {
        // TOML requires a table at the top level, so wrap the enum in a
        // single-field struct for the round-trip.
        #[derive(serde::Serialize, serde::Deserialize, PartialEq, Eq, Debug)]
        struct Wrap {
            kind: SourceKind,
        }
        for kind in [
            SourceKind::Markdown,
            SourceKind::Code,
            SourceKind::ClaudeCode,
            SourceKind::OstkProject,
            SourceKind::FileGlob,
            SourceKind::ZipExport,
        ] {
            let w = Wrap { kind };
            let s = toml::to_string(&w).unwrap();
            let parsed: Wrap = toml::from_str(&s).unwrap();
            assert_eq!(w, parsed);
        }
    }

    #[test]
    fn source_str_matches_serde() {
        for source in [
            Source::Markdown,
            Source::OstkDecision,
            Source::OstkAuditSignificant,
        ] {
            let json = serde_json::to_string(&source).unwrap();
            assert!(json.contains(source.as_str()));
        }
    }
}
