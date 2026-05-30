//! Record rules: config-driven interpretation overlay (P12).
//!
//! Apparatus filtering used to be hand-written functions chained inside each
//! scanner's `parse()` (`drop_system_reminders`, `tag_harness_orchestration`,
//! …). That had two problems: the patterns were Claude-Code-specific string
//! literals compiled into the binary, and — because the Tier-1 dedup key
//! (`cfg_overlay_hash`) tracks only operator facet overrides — a change to that
//! parser logic never re-applied to already-ingested, unchanged files (the
//! RT-7-3b bug class). See
//! `.ostk/threads/cognitive-memory/p12-interpretation-overlay.md`.
//!
//! Record rules lift that interpretation into a declarative, operator-owned
//! `[[record_rule]]` config section applied once in the pipeline overlay stage.
//! Each rule is `pattern → action` where action ∈ {drop, tag(record_kind)}.
//! The ruleset's per-source-kind digest folds into the Tier-1 freshness key, so
//! editing a rule self-propagates (the config content *is* the version).
//!
//! Apply semantics (see [`CompiledRecordRules::decide`]): **any matching `Drop`
//! short-circuits and drops the chunk regardless of position; otherwise the
//! first matching `Tag` sets `record_kind`.** `record_kind` is
//! `Cardinality::Single`, so tags do not accumulate — first-tag-wins is the
//! deterministic policy.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::error::{Error, Result};
use crate::source::{Source, SourceKind};

/// One operator-authored record rule (TOML `[[record_rule]]`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RecordRule {
    pub r#match: RuleMatch,
    pub action: RuleAction,
}

/// Match predicates. All present predicates must match (AND). Text predicates
/// are checked against `text.trim_start()` to preserve the behavior of the
/// hand-written `drop_*` filters (which trimmed before prefix-checking).
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RuleMatch {
    /// `text.trim_start().starts_with(prefix)`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prefix: Option<String>,
    /// `text.trim_start().contains(contains)`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub contains: Option<String>,
    /// Regex tested against `text.trim_start()`. Compiled once at build.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub regex: Option<String>,
    /// Scanner kinds this rule applies to (`SourceKind::as_str` values). A list
    /// so one rule can cover several kinds. `None` = any kind. `Some(empty)` is
    /// rejected at build (ambiguous).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_kind: Option<Vec<String>>,
    /// Concrete chunk source (`Source::as_str` value), e.g. `ostk_session`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source: Option<String>,
    /// Chunk role, e.g. `user` / `assistant`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
}

/// What a matching rule does. First cut: `record_kind`-only tagging, because
/// `record_kind` is in `EMBED_FACET_ALLOWLIST` so the change persists via the
/// Tier-2 re-embed path. A generic facet-tag action would need a metadata-only
/// corpus update path that does not exist yet.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum RuleAction {
    /// Drop the chunk (skip embedding; purge any previously-ingested copy).
    Drop,
    /// Stamp `record_kind = <value>` on the chunk's facets.
    Tag { record_kind: String },
}

/// The default ruleset, used when `[[record_rule]]` is absent from config.
/// Reproduces the pre-P12 hardcoded behavior, **source-scoped** to exactly the
/// scanners that applied each filter (applying them to all sources would drop
/// legitimate markdown/code starting with `<command-name>` etc.).
#[must_use]
pub fn default_record_rules() -> Vec<RecordRule> {
    // The five command-wrapper prefixes the old `drop_local_command_wrappers`
    // matched, as one anchored alternation.
    let command_wrapper_regex =
        r"^(<local-command-|<command-name>|</command-name>|<command-message>|<command-args>)"
            .to_string();
    let claude = || Some(vec!["claude_code".to_string()]);
    vec![
        // RT-4: pure `<system-reminder>` apparatus (claude_code only — the
        // ostk_project session sub-scan did not drop these).
        RecordRule {
            r#match: RuleMatch {
                prefix: Some("<system-reminder>".to_string()),
                source_kind: claude(),
                ..Default::default()
            },
            action: RuleAction::Drop,
        },
        // Local-command / slash-command wrappers (claude_code).
        RecordRule {
            r#match: RuleMatch {
                regex: Some(command_wrapper_regex.clone()),
                source_kind: claude(),
                ..Default::default()
            },
            action: RuleAction::Drop,
        },
        // Same wrappers, but only on ostk_project *session* chunks
        // (Source::OstkSession) — not decisions / needles / audit / memory.
        RecordRule {
            r#match: RuleMatch {
                regex: Some(command_wrapper_regex),
                source_kind: Some(vec!["ostk_project".to_string()]),
                source: Some("ostk_session".to_string()),
                ..Default::default()
            },
            action: RuleAction::Drop,
        },
        // RT-7: `<teammate-message>` multi-agent orchestration envelopes — tag,
        // don't drop (they carry task-description history); attenuated from the
        // lens via the denylist (claude_code only).
        RecordRule {
            r#match: RuleMatch {
                prefix: Some("<teammate-message".to_string()),
                source_kind: claude(),
                ..Default::default()
            },
            action: RuleAction::Tag {
                record_kind: "harness_orchestration".to_string(),
            },
        },
    ]
}

/// The decision a compiled ruleset reaches for a single chunk.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RuleDecision {
    /// No rule matched — keep the chunk unchanged.
    Keep,
    /// Drop the chunk (a `Drop` rule matched).
    Drop,
    /// Keep the chunk but stamp `record_kind = <value>` (first matching tag).
    Tag(String),
}

/// A validated, compiled rule (regex compiled, enum strings parsed to typed
/// `SourceKind` / `Source` so there is no string drift at match time).
#[derive(Debug)]
struct CompiledRule {
    prefix: Option<String>,
    contains: Option<String>,
    regex: Option<regex::Regex>,
    source_kinds: Option<Vec<SourceKind>>,
    source: Option<Source>,
    role: Option<String>,
    action: RuleAction,
}

impl CompiledRule {
    fn matches(&self, trimmed: &str, source: Source, role: Option<&str>, kind: SourceKind) -> bool {
        if let Some(p) = &self.prefix {
            if !trimmed.starts_with(p.as_str()) {
                return false;
            }
        }
        if let Some(c) = &self.contains {
            if !trimmed.contains(c.as_str()) {
                return false;
            }
        }
        if let Some(rx) = &self.regex {
            if !rx.is_match(trimmed) {
                return false;
            }
        }
        if let Some(ks) = &self.source_kinds {
            if !ks.contains(&kind) {
                return false;
            }
        }
        if let Some(s) = &self.source {
            if *s != source {
                return false;
            }
        }
        if let Some(r) = &self.role {
            if role != Some(r.as_str()) {
                return false;
            }
        }
        true
    }
}

/// A validated, compiled ruleset. Built once from the effective config rules
/// and shared (the pipeline holds it behind an `Arc`). Holds the raw rules too,
/// for the per-source-kind Tier-1 digest.
#[derive(Debug)]
pub struct CompiledRecordRules {
    compiled: Vec<CompiledRule>,
    raw: Vec<RecordRule>,
}

impl CompiledRecordRules {
    /// An empty ruleset — never matches, digest is constant. For test
    /// ergonomics and any caller that intentionally runs without rules.
    #[must_use]
    pub const fn empty() -> Self {
        Self {
            compiled: Vec::new(),
            raw: Vec::new(),
        }
    }

    /// Validate + compile a ruleset. Errors (config-level) on: unknown
    /// `source_kind` / `source` values, empty `source_kind = []`, an invalid
    /// regex, an empty `record_kind`, or a rule with **no predicates at all**
    /// (a match-everything rule is almost always an authoring mistake).
    pub fn build(rules: &[RecordRule]) -> Result<Self> {
        let mut compiled = Vec::with_capacity(rules.len());
        for (i, r) in rules.iter().enumerate() {
            let m = &r.r#match;

            let has_text = m.prefix.is_some() || m.contains.is_some() || m.regex.is_some();
            let has_struct = m.source_kind.is_some() || m.source.is_some() || m.role.is_some();
            if !has_text && !has_struct {
                return Err(Error::Config(format!(
                    "record_rule[{i}] has no predicates; a match-everything rule is \
                     rejected — add at least one of prefix/contains/regex/source_kind/source/role"
                )));
            }

            let source_kinds = match &m.source_kind {
                None => None,
                Some(v) => {
                    if v.is_empty() {
                        return Err(Error::Config(format!(
                            "record_rule[{i}] has empty `source_kind = []` (ambiguous; omit it for any-kind)"
                        )));
                    }
                    let mut ks = Vec::with_capacity(v.len());
                    for s in v {
                        let k = SourceKind::parse_str(s).ok_or_else(|| {
                            Error::Config(format!("record_rule[{i}] unknown source_kind {s:?}"))
                        })?;
                        ks.push(k);
                    }
                    Some(ks)
                }
            };

            let source = match &m.source {
                None => None,
                Some(s) => Some(Source::parse_str(s).ok_or_else(|| {
                    Error::Config(format!("record_rule[{i}] unknown source {s:?}"))
                })?),
            };

            let regex = match &m.regex {
                None => None,
                Some(rx) => Some(regex::Regex::new(rx).map_err(|e| {
                    Error::Config(format!("record_rule[{i}] invalid regex {rx:?}: {e}"))
                })?),
            };

            if let RuleAction::Tag { record_kind } = &r.action {
                if record_kind.is_empty() {
                    return Err(Error::Config(format!(
                        "record_rule[{i}] tag has empty `record_kind`"
                    )));
                }
            }

            compiled.push(CompiledRule {
                prefix: m.prefix.clone(),
                contains: m.contains.clone(),
                regex,
                source_kinds,
                source,
                role: m.role.clone(),
                action: r.action.clone(),
            });
        }
        Ok(Self {
            compiled,
            raw: rules.to_vec(),
        })
    }

    /// Decide what to do with a chunk. Drop wins over any tag regardless of
    /// rule order; otherwise the first matching tag sets `record_kind`.
    #[must_use]
    pub fn decide(
        &self,
        text: &str,
        source: Source,
        role: Option<&str>,
        kind: SourceKind,
    ) -> RuleDecision {
        let trimmed = text.trim_start();
        let mut first_tag: Option<String> = None;
        for rule in &self.compiled {
            if !rule.matches(trimmed, source, role, kind) {
                continue;
            }
            match &rule.action {
                // Drop short-circuits: a drop anywhere wins over earlier tags.
                RuleAction::Drop => return RuleDecision::Drop,
                RuleAction::Tag { record_kind } => {
                    if first_tag.is_none() {
                        first_tag = Some(record_kind.clone());
                    }
                }
            }
        }
        first_tag.map_or(RuleDecision::Keep, RuleDecision::Tag)
    }

    /// Stable digest of the rules that can match `kind` (scoped to it or
    /// unscoped). Folded into the per-source Tier-1 freshness key so editing a
    /// rule for one source_kind re-parses only that kind's sources, not the
    /// whole corpus.
    ///
    /// **Returns an empty string when no rule applies to `kind`.** The empty
    /// case is load-bearing: the pipeline maps it (with `parse_version == 0`)
    /// to an empty Tier-1 `extra_digest`, which reproduces the pre-P12 hash
    /// byte-for-byte so unaffected sources (markdown/code/…) keep Tier-1
    /// skipping instead of re-parsing the whole corpus.
    #[must_use]
    pub fn digest_for(&self, kind: SourceKind) -> String {
        let applicable: Vec<&RecordRule> = self
            .raw
            .iter()
            .filter(|r| match &r.r#match.source_kind {
                None => true,
                Some(v) => v.iter().any(|s| SourceKind::parse_str(s) == Some(kind)),
            })
            .collect();
        if applicable.is_empty() {
            return String::new();
        }
        let mut h = Sha256::new();
        h.update(b"record_rules_v1");
        for r in applicable {
            hash_rule(&mut h, r);
        }
        hex::encode(&h.finalize()[..16])
    }
}

/// Mix one rule's fields into the digest hasher deterministically.
fn hash_rule(h: &mut Sha256, r: &RecordRule) {
    let m = &r.r#match;
    h.update(b"|p:");
    h.update(m.prefix.as_deref().unwrap_or("").as_bytes());
    h.update(b"|c:");
    h.update(m.contains.as_deref().unwrap_or("").as_bytes());
    h.update(b"|rx:");
    h.update(m.regex.as_deref().unwrap_or("").as_bytes());
    h.update(b"|sk:");
    if let Some(v) = &m.source_kind {
        for s in v {
            h.update(s.as_bytes());
            h.update(b",");
        }
    }
    h.update(b"|src:");
    h.update(m.source.as_deref().unwrap_or("").as_bytes());
    h.update(b"|role:");
    h.update(m.role.as_deref().unwrap_or("").as_bytes());
    match &r.action {
        RuleAction::Drop => h.update(b"|a:drop"),
        RuleAction::Tag { record_kind } => {
            h.update(b"|a:tag:");
            h.update(record_kind.as_bytes());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn build_default() -> CompiledRecordRules {
        CompiledRecordRules::build(&default_record_rules()).expect("defaults compile")
    }

    #[test]
    fn defaults_drop_system_reminder_only_for_claude_code() {
        let rules = build_default();
        // claude_code system-reminder → drop.
        assert_eq!(
            rules.decide(
                "<system-reminder>do X</system-reminder>",
                Source::ClaudeCode,
                None,
                SourceKind::ClaudeCode
            ),
            RuleDecision::Drop
        );
        // Same text from a markdown source → kept (source-scoping; review finding 3).
        assert_eq!(
            rules.decide(
                "<system-reminder>do X</system-reminder>",
                Source::Markdown,
                None,
                SourceKind::Markdown
            ),
            RuleDecision::Keep
        );
    }

    #[test]
    fn defaults_tag_teammate_message() {
        let rules = build_default();
        assert_eq!(
            rules.decide(
                "<teammate-message from=lead>hi</teammate-message>",
                Source::ClaudeCode,
                None,
                SourceKind::ClaudeCode
            ),
            RuleDecision::Tag("harness_orchestration".to_string())
        );
    }

    #[test]
    fn defaults_command_wrappers_drop_for_claude_and_ostk_session_only() {
        let rules = build_default();
        for prefix in [
            "<local-command-stdout>x",
            "<command-name>foo",
            "</command-name>",
            "<command-message>m",
            "<command-args>a",
        ] {
            assert_eq!(
                rules.decide(prefix, Source::ClaudeCode, None, SourceKind::ClaudeCode),
                RuleDecision::Drop,
                "claude_code should drop {prefix:?}"
            );
            assert_eq!(
                rules.decide(prefix, Source::OstkSession, None, SourceKind::OstkProject),
                RuleDecision::Drop,
                "ostk_session should drop {prefix:?}"
            );
            // ostk_project NON-session chunk (e.g. a decision) is NOT dropped.
            assert_eq!(
                rules.decide(prefix, Source::OstkDecision, None, SourceKind::OstkProject),
                RuleDecision::Keep,
                "ostk decision should keep {prefix:?}"
            );
        }
    }

    #[test]
    fn leading_whitespace_is_trimmed_before_prefix_match() {
        let rules = build_default();
        assert_eq!(
            rules.decide(
                "\n  <system-reminder>x",
                Source::ClaudeCode,
                None,
                SourceKind::ClaudeCode
            ),
            RuleDecision::Drop
        );
    }

    #[test]
    fn drop_wins_over_tag_regardless_of_order() {
        // Tag rule first, drop rule second; drop must still win.
        let rules = CompiledRecordRules::build(&[
            RecordRule {
                r#match: RuleMatch {
                    contains: Some("KEEPME".to_string()),
                    ..Default::default()
                },
                action: RuleAction::Tag {
                    record_kind: "x".to_string(),
                },
            },
            RecordRule {
                r#match: RuleMatch {
                    contains: Some("DROPME".to_string()),
                    ..Default::default()
                },
                action: RuleAction::Drop,
            },
        ])
        .unwrap();
        assert_eq!(
            rules.decide("KEEPME DROPME", Source::Code, None, SourceKind::Code),
            RuleDecision::Drop
        );
    }

    #[test]
    fn first_matching_tag_wins() {
        let rules = CompiledRecordRules::build(&[
            RecordRule {
                r#match: RuleMatch {
                    contains: Some("a".to_string()),
                    ..Default::default()
                },
                action: RuleAction::Tag {
                    record_kind: "first".to_string(),
                },
            },
            RecordRule {
                r#match: RuleMatch {
                    contains: Some("a".to_string()),
                    ..Default::default()
                },
                action: RuleAction::Tag {
                    record_kind: "second".to_string(),
                },
            },
        ])
        .unwrap();
        assert_eq!(
            rules.decide("a", Source::Code, None, SourceKind::Code),
            RuleDecision::Tag("first".to_string())
        );
    }

    #[test]
    fn build_rejects_no_predicate_rule() {
        let err = CompiledRecordRules::build(&[RecordRule {
            r#match: RuleMatch::default(),
            action: RuleAction::Drop,
        }]);
        assert!(err.is_err(), "match-everything rule must be rejected");
    }

    #[test]
    fn build_rejects_empty_source_kind() {
        let err = CompiledRecordRules::build(&[RecordRule {
            r#match: RuleMatch {
                prefix: Some("x".to_string()),
                source_kind: Some(vec![]),
                ..Default::default()
            },
            action: RuleAction::Drop,
        }]);
        assert!(err.is_err(), "empty source_kind = [] must be rejected");
    }

    #[test]
    fn build_rejects_unknown_source_kind_and_source() {
        assert!(CompiledRecordRules::build(&[RecordRule {
            r#match: RuleMatch {
                prefix: Some("x".to_string()),
                source_kind: Some(vec!["not_a_kind".to_string()]),
                ..Default::default()
            },
            action: RuleAction::Drop,
        }])
        .is_err());
        assert!(CompiledRecordRules::build(&[RecordRule {
            r#match: RuleMatch {
                source: Some("not_a_source".to_string()),
                ..Default::default()
            },
            action: RuleAction::Drop,
        }])
        .is_err());
    }

    #[test]
    fn build_rejects_invalid_regex() {
        let err = CompiledRecordRules::build(&[RecordRule {
            r#match: RuleMatch {
                regex: Some("(unclosed".to_string()),
                ..Default::default()
            },
            action: RuleAction::Drop,
        }]);
        assert!(err.is_err());
    }

    #[test]
    fn digest_for_is_empty_when_no_rule_applies() {
        // Load-bearing for the Tier-1 hash: an empty digest (no applicable
        // rule) + parse_version 0 reproduces the pre-P12 hash, so unaffected
        // sources keep Tier-1 skipping instead of re-parsing the whole corpus.
        let rules = build_default();
        assert_eq!(rules.digest_for(SourceKind::Markdown), "");
        assert_eq!(rules.digest_for(SourceKind::Code), "");
        assert_eq!(rules.digest_for(SourceKind::Gemini), "");
        assert!(!rules.digest_for(SourceKind::ClaudeCode).is_empty());
        assert!(!rules.digest_for(SourceKind::OstkProject).is_empty());
    }

    #[test]
    fn digest_changes_when_a_matching_rule_changes_but_is_scoped_per_kind() {
        let base = build_default();
        // Add a markdown-scoped rule; claude_code digest must NOT change,
        // markdown digest MUST change.
        let mut extended = default_record_rules();
        extended.push(RecordRule {
            r#match: RuleMatch {
                prefix: Some("<!--".to_string()),
                source_kind: Some(vec!["markdown".to_string()]),
                ..Default::default()
            },
            action: RuleAction::Drop,
        });
        let extended = CompiledRecordRules::build(&extended).unwrap();

        assert_eq!(
            base.digest_for(SourceKind::ClaudeCode),
            extended.digest_for(SourceKind::ClaudeCode),
            "claude_code digest must be unaffected by a markdown-scoped rule"
        );
        assert_ne!(
            base.digest_for(SourceKind::Markdown),
            extended.digest_for(SourceKind::Markdown),
            "markdown digest must change when a markdown rule is added"
        );
    }

    #[test]
    fn toml_round_trip() {
        let rule = RecordRule {
            r#match: RuleMatch {
                prefix: Some("<teammate-message".to_string()),
                source_kind: Some(vec!["claude_code".to_string()]),
                ..Default::default()
            },
            action: RuleAction::Tag {
                record_kind: "harness_orchestration".to_string(),
            },
        };
        // Wrap in a table so toml has a top-level map.
        #[derive(Serialize, Deserialize, PartialEq, Eq, Debug)]
        struct Wrap {
            record_rule: Vec<RecordRule>,
        }
        let w = Wrap {
            record_rule: vec![rule],
        };
        let s = toml::to_string(&w).unwrap();
        let back: Wrap = toml::from_str(&s).unwrap();
        assert_eq!(w, back);
    }
}
