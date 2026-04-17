//! Shared filesystem walker.
//!
//! Honors `.gitignore`, `.ignore`, hidden-file filtering, and a custom
//! `.ostk-recall-ignore` filename. Per-source `ignore` patterns from
//! [`ostk_recall_core::SourceConfig`] are layered on top via
//! [`ignore::overrides::OverrideBuilder`].
//!
//! All filesystem scanners (code, markdown, `file_glob`, `ostk_project`)
//! funnel through [`walk_filtered`] so a single `ignore` field on a
//! source config has uniform effect.
//!
//! ## Why not `walkdir`?
//!
//! `walkdir` walks every entry; respecting `.gitignore` would mean
//! parsing it ourselves at every level. The `ignore` crate (the engine
//! ripgrep uses) bakes that in plus `.ignore`, parent-`.gitignore`s,
//! global git excludes, and hidden-file filtering.
//!
//! ## `OverrideBuilder` semantics
//!
//! `OverrideBuilder` defaults to a *whitelist* once any include pattern
//! is added. We use it purely for excludes: every pattern from the
//! user's `ignore = [...]` list gets a leading `!` so it becomes an
//! exclude rule. No include patterns are ever added, so the whitelist
//! mode never engages — the default `.gitignore` flow keeps working.

use std::path::Path;

use ignore::{DirEntry, WalkBuilder, overrides::OverrideBuilder};

/// Walk `root`, returning files only, with `.gitignore` / `.ignore` /
/// `.ostk-recall-ignore` applied plus the per-source `extra_ignore_patterns`.
///
/// The returned iterator silently drops `Err` entries (typically permission
/// denied on a single subdir); callers that need failure visibility should
/// use [`ignore::WalkBuilder`] directly.
pub fn walk_filtered(
    root: &Path,
    extra_ignore_patterns: &[String],
) -> Box<dyn Iterator<Item = DirEntry>> {
    let mut builder = WalkBuilder::new(root);
    builder
        .standard_filters(true)
        .hidden(true)
        .git_ignore(true)
        .git_global(true)
        .git_exclude(true)
        .add_custom_ignore_filename(".ostk-recall-ignore")
        .follow_links(false);

    if !extra_ignore_patterns.is_empty() {
        let mut overrides = OverrideBuilder::new(root);
        for pat in extra_ignore_patterns {
            // OverrideBuilder treats unprefixed patterns as INCLUDES
            // and `!` prefixes as EXCLUDES. A user-supplied "vendor/**"
            // means "exclude vendor", so prefix with `!` unless the
            // user already did.
            let needs_bang = !pat.starts_with('!');
            let normalized = if needs_bang {
                format!("!{pat}")
            } else {
                pat.clone()
            };
            if let Err(e) = overrides.add(&normalized) {
                tracing::warn!(pattern = %pat, error = %e, "ignore: bad override pattern");
            }
        }
        match overrides.build() {
            Ok(o) => {
                builder.overrides(o);
            }
            Err(e) => {
                tracing::warn!(error = %e, "ignore: failed to build overrides");
            }
        }
    }

    Box::new(
        builder
            .build()
            .filter_map(std::result::Result::ok)
            .filter(|e| e.file_type().is_some_and(|ft| ft.is_file())),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    fn collect_names(root: &Path, patterns: &[String]) -> Vec<String> {
        let mut names: Vec<String> = walk_filtered(root, patterns)
            .map(|e| {
                e.path()
                    .strip_prefix(root)
                    .unwrap_or_else(|_| e.path())
                    .to_string_lossy()
                    .into_owned()
            })
            .collect();
        names.sort();
        names
    }

    #[test]
    fn extra_ignore_excludes_vendor() {
        let tmp = TempDir::new().unwrap();
        fs::create_dir_all(tmp.path().join("vendor")).unwrap();
        fs::create_dir_all(tmp.path().join("src")).unwrap();
        fs::write(tmp.path().join("vendor/skip.rs"), "skip\n").unwrap();
        fs::write(tmp.path().join("src/keep.rs"), "keep\n").unwrap();

        let names = collect_names(tmp.path(), &["vendor/**".to_string()]);
        assert_eq!(names, vec!["src/keep.rs"]);
    }

    #[test]
    fn gitignore_excludes_target() {
        let tmp = TempDir::new().unwrap();
        // Init a real git repo so .gitignore is honored — `ignore` only
        // walks gitignore in directories that look git-tracked.
        fs::create_dir_all(tmp.path().join(".git")).unwrap();
        fs::create_dir_all(tmp.path().join("target")).unwrap();
        fs::create_dir_all(tmp.path().join("src")).unwrap();
        fs::write(tmp.path().join(".gitignore"), "target\n").unwrap();
        fs::write(tmp.path().join("target/build.rs"), "build\n").unwrap();
        fs::write(tmp.path().join("src/lib.rs"), "lib\n").unwrap();

        let names = collect_names(tmp.path(), &[]);
        assert!(
            !names.iter().any(|n| n.starts_with("target/")),
            "target/ should be filtered by .gitignore, got {names:?}"
        );
        assert!(names.contains(&"src/lib.rs".to_string()));
    }

    #[test]
    fn custom_ostk_recall_ignore_applies() {
        let tmp = TempDir::new().unwrap();
        fs::create_dir_all(tmp.path().join("secret")).unwrap();
        fs::create_dir_all(tmp.path().join("src")).unwrap();
        fs::write(tmp.path().join(".ostk-recall-ignore"), "secret/\n").unwrap();
        fs::write(tmp.path().join("secret/x.txt"), "secret\n").unwrap();
        fs::write(tmp.path().join("src/main.rs"), "fn main(){}\n").unwrap();

        let names = collect_names(tmp.path(), &[]);
        assert!(
            !names.iter().any(|n| n.starts_with("secret/")),
            "secret/ should be filtered by .ostk-recall-ignore, got {names:?}"
        );
        assert!(names.iter().any(|n| n.ends_with("main.rs")));
    }

    #[test]
    fn hidden_dot_git_filtered_by_default() {
        let tmp = TempDir::new().unwrap();
        fs::create_dir_all(tmp.path().join(".git/objects")).unwrap();
        fs::write(tmp.path().join(".git/HEAD"), "ref: x\n").unwrap();
        fs::write(tmp.path().join("plain.txt"), "hello\n").unwrap();

        let names = collect_names(tmp.path(), &[]);
        assert!(
            !names.iter().any(|n| n.starts_with(".git")),
            "hidden .git/ should be filtered, got {names:?}"
        );
        assert_eq!(names, vec!["plain.txt"]);
    }
}
