use std::path::PathBuf;

use crate::chunk::Chunk;
use crate::config::SourceConfig;
use crate::error::Result;
use crate::source::SourceKind;

/// A discovered artifact, ready to be parsed into chunks.
#[derive(Debug, Clone, Default)]
pub struct SourceItem {
    pub source_id: String,
    pub path: Option<PathBuf>,
    pub project: Option<String>,
    pub bytes: Option<Vec<u8>>,
    /// Per-source ignore patterns (from [`SourceConfig::ignore`]) carried
    /// through to scanners that do additional walks inside `parse`. Empty
    /// for scanners whose `discover` already finalized the path list.
    pub ignore: Vec<String>,
    /// Physical-identity discriminator, copied from
    /// [`SourceConfig::source_config_id`] by [`Pipeline::ingest_source`]
    /// before this item reaches `parse`. Scanners thread it into emitted
    /// [`Chunk::source_config_id`] for the chunk_id hash + Lance row.
    pub source_config_id: String,
}

/// Scanners are synchronous producers.
///
/// `discover` yields items. `parse` turns one item into zero or more chunks.
/// Scanners do not call the embedder or the store — the pipeline does. This
/// keeps scanners cheap to unit-test with in-memory fixtures.
pub trait Scanner: Send + Sync {
    fn kind(&self) -> SourceKind;

    fn discover<'a>(
        &'a self,
        cfg: &'a SourceConfig,
    ) -> Box<dyn Iterator<Item = Result<SourceItem>> + 'a>;

    fn parse(&self, item: SourceItem) -> Result<Vec<Chunk>>;

    /// Structural parse-logic version (P12). Folded into the Tier-1 freshness
    /// key (`cfg_overlay_hash`) so a change to *structural* parse behavior the
    /// config record-rules can't express — chunk boundaries, wire-format block
    /// handling — forces a one-time re-parse of already-ingested, otherwise
    /// unchanged files. Bump when a scanner's emitted-chunk set changes for
    /// reasons other than the config rule overlay. Defaults to `0`
    /// ("unversioned"); content/semantic apparatus changes ride the
    /// record-rule digest instead.
    fn parse_version(&self) -> u32 {
        0
    }

    /// Path-filtered discovery. Yields only items whose `path` is in
    /// `paths`. Default impl walks the full `discover()` output and
    /// filters — correct but no faster than full scan. Per-scanner
    /// overrides skip the walk and short-circuit to direct path-keyed
    /// discovery for O(|paths|) instead of O(|tree|).
    ///
    /// Substrate for `Pipeline::scan_paths` (gh#4) — the incremental
    /// scan path-aware trigger protocol (EPIC gh#8). Items without a
    /// `path` (e.g. zip-export streamed bytes) are dropped from the
    /// default; per-scanner overrides decide their own membership rule.
    fn discover_paths<'a>(
        &'a self,
        cfg: &'a SourceConfig,
        paths: &'a [PathBuf],
    ) -> Box<dyn Iterator<Item = Result<SourceItem>> + 'a> {
        Box::new(self.discover(cfg).filter(move |item| {
            item.as_ref().map_or(true, |it| {
                it.path
                    .as_deref()
                    .is_some_and(|p| paths.iter().any(|q| q == p || q.starts_with(p)))
            })
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::source::SourceKind;

    /// Scratch scanner whose `discover` yields one directory-level item.
    /// Used to verify the trait default's `starts_with` branch — without
    /// the refinement (gh#10), `discover_paths` would silently drop the
    /// directory item when callers ask about files inside it.
    struct DirYieldingScanner {
        dir: PathBuf,
    }

    impl Scanner for DirYieldingScanner {
        fn kind(&self) -> SourceKind {
            SourceKind::OstkProject
        }

        fn discover<'a>(
            &'a self,
            _cfg: &'a SourceConfig,
        ) -> Box<dyn Iterator<Item = Result<SourceItem>> + 'a> {
            Box::new(std::iter::once(Ok(SourceItem {
                source_id: self.dir.to_string_lossy().into_owned(),
                path: Some(self.dir.clone()),
                source_config_id: "test-cfg".into(),
                ..SourceItem::default()
            })))
        }

        fn parse(&self, _item: SourceItem) -> Result<Vec<Chunk>> {
            Ok(Vec::new())
        }
    }

    #[test]
    fn discover_paths_default_matches_directory_for_inner_files() {
        let dir = PathBuf::from("/tmp/X");
        let scanner = DirYieldingScanner { dir: dir.clone() };
        let cfg = SourceConfig {
            kind: SourceKind::OstkProject,
            project: Some("scratch".into()),
            paths: vec![dir.to_string_lossy().into_owned()],
            ignore: vec![],
            extensions: vec![],
            id: None,
            source_config_id: "test-cfg".to_string(),
            facets: Default::default(),
        };

        let requested = vec![dir.join("a.txt"), dir.join("b.txt")];
        let yielded: Vec<SourceItem> = scanner
            .discover_paths(&cfg, &requested)
            .filter_map(Result::ok)
            .collect();

        assert_eq!(
            yielded.len(),
            1,
            "directory item matches inner-file requests exactly once"
        );
        assert_eq!(yielded[0].path.as_deref(), Some(dir.as_path()));
    }

    #[test]
    fn discover_paths_default_does_not_match_unrelated_paths() {
        let dir = PathBuf::from("/tmp/X");
        let scanner = DirYieldingScanner { dir: dir.clone() };
        let cfg = SourceConfig {
            kind: SourceKind::OstkProject,
            project: Some("scratch".into()),
            paths: vec![dir.to_string_lossy().into_owned()],
            ignore: vec![],
            extensions: vec![],
            id: None,
            source_config_id: "test-cfg".to_string(),
            facets: Default::default(),
        };

        // /tmp/Y/a.txt is unrelated; /tmp/Xtra/a.txt would byte-prefix
        // /tmp/X but `Path::starts_with` is component-wise so it must
        // also be rejected.
        let requested = vec![
            PathBuf::from("/tmp/Y/a.txt"),
            PathBuf::from("/tmp/Xtra/a.txt"),
        ];
        let yielded: Vec<SourceItem> = scanner
            .discover_paths(&cfg, &requested)
            .filter_map(Result::ok)
            .collect();

        assert!(
            yielded.is_empty(),
            "no inner-file matches => no item yielded"
        );
    }
}
