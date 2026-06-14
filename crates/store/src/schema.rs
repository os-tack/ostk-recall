use std::collections::HashMap;
use std::sync::Arc;

use arrow_schema::{DataType, Field, Schema, TimeUnit};

/// Field-metadata key Lance reads to override a column's structural encoding,
/// and the value that selects the per-page "fullzip" layout. See
/// `lance-encoding::constants::{STRUCTURAL_ENCODING_META_KEY,
/// STRUCTURAL_ENCODING_FULLZIP}`.
pub(crate) const LANCE_STRUCTURAL_ENCODING_KEY: &str = "lance-encoding:structural-encoding";
pub(crate) const LANCE_STRUCTURAL_ENCODING_FULLZIP: &str = "fullzip";

/// Corpus columns pinned to Lance's `fullzip` structural encoding instead of
/// the default `miniblock`. Miniblock caps a chunk at 32 KiB, so `text`'s
/// occasional multi-hundred-KiB chunks (large tool outputs in transcripts)
/// trip its `chunk_bytes <= max_chunk_size` assertion during compaction;
/// `fullzip` has no such cap. `facets` is also pinned for safety (it is a flat
/// `Utf8` JSON column — see `corpus_schema` — so miniblock would suffice, but
/// fullzip keeps any future large facet payload off the miniblock path).
/// Pinning here covers table creation and every insert;
/// `CorpusStore::ensure_structural_encodings` stamps it onto corpora that
/// predate the pin.
pub(crate) const FULLZIP_FIELDS: &[&str] = &["text", "facets"];

/// Field metadata that pins a column to the `fullzip` structural encoding.
pub(crate) fn fullzip_metadata() -> HashMap<String, String> {
    HashMap::from([(
        LANCE_STRUCTURAL_ENCODING_KEY.to_string(),
        LANCE_STRUCTURAL_ENCODING_FULLZIP.to_string(),
    )])
}

pub const CORPUS_TABLE: &str = "corpus";

/// Arrow schema for the corpus table. `dim` is the embedding dimensionality;
/// it is fixed at table-create time.
///
/// `source_config_id` (P0, v0.6) is the physical-identity discriminator —
/// nullable so v0.5 corpora can be `add_column`-migrated without rewrite
/// (probe in P2). New rows always populate it.
pub fn corpus_schema(dim: usize) -> Arc<Schema> {
    let tz: Arc<str> = Arc::from("UTC");
    Arc::new(Schema::new(vec![
        Field::new("chunk_id", DataType::Utf8, false),
        Field::new("source", DataType::Utf8, false),
        Field::new("project", DataType::Utf8, true),
        Field::new("source_id", DataType::Utf8, false),
        Field::new("source_config_id", DataType::Utf8, true),
        // P1: per-chunk facets, serialized as `key:value` strings. Sorted
        // for stable round-trip. `array_contains(facets, 'project:auth')`
        // is the single-value filter primitive; OR-of-contains expresses
        // multi-value filters.
        // `facets` is stored as a JSON array string (like `links_json` /
        // `extra_json`), NOT a `List<Utf8>`. lance 7 mis-handles this column as
        // a variable list: most chunks carry an empty/default facet set, and a
        // sparse list's rep/def levels overflow the miniblock encoder's u16
        // buffer-length / 32 KiB chunk asserts during compaction (file format
        // 2.1), while format 2.2 writes it but then can't decode it
        // ("Incorrect array length for StructArray field facets"). A flat Utf8
        // has no rep/def, so it round-trips cleanly under the stable format.
        // Facet *filtering* is done Rust-side on the decoded `FacetSet`
        // (`query::lens::has_denylist_facet`), never via a lance
        // `array_contains`, so the column shape is transparent to queries.
        // fullzip keeps large facet payloads off the miniblock chunk path too.
        Field::new("facets", DataType::Utf8, true).with_metadata(fullzip_metadata()),
        // P1: drives Tier-2 dedupe. Changing an allowlisted facet flips
        // this hash and forces a re-embed; other-facet edits leave it
        // alone.
        Field::new("embedding_input_sha256", DataType::Utf8, true),
        Field::new("chunk_index", DataType::UInt32, false),
        Field::new(
            "ts",
            DataType::Timestamp(TimeUnit::Microsecond, Some(tz)),
            true,
        ),
        Field::new("role", DataType::Utf8, true),
        // `text` can hold very large chunks (multi-hundred-KiB tool outputs in
        // transcripts); pin fullzip so a single oversized value doesn't trip
        // the miniblock 32 KiB chunk assertion during compaction.
        Field::new("text", DataType::Utf8, false).with_metadata(fullzip_metadata()),
        Field::new("sha256", DataType::Utf8, false),
        Field::new("links_json", DataType::Utf8, false),
        Field::new("extra_json", DataType::Utf8, false),
        Field::new("stale", DataType::Boolean, false),
        Field::new(
            "embedding",
            DataType::FixedSizeList(
                Arc::new(Field::new("item", DataType::Float32, true)),
                i32::try_from(dim).expect("dim fits in i32"),
            ),
            false,
        ),
    ]))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn schema_has_expected_fields() {
        let s = corpus_schema(512);
        let names: Vec<&str> = s.fields().iter().map(|f| f.name().as_str()).collect();
        assert!(names.contains(&"chunk_id"));
        assert!(names.contains(&"embedding"));
        assert!(names.contains(&"text"));
        assert!(names.contains(&"links_json"));
        assert!(names.contains(&"stale"));
        assert!(names.contains(&"source_config_id"));
    }

    #[test]
    fn source_config_id_is_nullable() {
        let s = corpus_schema(128);
        let f = s.field_with_name("source_config_id").unwrap();
        assert!(
            f.is_nullable(),
            "source_config_id is nullable for migration"
        );
    }

    #[test]
    fn embedding_field_has_fixed_dim() {
        let s = corpus_schema(256);
        let emb = s.field_with_name("embedding").unwrap();
        match emb.data_type() {
            DataType::FixedSizeList(_, d) => assert_eq!(*d, 256),
            other => panic!("expected FixedSizeList, got {other:?}"),
        }
    }

    /// Regression sentinel for the compaction-overflow fix: every
    /// [`FULLZIP_FIELDS`] column MUST carry the `fullzip` structural-encoding
    /// hint, or Lance's miniblock encoder overflows on this corpus —
    /// `facets`'s sparse-list def-levels blow the u16 buffer-length field, and
    /// an oversized `text` value trips `assert!(chunk_bytes <= max_chunk_size)`.
    /// Either makes `optimize` fail forever. Dropping the metadata silently
    /// reintroduces the bug, so guard it explicitly.
    #[test]
    fn fullzip_fields_force_fullzip_encoding() {
        let s = corpus_schema(64);
        for name in FULLZIP_FIELDS {
            let field = s.field_with_name(name).unwrap();
            assert_eq!(
                field
                    .metadata()
                    .get(LANCE_STRUCTURAL_ENCODING_KEY)
                    .map(String::as_str),
                Some(LANCE_STRUCTURAL_ENCODING_FULLZIP),
                "{name} must pin fullzip encoding to avoid the miniblock overflow",
            );
        }
    }
}
