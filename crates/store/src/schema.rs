use std::sync::Arc;

use arrow_schema::{DataType, Field, Schema, TimeUnit};

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
        Field::new(
            "facets",
            DataType::List(Arc::new(Field::new("item", DataType::Utf8, true))),
            true,
        ),
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
        Field::new("text", DataType::Utf8, false),
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
        assert!(f.is_nullable(), "source_config_id is nullable for migration");
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
}
