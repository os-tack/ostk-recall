use std::sync::Arc;

use arrow_schema::{DataType, Field, Schema, TimeUnit};

pub const CORPUS_TABLE: &str = "corpus";

/// Arrow schema for the corpus table. `dim` is the embedding dimensionality;
/// it is fixed at table-create time.
pub fn corpus_schema(dim: usize) -> Arc<Schema> {
    let tz: Arc<str> = Arc::from("UTC");
    Arc::new(Schema::new(vec![
        Field::new("chunk_id", DataType::Utf8, false),
        Field::new("source", DataType::Utf8, false),
        Field::new("project", DataType::Utf8, true),
        Field::new("source_id", DataType::Utf8, false),
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
