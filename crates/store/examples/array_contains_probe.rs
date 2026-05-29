//! P2 Probe 1: does Lance 0.29.0 support `array_contains` on a
//! `List<Utf8>` column?
//!
//! Writes a tiny corpus with a `facets: List<Utf8>` column matching
//! P1's serialization (`key:value` strings), then issues
//! `only_if("array_contains(facets, 'project:auth')")` and asserts the
//! filtered row count.
//!
//! Exit code 0 = `array_contains` works → P1/P3 facet filtering uses
//! it natively. Exit code 1 = it doesn't → fall back to post-filter.

use std::sync::Arc;

use arrow_array::builder::{ListBuilder, StringBuilder};
use arrow_array::{RecordBatch, RecordBatchIterator, RecordBatchReader, StringArray};
use arrow_schema::{DataType, Field, Schema};
use futures::TryStreamExt;
use lancedb::query::{ExecutableQuery, QueryBase};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let tmp = tempfile::tempdir()?;
    let uri = tmp.path().join("probe.lance");
    let conn = lancedb::connect(uri.to_string_lossy().as_ref())
        .execute()
        .await?;

    let schema = Arc::new(Schema::new(vec![
        Field::new("chunk_id", DataType::Utf8, false),
        Field::new(
            "facets",
            DataType::List(Arc::new(Field::new("item", DataType::Utf8, true))),
            true,
        ),
    ]));

    // Three rows:
    //   r0: facets = ["project:auth", "lang:rust"]
    //   r1: facets = ["project:billing", "lang:rust"]
    //   r2: facets = ["lang:python"]
    let chunk_ids = StringArray::from(vec!["r0", "r1", "r2"]);
    let mut facets_builder = ListBuilder::new(StringBuilder::new());
    for row in [
        vec!["project:auth", "lang:rust"],
        vec!["project:billing", "lang:rust"],
        vec!["lang:python"],
    ] {
        for v in row {
            facets_builder.values().append_value(v);
        }
        facets_builder.append(true);
    }
    let facets_col = facets_builder.finish();
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![Arc::new(chunk_ids), Arc::new(facets_col)],
    )?;

    let reader: Box<dyn RecordBatchReader + Send> = Box::new(RecordBatchIterator::new(
        std::iter::once(Ok::<_, arrow::error::ArrowError>(batch)),
        schema,
    ));
    let table = conn.create_table("probe", reader).execute().await?;

    // The probe question.
    let stream = table
        .query()
        .only_if("array_contains(facets, 'project:auth')")
        .execute()
        .await?;
    let batches: Vec<RecordBatch> = stream.try_collect().await?;
    let rows: usize = batches.iter().map(|b| b.num_rows()).sum();

    if rows == 1 {
        println!("PASS: array_contains returned exactly 1 row (project:auth)");
        // Quick sanity: ANOTHER filter that should match 2 rows.
        let stream2 = table
            .query()
            .only_if("array_contains(facets, 'lang:rust')")
            .execute()
            .await?;
        let batches2: Vec<RecordBatch> = stream2.try_collect().await?;
        let rows2: usize = batches2.iter().map(|b| b.num_rows()).sum();
        if rows2 == 2 {
            println!("PASS: array_contains(facets, 'lang:rust') returned 2 rows");
            println!("RESULT: array_contains works in Lance 0.29.0");
            return Ok(());
        }
        eprintln!("FAIL: secondary check expected 2, got {rows2}");
        std::process::exit(1);
    }
    eprintln!("FAIL: array_contains expected 1 row, got {rows}");
    std::process::exit(1);
}
