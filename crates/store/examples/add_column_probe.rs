//! P2 Probe 3: does Lance 0.29.0 support online `add_column` schema
//! evolution?
//!
//! Creates a table with 3 rows + 2 columns, then attempts to add a
//! `facets: List<Utf8>` column. Asserts the column exists post-add and
//! that existing rows have NULL values for it. Exit 0 = supported →
//! P10 migration can evolve in-place; exit 1 = unsupported → P10 must
//! wipe-and-rescan.

use std::sync::Arc;

use arrow_array::{RecordBatch, RecordBatchIterator, RecordBatchReader, StringArray, UInt32Array};
use arrow_schema::{DataType, Field, Schema};
use futures::TryStreamExt;
use lancedb::query::ExecutableQuery;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let tmp = tempfile::tempdir()?;
    let uri = tmp.path().join("probe.lance");
    let conn = lancedb::connect(uri.to_string_lossy().as_ref())
        .execute()
        .await?;

    let initial_schema = Arc::new(Schema::new(vec![
        Field::new("chunk_id", DataType::Utf8, false),
        Field::new("chunk_index", DataType::UInt32, false),
    ]));
    let chunk_ids = StringArray::from(vec!["a", "b", "c"]);
    let chunk_index = UInt32Array::from(vec![0u32, 1, 2]);
    let batch = RecordBatch::try_new(
        initial_schema.clone(),
        vec![Arc::new(chunk_ids), Arc::new(chunk_index)],
    )?;
    let reader: Box<dyn RecordBatchReader + Send> = Box::new(RecordBatchIterator::new(
        std::iter::once(Ok::<_, arrow::error::ArrowError>(batch)),
        initial_schema,
    ));
    let table = conn.create_table("probe", reader).execute().await?;
    println!("initial row count: {}", table.count_rows(None).await?);

    // The probe: try a Lance NewColumnTransform::AllNulls add for a
    // List<Utf8> field. (lancedb 0.29.0 exposes `Table::add_columns`
    // taking a `NewColumnTransform`.)
    use lancedb::table::NewColumnTransform;
    let new_field = Field::new(
        "facets",
        DataType::List(Arc::new(Field::new("item", DataType::Utf8, true))),
        true,
    );
    let new_schema = Arc::new(Schema::new(vec![new_field]));
    match table
        .add_columns(NewColumnTransform::AllNulls(new_schema), None)
        .await
    {
        Ok(_) => {
            println!("add_columns(AllNulls List<Utf8>) returned Ok");
        }
        Err(e) => {
            eprintln!("FAIL: add_columns returned err: {e}");
            std::process::exit(1);
        }
    }

    // Re-open table; verify the column is now in the schema, and that
    // existing rows have NULL facets.
    let reopened = conn.open_table("probe").execute().await?;
    let live_schema = reopened.schema().await?;
    let has_facets = live_schema.fields().iter().any(|f| f.name() == "facets");
    if !has_facets {
        eprintln!("FAIL: `facets` column not in schema after add_columns");
        std::process::exit(1);
    }
    println!("PASS: facets column present in schema after add_columns");

    // Spot-check NULL fill: stream all rows and count NULLs in facets.
    let stream = reopened.query().execute().await?;
    let batches: Vec<RecordBatch> = stream.try_collect().await?;
    let mut total_rows = 0usize;
    let mut null_facets = 0usize;
    for batch in &batches {
        total_rows += batch.num_rows();
        if let Some(col) = batch.column_by_name("facets") {
            for i in 0..col.len() {
                if col.is_null(i) {
                    null_facets += 1;
                }
            }
        }
    }
    if total_rows == 3 && null_facets == 3 {
        println!("PASS: all 3 existing rows have NULL facets (online add_column works)");
        println!("RESULT: Lance 0.29.0 supports add_columns(AllNulls); P10 can evolve in-place");
        return Ok(());
    }
    eprintln!("FAIL: expected 3 rows / 3 nulls, got {total_rows} rows / {null_facets} nulls");
    std::process::exit(1);
}
