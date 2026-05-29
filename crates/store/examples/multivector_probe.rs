//! P2 Probe 2 (rerank-only): does Lance 0.29.0 support a
//! `List<FixedSizeList<F32, D>>` side-table column, bulk-fetched via
//! `chunk_id IN (...)` in a single query (NOT N queries), and decoded
//! into `Vec<Vec<f32>>` per chunk for in-memory MaxSim?
//!
//! This is the design P4 commits to: MaxSim is a rerank feature only;
//! a side table holds per-chunk token-vector lists; one bulk fetch
//! decodes into a `HashMap<chunk_id, Vec<Vec<f32>>>`; the feature's
//! prepare() runs the bulk fetch once per `RankEngine::rank` call.
//!
//! Exit 0 = side-table + bulk-fetch + decode all work and MaxSim
//! against a hand-computed reference matches; exit 1 = at least one
//! step fails (P4 must drop the MaxSim feature or take a fallback).

use std::collections::HashMap;
use std::sync::Arc;

use arrow_array::builder::{FixedSizeListBuilder, Float32Builder, ListBuilder};
use arrow_array::{
    Array, FixedSizeListArray, ListArray, RecordBatch, RecordBatchIterator, RecordBatchReader,
    StringArray,
};
use arrow_schema::{DataType, Field, Schema};
use futures::TryStreamExt;
use lancedb::query::{ExecutableQuery, QueryBase, Select};

const DIM: usize = 8;
const N_CHUNKS: usize = 1_000;
const SAMPLE: usize = 100;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let tmp = tempfile::tempdir()?;
    let uri = tmp.path().join("multivec.lance");
    let conn = lancedb::connect(uri.to_string_lossy().as_ref())
        .execute()
        .await?;

    // Schema: side table keyed by chunk_id, with a
    // List<FixedSizeList<F32, DIM>> token-vector column. Each row holds
    // a variable number of token vectors of dimension DIM.
    let schema = Arc::new(Schema::new(vec![
        Field::new("chunk_id", DataType::Utf8, false),
        Field::new(
            "tokens",
            DataType::List(Arc::new(Field::new(
                "item",
                DataType::FixedSizeList(
                    Arc::new(Field::new("item", DataType::Float32, true)),
                    DIM as i32,
                ),
                true,
            ))),
            false,
        ),
    ]));

    // Build N_CHUNKS rows; each row has 3..=7 token vectors.
    let chunk_ids: Vec<String> = (0..N_CHUNKS).map(|i| format!("c{i:04}")).collect();
    let chunk_id_col = StringArray::from(chunk_ids.iter().map(String::as_str).collect::<Vec<_>>());

    let inner_value_builder = FixedSizeListBuilder::new(Float32Builder::new(), DIM as i32);
    let mut tokens_builder = ListBuilder::new(inner_value_builder);
    for i in 0..N_CHUNKS {
        let n_tokens = 3 + (i % 5); // 3..=7
        for t in 0..n_tokens {
            let fsl = tokens_builder.values();
            for d in 0..DIM {
                // Deterministic synthetic embeddings — orthogonal-ish so
                // MaxSim is non-trivial.
                let v = ((i + t * 31 + d * 7) % 13) as f32 / 13.0;
                fsl.values().append_value(v);
            }
            fsl.append(true);
        }
        tokens_builder.append(true);
    }
    let tokens_col = tokens_builder.finish();
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![Arc::new(chunk_id_col), Arc::new(tokens_col)],
    )?;
    let reader: Box<dyn RecordBatchReader + Send> = Box::new(RecordBatchIterator::new(
        std::iter::once(Ok::<_, arrow::error::ArrowError>(batch)),
        schema,
    ));
    let table = conn.create_table("multivec", reader).execute().await?;
    println!("inserted {} rows", table.count_rows(None).await?);

    // Single bulk-fetch via chunk_id IN (...). The probe's whole point is
    // that this is ONE Lance call, not 100.
    let sample_ids: Vec<&str> = chunk_ids
        .iter()
        .step_by(N_CHUNKS / SAMPLE)
        .take(SAMPLE)
        .map(String::as_str)
        .collect();
    assert_eq!(sample_ids.len(), SAMPLE, "sample size {}", sample_ids.len());

    let ids_sql_list = sample_ids
        .iter()
        .map(|id| format!("'{id}'"))
        .collect::<Vec<_>>()
        .join(",");
    let filter = format!("chunk_id IN ({ids_sql_list})");

    let stream = table
        .query()
        .only_if(filter)
        .select(Select::Columns(vec!["chunk_id".into(), "tokens".into()]))
        .execute()
        .await?;
    let batches: Vec<RecordBatch> = stream.try_collect().await?;
    let returned: usize = batches.iter().map(|b| b.num_rows()).sum();
    if returned != SAMPLE {
        eprintln!("FAIL: bulk fetch returned {returned} rows, expected {SAMPLE}");
        std::process::exit(1);
    }
    println!("PASS: single chunk_id IN (...) query returned {returned} rows");

    // Decode into HashMap<chunk_id, Vec<Vec<f32>>>.
    let mut decoded: HashMap<String, Vec<Vec<f32>>> = HashMap::new();
    for batch in &batches {
        let ids = batch
            .column_by_name("chunk_id")
            .expect("chunk_id col")
            .as_any()
            .downcast_ref::<StringArray>()
            .expect("StringArray");
        let tokens = batch
            .column_by_name("tokens")
            .expect("tokens col")
            .as_any()
            .downcast_ref::<ListArray>()
            .expect("ListArray");
        for row in 0..batch.num_rows() {
            let chunk_id = ids.value(row).to_string();
            let token_list = tokens.value(row);
            let fsl = token_list
                .as_any()
                .downcast_ref::<FixedSizeListArray>()
                .expect("FixedSizeListArray");
            let n_tokens = fsl.len();
            let mut tvs: Vec<Vec<f32>> = Vec::with_capacity(n_tokens);
            for t in 0..n_tokens {
                let inner = fsl.value(t);
                let f32s = inner
                    .as_any()
                    .downcast_ref::<arrow_array::Float32Array>()
                    .expect("Float32Array");
                let v: Vec<f32> = (0..f32s.len()).map(|i| f32s.value(i)).collect();
                assert_eq!(v.len(), DIM);
                tvs.push(v);
            }
            decoded.insert(chunk_id, tvs);
        }
    }
    println!(
        "PASS: decoded into HashMap<chunk_id, Vec<Vec<f32>>> with {} entries",
        decoded.len()
    );

    // Reference MaxSim: pick one known chunk + a fixed query token set;
    // hand-compute the score and compare.
    let target_id = sample_ids[0].to_string();
    let target_tokens = decoded.get(&target_id).expect("target present");
    let query_tokens: Vec<Vec<f32>> = vec![vec![1.0; DIM], vec![0.5; DIM]];
    let max_sim = maxsim(&query_tokens, target_tokens);
    let reference = maxsim_reference(&query_tokens, target_tokens);
    let delta = (max_sim - reference).abs();
    if delta > 1e-6 {
        eprintln!("FAIL: maxsim={max_sim}, reference={reference}, delta={delta}");
        std::process::exit(1);
    }
    println!("PASS: in-memory MaxSim ({max_sim:.6}) matches reference ({reference:.6})");
    println!(
        "RESULT: Lance 0.29.0 supports List<FixedSizeList<F32,D>> side-table + bulk-fetch + decode for MaxSim"
    );
    Ok(())
}

/// MaxSim: for each query token, max dot-product over doc tokens; sum.
fn maxsim(query: &[Vec<f32>], doc: &[Vec<f32>]) -> f32 {
    query
        .iter()
        .map(|q| {
            doc.iter()
                .map(|d| dot(q, d))
                .fold(f32::NEG_INFINITY, f32::max)
        })
        .sum()
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Reference: identical implementation but written with explicit loops
/// so the two cross-check rather than being literally the same.
fn maxsim_reference(query: &[Vec<f32>], doc: &[Vec<f32>]) -> f32 {
    let mut total = 0.0f32;
    for q in query {
        let mut best = f32::NEG_INFINITY;
        for d in doc {
            let mut s = 0.0f32;
            for i in 0..q.len() {
                s += q[i] * d[i];
            }
            if s > best {
                best = s;
            }
        }
        total += best;
    }
    total
}
