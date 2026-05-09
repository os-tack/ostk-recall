// Smoke test: point the rewritten gemini scanner at a real corpus and
// print per-file chunk counts. Verifies both shapes (logs.json flat
// array + chats/session-*.json envelope) decode without falling back.
//
// Usage:  cargo run -p ostk-recall-scan --example scan_real_gemini -- ~/.gemini/tmp

use ostk_recall_core::{Scanner, SourceConfig, SourceKind};
use ostk_recall_scan::gemini::GeminiScanner;

fn main() {
    let arg = std::env::args().nth(1).expect("usage: scan_real_gemini <root>");
    let cfg = SourceConfig {
        kind: SourceKind::Gemini,
        project: Some("real".into()),
        paths: vec![arg],
        ignore: vec![],
        extensions: vec![],
    };

    let scanner = GeminiScanner;
    let mut total_files = 0usize;
    let mut total_chunks = 0usize;
    let mut fallback_files = 0usize;
    let mut shape_a_files = 0usize;
    let mut shape_b_files = 0usize;
    let mut zero_chunk_files = 0usize;

    for item_res in scanner.discover(&cfg) {
        let item = item_res.expect("discover error");
        let path = item.path.clone().unwrap();
        total_files += 1;
        match scanner.parse(item) {
            Ok(chunks) => {
                total_chunks += chunks.len();
                if chunks.is_empty() {
                    zero_chunk_files += 1;
                    println!("  zero chunks: {}", path.display());
                } else if chunks.iter().any(|c| c.extra.get("shape").and_then(|v| v.as_str()) == Some("fallback")) {
                    fallback_files += 1;
                    println!("  FALLBACK: {} -> {} chunk(s)", path.display(), chunks.len());
                } else if chunks.iter().any(|c| c.extra.get("shape").and_then(|v| v.as_str()) == Some("logs.json")) {
                    shape_a_files += 1;
                } else {
                    shape_b_files += 1;
                }
            }
            Err(e) => println!("  parse error {}: {}", path.display(), e),
        }
    }
    println!("---");
    println!("files discovered:   {}", total_files);
    println!("  shape A (logs):   {}", shape_a_files);
    println!("  shape B (chats):  {}", shape_b_files);
    println!("  fallback:         {}", fallback_files);
    println!("  zero chunks:      {}", zero_chunk_files);
    println!("total chunks:       {}", total_chunks);
}
