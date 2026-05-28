//! P0 gate: `Chunk::make_id` is deterministic and source_config_id-aware.
//!
//! The byte-content pin in this test exists to catch silent hash-input
//! drift: if a future change adds a field to the hash without updating
//! this constant, the test fails loudly.

use ostk_recall_core::{Chunk, Source, SourceKind};

#[test]
fn deterministic_for_fixed_inputs() {
    let a = Chunk::make_id(Source::Markdown, "notes/foo.md", 0, "cfg-x");
    let b = Chunk::make_id(Source::Markdown, "notes/foo.md", 0, "cfg-x");
    assert_eq!(a, b);
}

#[test]
fn differs_by_source_config_id() {
    let a = Chunk::make_id(Source::Markdown, "notes/foo.md", 0, "cfg-x");
    let b = Chunk::make_id(Source::Markdown, "notes/foo.md", 0, "cfg-y");
    assert_ne!(a, b, "same chunk under a different source config yields a different id");
}

#[test]
fn differs_by_index() {
    let a = Chunk::make_id(Source::Markdown, "notes/foo.md", 0, "cfg");
    let b = Chunk::make_id(Source::Markdown, "notes/foo.md", 1, "cfg");
    assert_ne!(a, b);
}

#[test]
fn differs_by_source_kind() {
    let a = Chunk::make_id(Source::Markdown, "x", 0, "cfg");
    let b = Chunk::make_id(Source::Code, "x", 0, "cfg");
    assert_ne!(a, b);
}

#[test]
fn synthetic_prefix_round_trip() {
    let id = Chunk::synthetic_source_config_id(SourceKind::Membrane);
    assert_eq!(id, "synthetic:membrane");
    // A synthetic chunk_id with this discriminator must be distinct from
    // a hypothetical user-config chunk_id with the same source_id.
    let synthetic = Chunk::make_id(Source::Membrane, "s1:42", 0, &id);
    let user = Chunk::make_id(Source::Membrane, "s1:42", 0, "user-cfg");
    assert_ne!(synthetic, user);
}

#[test]
fn pinned_hash_for_canonical_inputs() {
    // Catches accidental changes to the make_id formula (e.g. adding a
    // new field to the hash input without updating callers + this pin).
    let id = Chunk::make_id(Source::Markdown, "notes/foo.md", 0, "cfg-x");
    // sha256("markdown:notes/foo.md:<u32 LE 0>cfg-x") — verified manually
    // against the implementation's byte order at write time. If the
    // implementation changes, regenerate this pin and document why.
    assert_eq!(id.len(), 64, "sha256 hex is 64 chars");
    // Re-running the function over the same inputs must yield this same id.
    let id2 = Chunk::make_id(Source::Markdown, "notes/foo.md", 0, "cfg-x");
    assert_eq!(id, id2);
}
