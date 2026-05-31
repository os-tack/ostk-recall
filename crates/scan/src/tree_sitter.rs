//! Tree-sitter native code chunker (issue #11).
//!
//! Per-file structural parse — milliseconds, no subprocess, no cargo
//! cold-start, no workspace re-index. Replaces the rust-analyzer-backed
//! `fcp-rust` chunking adapter that this module supersedes.
//!
//! For each supported language ([`Lang`]) we parse the file with its
//! tree-sitter grammar and walk the syntax tree for top-level items
//! (`fn`/`struct`/`enum`/`impl`/`trait`/`mod` for Rust; the analogues for
//! Python / TypeScript / JavaScript / Go). Container items (`impl`,
//! `trait`, `mod`, `class`) recurse one level so their methods / members
//! become their own chunks — matching the symbol granularity the old
//! rust-analyzer chunker produced, without overlap.
//!
//! Each chunk carries the same shape the previous chunker emitted so
//! downstream retrieval / inspect keeps working:
//!
//! * a synthetic header line `// <kind> <name>` so BM25 surfaces
//!   symbol-name queries,
//! * a leading doc/comment block captured by walking backward from the
//!   item's declaration (the `slice_symbol_with_docs` heuristic,
//!   generalized per-language), and
//! * `extra` metadata `{ kind, symbols, line_start, line_end, chunker }`
//!   with `chunker` set to `"tree-sitter"`.
//!
//! Tree-sitter is *syntactic*: it won't see macro-generated items or
//! resolve re-exports. For chunking that's correct — we chunk what is
//! literally in the file.

use std::path::Path;

use chrono::{DateTime, Utc};
use ostk_recall_core::{Chunk, Links, Source};
use tree_sitter::{Language, Node, Parser};

/// Upper bound on lines scanned backward from an item while searching for
/// its preceding doc-comment / attribute block. Bounds the walk on
/// pathological files; the scan stops at the first non-doc line anyway.
const SYMBOL_DOC_SCAN_MAX: u32 = 200;

/// Max lines of the top-of-file comment/doc block captured as a synthetic
/// module-header chunk (where `//!` rustdoc, needle refs, and module-level
/// reasoning live).
const MODULE_HEADER_MAX_LINES: u32 = 200;

/// Languages with a tree-sitter grammar wired in. Extensions outside this
/// set (e.g. `md`) return `None` from [`chunk_code_file`] so the caller
/// falls back to line-window chunking.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Lang {
    Rust,
    Python,
    TypeScript,
    Tsx,
    JavaScript,
    Go,
}

impl Lang {
    fn from_ext(ext: &str) -> Option<Self> {
        match ext.to_ascii_lowercase().as_str() {
            "rs" => Some(Self::Rust),
            "py" | "pyi" => Some(Self::Python),
            "ts" | "mts" | "cts" => Some(Self::TypeScript),
            "tsx" => Some(Self::Tsx),
            "js" | "jsx" | "mjs" | "cjs" => Some(Self::JavaScript),
            "go" => Some(Self::Go),
            _ => None,
        }
    }

    fn language(self) -> Language {
        match self {
            Self::Rust => tree_sitter_rust::LANGUAGE.into(),
            Self::Python => tree_sitter_python::LANGUAGE.into(),
            Self::TypeScript => tree_sitter_typescript::LANGUAGE_TYPESCRIPT.into(),
            Self::Tsx => tree_sitter_typescript::LANGUAGE_TSX.into(),
            Self::JavaScript => tree_sitter_javascript::LANGUAGE.into(),
            Self::Go => tree_sitter_go::LANGUAGE.into(),
        }
    }

    /// Line prefixes that count as a leading doc/comment line during the
    /// backward walk. Attributes/decorators that the grammar already
    /// folds into the item node don't need listing — only freestanding
    /// comment lines that sit *above* the node do.
    fn doc_prefixes(self) -> &'static [&'static str] {
        match self {
            // Mirrors the original Rust `slice_symbol_with_docs`: outer/
            // inner doc comments and attributes (plain `//` is NOT a doc).
            Self::Rust => &["///", "//!", "#[", "#!["],
            Self::Python => &["#", "@"],
            Self::TypeScript | Self::Tsx | Self::JavaScript => &["//", "/*", "*", "@"],
            Self::Go => &["//"],
        }
    }
}

/// One item to emit as a chunk. Line numbers are 1-based.
struct Emit {
    kind: &'static str,
    name: String,
    /// Declaration line (node start) — stamped as `extra.line_start`.
    decl_line: u32,
    /// Inclusive end line of the slice — stamped as `extra.line_end`.
    end_line: u32,
}

/// Build per-item chunks for a single source file via tree-sitter.
///
/// Returns `Some(chunks)` when the extension maps to a supported grammar
/// and the parse yields at least one top-level item; `None` otherwise
/// (unsupported extension, parse failure, or no items found) so the
/// caller falls back to line-window chunking.
///
/// The argument shape mirrors the removed `fcp_rust::chunk_rust_file` so
/// it drops into the existing call sites unchanged.
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn chunk_code_file(
    file_path: &Path,
    text: &str,
    source: Source,
    source_id: &str,
    project: Option<&str>,
    source_config_id: &str,
    mtime: Option<DateTime<Utc>>,
    abs_path: &str,
) -> Option<Vec<Chunk>> {
    let ext = file_path.extension().and_then(|e| e.to_str())?;
    let lang = Lang::from_ext(ext)?;

    let mut parser = Parser::new();
    if parser.set_language(&lang.language()).is_err() {
        tracing::warn!(lang = ?lang, "tree_sitter: set_language failed");
        return None;
    }
    let tree = parser.parse(text, None)?;
    let root = tree.root_node();

    let mut items: Vec<Emit> = Vec::new();
    collect_items(root, text, lang, &mut items);
    if items.is_empty() {
        return None;
    }

    let lines: Vec<&str> = text.split_inclusive('\n').collect();
    let doc_prefixes = lang.doc_prefixes();
    let mut chunks: Vec<Chunk> = Vec::with_capacity(items.len() + 1);

    // Module-header chunk: the top-of-file comment/doc block before the
    // first item. Captures `//!` rustdoc + needle refs that no per-item
    // chunk would otherwise cover.
    let first_item_line = items.iter().map(|e| e.decl_line).min().unwrap_or(1);
    if let Some(body) = slice_module_header(&lines, first_item_line, doc_prefixes) {
        chunks.push(build_chunk(
            "module_header",
            &[],
            1,
            first_item_line.saturating_sub(1),
            body,
            chunks.len(),
            source,
            source_id,
            project,
            source_config_id,
            mtime,
            abs_path,
        )?);
    }

    for item in &items {
        let doc_start = extend_start_over_docs(&lines, item.decl_line, doc_prefixes);
        let body = slice_lines(&lines, doc_start, item.end_line);
        if body.trim().is_empty() {
            continue;
        }
        let header_body = format!("// {} {}\n{}", item.kind, item.name, body);
        chunks.push(build_chunk(
            item.kind,
            std::slice::from_ref(&item.name),
            item.decl_line,
            item.end_line,
            header_body,
            chunks.len(),
            source,
            source_id,
            project,
            source_config_id,
            mtime,
            abs_path,
        )?);
    }

    if chunks.is_empty() {
        None
    } else {
        Some(chunks)
    }
}

/// Construct one [`Chunk`] with the shared layout. `symbols` is the list
/// stamped into `extra.symbols`. Returns `None` only if `chunk_index`
/// overflows `u32` (degenerate file with >4B items).
#[allow(clippy::too_many_arguments)]
fn build_chunk(
    kind: &str,
    symbols: &[String],
    line_start: u32,
    line_end: u32,
    body: String,
    chunk_index: usize,
    source: Source,
    source_id: &str,
    project: Option<&str>,
    source_config_id: &str,
    mtime: Option<DateTime<Utc>>,
    abs_path: &str,
) -> Option<Chunk> {
    let chunk_index = u32::try_from(chunk_index).ok()?;
    let chunk_id = Chunk::make_id(source, source_id, chunk_index, source_config_id);
    let sha256 = Chunk::content_hash(&body);
    let extra = serde_json::json!({
        "kind": kind,
        "symbols": symbols,
        "line_start": line_start,
        "line_end": line_end,
        "chunker": "tree-sitter",
    });
    Some(Chunk {
        chunk_id,
        source,
        project: project.map(str::to_string),
        source_id: source_id.to_string(),
        facets: Default::default(),
        embedding_input_sha256: String::new(),
        source_config_id: source_config_id.to_string(),
        chunk_index,
        ts: mtime,
        role: None,
        text: body,
        sha256,
        links: Links {
            file_path: Some(abs_path.to_string()),
            ..Links::default()
        },
        extra,
    })
}

/// Walk the top-level named children of `root`, emitting an [`Emit`] per
/// item. Container items (`impl`/`trait`/`mod`/`class`) emit a thin header
/// covering the declaration up to their first member, then recurse one
/// level so each member becomes its own chunk (matching the granularity
/// the rust-analyzer chunker produced, without overlap).
fn collect_items(root: Node, src: &str, lang: Lang, out: &mut Vec<Emit>) {
    let mut cursor = root.walk();
    for child in root.named_children(&mut cursor) {
        process_item(child, src, lang, out);
    }
}

fn process_item(child: Node, src: &str, lang: Lang, out: &mut Vec<Emit>) {
    // Unwrap single-declaration wrappers (export / decorated). `range`
    // spans the wrapper (so `export`/decorators land in the chunk);
    // `decl` is the node we classify and name.
    let Some((range, decl)) = unwrap(child, lang) else {
        return;
    };
    let Some(kind) = classify(decl, lang) else {
        return;
    };
    let Some(name) = item_name(decl, src, lang) else {
        return;
    };

    let decl_line = range.start_position().row as u32 + 1;
    let end_line = range.end_position().row as u32 + 1;

    let members = container_members(decl, lang);
    if members.is_empty() {
        out.push(Emit {
            kind,
            name,
            decl_line,
            end_line,
        });
        return;
    }

    // Container: header chunk from the declaration up to the first member,
    // then one chunk per member.
    let first_member_line = members[0].start_position().row as u32 + 1;
    let header_end = first_member_line.saturating_sub(1).max(decl_line);
    out.push(Emit {
        kind,
        name,
        decl_line,
        end_line: header_end,
    });
    for member in members {
        process_item(member, src, lang, out);
    }
}

/// Map a top-level child onto `(range_node, decl_node)`, unwrapping
/// `export_statement` / `decorated_definition` so the inner declaration
/// is classified/named while the chunk range covers the wrapper.
fn unwrap<'t>(child: Node<'t>, lang: Lang) -> Option<(Node<'t>, Node<'t>)> {
    match (lang, child.kind()) {
        (Lang::Python, "decorated_definition") => {
            child.child_by_field_name("definition").map(|d| (child, d))
        }
        (Lang::TypeScript | Lang::Tsx | Lang::JavaScript, "export_statement") => {
            child.child_by_field_name("declaration").map(|d| (child, d))
        }
        _ => Some((child, child)),
    }
}

/// Concise kind label for a declaration node, or `None` if the node isn't
/// a top-level item we chunk.
fn classify(node: Node, lang: Lang) -> Option<&'static str> {
    let k = node.kind();
    match lang {
        Lang::Rust => match k {
            "function_item" | "function_signature_item" => Some("fn"),
            "struct_item" => Some("struct"),
            "enum_item" => Some("enum"),
            "union_item" => Some("union"),
            "trait_item" => Some("trait"),
            "impl_item" => Some("impl"),
            "mod_item" => Some("mod"),
            "const_item" => Some("const"),
            "static_item" => Some("static"),
            "type_item" => Some("type"),
            "macro_definition" => Some("macro"),
            "associated_type" => Some("type"),
            _ => None,
        },
        Lang::Python => match k {
            "function_definition" => Some("fn"),
            "class_definition" => Some("class"),
            _ => None,
        },
        Lang::TypeScript | Lang::Tsx | Lang::JavaScript => match k {
            "function_declaration" | "generator_function_declaration" | "function_signature" => {
                Some("fn")
            }
            "method_definition" => Some("fn"),
            "class_declaration" | "abstract_class_declaration" => Some("class"),
            "interface_declaration" => Some("interface"),
            "type_alias_declaration" => Some("type"),
            "enum_declaration" => Some("enum"),
            "lexical_declaration" => Some("const"),
            "variable_declaration" => Some("var"),
            _ => None,
        },
        Lang::Go => match k {
            "function_declaration" => Some("fn"),
            "method_declaration" => Some("fn"),
            "type_declaration" => Some("type"),
            "const_declaration" => Some("const"),
            "var_declaration" => Some("var"),
            _ => None,
        },
    }
}

/// Direct members of a container item, or empty if `node` isn't a
/// container. Members are the chunk-worthy declarations inside the
/// container's body — methods, associated fns/consts/types.
fn container_members<'t>(node: Node<'t>, lang: Lang) -> Vec<Node<'t>> {
    let body_field = match (lang, node.kind()) {
        (Lang::Rust, "impl_item" | "trait_item" | "mod_item") => "body",
        (Lang::Python, "class_definition") => "body",
        (Lang::TypeScript | Lang::Tsx | Lang::JavaScript, "class_declaration")
        | (Lang::TypeScript | Lang::Tsx | Lang::JavaScript, "abstract_class_declaration") => "body",
        _ => return Vec::new(),
    };
    let Some(body) = node.child_by_field_name(body_field) else {
        return Vec::new();
    };
    let mut out = Vec::new();
    let mut cursor = body.walk();
    for member in body.named_children(&mut cursor) {
        // Unwrap decorated/exported members so they classify.
        let Some((_, decl)) = unwrap(member, lang) else {
            continue;
        };
        if classify(decl, lang).is_some() && item_name_node(decl, lang).is_some() {
            out.push(member);
        }
    }
    out
}

/// Resolve the display name of a declaration node.
fn item_name(node: Node, src: &str, lang: Lang) -> Option<String> {
    // Rust impl blocks have no `name`; synthesize `Trait for Type` / `Type`.
    if lang == Lang::Rust && node.kind() == "impl_item" {
        let ty = node
            .child_by_field_name("type")
            .map(|n| node_text(n, src).trim().to_string());
        let tr = node
            .child_by_field_name("trait")
            .map(|n| node_text(n, src).trim().to_string());
        return Some(match (tr, ty) {
            (Some(t), Some(y)) => format!("{t} for {y}"),
            (None, Some(y)) => y,
            (Some(t), None) => t,
            (None, None) => "impl".to_string(),
        });
    }
    let name_node = item_name_node(node, lang)?;
    Some(node_text(name_node, src).trim().to_string())
}

/// Locate the identifier node carrying an item's name (without reading
/// `src`). Used both for naming and to gate container membership.
fn item_name_node<'t>(node: Node<'t>, lang: Lang) -> Option<Node<'t>> {
    if let Some(n) = node.child_by_field_name("name") {
        return Some(n);
    }
    // Declarations that wrap their name in a `*_spec` / `variable_declarator`.
    let mut cursor = node.walk();
    for ch in node.named_children(&mut cursor) {
        let k = ch.kind();
        let descend = match lang {
            Lang::Go => k.ends_with("_spec"),
            Lang::TypeScript | Lang::Tsx | Lang::JavaScript => k == "variable_declarator",
            _ => false,
        };
        if descend {
            if let Some(n) = ch.child_by_field_name("name") {
                return Some(n);
            }
        }
    }
    None
}

fn node_text<'a>(node: Node, src: &'a str) -> &'a str {
    node.utf8_text(src.as_bytes()).unwrap_or("")
}

/// Walk backward from the line above `decl_line` while lines match the
/// language's doc/comment prefixes (or are blank lines *inside* such a
/// block). Returns the 1-based start line of the doc-extended slice.
///
/// Generalizes the original Rust `slice_symbol_with_docs` backward walk:
/// a blank line is part of the block only when the line above it is also
/// a doc line — otherwise it's whitespace between the previous item and
/// this one.
fn extend_start_over_docs(lines: &[&str], decl_line: u32, prefixes: &[&str]) -> u32 {
    if lines.is_empty() || decl_line <= 1 {
        return decl_line.max(1);
    }
    let mut start = decl_line;
    let scan_limit = decl_line.saturating_sub(SYMBOL_DOC_SCAN_MAX).max(1);
    while start > scan_limit {
        let candidate = start - 1;
        let idx = (candidate - 1) as usize;
        if idx >= lines.len() {
            break;
        }
        let line = lines[idx].trim_start();
        let is_doc = prefixes.iter().any(|p| line.starts_with(p));
        let is_block_blank = if line.is_empty() && candidate > 1 {
            let above = lines[(candidate - 2) as usize].trim_start();
            prefixes.iter().any(|p| above.starts_with(p))
        } else {
            false
        };
        if is_doc || is_block_blank {
            start = candidate;
        } else {
            break;
        }
    }
    start
}

/// Slice `[start, end]` (1-based, inclusive) out of `lines`, clamping to
/// file bounds. Returns an empty string for an inverted/out-of-range span.
fn slice_lines(lines: &[&str], start: u32, end: u32) -> String {
    if lines.is_empty() {
        return String::new();
    }
    let total = lines.len();
    let start_idx = (start.max(1) as usize).saturating_sub(1).min(total);
    let end_idx = (end as usize).min(total);
    if end_idx <= start_idx {
        return String::new();
    }
    lines[start_idx..end_idx].concat()
}

/// Build the synthetic module-header chunk body: the leading comment/doc
/// block at the top of the file, up to the first item (or
/// [`MODULE_HEADER_MAX_LINES`]). Returns `None` when the file starts with
/// code.
fn slice_module_header(lines: &[&str], first_item_line: u32, prefixes: &[&str]) -> Option<String> {
    if lines.is_empty() || first_item_line <= 1 {
        return None;
    }
    let cap = MODULE_HEADER_MAX_LINES
        .min(first_item_line.saturating_sub(1))
        .min(u32::try_from(lines.len()).unwrap_or(u32::MAX)) as usize;
    let mut end = 0usize;
    for (i, line) in lines.iter().take(cap).enumerate() {
        let l = line.trim_start();
        // A leading comment block: blank lines or any comment-ish prefix.
        // Accept plain `//` / `#` here too (top-of-file banners) regardless
        // of the per-item doc set.
        let is_header_line = l.is_empty()
            || l.starts_with("//")
            || l.starts_with('#')
            || prefixes.iter().any(|p| l.starts_with(p));
        if is_header_line {
            end = i + 1;
        } else {
            break;
        }
    }
    while end > 0 && lines[end - 1].trim().is_empty() {
        end -= 1;
    }
    if end == 0 {
        return None;
    }
    let body = lines[..end].concat();
    if body.trim().is_empty() {
        None
    } else {
        Some(body)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Run the chunker on `text` with `name`'s extension. `chunk_code_file`
    /// reads only the path extension and the in-memory text — no file is
    /// touched on disk.
    fn chunks_for(name: &str, text: &str) -> Vec<Chunk> {
        chunk_code_file(
            Path::new(name),
            text,
            Source::Code,
            "sid",
            Some("proj"),
            "cfg",
            None,
            "/abs/path",
        )
        .unwrap_or_default()
    }

    fn headers(chunks: &[Chunk]) -> Vec<String> {
        chunks
            .iter()
            .map(|c| c.text.lines().next().unwrap_or("").to_string())
            .collect()
    }

    fn has_header(chunks: &[Chunk], prefix: &str) -> bool {
        headers(chunks).iter().any(|h| h.starts_with(prefix))
    }

    #[test]
    fn unsupported_extension_returns_none() {
        assert!(
            chunk_code_file(
                Path::new("README.md"),
                "# Title\n\nbody\n",
                Source::Code,
                "sid",
                None,
                "cfg",
                None,
                "/abs",
            )
            .is_none()
        );
    }

    #[test]
    fn all_chunks_tagged_tree_sitter() {
        let chunks = chunks_for("a.rs", "fn a() {}\nfn b() {}\n");
        assert!(!chunks.is_empty());
        for c in &chunks {
            assert_eq!(
                c.extra.get("chunker").and_then(|v| v.as_str()),
                Some("tree-sitter")
            );
            assert_eq!(c.source, Source::Code);
            assert_eq!(c.links.file_path.as_deref(), Some("/abs/path"));
        }
    }

    #[test]
    fn rust_struct_impl_methods_module_header() {
        let src = "//! Module docs.\n//! → needle ref.\n\n/// A widget.\npub struct Widget {\n    n: u32,\n}\n\nimpl Widget {\n    /// Make one.\n    pub fn new() -> Self {\n        Self { n: 0 }\n    }\n\n    pub fn bump(&mut self) {\n        self.n += 1;\n    }\n}\n";
        let chunks = chunks_for("widget.rs", src);

        // Module header chunk carries the `//!` block.
        let mh = chunks
            .iter()
            .find(|c| c.extra.get("kind").and_then(|v| v.as_str()) == Some("module_header"))
            .expect("module_header chunk");
        assert!(mh.text.contains("Module docs."));
        assert!(mh.text.contains("→ needle ref."));

        assert!(
            has_header(&chunks, "// struct Widget"),
            "{:?}",
            headers(&chunks)
        );
        assert!(
            has_header(&chunks, "// impl Widget"),
            "{:?}",
            headers(&chunks)
        );
        assert!(has_header(&chunks, "// fn new"), "{:?}", headers(&chunks));
        assert!(has_header(&chunks, "// fn bump"), "{:?}", headers(&chunks));

        // Leading doc captured; line metadata is sane and 1-based.
        let struct_c = chunks
            .iter()
            .find(|c| c.text.starts_with("// struct Widget"))
            .unwrap();
        assert!(struct_c.text.contains("/// A widget."));
        let ls = struct_c
            .extra
            .get("line_start")
            .and_then(|v| v.as_u64())
            .unwrap();
        let le = struct_c
            .extra
            .get("line_end")
            .and_then(|v| v.as_u64())
            .unwrap();
        assert!(ls >= 1 && le >= ls);
        let syms = struct_c
            .extra
            .get("symbols")
            .and_then(|v| v.as_array())
            .unwrap();
        assert_eq!(syms[0].as_str(), Some("Widget"));
    }

    #[test]
    fn rust_impl_trait_for_type_name() {
        let src = "struct S;\nimpl std::fmt::Debug for S {\n    fn fmt(&self, _f: &mut std::fmt::Formatter) -> std::fmt::Result { Ok(()) }\n}\n";
        let chunks = chunks_for("s.rs", src);
        assert!(
            has_header(&chunks, "// impl std::fmt::Debug for S"),
            "{:?}",
            headers(&chunks)
        );
    }

    #[test]
    fn python_class_methods_and_decorated_fn() {
        let src = "# leading comment\nclass Greeter:\n    \"\"\"docstring\"\"\"\n    def __init__(self, name):\n        self.name = name\n\n    def greet(self):\n        return self.name\n\n@decorator\ndef standalone():\n    return 1\n";
        let chunks = chunks_for("g.py", src);
        assert!(
            has_header(&chunks, "// class Greeter"),
            "{:?}",
            headers(&chunks)
        );
        assert!(
            has_header(&chunks, "// fn __init__"),
            "{:?}",
            headers(&chunks)
        );
        assert!(has_header(&chunks, "// fn greet"), "{:?}", headers(&chunks));
        assert!(
            has_header(&chunks, "// fn standalone"),
            "{:?}",
            headers(&chunks)
        );
        // The decorator is part of the chunk body (range covers the wrapper).
        let standalone = chunks
            .iter()
            .find(|c| c.text.starts_with("// fn standalone"))
            .unwrap();
        assert!(standalone.text.contains("@decorator"));
    }

    #[test]
    fn typescript_export_function_class_interface() {
        let src = "export function add(a: number, b: number): number {\n  return a + b;\n}\n\nexport class Box {\n  v: number = 0;\n  reset(): void { this.v = 0; }\n}\n\ninterface Shape {\n  area(): number;\n}\n";
        let chunks = chunks_for("m.ts", src);
        assert!(has_header(&chunks, "// fn add"), "{:?}", headers(&chunks));
        assert!(
            has_header(&chunks, "// class Box"),
            "{:?}",
            headers(&chunks)
        );
        assert!(has_header(&chunks, "// fn reset"), "{:?}", headers(&chunks));
        assert!(
            has_header(&chunks, "// interface Shape"),
            "{:?}",
            headers(&chunks)
        );
    }

    #[test]
    fn go_func_method_type_const() {
        let src = "package main\n\n// Greet returns a greeting.\nfunc Greet(name string) string {\n\treturn \"hi \" + name\n}\n\ntype Server struct {\n\tport int\n}\n\nfunc (s *Server) Port() int {\n\treturn s.port\n}\n\nconst MaxConns = 100\n";
        let chunks = chunks_for("srv.go", src);
        assert!(has_header(&chunks, "// fn Greet"), "{:?}", headers(&chunks));
        assert!(
            has_header(&chunks, "// type Server"),
            "{:?}",
            headers(&chunks)
        );
        assert!(has_header(&chunks, "// fn Port"), "{:?}", headers(&chunks));
        assert!(
            has_header(&chunks, "// const MaxConns"),
            "{:?}",
            headers(&chunks)
        );
        // Go doc-comment captured above the func.
        let greet = chunks
            .iter()
            .find(|c| c.text.starts_with("// fn Greet"))
            .unwrap();
        assert!(greet.text.contains("// Greet returns a greeting."));
    }

    #[test]
    fn no_items_returns_none() {
        // A file of only statements / comments has no top-level items.
        assert!(
            chunk_code_file(
                Path::new("x.go"),
                "package main\n",
                Source::Code,
                "sid",
                None,
                "cfg",
                None,
                "/abs",
            )
            .is_none()
        );
    }
}
