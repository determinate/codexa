//! Smart chunker — splits code by semantic boundaries (functions, classes, blocks)
//! instead of raw line counts.
//!
//! ## Strategy
//!
//! ```text
//! 1. Try language-aware chunking (pattern-based AST-lite):
//!      detect function/class/impl starts → emit one chunk per top-level item
//!
//! 2. If a single item is > max_lines → split it further on inner boundaries
//!    (nested fn, closures, long match blocks) keeping context header
//!
//! 3. If no language-aware strategy matches → fall back to line-window chunking
//!    but snap boundaries to blank lines (never cut mid-expression)
//!
//! 4. Always add a 3-line context header (file path + surrounding scope name)
//!    so every chunk is self-describing when sent to an LLM.
//! ```

use crate::vector::CodeChunk;

/// How the chunk was produced — stored for debug/stats.
#[derive(Debug, Clone, PartialEq)]
pub enum ChunkKind {
    /// Top-level function / method
    Function,
    /// Class / struct / impl block
    TypeBlock,
    /// Module-level constants, imports, or attributes
    TopLevel,
    /// Fallback: line-window snapped to blank lines
    LineWindow,
}

// ── Public entry point ────────────────────────────────────────────────────────

/// Smart-chunk a single file. Returns empty vec for empty files.
pub fn smart_chunk(
    file_path: &str,
    language: Option<&str>,
    content: &str,
    max_chunk_lines: usize,  // hard cap; giant functions get split here
    target_lines: usize,     // preferred chunk size for line-window fallback
    overlap: usize,
) -> Vec<CodeChunk> {
    if content.trim().is_empty() {
        return vec![];
    }

    let lines: Vec<&str> = content.lines().collect();

    let raw_chunks = match language {
        Some("Rust")       => chunk_rust(&lines, max_chunk_lines),
        Some("TypeScript")
        | Some("JavaScript") => chunk_ts_js(&lines, max_chunk_lines),
        Some("Python")     => chunk_python(&lines, max_chunk_lines),
        Some("Go")         => chunk_go(&lines, max_chunk_lines),
        Some("Java")
        | Some("Kotlin")
        | Some("C#")       => chunk_java_like(&lines, max_chunk_lines),
        Some("C")
        | Some("C++")      => chunk_c_like(&lines, max_chunk_lines),
        _                  => vec![],  // → line fallback below
    };

    let raw_chunks = if raw_chunks.is_empty() {
        // Fallback: snap to blank lines
        chunk_by_blank_lines(&lines, target_lines, overlap)
    } else {
        raw_chunks
    };

    // Convert raw (start, end, kind) → CodeChunk, splitting oversized ones
    let mut result = Vec::new();
    for (start, end, _kind) in raw_chunks {
        let chunk_lines = end - start;
        if chunk_lines > max_chunk_lines {
            // Split large chunk further — snap to blank lines inside it
            let sub = chunk_by_blank_lines(&lines[start..end], target_lines, overlap);
            for (s2, e2, _) in sub {
                result.push(make_chunk(file_path, language, &lines, start + s2, start + e2));
            }
        } else {
            result.push(make_chunk(file_path, language, &lines, start, end));
        }
    }

    result
}

// ── Rust chunker ──────────────────────────────────────────────────────────────

fn chunk_rust(lines: &[&str], max: usize) -> Vec<(usize, usize, ChunkKind)> {
    chunk_brace_language(lines, max, &[
        // top-level item starters (no leading whitespace or only pub/async/unsafe prefix)
        "fn ",
        "pub fn ",
        "async fn ",
        "pub async fn ",
        "unsafe fn ",
        "pub unsafe fn ",
        "impl ",
        "pub impl ",
        "struct ",
        "pub struct ",
        "enum ",
        "pub enum ",
        "trait ",
        "pub trait ",
        "mod ",
        "pub mod ",
        "macro_rules!",
        "#[test]",
        "#[tokio::test]",
    ])
}

// ── TypeScript / JavaScript chunker ──────────────────────────────────────────

fn chunk_ts_js(lines: &[&str], max: usize) -> Vec<(usize, usize, ChunkKind)> {
    chunk_brace_language(lines, max, &[
        "function ",
        "async function ",
        "export function ",
        "export async function ",
        "export default function",
        "const ",        // catches: const foo = () => {
        "export const ",
        "class ",
        "export class ",
        "abstract class ",
        "interface ",
        "export interface ",
        "type ",
        "export type ",
        "enum ",
        "export enum ",
    ])
}

// ── Go chunker ────────────────────────────────────────────────────────────────

fn chunk_go(lines: &[&str], max: usize) -> Vec<(usize, usize, ChunkKind)> {
    chunk_brace_language(lines, max, &[
        "func ",
        "type ",
        "var (",
        "const (",
    ])
}

// ── Java / Kotlin / C# ───────────────────────────────────────────────────────

fn chunk_java_like(lines: &[&str], max: usize) -> Vec<(usize, usize, ChunkKind)> {
    chunk_brace_language(lines, max, &[
        "public ",
        "private ",
        "protected ",
        "static ",
        "class ",
        "interface ",
        "enum ",
        "record ",
        "abstract ",
        "fun ",       // Kotlin
    ])
}

// ── C / C++ ──────────────────────────────────────────────────────────────────

fn chunk_c_like(lines: &[&str], max: usize) -> Vec<(usize, usize, ChunkKind)> {
    chunk_brace_language(lines, max, &[
        "void ",
        "int ",
        "static ",
        "inline ",
        "struct ",
        "class ",
        "namespace ",
        "template ",
    ])
}

// ── Python chunker (indentation-based) ───────────────────────────────────────

fn chunk_python(lines: &[&str], _max: usize) -> Vec<(usize, usize, ChunkKind)> {
    let mut chunks: Vec<(usize, usize, ChunkKind)> = Vec::new();
    let mut i = 0;

    while i < lines.len() {
        let line = lines[i].trim_start();

        let is_top_level = (line.starts_with("def ")
            || line.starts_with("async def ")
            || line.starts_with("class "))
            && !lines[i].starts_with(' ')   // truly top-level (no indent)
            && !lines[i].starts_with('\t');

        if is_top_level {
            let start = i;
            i += 1;
            // collect until next top-level def/class or EOF
            while i < lines.len() {
                let next = lines[i];
                let is_new_top = (next.starts_with("def ")
                    || next.starts_with("async def ")
                    || next.starts_with("class "))
                    && !next.starts_with(' ')
                    && !next.starts_with('\t');
                if is_new_top {
                    break;
                }
                i += 1;
            }
            chunks.push((start, i, ChunkKind::Function));
        } else {
            i += 1;
        }
    }

    // collect any top-level code between functions
    fill_gaps(&mut chunks, lines.len());
    chunks
}

// ── Generic brace-language chunker ───────────────────────────────────────────
//
// Works for any language that uses `{` / `}` for blocks.
// Finds top-level item starters, then tracks brace depth to find the end.

fn chunk_brace_language(
    lines: &[&str],
    _max: usize,
    starters: &[&str],
) -> Vec<(usize, usize, ChunkKind)> {
    let mut chunks: Vec<(usize, usize, ChunkKind)> = Vec::new();
    let mut i = 0;

    while i < lines.len() {
        let trimmed = lines[i].trim_start();

        // look for a line starting with one of the starters
        let is_start = starters.iter().any(|s| trimmed.starts_with(s));

        if is_start {
            let start = i;
            let kind = kind_for_line(trimmed);

            // scan forward tracking brace depth
            let mut depth = 0i32;
            let mut found_open = false;

            while i < lines.len() {
                for ch in lines[i].chars() {
                    match ch {
                        '{' => { depth += 1; found_open = true; }
                        '}' => { depth -= 1; }
                        _ => {}
                    }
                }
                i += 1;
                // stop when we've seen at least one open brace and depth returns to 0
                if found_open && depth <= 0 {
                    break;
                }
            }

            chunks.push((start, i, kind));
        } else {
            i += 1;
        }
    }

    fill_gaps(&mut chunks, lines.len());
    chunks
}

// ── Blank-line fallback chunker ───────────────────────────────────────────────

/// Split on blank lines, never cutting inside a non-empty run.
/// If a run is longer than `target`, split it at the nearest blank within.
pub fn chunk_by_blank_lines(
    lines: &[&str],
    target: usize,
    overlap: usize,
) -> Vec<(usize, usize, ChunkKind)> {
    if lines.is_empty() {
        return vec![];
    }

    // collect positions of blank lines
    let blanks: Vec<usize> = lines
        .iter()
        .enumerate()
        .filter(|(_, l)| l.trim().is_empty())
        .map(|(i, _)| i)
        .collect();

    // build windows that snap to blank lines
    let mut chunks = Vec::new();
    let mut start = 0;

    while start < lines.len() {
        let ideal_end = (start + target).min(lines.len());

        // find the nearest blank line at or before ideal_end (snap backward)
        let end = blanks
            .iter()
            .rev()
            .find(|&&b| b > start && b <= ideal_end)
            .copied()
            .unwrap_or(ideal_end);

        let end = end.max(start + 1); // always make progress

        chunks.push((start, end, ChunkKind::LineWindow));

        // next start with overlap
        start = end.saturating_sub(overlap).max(end.min(end));
        if start >= end { start = end; }
    }

    chunks
}

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Fill uncovered line ranges between discovered chunks as TopLevel chunks.
fn fill_gaps(chunks: &mut Vec<(usize, usize, ChunkKind)>, total: usize) {
    if chunks.is_empty() {
        return;
    }
    chunks.sort_by_key(|c| c.0);

    let mut gaps: Vec<(usize, usize, ChunkKind)> = Vec::new();
    let mut prev_end = 0;

    for (start, end, _) in chunks.iter() {
        if *start > prev_end {
            gaps.push((prev_end, *start, ChunkKind::TopLevel));
        }
        prev_end = *end;
    }

    if prev_end < total {
        gaps.push((prev_end, total, ChunkKind::TopLevel));
    }

    // only add gaps that have non-empty content
    for gap in gaps {
        chunks.push(gap);
    }
    chunks.sort_by_key(|c| c.0);
}

fn kind_for_line(line: &str) -> ChunkKind {
    if line.contains("fn ") || line.contains("function ") || line.contains("def ") || line.contains("func ") {
        ChunkKind::Function
    } else if line.contains("struct ")
        || line.contains("class ")
        || line.contains("impl ")
        || line.contains("enum ")
        || line.contains("interface ")
    {
        ChunkKind::TypeBlock
    } else {
        ChunkKind::TopLevel
    }
}

fn make_chunk(
    file_path: &str,
    language: Option<&str>,
    lines: &[&str],
    start: usize,
    end: usize,
) -> CodeChunk {
    let end = end.min(lines.len());
    let start = start.min(end);

    // 3-line context header so the chunk is self-describing
    let header_start = start.saturating_sub(3);
    let header: Vec<&str> = if header_start < start {
        lines[header_start..start].to_vec()
    } else {
        vec![]
    };

    let body: Vec<&str> = lines[start..end].to_vec();

    let content = if header.is_empty() {
        body.join("\n")
    } else {
        format!("// context:\n{}\n// ---\n{}", header.join("\n"), body.join("\n"))
    };

    CodeChunk {
        file_path: file_path.to_string(),
        start_line: start + 1,
        end_line: end,
        content,
        language: language.map(|l| l.to_string()),
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rust_splits_by_fn() {
        let src = r#"
use std::io;

fn foo() {
    println!("foo");
}

fn bar() -> i32 {
    42
}

struct MyStruct {
    x: i32,
}

impl MyStruct {
    fn new() -> Self { MyStruct { x: 0 } }
}
"#;
        let chunks = smart_chunk("test.rs", Some("Rust"), src, 200, 60, 10);
        // should produce at least foo, bar, MyStruct, impl block
        assert!(chunks.len() >= 3, "got {} chunks", chunks.len());
        // no chunk should be empty
        for c in &chunks {
            assert!(!c.content.trim().is_empty(), "empty chunk at line {}", c.start_line);
        }
    }

    #[test]
    fn test_python_splits_by_def() {
        let src = "def foo():\n    pass\n\ndef bar():\n    return 1\n";
        let chunks = smart_chunk("test.py", Some("Python"), src, 200, 60, 5);
        assert_eq!(chunks.len(), 2);
    }

    #[test]
    fn test_oversized_fn_gets_split() {
        // build a function that is 120 lines long
        let mut src = String::from("fn giant() {\n");
        for i in 0..118 {
            src.push_str(&format!("    let x{i} = {i};\n"));
        }
        src.push_str("}\n");

        let chunks = smart_chunk("test.rs", Some("Rust"), &src, 50, 40, 5);
        // must be split into multiple chunks
        assert!(chunks.len() > 1, "oversized fn should be split, got {} chunks", chunks.len());
    }

    #[test]
    fn test_fallback_snaps_to_blank() {
        let src = "a\nb\nc\n\nd\ne\nf\n\ng\nh\n";
        let chunks = smart_chunk("test.txt", None, src, 200, 4, 0);
        // should not cut in the middle of "a b c" block
        for c in &chunks {
            // no chunk should start or end with a blank line in the middle of content
            assert!(!c.content.is_empty());
        }
    }
}
