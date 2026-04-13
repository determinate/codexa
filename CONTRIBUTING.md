# Contributing to codexa

First off — thank you! Every PR, bug report, and idea makes codexa better.

---

## Before you start

1. **Read the [CLA](../CLA.md).** By submitting a PR you agree to it.
   Add your name to [CONTRIBUTORS.md](../CONTRIBUTORS.md) in your PR.

2. **Open an issue first** for large changes (new features, refactors).
   Small fixes (typos, docs, obvious bugs) can go straight to a PR.

3. **Check existing issues** — your idea might already be discussed.

---

## Development setup

```bash
git clone https://github.com/determinate/codexa
cd codexa
cargo build
cargo test
```

Requirements:
- Rust stable (1.75+)
- No other dependencies — codexa is self-contained

---

## Making a PR

1. Fork the repo
2. Create a branch: `git checkout -b feat/your-feature`
3. Make your changes
4. Run checks:
   ```bash
   cargo fmt
   cargo clippy -- -D warnings
   cargo test
   ```
5. Add your name to `CONTRIBUTORS.md`
6. Open the PR — describe what and why

---

## What we welcome

- 🐛 Bug fixes
- 📖 Documentation improvements
- 🌍 New language support in `src/chunker.rs`
- 🔌 New embedding providers in `src/local_embed.rs`
- ⚡ Performance improvements
- 🧪 More tests

## What needs discussion first

- Breaking changes to CLI commands or config format
- New dependencies (we keep the binary small)
- Changes to the license or CLA

---

## Code style

- `cargo fmt` before committing (enforced in CI)
- `cargo clippy` must pass with no warnings
- Add tests for new functionality
- Keep modules focused — each file does one thing

---

## Project structure

```
src/
  main.rs          CLI commands (thin layer — logic lives in modules)
  config.rs        codexa.toml config + interactive wizard
  indexer.rs       File scanner, language detection, entry points
  chunker.rs       Smart code chunker (by fn/class/blank-line)
  vector.rs        Embedding client + SQLite/pg vector stores
  local_embed.rs   Built-in ONNX embedder (fastembed)
  generator.rs     AGENTS.md / JSON output writer
```

---

## Adding a new language to the chunker

Edit `src/chunker.rs`:

```rust
// 1. Add a match arm in smart_chunk()
Some("YourLang") => chunk_your_lang(&lines, max_chunk_lines),

// 2. Implement the function
fn chunk_your_lang(lines: &[&str], max: usize) -> Vec<(usize, usize, ChunkKind)> {
    chunk_brace_language(lines, max, &[
        "fn ",      // function starters in YourLang
        "class ",
    ])
}
```

---

## Questions?

Open an issue or email evrey2696@gmail.com.
