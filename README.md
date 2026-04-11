# codexa

**AI-ready codebase indexer.** Scans your project and generates a minimal, structured context file (`AGENTS.md` / `CLAUDE.md`) so any AI agent — Claude Code, GPT-4, Gemini, or your own — instantly understands your codebase.

```bash
codexa init          # scan project → generate AGENTS.md
codexa update        # re-scan and refresh context
codexa stats         # show project summary
codexa configure     # interactive setup wizard
codexa index         # build vector semantic index
codexa search "query" # semantic search over codebase
```

---

## Why

AI agents work better when they know:

- What language / stack the project uses
- Where entry points and key files are
- What the coding conventions are
- What they should and shouldn't touch

Writing this by hand is tedious. `codexa` does it in one command.

---

## Advanced Features

### Semantic Search (Vector DB)
For large projects, codexa can index your code into a vector database so agents can find relevant context by semantic similarity — without re-scanning the whole project every time.

### Smart Chunker
Unlike simple encoders, `codexa` uses a language-aware chunker (`chunker.rs`) that respects code boundaries (functions, classes, blocks) to keep semantic units together.

---

## Configuration

Running `codexa configure` creates `codexa.toml` in your project. You can enable SQLite or PostgreSQL backends for vector storage.

---

## Structure

```
src/
  main.rs       # CLI entry point and commands
  config.rs     # codexa.toml config + wizard
  indexer.rs    # file scanner, language detection
  generator.rs  # markdown + json output
  vector.rs     # vector store + embeddings integration
  chunker.rs    # semantic code splitting logic (Smart Chunker)
```

---

## Contributing

PRs are welcome. The codebase is modular and designed to be easily extensible.

To add a new language — edit the `detect_language()` match in `indexer.rs`.  
To improve chunking — look into `chunker.rs`.

---

## License

MIT
