mod config;
mod indexer;
mod generator;
mod vector;
mod chunker;

use anyhow::Result;
use clap::{Parser, Subcommand};
use colored::Colorize;

#[derive(Parser)]
#[command(
    name = "codexa",
    about = "AI-ready codebase indexer — generates context + vector search for Claude, GPT, and other agents",
    version,
    long_about = None
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Scan project and generate AI context (AGENTS.md / CLAUDE.md)
    Init {
        #[arg(default_value = ".")]
        path: String,
        #[arg(short, long, default_value = "markdown")]
        format: String,
        #[arg(short, long, default_value = "AGENTS.md")]
        output: String,
    },
    /// Re-scan and update existing context file
    Update {
        #[arg(default_value = ".")]
        path: String,
    },
    /// Show project stats
    Stats {
        #[arg(default_value = ".")]
        path: String,
    },
    /// Interactive configuration wizard (includes vector DB setup)
    Configure {
        #[arg(default_value = ".")]
        path: String,
    },
    /// Chunk all source files and write embeddings to the vector DB
    Index {
        #[arg(default_value = ".")]
        path: String,
        /// Re-index even if DB already has entries
        #[arg(long)]
        force: bool,
    },
    /// Semantic search over the vector DB
    Search {
        query: String,
        #[arg(default_value = ".")]
        path: String,
        #[arg(short = 'k', long, default_value = "5")]
        top_k: usize,
        /// Output as JSON (for piping into agents)
        #[arg(long)]
        json: bool,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    println!("{}", "  codexa  ".on_black().white().bold());
    println!("{}", "AI context generator for your codebase\n".dimmed());

    match cli.command {
        Commands::Init { path, format, output } => cmd_init(&path, &format, &output),
        Commands::Update { path } => cmd_update(&path),
        Commands::Stats { path } => cmd_stats(&path),
        Commands::Configure { path } => cmd_configure(&path),
        Commands::Index { path, force } => cmd_index(&path, force).await,
        Commands::Search { query, path, top_k, json } => cmd_search(&query, &path, top_k, json).await,
    }
}

fn cmd_init(path: &str, format: &str, output: &str) -> Result<()> {
    println!("{} {}", "→ Scanning".cyan(), path.bold());
    let cfg = config::Config::load_or_default(path);
    let index = indexer::scan(path, &cfg)?;

    println!(
        "  {} files indexed ({} languages detected)",
        index.total_files.to_string().green().bold(),
        index.languages.len().to_string().yellow()
    );

    let has_vector = cfg.vector.is_some();
    vector::check_size_and_warn(index.total_files, index.total_lines, has_vector);

    match format {
        "json" => {
            let json_out = format!("{}.json", output.trim_end_matches(".md"));
            generator::write_json(&index, &json_out)?;
            println!("{} {}", "✓ Written:".green(), json_out);
        }
        "both" => {
            generator::write_markdown(&index, output, &cfg)?;
            let json_out = format!("{}.json", output.trim_end_matches(".md"));
            generator::write_json(&index, &json_out)?;
            println!("{} {} and {}", "✓ Written:".green(), output, json_out);
        }
        _ => {
            generator::write_markdown(&index, output, &cfg)?;
            println!("{} {}", "✓ Written:".green(), output);
        }
    }

    if has_vector {
        println!("\n{} Run {} to build the vector index.", "Tip:".yellow().bold(), "codexa index".cyan());
    } else {
        println!("\n{} Pass {} to your AI agent as context.", "Tip:".yellow().bold(), output.cyan());
    }
    Ok(())
}

fn cmd_update(path: &str) -> Result<()> {
    println!("{} {}", "→ Updating context for".cyan(), path.bold());
    let cfg = config::Config::load_or_default(path);
    let index = indexer::scan(path, &cfg)?;
    let output = if std::path::Path::new(&format!("{}/CLAUDE.md", path)).exists() { "CLAUDE.md" } else { "AGENTS.md" };
    generator::write_markdown(&index, &format!("{}/{}", path, output), &cfg)?;
    println!("{} {} updated", "✓".green(), output);
    Ok(())
}

fn cmd_stats(path: &str) -> Result<()> {
    let cfg = config::Config::load_or_default(path);
    let index = indexer::scan(path, &cfg)?;

    println!("{}", "── Project Stats ──".bold());
    println!("  Files:  {}", index.total_files.to_string().green());
    println!("  Lines:  {}", index.total_lines.to_string().green());

    println!("\n{}", "── Languages ──".bold());
    let mut langs: Vec<_> = index.languages.iter().collect();
    langs.sort_by(|a, b| b.1.cmp(a.1));
    for (lang, count) in langs.iter().take(10) {
        println!("  {:12} {} files", lang.cyan(), count.to_string().yellow());
    }

    if !index.entry_points.is_empty() {
        println!("\n{}", "── Entry Points ──".bold());
        for ep in &index.entry_points { println!("  {}", ep.dimmed()); }
    }

    let has_vector = cfg.vector.is_some();
    println!("\n{}", "── Vector DB ──".bold());
    if has_vector {
        println!("  Status: {}", "enabled".green());
    } else {
        println!("  Status: {} (run `codexa configure` to enable)", "disabled".dimmed());
    }
    vector::check_size_and_warn(index.total_files, index.total_lines, has_vector);
    Ok(())
}

fn cmd_configure(path: &str) -> Result<()> {
    println!("{}", "── codexa configuration wizard ──\n".bold());
    let mut cfg = config::Config::interactive_create(path)?;
    cfg.configure_vector_wizard()?;
    cfg.save(path)?;
    println!("\n{} Saved to {}/codexa.toml", "✓".green(), path);
    if cfg.vector.is_some() {
        println!("  Run {} to build the vector index.", "codexa index".cyan().bold());
    } else {
        println!("  Run {} to generate context.", "codexa init".cyan().bold());
    }
    Ok(())
}

async fn cmd_index(path: &str, force: bool) -> Result<()> {
    let cfg = config::Config::load_or_default(path);
    let vcfg = cfg.vector.as_ref().ok_or_else(|| {
        anyhow::anyhow!("Vector DB not configured. Run `codexa configure` first.")
    })?;

    println!("{} {}", "→ Scanning project".cyan(), path.bold());
    let index = indexer::scan(path, &cfg)?;
    println!("  {} files, {} lines", index.total_files.to_string().green(), index.total_lines.to_string().green());

    let db_path = match &vcfg.backend {
        config::VectorBackend::None => anyhow::bail!("Vector backend is 'none'."),
        config::VectorBackend::Sqlite { path: p } => {
            let full = format!("{}/{}", path, p);
            if let Some(parent) = std::path::Path::new(&full).parent() {
                std::fs::create_dir_all(parent)?;
            }
            full
        }
        config::VectorBackend::Postgres { connection_string } => {
            return cmd_index_pg(path, connection_string, vcfg, &index, force).await;
        }
    };

    let mut emb_client = vector::EmbeddingClient::from_config(vcfg);
    let mut store = vector::SqliteStore::open(&db_path, emb_client.dimensions)?;

    if !force {
        let existing = { use vector::VectorStore; store.chunk_count()? };
        if existing > 0 {
            println!("  {} chunks already indexed. Use {} to re-index.", existing.to_string().yellow(), "--force".bold());
            return Ok(());
        }
    } else {
        use vector::VectorStore;
        store.clear()?;
        println!("  Cleared existing index.");
    }

    println!("{}", "→ Chunking source files…".cyan());
    let chunks = vector::chunk_project(path, &index, vcfg.chunk_size, vcfg.overlap, cfg.scan.max_file_size_kb);
    println!("  {} chunks created", chunks.len().to_string().green());
    if chunks.is_empty() {
        println!("{}", "  No source chunks to embed. Done.".dimmed());
        return Ok(());
    }

    println!("{}", "→ Generating embeddings…".cyan());
    let embedded = vector::embed_chunks(&mut emb_client, chunks, vcfg.batch_size).await?;

    println!("{}", "→ Writing to vector store…".cyan());
    { use vector::VectorStore; store.upsert(&embedded)?; }

    println!("{} {} chunks indexed to {}", "✓".green(), embedded.len().to_string().green().bold(), db_path.cyan());
    println!("\n{} Run {} to query the index.", "Tip:".yellow().bold(), "codexa search \"your query\"".cyan());
    Ok(())
}

#[cfg(feature = "postgres")]
async fn cmd_index_pg(
    project_root: &str,
    conn_str: &str,
    vcfg: &config::VectorConfig,
    index: &indexer::ProjectIndex,
    force: bool,
) -> Result<()> {
    let mut emb_client = vector::EmbeddingClient::from_config(vcfg);
    let store = vector::pg::PgStore::connect(conn_str, emb_client.dimensions).await?;

    if force { store.clear_async().await?; println!("  Cleared existing index."); }
    else {
        let n = store.chunk_count_async().await?;
        if n > 0 { println!("  {} chunks already indexed. Use --force.", n); return Ok(()); }
    }

    let chunks = vector::chunk_project(project_root, index, vcfg.chunk_size, vcfg.overlap, 512);
    let embedded = vector::embed_chunks(&mut emb_client, chunks, vcfg.batch_size).await?;
    store.upsert_async(&embedded).await?;
    println!("✓ {} chunks indexed to PostgreSQL", embedded.len());
    Ok(())
}

#[cfg(not(feature = "postgres"))]
async fn cmd_index_pg(_: &str, _: &str, _: &config::VectorConfig, _: &indexer::ProjectIndex, _: bool) -> Result<()> {
    anyhow::bail!("PostgreSQL not compiled in. Rebuild with: cargo build --features postgres")
}

async fn cmd_search(query: &str, path: &str, top_k: usize, as_json: bool) -> Result<()> {
    let cfg = config::Config::load_or_default(path);
    let vcfg = cfg.vector.as_ref().ok_or_else(|| {
        anyhow::anyhow!("Vector DB not configured. Run `codexa configure` first.")
    })?;

    let db_path = match &vcfg.backend {
        config::VectorBackend::Sqlite { path: p } => format!("{}/{}", path, p),
        config::VectorBackend::Postgres { connection_string } => {
            return cmd_search_pg(query, connection_string, vcfg, top_k, as_json).await;
        }
        config::VectorBackend::None => anyhow::bail!("Vector backend is 'none'."),
    };

    let mut emb_client = vector::EmbeddingClient::from_config(vcfg);
    let store = vector::SqliteStore::open(&db_path, emb_client.dimensions)?;

    let embeddings = emb_client.embed_batch(&[query]).await?;
    let query_vec = embeddings.into_iter().next().ok_or_else(|| anyhow::anyhow!("Empty embedding response"))?;

    use vector::VectorStore;
    let results = store.search(&query_vec, top_k)?;

    if results.is_empty() {
        println!("{}", "No results found. Run `codexa index` first.".yellow());
        return Ok(());
    }

    if as_json {
        println!("{}", serde_json::to_string_pretty(&results)?);
        return Ok(());
    }

    println!("{} {} result(s) for {}\n", "→".cyan(), results.len(), format!("\"{}\"", query).bold());
    for (i, r) in results.iter().enumerate() {
        println!(
            "{}  {} (lines {}-{})  score: {:.4}",
            format!("[{}]", i + 1).yellow().bold(),
            r.file_path.cyan(), r.start_line, r.end_line, r.score
        );
        println!("{}", "─".repeat(60).dimmed());
        for line in r.content.lines().take(8) { println!("  {}", line); }
        if r.content.lines().count() > 8 { println!("  {}", "...".dimmed()); }
        println!();
    }
    Ok(())
}

#[cfg(feature = "postgres")]
async fn cmd_search_pg(query: &str, conn_str: &str, vcfg: &config::VectorConfig, top_k: usize, as_json: bool) -> Result<()> {
    let mut emb_client = vector::EmbeddingClient::from_config(vcfg);
    let store = vector::pg::PgStore::connect(conn_str, emb_client.dimensions).await?;
    let embeddings = emb_client.embed_batch(&[query]).await?;
    let qv = embeddings.into_iter().next().unwrap();
    let results = store.search_async(&qv, top_k).await?;
    if as_json { println!("{}", serde_json::to_string_pretty(&results)?); }
    else {
        for (i, r) in results.iter().enumerate() {
            println!("[{}] {} lines {}-{} score={:.4}", i+1, r.file_path, r.start_line, r.end_line, r.score);
            for line in r.content.lines().take(6) { println!("  {}", line); }
            println!();
        }
    }
    Ok(())
}

#[cfg(not(feature = "postgres"))]
async fn cmd_search_pg(_: &str, _: &str, _: &config::VectorConfig, _: usize, _: bool) -> Result<()> {
    anyhow::bail!("PostgreSQL not compiled in. Rebuild with: cargo build --features postgres")
}
