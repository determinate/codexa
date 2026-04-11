//! Vector store layer for codexa.
//!
//! Splits code files into semantic chunks (by function/class/impl boundaries),
//! generates embeddings via OpenAI-compatible API or Ollama, and stores them
//! in either SQLite (sqlite-vec) or PostgreSQL (pgvector).
//!
//! ## Flow
//!
//! ```text
//! FileNode → chunker::smart_chunk() → [CodeChunk]   ← by fn/class/blank-line
//!                                          ↓
//!                                   embed_chunks()  ← OpenAI / Ollama
//!                                          ↓
//!                                   VectorStore::upsert()
//!                                          ↓
//!                              SQLiteStore  /  PgStore
//! ```

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

use crate::chunker::smart_chunk;
use crate::config::{EmbeddingProvider, VectorBackend, VectorConfig};
use crate::indexer::ProjectIndex;
use ndarray::Array2;
use ort::session::Session;
use ort::value::Value;
use tokenizers::Tokenizer;
use colored::Colorize;

// ── Data types ────────────────────────────────────────────────────────────────

/// One indexed chunk of source code.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeChunk {
    /// Relative file path
    pub file_path: String,
    /// Start line (1-based)
    pub start_line: usize,
    /// End line (1-based, inclusive)
    pub end_line: usize,
    /// Raw text content
    pub content: String,
    /// Language tag (e.g. "Rust")
    pub language: Option<String>,
}

/// A chunk enriched with its embedding vector.
#[derive(Debug, Clone)]
pub struct EmbeddedChunk {
    pub chunk: CodeChunk,
    pub embedding: Vec<f32>,
}

/// A search result returned from the vector store.
#[derive(Debug, Clone, Serialize)]
pub struct SearchResult {
    pub file_path: String,
    pub start_line: usize,
    pub end_line: usize,
    pub content: String,
    pub score: f32,
}

// ── Chunking ─────────────────────────────────────────────────────────────────

/// Chunk all indexable files using smart semantic boundaries.
///
/// Each file is split by function/class/impl blocks (language-aware).
/// If a single block exceeds `max_chunk_lines`, it is sub-split further.
/// Falls back to blank-line-snapped windows for unknown languages.
///
/// Large files (> `max_file_kb` KB) are skipped with a warning.
pub fn chunk_project(
    root: &str,
    index: &ProjectIndex,
    max_chunk_lines: usize,
    overlap: usize,
    max_file_kb: u64,
) -> Vec<CodeChunk> {
    let mut all_chunks = Vec::new();

    for node in &index.file_tree {
        if node.language.is_none() {
            continue;
        }
        if node.size_bytes > max_file_kb * 1024 {
            eprintln!(
                "  skip (too large): {} ({} KB)",
                node.path,
                node.size_bytes / 1024
            );
            continue;
        }

        let full_path = format!("{}/{}", root, node.path);
        if let Ok(content) = std::fs::read_to_string(&full_path) {
            // target_lines = 2/3 of max so chunks have breathing room
            let target = (max_chunk_lines * 2 / 3).max(20);
            let chunks = smart_chunk(
                &node.path,
                node.language.as_deref(),
                &content,
                max_chunk_lines,
                target,
                overlap,
            );
            all_chunks.extend(chunks);
        }
    }

    all_chunks
}

// ── Embedding client ──────────────────────────────────────────────────────────

pub struct EmbeddingClient {
    provider: EmbeddingProvider,
    api_base: String,
    api_key: Option<String>,
    model: String,
    pub dimensions: usize,
    onnx_client: Option<OnnxClient>,
}

struct OnnxClient {
    session: Session,
    tokenizer: Tokenizer,
}

impl EmbeddingClient {
    pub fn from_config(vcfg: &VectorConfig) -> Self {
        match &vcfg.embedding {
            EmbeddingProvider::OpenAI { api_key, model } => EmbeddingClient {
                provider: vcfg.embedding.clone(),
                api_base: "https://api.openai.com/v1".into(),
                api_key: Some(api_key.clone()),
                model: model.clone(),
                dimensions: 1536,
                onnx_client: None,
            },
            EmbeddingProvider::OpenAICompatible {
                api_base,
                api_key,
                model,
                dimensions,
            } => EmbeddingClient {
                provider: vcfg.embedding.clone(),
                api_base: api_base.clone(),
                api_key: api_key.clone(),
                model: model.clone(),
                dimensions: *dimensions,
                onnx_client: None,
            },
            EmbeddingProvider::Ollama {
                base_url,
                model,
                dimensions,
            } => EmbeddingClient {
                provider: vcfg.embedding.clone(),
                api_base: base_url.clone(),
                api_key: None,
                model: model.clone(),
                dimensions: *dimensions,
                onnx_client: None,
            },
            EmbeddingProvider::LocalOnnx {
                model_path,
                tokenizer_path,
                dimensions,
            } => {
                let onnx_client = match OnnxClient::new(model_path, tokenizer_path) {
                    Ok(client) => Some(client),
                    Err(e) => {
                        eprintln!(
                            "  {} Failed to initialize Local ONNX client: {}",
                            "error:".red().bold(),
                            e
                        );
                        None
                    }
                };
                EmbeddingClient {
                    provider: vcfg.embedding.clone(),
                    api_base: "".into(),
                    api_key: None,
                    model: "local".into(),
                    dimensions: *dimensions,
                    onnx_client,
                }
            }
        }
    }

    /// Embed a batch of texts. Returns one Vec<f32> per input text.
    pub async fn embed_batch(&mut self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        match &self.provider {
            EmbeddingProvider::Ollama { .. } => self.embed_ollama(texts).await,
            EmbeddingProvider::LocalOnnx { .. } => self.embed_onnx_local(texts),
            _ => self.embed_openai_compat(texts).await,
        }
    }

    async fn embed_openai_compat(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        let client = reqwest::Client::new();

        #[derive(Serialize)]
        struct Req<'a> {
            model: &'a str,
            input: &'a [&'a str],
        }

        #[derive(Deserialize)]
        struct Resp {
            data: Vec<EmbData>,
        }
        #[derive(Deserialize)]
        struct EmbData {
            embedding: Vec<f32>,
        }

        let mut req = client
            .post(format!("{}/embeddings", self.api_base))
            .header("Content-Type", "application/json")
            .json(&Req { model: &self.model, input: texts });

        if let Some(ref key) = self.api_key {
            req = req.header("Authorization", format!("Bearer {}", key));
        }

        let resp: Resp = req.send().await?.json().await
            .context("Failed to parse embedding response")?;

        Ok(resp.data.into_iter().map(|d| d.embedding).collect())
    }

    async fn embed_ollama(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        // Ollama embed endpoint: POST /api/embed  { model, input: [..] }
        let client = reqwest::Client::new();

        #[derive(Serialize)]
        struct Req<'a> {
            model: &'a str,
            input: &'a [&'a str],
        }

        #[derive(Deserialize)]
        struct Resp {
            embeddings: Vec<Vec<f32>>,
        }

        let url = format!("{}/api/embed", self.api_base);
        let resp: Resp = client
            .post(&url)
            .json(&Req { model: &self.model, input: texts })
            .send()
            .await?
            .json()
            .await
            .context("Failed to parse Ollama embed response — is Ollama running?")?;

        Ok(resp.embeddings)
    }

    fn embed_onnx_local(&mut self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        let client = self.onnx_client.as_mut()
            .context("Local ONNX client failed to initialize (check model/tokenizer paths)")?;
        client.embed(texts)
    }
}

impl OnnxClient {
    pub fn new(model_path: &str, tokenizer_path: &str) -> Result<Self> {
        let tokenizer = Tokenizer::from_file(tokenizer_path)
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer from {}: {}", tokenizer_path, e))?;
        
        let session = Session::builder()
            .map_err(|e| anyhow::anyhow!("Failed to create SessionBuilder: {}", e))?
            .with_intra_threads(4)
            .map_err(|e| anyhow::anyhow!("Failed to set intra threads: {}", e))?
            .commit_from_file(model_path)
            .map_err(|e| anyhow::anyhow!("Failed to load model from {}: {}", model_path, e))?;

        Ok(OnnxClient { session, tokenizer })
    }

    pub fn embed(&mut self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        // 1. Tokenization
        let encodings = self.tokenizer.encode_batch(texts.to_vec(), true)
            .map_err(|e| anyhow::anyhow!("Tokenization failed: {}", e))?;

        if encodings.is_empty() { return Ok(vec![]); }

        let seq_len = encodings[0].get_ids().len();
        let batch_size = encodings.len();

        let mut input_ids = Vec::with_capacity(batch_size * seq_len);
        let mut attention_mask = Vec::with_capacity(batch_size * seq_len);
        let mut token_type_ids = Vec::with_capacity(batch_size * seq_len);

        for enc in &encodings {
            input_ids.extend(enc.get_ids().iter().map(|&id| id as i64));
            attention_mask.extend(enc.get_attention_mask().iter().map(|&id| id as i64));
            token_type_ids.extend(enc.get_type_ids().iter().map(|&id| id as i64));
        }

        // 2. Inference
        let input_ids_tensor = Array2::from_shape_vec((batch_size, seq_len), input_ids)?;
        let attention_mask_tensor = Array2::from_shape_vec((batch_size, seq_len), attention_mask)?;
        let token_type_ids_tensor = Array2::from_shape_vec((batch_size, seq_len), token_type_ids)?;

        let input_ids_value = Value::from_array(input_ids_tensor)
            .map_err(|e| anyhow::anyhow!("Failed to create input_ids Value: {}", e))?;
        let attention_mask_value = Value::from_array(attention_mask_tensor)
            .map_err(|e| anyhow::anyhow!("Failed to create attention_mask Value: {}", e))?;
        let token_type_ids_value = Value::from_array(token_type_ids_tensor)
            .map_err(|e| anyhow::anyhow!("Failed to create token_type_ids Value: {}", e))?;

        let inputs = ort::inputs![
            "input_ids" => &input_ids_value,
            "attention_mask" => &attention_mask_value,
            "token_type_ids" => &token_type_ids_value,
        ];

        let outputs = self.session.run(inputs)
            .map_err(|e| anyhow::anyhow!("ONNX inference failed: {}", e))?;
        let last_hidden_state = outputs["last_hidden_state"]
            .try_extract_array::<f32>()
            .map_err(|e| anyhow::anyhow!("Failed to extract output tensor: {}", e))?;

        // 3. Mean Pooling
        // last_hidden_state shape: (batch_size, seq_len, hidden_size)
        // attention_mask shape: (batch_size, seq_len)
        let hidden_size = last_hidden_state.shape()[2];
        let mut result_embeddings = Vec::with_capacity(batch_size);

        for b in 0..batch_size {
            let mut sum = vec![0.0f32; hidden_size];
            let mut count = 0.0f32;

            for s in 0..seq_len {
                let mask = encodings[b].get_attention_mask()[s] as f32;
                if mask > 0.0 {
                    for h in 0..hidden_size {
                        sum[h] += last_hidden_state[[b, s, h]];
                    }
                    count += 1.0;
                }
            }

            // Average and Normalize
            if count > 0.0 {
                for h in 0..hidden_size {
                    sum[h] /= count;
                }
            }

            // L2 Normalization
            let norm = sum.iter().map(|v| v * v).sum::<f32>().sqrt();
            if norm > 0.0 {
                for h in 0..hidden_size {
                    sum[h] /= norm;
                }
            }

            result_embeddings.push(sum);
        }

        Ok(result_embeddings)
    }
}

/// Embed all chunks in batches.
pub async fn embed_chunks(
    client: &mut EmbeddingClient,
    chunks: Vec<CodeChunk>,
    batch_size: usize,
) -> Result<Vec<EmbeddedChunk>> {
    let mut result = Vec::with_capacity(chunks.len());

    for batch in chunks.chunks(batch_size) {
        let texts: Vec<&str> = batch.iter().map(|c| c.content.as_str()).collect();
        let embeddings = client.embed_batch(&texts).await?;

        for (chunk, embedding) in batch.iter().zip(embeddings) {
            result.push(EmbeddedChunk {
                chunk: chunk.clone(),
                embedding,
            });
        }
    }

    Ok(result)
}

// ── Vector store trait ────────────────────────────────────────────────────────

/// Generic interface — implemented by SQLiteStore and PgStore.
pub trait VectorStore: Send {
    fn upsert(&mut self, chunks: &[EmbeddedChunk]) -> Result<()>;
    fn search(&self, query_vec: &[f32], top_k: usize) -> Result<Vec<SearchResult>>;
    fn chunk_count(&self) -> Result<usize>;
    fn clear(&mut self) -> Result<()>;
}

// ── SQLite backend (sqlite-vec) ────────────────────────────────────────────────

pub struct SqliteStore {
    conn: rusqlite::Connection,
    _dimensions: usize,
}

impl SqliteStore {
    /// Open (or create) a SQLite vector store at `db_path`.
    pub fn open(db_path: &str, dimensions: usize) -> Result<Self> {
        // Register sqlite-vec extension globally.
        // This must be done before opening any connections.
        static INIT: std::sync::Once = std::sync::Once::new();
        INIT.call_once(|| {
            unsafe {
                rusqlite::ffi::sqlite3_auto_extension(Some(std::mem::transmute(
                    sqlite_vec::sqlite3_vec_init as *const (),
                )));
            }
        });

        let conn = rusqlite::Connection::open(db_path)
            .context("Cannot open SQLite database")?;

        // metadata table
        conn.execute_batch("
            CREATE TABLE IF NOT EXISTS codexa_chunks (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                file_path   TEXT NOT NULL,
                start_line  INTEGER NOT NULL,
                end_line    INTEGER NOT NULL,
                language    TEXT,
                content     TEXT NOT NULL
            );
        ")?;

        // virtual vector table — dimensions are fixed at creation
        conn.execute_batch(&format!("
            CREATE VIRTUAL TABLE IF NOT EXISTS codexa_vec
            USING vec0(embedding float[{dimensions}]);
        "))?;

        Ok(SqliteStore { conn, _dimensions: dimensions })
    }
}

impl VectorStore for SqliteStore {
    fn upsert(&mut self, chunks: &[EmbeddedChunk]) -> Result<()> {
        let tx = self.conn.transaction()?;

        for ec in chunks {
            // insert metadata
            tx.execute(
                "INSERT INTO codexa_chunks (file_path, start_line, end_line, language, content)
                 VALUES (?1, ?2, ?3, ?4, ?5)",
                rusqlite::params![
                    ec.chunk.file_path,
                    ec.chunk.start_line,
                    ec.chunk.end_line,
                    ec.chunk.language,
                    ec.chunk.content,
                ],
            )?;

            let rowid = tx.last_insert_rowid();

            // insert vector — sqlite-vec expects raw f32 bytes
            let bytes: Vec<u8> = ec.embedding.iter()
                .flat_map(|f| f.to_le_bytes())
                .collect();

            tx.execute(
                "INSERT INTO codexa_vec (rowid, embedding) VALUES (?1, ?2)",
                rusqlite::params![rowid, bytes],
            )?;
        }

        tx.commit()?;
        Ok(())
    }

    fn search(&self, query_vec: &[f32], top_k: usize) -> Result<Vec<SearchResult>> {
        let bytes: Vec<u8> = query_vec.iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();

        let mut stmt = self.conn.prepare(
            "SELECT c.file_path, c.start_line, c.end_line, c.content, v.distance
             FROM codexa_vec v
             JOIN codexa_chunks c ON c.id = v.rowid
             WHERE v.embedding MATCH ?1
             ORDER BY v.distance
             LIMIT ?2",
        )?;

        let results = stmt.query_map(rusqlite::params![bytes, top_k as i64], |row| {
            Ok(SearchResult {
                file_path: row.get(0)?,
                start_line: row.get::<_, usize>(1)?,
                end_line: row.get::<_, usize>(2)?,
                content: row.get(3)?,
                score: row.get::<_, f32>(4)?,
            })
        })?
        .filter_map(|r| r.ok())
        .collect();

        Ok(results)
    }

    fn chunk_count(&self) -> Result<usize> {
        let n: usize = self.conn
            .query_row("SELECT COUNT(*) FROM codexa_chunks", [], |r| r.get(0))?;
        Ok(n)
    }

    fn clear(&mut self) -> Result<()> {
        self.conn.execute_batch(
            "DELETE FROM codexa_chunks; DELETE FROM codexa_vec;",
        )?;
        Ok(())
    }
}

// ── PostgreSQL backend (pgvector) ─────────────────────────────────────────────
//
// Compiled only when `--features postgres` is passed.
// Uses tokio-postgres + pgvector crate.

#[cfg(feature = "postgres")]
pub mod pg {
    use super::*;
    use pgvector::Vector;
    use tokio_postgres::{Client, NoTls};

    pub struct PgStore {
        client: Client,
        dimensions: usize,
    }

    impl PgStore {
        pub async fn connect(conn_str: &str, dimensions: usize) -> Result<Self> {
            let (client, connection) = tokio_postgres::connect(conn_str, NoTls)
                .await
                .context("Cannot connect to PostgreSQL")?;

            // drive the connection in a background task
            tokio::spawn(async move {
                if let Err(e) = connection.await {
                    eprintln!("pg connection error: {e}");
                }
            });

            // bootstrap schema
            client.execute("CREATE EXTENSION IF NOT EXISTS vector", &[]).await?;

            client.execute(&format!(
                "CREATE TABLE IF NOT EXISTS codexa_chunks (
                    id          BIGSERIAL PRIMARY KEY,
                    file_path   TEXT NOT NULL,
                    start_line  INT  NOT NULL,
                    end_line    INT  NOT NULL,
                    language    TEXT,
                    content     TEXT NOT NULL,
                    embedding   vector({dimensions})
                )"
            ), &[]).await?;

            client.execute(
                "CREATE INDEX IF NOT EXISTS codexa_chunks_vec_idx
                 ON codexa_chunks USING hnsw (embedding vector_cosine_ops)",
                &[],
            ).await?;

            Ok(PgStore { client, dimensions })
        }
    }

    impl PgStore {
        pub async fn upsert_async(&self, chunks: &[EmbeddedChunk]) -> Result<()> {
            for ec in chunks {
                let vec = Vector::from(ec.embedding.clone());
                self.client.execute(
                    "INSERT INTO codexa_chunks
                     (file_path, start_line, end_line, language, content, embedding)
                     VALUES ($1,$2,$3,$4,$5,$6)",
                    &[
                        &ec.chunk.file_path,
                        &(ec.chunk.start_line as i32),
                        &(ec.chunk.end_line as i32),
                        &ec.chunk.language,
                        &ec.chunk.content,
                        &vec,
                    ],
                ).await?;
            }
            Ok(())
        }

        pub async fn search_async(&self, query_vec: &[f32], top_k: usize) -> Result<Vec<SearchResult>> {
            let vec = Vector::from(query_vec.to_vec());
            let rows = self.client.query(
                "SELECT file_path, start_line, end_line, content,
                        (embedding <=> $1)::float4 AS score
                 FROM codexa_chunks
                 ORDER BY embedding <=> $1
                 LIMIT $2",
                &[&vec, &(top_k as i64)],
            ).await?;

            Ok(rows.iter().map(|r| SearchResult {
                file_path: r.get(0),
                start_line: r.get::<_, i32>(1) as usize,
                end_line: r.get::<_, i32>(2) as usize,
                content: r.get(3),
                score: r.get(4),
            }).collect())
        }

        pub async fn clear_async(&self) -> Result<()> {
            self.client.execute("TRUNCATE codexa_chunks", &[]).await?;
            Ok(())
        }

        pub async fn chunk_count_async(&self) -> Result<usize> {
            let row = self.client
                .query_one("SELECT COUNT(*)::bigint FROM codexa_chunks", &[])
                .await?;
            Ok(row.get::<_, i64>(0) as usize)
        }
    }
}

// ── Factory ───────────────────────────────────────────────────────────────────

/// Create the right store based on config, or return None if vector DB is disabled.
#[allow(dead_code)]
pub fn open_store(vcfg: &VectorConfig, dimensions: usize) -> Result<Option<Box<dyn VectorStore>>> {
    match &vcfg.backend {
        VectorBackend::None => Ok(None),
        VectorBackend::Sqlite { path } => {
            let store = SqliteStore::open(path, dimensions)?;
            Ok(Some(Box::new(store)))
        }
        VectorBackend::Postgres { .. } => {
            // Postgres store is async-only; caller should use pg::PgStore directly.
            // For the sync VectorStore trait we return an error hinting the user.
            anyhow::bail!(
                "PostgreSQL backend requires async mode. Use `codexa index --async` or feature flag `postgres`."
            )
        }
    }
}

// ── Large-project warning ─────────────────────────────────────────────────────

/// Warn if the project is large and no vector DB is configured.
pub fn check_size_and_warn(total_files: usize, total_lines: usize, has_vector_db: bool) {
    let large = total_files > 500 || total_lines > 50_000;
    if large && !has_vector_db {
        eprintln!();
        eprintln!("  ⚠  Large project detected ({total_files} files, {total_lines} lines).");
        eprintln!("     Re-scanning everything on each query will be slow.");
        eprintln!("     Enable a vector database in codexa.toml for fast semantic search:");
        eprintln!();
        eprintln!("     [vector]");
        eprintln!("     backend = \"sqlite\"          # or \"postgres\"");
        eprintln!("     db_path = \".codexa/index.db\"");
        eprintln!();
        eprintln!("     [vector.embedding]");
        eprintln!("     provider = \"ollama\"          # free, local");
        eprintln!("     model    = \"nomic-embed-text\"");
        eprintln!();
    }
}
