use crate::config::Config;
use anyhow::Result;
use ignore::WalkBuilder;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ProjectIndex {
    pub root: String,
    pub total_files: usize,
    pub total_lines: usize,
    pub languages: HashMap<String, usize>,
    pub file_tree: Vec<FileNode>,
    pub entry_points: Vec<String>,
    pub key_files: Vec<KeyFile>,
    pub dependencies: Vec<String>,
    pub scanned_at: String,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct FileNode {
    pub path: String,
    pub size_bytes: u64,
    pub language: Option<String>,
    pub lines: Option<usize>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct KeyFile {
    pub path: String,
    pub reason: String,
    pub snippet: Option<String>,
}

pub fn scan(root: &str, cfg: &Config) -> Result<ProjectIndex> {
    let root_path = Path::new(root).canonicalize().unwrap_or_else(|_| Path::new(root).to_path_buf());

    let mut file_tree: Vec<FileNode> = Vec::new();
    let mut languages: HashMap<String, usize> = HashMap::new();
    let mut total_lines = 0usize;
    let mut key_files: Vec<KeyFile> = Vec::new();
    let mut entry_points: Vec<String> = Vec::new();

    let walker = WalkBuilder::new(&root_path)
        .hidden(false)
        .ignore(true)      // respects .gitignore
        .git_ignore(true)
        .max_depth(Some(cfg.scan.max_depth))
        .build();

    for result in walker {
        let entry = match result {
            Ok(e) => e,
            Err(_) => continue,
        };

        let path = entry.path();

        // skip excluded dirs
        if path.is_dir() {
            let dir_name = path.file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("");
            if cfg.scan.exclude_dirs.iter().any(|ex| ex == dir_name) {
                continue;
            }
            continue;
        }

        if !path.is_file() {
            continue;
        }

        let rel_path = path
            .strip_prefix(&root_path)
            .map(|p| p.to_string_lossy().to_string())
            .unwrap_or_else(|_| path.to_string_lossy().to_string());

        let ext = path.extension()
            .and_then(|e| e.to_str())
            .unwrap_or("")
            .to_lowercase();

        // extension filter
        if !cfg.scan.include_extensions.is_empty()
            && !cfg.scan.include_extensions.contains(&ext)
        {
            continue;
        }

        let size_bytes = path.metadata().map(|m| m.len()).unwrap_or(0);
        let language = detect_language(&ext, path);

        let lines = if size_bytes < cfg.scan.max_file_size_kb * 1024 {
            count_lines(path)
        } else {
            None
        };

        if let Some(l) = lines {
            total_lines += l;
        }

        if let Some(ref lang) = language {
            *languages.entry(lang.clone()).or_insert(0) += 1;
        }

        // detect entry points
        if is_entry_point(&rel_path, &ext) {
            entry_points.push(rel_path.clone());
        }

        // detect key files
        if let Some(reason) = is_key_file(&rel_path) {
            let snippet = if cfg.output.include_snippets && size_bytes < 32 * 1024 {
                read_snippet(path, cfg.output.snippet_lines)
            } else {
                None
            };
            key_files.push(KeyFile {
                path: rel_path.clone(),
                reason,
                snippet,
            });
        }

        file_tree.push(FileNode {
            path: rel_path,
            size_bytes,
            language,
            lines,
        });
    }

    // detect dependencies from manifest files
    let dependencies = detect_dependencies(&root_path);

    Ok(ProjectIndex {
        root: root_path.to_string_lossy().to_string(),
        total_files: file_tree.len(),
        total_lines,
        languages,
        file_tree,
        entry_points,
        key_files,
        dependencies,
        scanned_at: chrono::Utc::now().to_rfc3339(),
    })
}

fn detect_language(ext: &str, path: &Path) -> Option<String> {
    let filename = path.file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("");

    // special filenames
    match filename {
        "Makefile" | "makefile" | "GNUmakefile" => return Some("Makefile".into()),
        "Dockerfile" => return Some("Docker".into()),
        "docker-compose.yml" | "docker-compose.yaml" => return Some("Docker".into()),
        _ => {}
    }

    let lang = match ext {
        "rs"                        => "Rust",
        "ts" | "tsx"                => "TypeScript",
        "js" | "jsx" | "mjs"       => "JavaScript",
        "py"                        => "Python",
        "go"                        => "Go",
        "java"                      => "Java",
        "cs"                        => "C#",
        "cpp" | "cc" | "cxx"       => "C++",
        "c" | "h"                  => "C",
        "rb"                        => "Ruby",
        "php"                       => "PHP",
        "swift"                     => "Swift",
        "kt" | "kts"               => "Kotlin",
        "scala"                     => "Scala",
        "zig"                       => "Zig",
        "lua"                       => "Lua",
        "ex" | "exs"               => "Elixir",
        "erl"                       => "Erlang",
        "hs"                        => "Haskell",
        "ml" | "mli"               => "OCaml",
        "clj" | "cljs"             => "Clojure",
        "sh" | "bash" | "zsh"      => "Shell",
        "ps1"                       => "PowerShell",
        "sql"                       => "SQL",
        "html" | "htm"             => "HTML",
        "css" | "scss" | "sass"    => "CSS",
        "json"                      => "JSON",
        "toml"                      => "TOML",
        "yaml" | "yml"             => "YAML",
        "md" | "mdx"               => "Markdown",
        "proto"                     => "Protobuf",
        "graphql" | "gql"          => "GraphQL",
        "tf" | "tfvars"            => "Terraform",
        _                           => return None,
    };

    Some(lang.into())
}

fn is_entry_point(rel_path: &str, ext: &str) -> bool {
    let filename = Path::new(rel_path)
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("");

    matches!(
        filename,
        "main.rs" | "main.go" | "main.py" | "main.ts" | "main.js"
        | "index.ts" | "index.js" | "index.tsx"
        | "app.ts" | "app.js" | "app.tsx" | "app.py"
        | "server.ts" | "server.js" | "server.py"
        | "lib.rs" | "mod.rs"
    ) || (ext == "rs" && rel_path.starts_with("src/bin/"))
}

fn is_key_file(rel_path: &str) -> Option<String> {
    let filename = Path::new(rel_path)
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("");

    match filename {
        "Cargo.toml"        => Some("Rust manifest".into()),
        "package.json"      => Some("Node.js manifest".into()),
        "pyproject.toml"    => Some("Python project config".into()),
        "setup.py"          => Some("Python setup".into()),
        "go.mod"            => Some("Go module".into()),
        "pom.xml"           => Some("Maven project".into()),
        "build.gradle"      => Some("Gradle build".into()),
        "Makefile"          => Some("Build system".into()),
        "Dockerfile"        => Some("Container definition".into()),
        "docker-compose.yml" | "docker-compose.yaml" => Some("Docker Compose config".into()),
        "README.md" | "README.rst" => Some("Project documentation".into()),
        ".env.example"      => Some("Environment variables template".into()),
        "schema.sql" | "schema.prisma" => Some("Database schema".into()),
        "openapi.yml" | "openapi.yaml" | "swagger.yml" => Some("API specification".into()),
        _                   => None,
    }
}

fn count_lines(path: &Path) -> Option<usize> {
    std::fs::read_to_string(path)
        .ok()
        .map(|content| content.lines().count())
}

fn read_snippet(path: &Path, max_lines: usize) -> Option<String> {
    std::fs::read_to_string(path).ok().map(|content| {
        content
            .lines()
            .take(max_lines)
            .collect::<Vec<_>>()
            .join("\n")
    })
}

fn detect_dependencies(root: &Path) -> Vec<String> {
    let mut deps = Vec::new();

    // Cargo.toml
    if let Ok(content) = std::fs::read_to_string(root.join("Cargo.toml")) {
        if let Ok(parsed) = content.parse::<toml::Value>() {
            if let Some(table) = parsed.get("dependencies").and_then(|d| d.as_table()) {
                for key in table.keys().take(20) {
                    deps.push(format!("rust:{}", key));
                }
            }
        }
    }

    // package.json
    if let Ok(content) = std::fs::read_to_string(root.join("package.json")) {
        if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(&content) {
            if let Some(obj) = parsed.get("dependencies").and_then(|d| d.as_object()) {
                for key in obj.keys().take(20) {
                    deps.push(format!("npm:{}", key));
                }
            }
        }
    }

    // go.mod
    if let Ok(content) = std::fs::read_to_string(root.join("go.mod")) {
        for line in content.lines() {
            let line = line.trim();
            if line.starts_with("require") || line.is_empty() || line == ")" {
                continue;
            }
            if !line.starts_with("module") && !line.starts_with("go ") {
                let dep = line.split_whitespace().next().unwrap_or("").to_string();
                if !dep.is_empty() {
                    deps.push(format!("go:{}", dep));
                }
            }
        }
    }

    deps
}
