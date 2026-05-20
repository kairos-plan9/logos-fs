use std::path::PathBuf;
use std::sync::Arc;

use async_trait::async_trait;
use logos_vfs::{Namespace, VfsError};

use crate::proc::ProcTool;
#[cfg(feature = "sandbox")]
use crate::sandbox::SandboxNs;

/// ProcStore namespace — `logos://proc-store/`.
///
/// Filesystem-backed persistent storage for proc tool declarations.
/// Each tool gets a directory: `{root}/{tool_name}/schema.json`
pub struct ProcStoreNs {
    root: PathBuf,
}

impl ProcStoreNs {
    pub fn init(root: PathBuf) -> Result<Self, VfsError> {
        std::fs::create_dir_all(&root)
            .map_err(|e| VfsError::Io(format!("create proc-store dir: {e}")))?;
        Ok(Self { root })
    }

    #[cfg(feature = "sandbox")]
    /// Scan proc-store and create ExternalProcTool instances for each tool.
    ///
    /// For tools with a `git` field, clones/updates the repo in the system sandbox.
    pub async fn restore_tools(
        &self,
        sandbox: &Arc<SandboxNs>,
    ) -> Result<Vec<Arc<dyn ProcTool>>, VfsError> {
        let dir = std::fs::read_dir(&self.root)
            .map_err(|e| VfsError::Io(format!("read proc-store: {e}")))?;

        let mut tools: Vec<Arc<dyn ProcTool>> = Vec::new();

        for entry in dir.flatten() {
            if !entry.file_type().map(|t| t.is_dir()).unwrap_or(false) {
                continue;
            }
            let tool_name = entry.file_name().to_string_lossy().to_string();
            let schema_path = entry.path().join("schema.json");

            if !schema_path.exists() {
                continue;
            }

            let content = match std::fs::read_to_string(&schema_path) {
                Ok(c) => c,
                Err(_) => continue,
            };

            let schema: serde_json::Value = match serde_json::from_str(&content) {
                Ok(v) => v,
                Err(_) => continue,
            };

            let run_cmd = match schema["run"].as_str() {
                Some(r) => r.to_string(),
                None => continue, // no run command, skip
            };

            // If git field exists, clone or pull in the system sandbox
            if let Some(git_url) = schema["git"].as_str() {
                let sandbox_path = format!("logos://sandbox/__system__/proc/{tool_name}");
                // Check if already cloned by trying to list
                let exists = sandbox
                    .read(&["__system__", "proc", &tool_name])
                    .await
                    .is_ok();

                if !exists {
                    println!("[logos] cloning proc tool {tool_name} from {git_url}...");
                    let result = sandbox
                        .exec(&format!("git clone {git_url} {sandbox_path}"), "__system__", "__system__")
                        .await;
                    if let Err(e) = result {
                        eprintln!("[logos] failed to clone {tool_name}: {e}");
                        continue;
                    }
                } else {
                    // Pull latest
                    if let Err(e) = sandbox
                        .exec(&format!("cd {sandbox_path} && git pull --ff-only"), "__system__", "__system__")
                        .await
                    {
                        eprintln!("[logos] WARNING: git pull failed for {tool_name}: {e}");
                    }
                }
            }

            tools.push(Arc::new(ExternalProcTool {
                tool_name: tool_name.clone(),
                schema: schema.clone(),
                run_cmd,
                sandbox: Arc::clone(sandbox),
            }));

            println!("[logos] loaded external proc tool: {tool_name}");
        }

        Ok(tools)
    }
}

#[async_trait]
impl Namespace for ProcStoreNs {
    fn name(&self) -> &str {
        "proc-store"
    }

    async fn read(&self, path: &[&str]) -> Result<String, VfsError> {
        if path.is_empty() {
            let mut entries = Vec::new();
            let dir = std::fs::read_dir(&self.root)
                .map_err(|e| VfsError::Io(format!("read proc-store: {e}")))?;
            for entry in dir.flatten() {
                if let Some(name) = entry.file_name().to_str() {
                    entries.push(name.to_string());
                }
            }
            return Ok(serde_json::to_string(&entries).unwrap_or_else(|_| "[]".to_string()));
        }

        let file_path = self.root.join(path.join("/"));
        if file_path.is_dir() {
            let mut entries = Vec::new();
            let dir = std::fs::read_dir(&file_path)
                .map_err(|e| VfsError::Io(format!("read dir: {e}")))?;
            for entry in dir.flatten() {
                if let Some(name) = entry.file_name().to_str() {
                    entries.push(name.to_string());
                }
            }
            return Ok(serde_json::to_string(&entries).unwrap_or_else(|_| "[]".to_string()));
        }

        tokio::fs::read_to_string(&file_path)
            .await
            .map_err(|e| match e.kind() {
                std::io::ErrorKind::NotFound => {
                    VfsError::NotFound(format!("logos://proc-store/{}", path.join("/")))
                }
                _ => VfsError::Io(format!("read: {e}")),
            })
    }

    async fn write(&self, path: &[&str], content: &str) -> Result<(), VfsError> {
        if path.is_empty() {
            return Err(VfsError::InvalidPath("empty proc-store path".to_string()));
        }
        let file_path = self.root.join(path.join("/"));
        if let Some(parent) = file_path.parent() {
            tokio::fs::create_dir_all(parent)
                .await
                .map_err(|e| VfsError::Io(format!("mkdir: {e}")))?;
        }
        tokio::fs::write(&file_path, content)
            .await
            .map_err(|e| VfsError::Io(format!("write: {e}")))
    }

    async fn patch(&self, path: &[&str], partial: &str) -> Result<(), VfsError> {
        self.write(path, partial).await
    }
}

#[cfg(feature = "sandbox")]
/// An external proc tool backed by a git project in the sandbox.
///
/// Execution: writes params to stdin, runs the `run` command in the
/// tool's sandbox directory, returns stdout as result.
struct ExternalProcTool {
    tool_name: String,
    schema: serde_json::Value,
    run_cmd: String,
    sandbox: Arc<SandboxNs>,
}

#[cfg(feature = "sandbox")]
#[async_trait]
impl ProcTool for ExternalProcTool {
    fn name(&self) -> &str {
        &self.tool_name
    }

    fn schema(&self) -> serde_json::Value {
        self.schema.clone()
    }

    async fn call(&self, params: &str) -> Result<String, VfsError> {
        let tool_dir = format!("logos://sandbox/__system__/proc/{}", self.tool_name);

        // Write params to input.json
        self.sandbox
            .write(
                &["__system__", "proc", &self.tool_name, "input.json"],
                params,
            )
            .await?;

        // Execute the run command with input piped from input.json
        let command = format!(
            "cd {tool_dir} && {run_cmd} < {tool_dir}/input.json",
            tool_dir = tool_dir,
            run_cmd = self.run_cmd,
        );

        let result = self.sandbox.exec(&command, "__system__", "__system__").await?;

        if result.exit_code != 0 {
            return Err(VfsError::Io(format!(
                "proc tool {} exited with code {}: {}",
                self.tool_name, result.exit_code, result.stderr
            )));
        }

        Ok(result.stdout)
    }
}
