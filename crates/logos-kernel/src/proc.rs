use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use async_trait::async_trait;
use logos_vfs::{Namespace, VfsError};
use tokio::sync::Mutex;

use crate::memory_store::{CrystalMemoryStore, ToolMemoryStore};

/// A proc tool — stateless, callable via `logos_call`.
///
/// Built-in tools implement this trait directly in Rust.
/// External tools delegate to a git project process in the sandbox.
#[async_trait]
pub trait ProcTool: Send + Sync {
    fn name(&self) -> &str;
    fn schema(&self) -> serde_json::Value;
    async fn call(&self, params: &str) -> Result<String, VfsError>;
}

/// Per-call session slot (RFC 002 §3.2 call_id model).
struct CallSlot {
    tool_name: String,
    input: String,
    output: Option<String>,
    error: Option<String>,
    #[allow(dead_code)]
    created_at: Instant,
}

/// The Proc namespace — `logos://proc/`.
///
/// Provides tool discovery, call dispatch, and per-call session state.
///
/// URI routing:
///   logos://proc/                                → read: tool name list
///   logos://proc/{tool_name}                     → read: tool schema
///   logos://proc/{tool_name}/.schema             → read: tool schema
///   logos://proc/{tool_name}/{call_id}/input     → write: submit params and trigger execution
///   logos://proc/{tool_name}/{call_id}/output    → read: call result
///   logos://proc/{tool_name}/{call_id}/error     → read: call error (if any)
pub struct ProcNs {
    tools: HashMap<String, Arc<dyn ProcTool>>,
    call_slots: Mutex<HashMap<String, CallSlot>>,
    memory_slots: Mutex<HashMap<String, String>>,
    crystal_store: Option<Arc<CrystalMemoryStore>>,
    tool_memory_store: Option<Arc<ToolMemoryStore>>,
}

impl ProcNs {
    pub fn new() -> Self {
        Self {
            tools: HashMap::new(),
            call_slots: Mutex::new(HashMap::new()),
            memory_slots: Mutex::new(HashMap::new()),
            crystal_store: None,
            tool_memory_store: None,
        }
    }

    pub fn set_memory_stores(
        &mut self,
        crystal: Arc<CrystalMemoryStore>,
        tool_mem: Arc<ToolMemoryStore>,
    ) {
        self.crystal_store = Some(crystal);
        self.tool_memory_store = Some(tool_mem);
    }

    /// Register a tool. Overwrites if name already exists.
    pub fn register(&mut self, tool: Arc<dyn ProcTool>) {
        self.tools.insert(tool.name().to_string(), tool);
    }

    /// Dispatch a `logos_call` to the named tool (synchronous sugar).
    ///
    /// Internally generates a call_id, writes input, executes, returns output.
    pub async fn call(&self, tool_name: &str, params: &str) -> Result<String, VfsError> {
        let tool = self.tools.get(tool_name).ok_or_else(|| {
            VfsError::NotFound(format!("unknown proc tool: {tool_name}"))
        })?;
        tool.call(params).await
    }

    /// Execute a call via the call_id session model.
    async fn execute_call(&self, tool_name: &str, call_id: &str, input: &str) -> Result<(), VfsError> {
        let tool = self.tools.get(tool_name).ok_or_else(|| {
            VfsError::NotFound(format!("unknown proc tool: {tool_name}"))
        })?;

        let result = tool.call(input).await;

        let success = result.is_ok();
        let result_summary = match &result {
            Ok(o) => {
                let truncated = if o.len() > 200 { &o[..200] } else { o.as_str() };
                truncated.to_string()
            }
            Err(e) => format!("ERROR: {e}"),
        };

        let mut slots = self.call_slots.lock().await;
        let slot = slots.entry(format!("{tool_name}/{call_id}")).or_insert(CallSlot {
            tool_name: tool_name.to_string(),
            input: input.to_string(),
            output: None,
            error: None,
            created_at: Instant::now(),
        });
        slot.input = input.to_string();

        match result {
            Ok(output) => {
                slot.output = Some(output);
                slot.error = None;
            }
            Err(e) => {
                slot.output = None;
                slot.error = Some(e.to_string());
            }
        }

        // Lazy cleanup: remove slots older than 5 minutes
        let cutoff = Instant::now() - std::time::Duration::from_secs(300);
        slots.retain(|_, s| s.created_at > cutoff);
        drop(slots);

        if let Some(ref store) = self.tool_memory_store {
            let store = Arc::clone(store);
            let tool_name = tool_name.to_string();
            let params_preview = if input.len() > 200 { &input[..200] } else { input };
            let condition = format!("{tool_name}: {params_preview}");
            let bullets = if success {
                format!("- {tool_name} succeeded: {result_summary}")
            } else {
                format!("- AVOID: {tool_name} failed: {result_summary}")
            };
            tokio::spawn(async move {
                if let Err(e) = store.record(&tool_name, &condition, &bullets, success).await {
                    eprintln!("[tool-memory] record failed: {e}");
                }
            });
        }

        Ok(())
    }

    // memory/* VFS routing
    // write memory/{crystal,tool}/query/input → execute query, store result
    // read  memory/{crystal,tool}/query/output → return result

    async fn write_memory(&self, path: &[&str], content: &str) -> Result<(), VfsError> {
        if path.len() != 4 || path[3] != "input" {
            return Err(VfsError::InvalidPath(format!(
                "memory write expects memory/{{type}}/{{action}}/input, got: {}",
                path.join("/")
            )));
        }

        let (mem_type, action) = (path[1], path[2]);

        match (mem_type, action) {
            ("crystal", "query") => {
                let store = self.crystal_store.as_ref().ok_or_else(|| {
                    VfsError::Io("crystal memory not available".into())
                })?;
                let crystals = store.query(content, 10, 0.55).await?;
                let json = serde_json::to_string(&serde_json::json!({ "crystals": crystals }))
                    .map_err(|e| VfsError::Io(format!("serialize: {e}")))?;
                self.memory_slots.lock().await.insert("crystal/query".into(), json);
            }
            ("tool", "query") => {
                let store = self.tool_memory_store.as_ref().ok_or_else(|| {
                    VfsError::Io("tool memory not available".into())
                })?;
                let all_tools: Vec<String> = self.tools.keys().cloned().collect();
                let ranked = store.rank_tools(content, &all_tools, 0).await?;
                let json = serde_json::to_string(&serde_json::json!({ "ranked_tools": ranked }))
                    .map_err(|e| VfsError::Io(format!("serialize: {e}")))?;
                self.memory_slots.lock().await.insert("tool/query".into(), json);
            }
            ("tool", "record") => {
                let store = self.tool_memory_store.as_ref().ok_or_else(|| {
                    VfsError::Io("tool memory not available".into())
                })?;
                #[derive(serde::Deserialize)]
                struct RecordInput {
                    tool_name: String,
                    condition: String,
                    bullets: String,
                    success: bool,
                }
                let input: RecordInput = serde_json::from_str(content)
                    .map_err(|e| VfsError::InvalidJson(format!("tool record: {e}")))?;
                store.record(&input.tool_name, &input.condition, &input.bullets, input.success).await?;
                self.memory_slots.lock().await.insert("tool/record".into(), r#"{"ok":true}"#.into());
            }
            _ => {
                return Err(VfsError::InvalidPath(format!(
                    "unknown memory path: {}/{}",
                    mem_type, action
                )));
            }
        }

        Ok(())
    }

    async fn read_memory(&self, path: &[&str]) -> Result<String, VfsError> {
        if path.len() != 4 || path[3] != "output" {
            return Err(VfsError::InvalidPath(format!(
                "memory read expects memory/{{type}}/{{action}}/output, got: {}",
                path.join("/")
            )));
        }

        let key = format!("{}/{}", path[1], path[2]);
        let mut slots = self.memory_slots.lock().await;
        slots.remove(&key).ok_or_else(|| {
            VfsError::NotFound(format!("no memory result for {key}"))
        })
    }
}

#[async_trait]
impl Namespace for ProcNs {
    fn name(&self) -> &str {
        "proc"
    }

    async fn read(&self, path: &[&str]) -> Result<String, VfsError> {
        // memory/* routing
        if path.first() == Some(&"memory") {
            return self.read_memory(path).await;
        }

        match path.len() {
            // logos://proc/ → list all tools
            0 => {
                let names: Vec<&str> = self.tools.keys().map(|s| s.as_str()).collect();
                Ok(serde_json::to_string(&names).unwrap_or_else(|_| "[]".to_string()))
            }
            // logos://proc/{tool_name} → schema
            1 => {
                let tool = self.tools.get(path[0]).ok_or_else(|| {
                    VfsError::NotFound(format!("unknown tool: {}", path[0]))
                })?;
                Ok(tool.schema().to_string())
            }
            // logos://proc/{tool_name}/.schema → schema
            2 if path[1] == ".schema" => {
                let tool = self.tools.get(path[0]).ok_or_else(|| {
                    VfsError::NotFound(format!("unknown tool: {}", path[0]))
                })?;
                Ok(tool.schema().to_string())
            }
            // logos://proc/{tool_name}/{call_id}/output → read result, then cleanup
            3 if path[2] == "output" => {
                let key = format!("{}/{}", path[0], path[1]);
                let mut slots = self.call_slots.lock().await;
                let slot = slots.get(&key).ok_or_else(|| {
                    VfsError::NotFound(format!("call not found: {key}"))
                })?;
                let result = match &slot.output {
                    Some(o) => Ok(o.clone()),
                    None => Err(VfsError::NotFound(format!("call {key}: no output yet"))),
                };
                // RFC: cleanup slot after output is read
                if result.is_ok() {
                    slots.remove(&key);
                }
                result
            }
            // logos://proc/{tool_name}/{call_id}/error → read error
            3 if path[2] == "error" => {
                let key = format!("{}/{}", path[0], path[1]);
                let slots = self.call_slots.lock().await;
                let slot = slots.get(&key).ok_or_else(|| {
                    VfsError::NotFound(format!("call not found: {key}"))
                })?;
                match &slot.error {
                    Some(e) => Ok(e.clone()),
                    None => Ok("null".to_string()),
                }
            }
            // logos://proc/{tool_name}/{call_id}/input → read back input
            3 if path[2] == "input" => {
                let key = format!("{}/{}", path[0], path[1]);
                let slots = self.call_slots.lock().await;
                let slot = slots.get(&key).ok_or_else(|| {
                    VfsError::NotFound(format!("call not found: {key}"))
                })?;
                Ok(slot.input.clone())
            }
            _ => Err(VfsError::InvalidPath(format!(
                "unexpected proc path: {}",
                path.join("/")
            ))),
        }
    }

    async fn write(&self, path: &[&str], content: &str) -> Result<(), VfsError> {
        // memory/* routing
        if path.first() == Some(&"memory") {
            return self.write_memory(path, content).await;
        }

        // logos://proc/{tool_name}/{call_id}/input → submit and execute
        if path.len() == 3 && path[2] == "input" {
            return self.execute_call(path[0], path[1], content).await;
        }
        Err(VfsError::InvalidPath(format!(
            "proc write only supports .../{{call_id}}/input: {}",
            path.join("/")
        )))
    }

    async fn patch(&self, path: &[&str], _partial: &str) -> Result<(), VfsError> {
        Err(VfsError::InvalidPath(format!(
            "proc does not support patch: {}",
            path.join("/")
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct DummyTool;

    #[async_trait]
    impl ProcTool for DummyTool {
        fn name(&self) -> &str { "test.echo" }
        fn schema(&self) -> serde_json::Value {
            serde_json::json!({
                "name": "test.echo",
                "description": "Echo input back",
                "parameters": { "type": "object", "properties": { "msg": { "type": "string" } } }
            })
        }
        async fn call(&self, params: &str) -> Result<String, VfsError> {
            Ok(params.to_string())
        }
    }

    #[tokio::test]
    async fn list_tools() {
        let mut ns = ProcNs::new();
        ns.register(Arc::new(DummyTool));
        let list = ns.read(&[]).await.unwrap();
        assert!(list.contains("test.echo"));
    }

    #[tokio::test]
    async fn read_schema() {
        let mut ns = ProcNs::new();
        ns.register(Arc::new(DummyTool));
        let schema = ns.read(&["test.echo", ".schema"]).await.unwrap();
        assert!(schema.contains("Echo input back"));
    }

    #[tokio::test]
    async fn call_tool() {
        let mut ns = ProcNs::new();
        ns.register(Arc::new(DummyTool));
        let result = ns.call("test.echo", r#"{"msg":"hello"}"#).await.unwrap();
        assert_eq!(result, r#"{"msg":"hello"}"#);
    }

    #[tokio::test]
    async fn unknown_tool() {
        let ns = ProcNs::new();
        assert!(ns.call("nope", "{}").await.is_err());
    }

    #[tokio::test]
    async fn call_id_session() {
        let mut ns = ProcNs::new();
        ns.register(Arc::new(DummyTool));

        // Write input triggers execution
        ns.write(&["test.echo", "call-001", "input"], r#"{"msg":"hi"}"#)
            .await
            .unwrap();

        // Read error before output (output read triggers cleanup)
        let error = ns.read(&["test.echo", "call-001", "error"]).await.unwrap();
        assert_eq!(error, "null");

        // Read input back
        let input = ns.read(&["test.echo", "call-001", "input"]).await.unwrap();
        assert_eq!(input, r#"{"msg":"hi"}"#);

        // Read output (triggers slot cleanup per RFC)
        let output = ns.read(&["test.echo", "call-001", "output"]).await.unwrap();
        assert_eq!(output, r#"{"msg":"hi"}"#);

        // Slot should be cleaned up — reading again should fail
        assert!(ns.read(&["test.echo", "call-001", "output"]).await.is_err());
    }

    // --- Integration tests (require Milvus + Ollama) ---

    async fn setup_ns_with_memory() -> ProcNs {
        let client = milvus::client::Client::new("http://localhost:19530").await.unwrap();
        let client = Arc::new(client);
        let embedder = Arc::new(crate::embedder::OllamaEmbedder::from_env());

        let crystal_store = Arc::new(crate::memory_store::CrystalMemoryStore::new(
            Arc::clone(&client), Arc::clone(&embedder),
        ));
        let tool_store = Arc::new(crate::memory_store::ToolMemoryStore::new(
            Arc::clone(&client), Arc::clone(&embedder),
        ));

        crystal_store.seed(&[crate::memory_store::CrystalSeed {
            id: "test_alpine".into(),
            condition: "When installing packages in Alpine Linux".into(),
            label: "Alpine packages".into(),
            bullets: "- Use apk add, not apt-get.\n- AVOID: apt-get silently fails.".into(),
        }]).await.unwrap();

        let mut ns = ProcNs::new();
        ns.register(Arc::new(DummyTool));
        ns.set_memory_stores(crystal_store, tool_store);
        ns
    }

    #[tokio::test]
    #[ignore] // requires Milvus on :19530 + Ollama
    async fn crystal_query_roundtrip() {
        let ns = setup_ns_with_memory().await;

        ns.write(&["memory", "crystal", "query", "input"], "install packages on alpine linux")
            .await
            .unwrap();

        let result = ns.read(&["memory", "crystal", "query", "output"]).await.unwrap();
        let data: serde_json::Value = serde_json::from_str(&result).unwrap();
        let crystals = data["crystals"].as_array().unwrap();
        assert!(!crystals.is_empty(), "should find alpine crystal");
        assert!(crystals[0]["label"].as_str().unwrap().contains("Alpine"));
        println!("crystal query result: {result}");
    }

    #[tokio::test]
    #[ignore] // requires Milvus on :19530 + Ollama
    async fn tool_memory_record_and_rank() {
        let ns = setup_ns_with_memory().await;

        ns.write(
            &["memory", "tool", "record", "input"],
            r#"{"tool_name":"test.echo","condition":"echo test in alpine","bullets":"- echo works fine","success":true}"#,
        ).await.unwrap();

        let ack = ns.read(&["memory", "tool", "record", "output"]).await.unwrap();
        assert!(ack.contains("ok"));

        // small delay for milvus to index
        tokio::time::sleep(std::time::Duration::from_secs(2)).await;

        ns.write(&["memory", "tool", "query", "input"], "echo test in alpine")
            .await
            .unwrap();

        let result = ns.read(&["memory", "tool", "query", "output"]).await.unwrap();
        let data: serde_json::Value = serde_json::from_str(&result).unwrap();
        let ranked = data["ranked_tools"].as_array().unwrap();
        assert!(!ranked.is_empty());
        println!("tool ranking result: {result}");
    }

    #[tokio::test]
    #[ignore] // requires Milvus on :19530 + Ollama
    async fn tool_call_auto_records_memory() {
        let ns = setup_ns_with_memory().await;

        ns.write(&["test.echo", "auto-001", "input"], r#"{"msg":"hello"}"#)
            .await
            .unwrap();

        let output = ns.read(&["test.echo", "auto-001", "output"]).await.unwrap();
        assert_eq!(output, r#"{"msg":"hello"}"#);

        // wait for async recording
        tokio::time::sleep(std::time::Duration::from_secs(3)).await;

        ns.write(&["memory", "tool", "query", "input"], "echo hello")
            .await
            .unwrap();

        let result = ns.read(&["memory", "tool", "query", "output"]).await.unwrap();
        println!("auto-recorded tool memory: {result}");
        let data: serde_json::Value = serde_json::from_str(&result).unwrap();
        let ranked = data["ranked_tools"].as_array().unwrap();
        let echo_tool = ranked.iter().find(|r| r["tool_name"] == "test.echo");
        assert!(echo_tool.is_some(), "test.echo should appear in rankings");
    }
}
