use std::sync::Arc;

use async_trait::async_trait;
use logos_vfs::{Namespace, VfsError};

use crate::proc::ProcTool;
#[cfg(feature = "sandbox")]
use crate::sandbox::SandboxNs;

/// system.search_tasks — multi-level experience retrieval (RFC 003 §6.2)
pub struct SystemSearchTasksTool {
    pub system: Arc<logos_system::SystemModule>,
}

#[async_trait]
impl ProcTool for SystemSearchTasksTool {
    fn name(&self) -> &str {
        "system.search_tasks"
    }
    fn schema(&self) -> serde_json::Value {
        serde_json::json!({
            "name": "system.search_tasks",
            "description": "Search tasks and anchors for relevant experience. Returns L1 (anchor facts) and L2 (task metadata) hits.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": { "type": "string", "description": "Search query (keyword or error snippet)" },
                    "limit": { "type": "integer", "default": 10 }
                },
                "required": ["query"]
            }
        })
    }
    async fn call(&self, params: &str) -> Result<String, VfsError> {
        let val: serde_json::Value =
            serde_json::from_str(params).map_err(|e| VfsError::InvalidJson(e.to_string()))?;
        let query = val["query"].as_str().unwrap_or_default();
        let limit = val["limit"].as_i64().unwrap_or(10);
        self.system.search_tasks(query, limit).await
    }
}

#[cfg(feature = "chat")]
/// system.get_context — auto-inject context for agent turn (RFC 003 §5.5)
pub struct SystemGetContextTool {
    pub mm: Arc<logos_mm::MemoryModule>,
    pub sessions: Arc<logos_mm::SessionStore>,
}

#[cfg(feature = "chat")]
#[async_trait]
impl ProcTool for SystemGetContextTool {
    fn name(&self) -> &str {
        "system.get_context"
    }
    fn schema(&self) -> serde_json::Value {
        serde_json::json!({
            "name": "system.get_context",
            "description": "Assemble context for an agent turn: session + recent summary + sender persona.",
            "parameters": {
                "type": "object",
                "properties": {
                    "chat_id": { "type": "string", "description": "Group chat ID" },
                    "sender_uid": { "type": "string", "description": "Message sender's user ID" },
                    "msg_id": { "type": "integer", "description": "Trigger message ID (optional)" }
                },
                "required": ["chat_id", "sender_uid"]
            }
        })
    }
    async fn call(&self, params: &str) -> Result<String, VfsError> {
        let val: serde_json::Value =
            serde_json::from_str(params).map_err(|e| VfsError::InvalidJson(e.to_string()))?;
        let chat_id = val["chat_id"].as_str().unwrap_or_default();
        let sender_uid = val["sender_uid"].as_str().unwrap_or_default();
        let msg_id = val["msg_id"].as_i64();

        // 1. Session from clustering module
        let session = if let Some(mid) = msg_id {
            self.sessions.get_session_for_msg(mid).await.map(|s| s.to_json())
        } else {
            self.sessions.get_active_session(chat_id).await.map(|s| s.to_json())
        }
        .unwrap_or(serde_json::json!(null));

        // 2. Latest short summary
        let summary = self.mm
            .read(&["groups", chat_id, "summary", "short", "latest"])
            .await
            .unwrap_or_else(|_| "null".to_string());

        // 3+4. Sender persona paths (injected by runtime)
        let context = serde_json::json!({
            "session": session,
            "recent_summary": serde_json::from_str::<serde_json::Value>(&summary)
                .unwrap_or(serde_json::json!(null)),
            "sender_uid": sender_uid,
            "persona_paths": {
                "long": format!("logos://users/{sender_uid}/persona/long.md"),
                "mid": format!("logos://users/{sender_uid}/persona/mid/"),
            }
        });
        Ok(context.to_string())
    }
}

#[cfg(feature = "sandbox")]
/// system.complete — turn terminator (RFC 002 §9.1)
///
/// Finishes, sleeps, or resumes a task. Optionally creates an anchor,
/// writes task_log to sandbox, and/or submits a plan for scheduler pickup.
pub struct SystemCompleteTool {
    pub system: Arc<logos_system::SystemModule>,
    pub sandbox: Arc<SandboxNs>,
}

#[cfg(feature = "sandbox")]
#[async_trait]
impl ProcTool for SystemCompleteTool {
    fn name(&self) -> &str {
        "system.complete"
    }
    fn schema(&self) -> serde_json::Value {
        serde_json::json!({
            "name": "system.complete",
            "description": "Turn terminator. Finishes/sleeps/resumes a task. The mandatory final call of every agent turn.",
            "parameters": {
                "type": "object",
                "properties": {
                    "task_id": { "type": "string", "description": "Current task ID" },
                    "summary": { "type": "string", "description": "What happened this turn" },
                    "reply": { "type": "string", "description": "Message to deliver to the user (optional)" },
                    "anchor": { "type": "boolean", "description": "Create a completion anchor", "default": false },
                    "anchor_facts": { "type": "string", "description": "JSON array of fact objects for the anchor" },
                    "task_log": { "type": "string", "description": "Execution log to append to sandbox" },
                    "sleep_reason": { "type": "string", "description": "If set, task sleeps instead of finishing" },
                    "sleep_retry": { "type": "boolean", "description": "Hint: adapter should auto-retry", "default": false },
                    "resume": { "type": "string", "description": "Task ID to resume (abandons current task)" },
                    "plan": {
                        "type": "object",
                        "description": "Plan mode: submit subtasks for scheduler",
                        "properties": {
                            "todo": {
                                "type": "array",
                                "items": { "type": "string" },
                                "description": "Ordered list of subtask descriptions"
                            }
                        }
                    }
                },
                "required": ["task_id", "summary"]
            }
        })
    }
    async fn call(&self, params: &str) -> Result<String, VfsError> {
        let val: serde_json::Value =
            serde_json::from_str(params).map_err(|e| VfsError::InvalidJson(e.to_string()))?;

        let plan_todo: Vec<String> = val["plan"]["todo"]
            .as_array()
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(|s| s.to_string()))
                    .collect()
            })
            .unwrap_or_default();

        let complete_params = logos_system::CompleteParams {
            task_id: val["task_id"].as_str().unwrap_or_default().to_string(),
            summary: val["summary"].as_str().unwrap_or_default().to_string(),
            reply: val["reply"].as_str().unwrap_or_default().to_string(),
            anchor: val["anchor"].as_bool().unwrap_or(false),
            anchor_facts: val["anchor_facts"].as_str().unwrap_or("[]").to_string(),
            task_log: val["task_log"].as_str().unwrap_or_default().to_string(),
            sleep_reason: val["sleep_reason"].as_str().unwrap_or_default().to_string(),
            sleep_retry: val["sleep_retry"].as_bool().unwrap_or(false),
            resume_task_id: val["resume"].as_str().unwrap_or_default().to_string(),
            plan_todo,
        };

        let result = self.system.complete(complete_params).await?;

        // Write task_log to sandbox (RFC 002 §9.1)
        if !result.task_log.is_empty() && !result.task_id.is_empty() {
            let log_path = format!("{}/log", result.task_id);
            if let Err(e) = self.sandbox.write(&[&log_path], &result.task_log).await {
                eprintln!("[logos] WARNING: failed to write task_log to sandbox: {e}");
            }
        }

        Ok(serde_json::json!({
            "reply": result.reply,
            "anchor_id": result.anchor_id,
            "task_id": result.task_id,
        })
        .to_string())
    }
}
