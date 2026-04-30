mod messages;
pub mod plugin;
mod summaries;
mod graph;
pub mod views;

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use async_trait::async_trait;
use logos_vfs::{Namespace, VfsError};

use messages::MessageDb;
use plugin::MemoryPlugin;
use views::MemoryView;

pub use logos_session::SessionStore;

/// The Memory Module — handles `logos://memory/`.
///
/// Path-mapped reads (RFC 003 §3.1):
///   logos://memory/groups/{gid}/messages/{msg_id}           → single message
///   logos://memory/groups/{gid}/summary/{layer}/latest      → latest summary
///   logos://memory/groups/{gid}/summary/{layer}/{period}    → summary by period
///
/// Path-mapped writes:
///   logos://memory/groups/{gid}/messages                    → insert message (JSON)
///   logos://memory/groups/{gid}/summary/{layer}/{period}    → insert summary (JSON)
///
/// Complex queries go through logos_call (proc tools) — RFC 003 §3.2:
///   call("memory.search",      { chat_id, query, limit })
///   call("memory.range_fetch", { chat_id, ranges, limit, offset })
pub struct MemoryModule {
    messages: MessageDb,
    sessions: Arc<SessionStore>,
    views: HashMap<String, Box<dyn views::MemoryView>>,
    plugins: Vec<Box<dyn MemoryPlugin>>,
}

impl MemoryModule {
    pub fn init(db_root: PathBuf, sessions: Arc<SessionStore>) -> Result<Self, VfsError> {
        let messages = MessageDb::new(db_root)?;
        let mut view_map: HashMap<String, Box<dyn views::MemoryView>> = HashMap::new();
        let by_actor = views::ByActorView;
        view_map.insert(by_actor.name().to_string(), Box::new(by_actor));
        let by_speaker = views::BySpeakerView;
        view_map.insert(by_speaker.name().to_string(), Box::new(by_speaker));
        let recent = views::RecentView;
        view_map.insert(recent.name().to_string(), Box::new(recent));

        // Mount default plugins (RFC 003 §11)
        let plugins: Vec<Box<dyn MemoryPlugin>> = vec![
            Box::new(summaries::HierarchicalPlugin),
            Box::new(graph::GraphPlugin),
        ];

        Ok(Self {
            messages,
            sessions,
            views: view_map,
            plugins,
        })
    }

    /// List mounted plugins (for logos://memory/plugins/ discovery).
    pub fn plugin_list(&self) -> Vec<serde_json::Value> {
        self.plugins.iter().map(|p| serde_json::json!({
            "name": p.name(),
            "docs": p.docs(),
        })).collect()
    }

    /// Access the session store (for runtime context injection — RFC 003 §5.5).
    pub fn sessions(&self) -> &Arc<SessionStore> {
        &self.sessions
    }

    // --- call handlers (proc tools) ---

    /// Dispatch a logos_call to the appropriate handler.
    pub async fn handle_call(&self, tool: &str, params: &str) -> Result<String, VfsError> {
        let val: serde_json::Value =
            serde_json::from_str(params).map_err(|e| VfsError::InvalidJson(e.to_string()))?;
        let chat_id = val["chat_id"].as_str().unwrap_or_default();

        match tool {
            "memory.search" => {
                let query = val["query"].as_str().unwrap_or_default();
                let limit = val["limit"].as_i64().unwrap_or(10);
                self.messages.search_fts(chat_id, query, limit).await
            }
            "memory.range_fetch" => {
                self.messages.range_fetch(chat_id, params).await
            }
            _ if tool.starts_with("memory.view.") => {
                let view_name = tool.strip_prefix("memory.view.").unwrap_or("");
                let view = self.views.get(view_name).ok_or_else(|| {
                    VfsError::NotFound(format!("unknown view: {view_name}"))
                })?;
                let pool = self.messages.pool(chat_id).await?;
                view.query(&pool, chat_id, params).await
            }
            _ => Err(VfsError::NotFound(format!("unknown memory tool: {tool}"))),
        }
    }
}

#[async_trait]
impl Namespace for MemoryModule {
    fn name(&self) -> &str {
        "memory"
    }

    async fn read(&self, path: &[&str]) -> Result<String, VfsError> {
        // logos://memory/plugins/ → plugin discovery (RFC 003 §11.3)
        if path.first() == Some(&"plugins") {
            return Ok(serde_json::to_string(&self.plugin_list())
                .unwrap_or_else(|_| "[]".to_string()));
        }

        if path.len() < 3 || path[0] != "groups" {
            return Err(VfsError::InvalidPath(
                "expected logos://memory/groups/{gid}/...".to_string(),
            ));
        }
        let gid = path[1];
        let resource = path[2];

        // Core: messages
        if resource == "messages" {
            let msg_id_str = path.get(3).ok_or_else(|| {
                VfsError::InvalidPath("expected logos://memory/groups/{gid}/messages/{msg_id}".to_string())
            })?;
            let msg_id: i64 = msg_id_str
                .parse()
                .map_err(|_| VfsError::InvalidPath(format!("invalid msg_id: {msg_id_str}")))?;
            return self.messages
                .get_by_id(gid, msg_id)
                .await
                .map(|opt| opt.unwrap_or_else(|| "null".to_string()));
        }

        // Core: views
        if resource == "views" {
            let view_name = path.get(3).ok_or_else(|| {
                VfsError::InvalidPath("missing view name".to_string())
            })?;
            let view = self.views.get(*view_name).ok_or_else(|| {
                VfsError::NotFound(format!("unknown view: {view_name}"))
            })?;
            let params = path.get(4).copied().unwrap_or("{}");
            let pool = self.messages.pool(gid).await?;
            return view.query(&pool, gid, params).await;
        }

        // Check plugins
        for plugin in &self.plugins {
            if plugin.name() == resource {
                let pool = self.messages.pool(gid).await?;
                plugin.init_schema(&pool).await?;
                return plugin.read(&pool, gid, &path[3..]).await;
            }
        }

        Err(VfsError::InvalidPath(format!("unknown memory resource: {resource}")))
    }

    async fn write(&self, path: &[&str], content: &str) -> Result<(), VfsError> {
        if path.len() < 3 || path[0] != "groups" {
            return Err(VfsError::InvalidPath(
                "expected logos://memory/groups/{gid}/...".to_string(),
            ));
        }
        let gid = path[1];
        let resource = path[2];

        // Core: messages
        if resource == "messages" {
            let msg_id = self.messages.insert(gid, content).await?;

            if let Ok(val) = serde_json::from_str::<serde_json::Value>(content) {
                let msg_ref = logos_session::MsgRef {
                    msg_id,
                    chat_id: gid.to_string(),
                    reply_to: val["reply_to"].as_i64(),
                    text: val["text"].as_str().unwrap_or_default().to_string(),
                    speaker: val["actor_id"]
                        .as_str()
                        .or_else(|| val["speaker"].as_str())
                        .unwrap_or_default()
                        .to_string(),
                    ts: val["ts"].as_str().unwrap_or_default().to_string(),
                };
                self.sessions.observe(msg_ref).await;
            }
            return Ok(());
        }

        // Check plugins
        for plugin in &self.plugins {
            if plugin.name() == resource {
                let pool = self.messages.pool(gid).await?;
                plugin.init_schema(&pool).await?;
                return plugin.write(&pool, gid, &path[3..], content).await;
            }
        }

        Err(VfsError::InvalidPath(format!("unknown memory resource for write: {resource}")))
    }

    async fn patch(&self, path: &[&str], partial: &str) -> Result<(), VfsError> {
        if path.len() < 3 || path[0] != "groups" {
            return Err(VfsError::InvalidPath(
                "expected logos://memory/groups/{gid}/...".to_string(),
            ));
        }
        let gid = path[1];
        let resource = path[2];

        // Check plugins
        for plugin in &self.plugins {
            if plugin.name() == resource {
                let pool = self.messages.pool(gid).await?;
                plugin.init_schema(&pool).await?;
                return plugin.patch(&pool, gid, &path[3..], partial).await;
            }
        }

        // Fallback: messages (append-only, patch = write)
        self.write(path, partial).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use logos_vfs::Namespace;

    async fn test_mm() -> (MemoryModule, tempfile::TempDir) {
        let dir = tempfile::tempdir().unwrap();
        let sessions = Arc::new(SessionStore::new(10, 10));
        let mm = MemoryModule::init(dir.path().to_path_buf(), sessions).unwrap();
        (mm, dir)
    }

    #[tokio::test]
    async fn insert_and_get_message() {
        let (mm, _dir) = test_mm().await;
        let msg = r#"{"ts":"2026-03-20T10:00:00Z","actor_id":"alice","text":"hello","reply_to":null,"meta":{"username":"Alice","senderEntityType":"user","replyToUserId":"bob"}}"#;
        mm.write(&["groups", "chat-1", "messages"], msg).await.unwrap();

        let result = mm.read(&["groups", "chat-1", "messages", "1"]).await.unwrap();
        let val: serde_json::Value = serde_json::from_str(&result).unwrap();
        assert_eq!(val["actor_id"], "alice");
        assert_eq!(val["speaker"], "alice");
        assert_eq!(val["text"], "hello");
        assert_eq!(val["meta"]["username"], "Alice");
        assert_eq!(val["meta"]["senderEntityType"], "user");
        assert_eq!(val["meta"]["replyToUserId"], "bob");
    }

    #[tokio::test]
    async fn search_fts() {
        let (mm, _dir) = test_mm().await;
        mm.write(&["groups", "chat-1", "messages"],
            r#"{"ts":"2026-03-20T10:00:00Z","speaker":"bob","text":"rust is great"}"#
        ).await.unwrap();
        mm.write(&["groups", "chat-1", "messages"],
            r#"{"ts":"2026-03-20T10:01:00Z","speaker":"alice","text":"python is nice"}"#
        ).await.unwrap();

        let result = mm.handle_call("memory.search",
            r#"{"chat_id":"chat-1","query":"rust","limit":10}"#
        ).await.unwrap();
        assert!(result.contains("rust is great"));
        assert!(!result.contains("python"));
    }

    #[tokio::test]
    async fn range_fetch() {
        let (mm, _dir) = test_mm().await;
        for i in 0..5 {
            mm.write(&["groups", "chat-1", "messages"],
                &format!(r#"{{"ts":"2026-03-20T10:{i:02}:00Z","speaker":"user","text":"msg {i}"}}"#)
            ).await.unwrap();
        }

        let result = mm.handle_call("memory.range_fetch",
            r#"{"chat_id":"chat-1","ranges":[[2,4]],"limit":10}"#
        ).await.unwrap();
        let arr: Vec<serde_json::Value> = serde_json::from_str(&result).unwrap();
        assert_eq!(arr.len(), 3); // msg_id 2,3,4
    }

    #[tokio::test]
    async fn summary_write_and_read() {
        let (mm, _dir) = test_mm().await;
        let summary = r#"{"layer":"short","period_start":"2026-03-20T10","period_end":"2026-03-20T10","source_refs":"[[1,5]]","content":"test summary"}"#;
        mm.write(&["groups", "chat-1", "summary", "short", "2026-03-20T10"], summary)
            .await.unwrap();

        let result = mm.read(&["groups", "chat-1", "summary", "short", "latest"]).await.unwrap();
        assert!(result.contains("test summary"));
    }

    #[tokio::test]
    async fn view_recent() {
        let (mm, _dir) = test_mm().await;
        for i in 0..3 {
            mm.write(&["groups", "chat-1", "messages"],
                &format!(r#"{{"ts":"2026-03-20T10:{i:02}:00Z","speaker":"user","text":"msg {i}"}}"#)
            ).await.unwrap();
        }

        let result = mm.handle_call("memory.view.recent",
            r#"{"chat_id":"chat-1","limit":2}"#
        ).await.unwrap();
        let arr: Vec<serde_json::Value> = serde_json::from_str(&result).unwrap();
        assert_eq!(arr.len(), 2);
    }

    #[tokio::test]
    async fn plugin_discovery() {
        let (mm, _dir) = test_mm().await;
        let result = mm.read(&["plugins"]).await.unwrap();
        let plugins: Vec<serde_json::Value> = serde_json::from_str(&result).unwrap();
        let names: Vec<&str> = plugins.iter().map(|p| p["name"].as_str().unwrap()).collect();
        assert!(names.contains(&"summary"));
        assert!(names.contains(&"graph"));
    }

    #[tokio::test]
    async fn graph_write_and_read() {
        let (mm, _dir) = test_mm().await;

        // Write triples about alice
        mm.write(&["groups", "chat-1", "graph", "alice"], r#"[
            {"predicate": "是", "object": "后端工程师"},
            {"predicate": "works_on", "object": "logos"},
            {"predicate": "likes", "object": "Rust"}
        ]"#).await.unwrap();

        // Read all triples for alice
        let result = mm.read(&["groups", "chat-1", "graph", "alice"]).await.unwrap();
        let triples: Vec<serde_json::Value> = serde_json::from_str(&result).unwrap();
        assert_eq!(triples.len(), 3);
        assert_eq!(triples[0]["subject"], "alice");

        // List all subjects
        let list = mm.read(&["groups", "chat-1", "graph"]).await.unwrap();
        let subjects: Vec<String> = serde_json::from_str(&list).unwrap();
        assert_eq!(subjects, vec!["alice"]);
    }

    #[tokio::test]
    async fn graph_not_found() {
        let (mm, _dir) = test_mm().await;
        let result = mm.read(&["groups", "chat-1", "graph", "nonexistent"]).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn view_by_speaker() {
        let (mm, _dir) = test_mm().await;
        mm.write(&["groups", "chat-1", "messages"],
            r#"{"ts":"2026-03-20T10:00:00Z","speaker":"alice","text":"hello from alice"}"#
        ).await.unwrap();
        mm.write(&["groups", "chat-1", "messages"],
            r#"{"ts":"2026-03-20T10:01:00Z","speaker":"bob","text":"hello from bob"}"#
        ).await.unwrap();

        let result = mm.handle_call("memory.view.by_speaker",
            r#"{"chat_id":"chat-1","speaker":"alice","limit":10}"#
        ).await.unwrap();
        assert!(result.contains("alice"));
        assert!(!result.contains("bob"));
    }

    #[tokio::test]
    async fn view_by_actor() {
        let (mm, _dir) = test_mm().await;
        mm.write(&["groups", "chat-1", "messages"],
            r#"{"ts":"2026-03-20T10:00:00Z","actor_id":"user:1","speaker":"alice","text":"hello from alice"}"#
        ).await.unwrap();
        mm.write(&["groups", "chat-1", "messages"],
            r#"{"ts":"2026-03-20T10:01:00Z","actor_id":"user:2","speaker":"bob","text":"hello from bob"}"#
        ).await.unwrap();

        let result = mm.handle_call("memory.view.by_actor",
            r#"{"chat_id":"chat-1","actor_id":"user:1","limit":10}"#
        ).await.unwrap();
        assert!(result.contains("user:1"));
        assert!(!result.contains("user:2"));
    }
}
