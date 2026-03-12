use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Mutex;
use std::time::Instant;

use tonic::{Request, Response, Status};

use crate::pb::{
    memory_vfs_server::MemoryVfs, ArchiveRequest, ArchiveResponse, PatchRequest, PatchResponse,
    ReadRequest, ReadResponse, SearchRequest, SearchResponse, WriteRequest, WriteResponse,
};
use crate::memory_store::MemoryStore;
use crate::sessions_store::SessionsStore;
use crate::users_store::UsersStore;

pub use crate::sessions_store::EmbeddingConfig;

const LOGOS_SCHEME: &str = "logos://";
const MEM_SCHEME: &str = "mem://";

#[derive(Debug, Clone, Copy)]
enum Namespace {
    Users,
    Memory,
    Proc,
}

fn resolve_namespace(path: &str) -> Result<Namespace, VfsError> {
    let after_scheme = path
        .strip_prefix(LOGOS_SCHEME)
        .or_else(|| path.strip_prefix(MEM_SCHEME))
        .ok_or_else(|| {
            VfsError::InvalidPath(format!(
                "path must start with \"{LOGOS_SCHEME}\" or \"{MEM_SCHEME}\""
            ))
        })?;

    match after_scheme.split('/').next().unwrap_or("") {
        "users" => Ok(Namespace::Users),
        "memory" => Ok(Namespace::Memory),
        "proc" => Ok(Namespace::Proc),
        other => Err(VfsError::InvalidPath(format!(
            "unsupported namespace: \"{other}\""
        ))),
    }
}

enum ProcRequest {
    Invoke {
        tool: String,
        call_id: String,
        endpoint: InvokeEndpoint,
    },
    Schema {
        tool: String,
    },
    List,
}

enum InvokeEndpoint {
    Input,
    Output,
    Error,
}

fn parse_proc_path(raw_path: &str) -> Result<ProcRequest, VfsError> {
    let after_scheme = raw_path
        .strip_prefix(LOGOS_SCHEME)
        .or_else(|| raw_path.strip_prefix(MEM_SCHEME))
        .ok_or_else(|| VfsError::InvalidPath("invalid scheme".into()))?;

    let rest = after_scheme
        .strip_prefix("proc/")
        .ok_or_else(|| VfsError::InvalidPath("expected proc namespace".into()))?;

    if rest.is_empty() {
        return Ok(ProcRequest::List);
    }

    let seg: Vec<&str> = rest.split('/').collect();

    // logos://proc/{tool}/.schema
    if seg.len() == 2 && seg[1] == ".schema" {
        return Ok(ProcRequest::Schema {
            tool: seg[0].to_string(),
        });
    }

    // logos://proc/{tool}/{call_id}/{input|output|error}
    if seg.len() == 3 {
        let endpoint = match seg[2] {
            "input" => InvokeEndpoint::Input,
            "output" => InvokeEndpoint::Output,
            "error" => InvokeEndpoint::Error,
            _ => {
                return Err(VfsError::InvalidPath(
                    "expected 'input', 'output', or 'error'".into(),
                ))
            }
        };
        return Ok(ProcRequest::Invoke {
            tool: seg[0].to_string(),
            call_id: seg[1].to_string(),
            endpoint,
        });
    }

    Err(VfsError::InvalidPath(
        "expected logos://proc/{tool}/{call_id}/{input|output|error} \
         or logos://proc/{tool}/.schema"
            .into(),
    ))
}

const PROC_TOOLS: &[&str] = &[
    "memory.range_fetch",
    "memory.range_summary",
    "memory.search",
];

fn proc_schema(tool: &str) -> Result<String, VfsError> {
    let schema = match tool {
        "memory.range_fetch" => serde_json::json!({
            "name": "memory.range_fetch",
            "description": "Fetch raw messages from given msg_id ranges with mandatory pagination.",
            "parameters": {
                "type": "object",
                "properties": {
                    "chat_id": { "type": "string" },
                    "ranges":  { "type": "array", "items": { "type": "array", "items": { "type": "integer" } } },
                    "limit":   { "type": "integer", "default": 20 },
                    "offset":  { "type": "integer", "default": 0 }
                },
                "required": ["chat_id", "ranges"]
            }
        }),
        "memory.range_summary" => serde_json::json!({
            "name": "memory.range_summary",
            "description": "Return a compressed summary of messages in the given ranges.",
            "parameters": {
                "type": "object",
                "properties": {
                    "chat_id":    { "type": "string" },
                    "ranges":     { "type": "array", "items": { "type": "array", "items": { "type": "integer" } } },
                    "max_tokens": { "type": "integer", "default": 500 }
                },
                "required": ["chat_id", "ranges"]
            }
        }),
        "memory.search" => serde_json::json!({
            "name": "memory.search",
            "description": "Full-text search over messages.",
            "parameters": {
                "type": "object",
                "properties": {
                    "chat_id": { "type": "string" },
                    "query":   { "type": "string" },
                    "limit":   { "type": "integer", "default": 10 }
                },
                "required": ["chat_id", "query"]
            }
        }),
        _ => return Err(VfsError::NotFound(format!("unknown tool: \"{tool}\""))),
    };
    serde_json::to_string(&schema)
        .map_err(|e| VfsError::Io(format!("serialize error: {e}")))
}

pub struct MemoryVfsService {
    users: UsersStore,
    memory: MemoryStore,
    sessions: SessionsStore,
    proc_results: Mutex<HashMap<String, Result<String, String>>>,
}

#[derive(Debug)]
pub(crate) enum VfsError {
    InvalidPath(String),
    NotFound(String),
    InvalidJson(String),
    InvalidRequest(String),
    Io(String),
    Http(String),
    Lance(String),
}

impl VfsError {
    pub(crate) fn message(&self) -> String {
        match self {
            Self::InvalidPath(msg)
            | Self::NotFound(msg)
            | Self::InvalidJson(msg)
            | Self::InvalidRequest(msg)
            | Self::Io(msg)
            | Self::Http(msg)
            | Self::Lance(msg) => msg.clone(),
        }
    }
}

impl MemoryVfsService {
    pub fn new(
        users_root: PathBuf,
        memory_root: PathBuf,
        embedding: EmbeddingConfig,
    ) -> std::io::Result<Self> {
        let users = UsersStore::new(users_root)?;
        let memory = MemoryStore::new(memory_root)?;
        let state_root = users
            .users_root()
            .parent()
            .map(|p| p.to_path_buf())
            .unwrap_or_else(|| PathBuf::from("."));
        let lancedb_dir = state_root.join("lancedb");
        std::fs::create_dir_all(&lancedb_dir)?;

        Ok(Self {
            users,
            memory,
            sessions: SessionsStore::new(lancedb_dir.to_string_lossy().to_string(), embedding),
            proc_results: Mutex::new(HashMap::new()),
        })
    }

    async fn handle_proc_write(&self, path: &str, content: &str) -> Result<(), VfsError> {
        match parse_proc_path(path)? {
            ProcRequest::Invoke {
                tool,
                call_id,
                endpoint: InvokeEndpoint::Input,
            } => {
                let result = match tool.as_str() {
                    "memory.range_fetch" => self.memory.range_fetch(content).await,
                    "memory.range_summary" => self.memory.range_summary(content).await,
                    "memory.search" => self.memory.search_messages(content).await,
                    _ => {
                        return Err(VfsError::InvalidPath(format!(
                            "unknown proc tool: \"{tool}\""
                        )))
                    }
                };
                let key = format!("{tool}:{call_id}");
                let entry = result.map_err(|e| e.message());
                self.proc_results.lock().unwrap().insert(key, entry);
                Ok(())
            }
            ProcRequest::Invoke { .. } => Err(VfsError::InvalidRequest(
                "can only write to proc input".into(),
            )),
            ProcRequest::Schema { .. } | ProcRequest::List => Err(VfsError::InvalidRequest(
                "cannot write to proc schema or tool list".into(),
            )),
        }
    }

    fn handle_proc_read(&self, path: &str) -> Result<String, VfsError> {
        match parse_proc_path(path)? {
            ProcRequest::List => {
                serde_json::to_string(PROC_TOOLS)
                    .map_err(|e| VfsError::Io(format!("serialize error: {e}")))
            }
            ProcRequest::Schema { tool } => proc_schema(&tool),
            ProcRequest::Invoke {
                tool,
                call_id,
                endpoint: InvokeEndpoint::Output,
            } => {
                let key = format!("{tool}:{call_id}");
                let mut map = self.proc_results.lock().unwrap();
                match map.remove(&key) {
                    Some(Ok(output)) => Ok(output),
                    Some(Err(err_msg)) => {
                        map.insert(key, Err(err_msg));
                        Err(VfsError::NotFound(
                            "tool execution failed; read the error endpoint".into(),
                        ))
                    }
                    None => Err(VfsError::NotFound(format!(
                        "no result for {tool} call {call_id}"
                    ))),
                }
            }
            ProcRequest::Invoke {
                tool,
                call_id,
                endpoint: InvokeEndpoint::Error,
            } => {
                let key = format!("{tool}:{call_id}");
                let mut map = self.proc_results.lock().unwrap();
                match map.remove(&key) {
                    Some(Err(err_msg)) => Ok(err_msg),
                    Some(Ok(output)) => {
                        map.insert(key, Ok(output));
                        Err(VfsError::NotFound("no error for this call".into()))
                    }
                    None => Err(VfsError::NotFound(format!(
                        "no result for {tool} call {call_id}"
                    ))),
                }
            }
            ProcRequest::Invoke {
                endpoint: InvokeEndpoint::Input,
                ..
            } => Err(VfsError::InvalidRequest(
                "cannot read from proc input".into(),
            )),
        }
    }
}

fn log_vfs_ok(op: &str, detail: &str, started_at: Instant) {
    println!(
        "[vfs] op={} status=ok elapsed_ms={} {}",
        op,
        started_at.elapsed().as_millis(),
        detail
    );
}

fn log_vfs_err(op: &str, detail: &str, err: &VfsError, started_at: Instant) {
    eprintln!(
        "[vfs] op={} status=error elapsed_ms={} {} err=\"{}\"",
        op,
        started_at.elapsed().as_millis(),
        detail,
        err.message()
    );
}

#[tonic::async_trait]
impl MemoryVfs for MemoryVfsService {
    async fn read(&self, request: Request<ReadRequest>) -> Result<Response<ReadResponse>, Status> {
        let started_at = Instant::now();
        let req = request.into_inner();
        let detail = format!("path={}", req.path);
        let result = match resolve_namespace(&req.path) {
            Ok(Namespace::Users) => self.users.read(&req.path).await,
            Ok(Namespace::Memory) => self.memory.read(&req.path).await,
            Ok(Namespace::Proc) => self.handle_proc_read(&req.path),
            Err(e) => Err(e),
        };
        match result {
            Ok(content) => {
                log_vfs_ok("read", &detail, started_at);
                Ok(Response::new(ReadResponse {
                    success: true,
                    content,
                    error_msg: String::new(),
                }))
            }
            Err(err) => {
                log_vfs_err("read", &detail, &err, started_at);
                Ok(Response::new(ReadResponse {
                    success: false,
                    content: String::new(),
                    error_msg: err.message(),
                }))
            }
        }
    }

    async fn write(
        &self,
        request: Request<WriteRequest>,
    ) -> Result<Response<WriteResponse>, Status> {
        let started_at = Instant::now();
        let req = request.into_inner();
        let detail = format!("path={} content_len={}", req.path, req.content.len());
        let result = match resolve_namespace(&req.path) {
            Ok(Namespace::Users) => self.users.write(&req.path, &req.content).await,
            Ok(Namespace::Memory) => self.memory.write(&req.path, &req.content).await,
            Ok(Namespace::Proc) => self.handle_proc_write(&req.path, &req.content).await,
            Err(e) => Err(e),
        };
        match result {
            Ok(_) => {
                log_vfs_ok("write", &detail, started_at);
                Ok(Response::new(WriteResponse {
                    success: true,
                    error_msg: String::new(),
                }))
            }
            Err(err) => {
                log_vfs_err("write", &detail, &err, started_at);
                Ok(Response::new(WriteResponse {
                    success: false,
                    error_msg: err.message(),
                }))
            }
        }
    }

    async fn patch(
        &self,
        request: Request<PatchRequest>,
    ) -> Result<Response<PatchResponse>, Status> {
        let started_at = Instant::now();
        let req = request.into_inner();
        let detail = format!(
            "path={} partial_content_len={}",
            req.path,
            req.partial_content.len()
        );
        let result = match resolve_namespace(&req.path) {
            Ok(Namespace::Users) => self.users.patch(&req.path, &req.partial_content).await,
            Ok(Namespace::Memory) | Ok(Namespace::Proc) => Err(VfsError::InvalidRequest(
                "patch is not supported for this namespace".into(),
            )),
            Err(e) => Err(e),
        };
        match result {
            Ok(_) => {
                log_vfs_ok("patch", &detail, started_at);
                Ok(Response::new(PatchResponse {
                    success: true,
                    error_msg: String::new(),
                }))
            }
            Err(err) => {
                log_vfs_err("patch", &detail, &err, started_at);
                Ok(Response::new(PatchResponse {
                    success: false,
                    error_msg: err.message(),
                }))
            }
        }
    }

    async fn search(
        &self,
        request: Request<SearchRequest>,
    ) -> Result<Response<SearchResponse>, Status> {
        let started_at = Instant::now();
        let req = request.into_inner();
        let detail = format!(
            "scope={} limit={} query_len={}",
            req.scope,
            req.limit,
            req.query.len()
        );
        match self.sessions.search(req).await {
            Ok(results) => {
                let result_count = results.len();
                log_vfs_ok(
                    "search",
                    &format!("{} result_count={}", detail, result_count),
                    started_at,
                );
                Ok(Response::new(SearchResponse { results }))
            }
            Err(err) => {
                log_vfs_err("search", &detail, &err, started_at);
                Err(Status::internal(err.message()))
            }
        }
    }

    async fn archive(
        &self,
        request: Request<ArchiveRequest>,
    ) -> Result<Response<ArchiveResponse>, Status> {
        let started_at = Instant::now();
        let req = request.into_inner();
        let detail = format!(
            "session_id={} chat_id={} messages_count={}",
            req.session_id,
            req.chat_id,
            req.messages.len()
        );
        match self.sessions.archive(req).await {
            Ok(_) => {
                log_vfs_ok("archive", &detail, started_at);
                Ok(Response::new(ArchiveResponse {
                    success: true,
                    error_msg: String::new(),
                }))
            }
            Err(err) => {
                log_vfs_err("archive", &detail, &err, started_at);
                Ok(Response::new(ArchiveResponse {
                    success: false,
                    error_msg: err.message(),
                }))
            }
        }
    }
}
