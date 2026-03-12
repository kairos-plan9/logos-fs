use std::path::PathBuf;
use std::time::Instant;

use tonic::{Request, Response, Status};

use crate::pb::{
    memory_vfs_server::MemoryVfs, ArchiveRequest, ArchiveResponse, PatchRequest, PatchResponse,
    ReadRequest, ReadResponse, SearchRequest, SearchResponse, WriteRequest, WriteResponse,
};
use crate::sessions_store::SessionsStore;
use crate::users_store::UsersStore;

pub use crate::sessions_store::EmbeddingConfig;

const LOGOS_SCHEME: &str = "logos://";
const MEM_SCHEME: &str = "mem://";

#[derive(Debug, Clone, Copy)]
enum Namespace {
    Users,
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
        other => Err(VfsError::InvalidPath(format!(
            "unsupported namespace: \"{other}\""
        ))),
    }
}

pub struct MemoryVfsService {
    users: UsersStore,
    sessions: SessionsStore,
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
    pub fn new(users_root: PathBuf, embedding: EmbeddingConfig) -> std::io::Result<Self> {
        let users = UsersStore::new(users_root)?;
        let state_root = users
            .users_root()
            .parent()
            .map(|p| p.to_path_buf())
            .unwrap_or_else(|| PathBuf::from("."));
        let lancedb_dir = state_root.join("lancedb");
        std::fs::create_dir_all(&lancedb_dir)?;

        Ok(Self {
            users,
            sessions: SessionsStore::new(lancedb_dir.to_string_lossy().to_string(), embedding),
        })
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
