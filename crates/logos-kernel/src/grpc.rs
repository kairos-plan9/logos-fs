use std::sync::Arc;

use tonic::{Request, Response, Status};

use crate::pb::logos_server::Logos;
use crate::pb::*;
#[cfg(feature = "sandbox")]
use crate::sandbox::SandboxNs;
use crate::token::TokenRegistry;
use logos_vfs::{Namespace, RoutingTable, VfsError};

const SESSION_KEY_HEADER: &str = "x-logos-session";

pub struct LogosService {
    table: Arc<RoutingTable>,
    system: Arc<logos_system::SystemModule>,
    #[cfg(feature = "sandbox")]
    sandbox: Option<Arc<SandboxNs>>,
    proc_ns: Arc<crate::proc::ProcNs>,
    tokens: Arc<TokenRegistry>,
    session_ns: Option<Arc<logos_session_ns::SessionNs>>,
}

impl LogosService {
    pub fn new(
        table: Arc<RoutingTable>,
        system: Arc<logos_system::SystemModule>,
        proc_ns: Arc<crate::proc::ProcNs>,
        tokens: Arc<TokenRegistry>,
    ) -> Self {
        Self {
            table,
            system,
            #[cfg(feature = "sandbox")]
            sandbox: None,
            proc_ns,
            tokens,
            session_ns: None,
        }
    }

    #[cfg(feature = "sandbox")]
    pub fn with_sandbox(mut self, sandbox: Arc<SandboxNs>) -> Self {
        self.sandbox = Some(sandbox);
        self
    }

    pub fn with_session_ns(mut self, session_ns: Arc<logos_session_ns::SessionNs>) -> Self {
        self.session_ns = Some(session_ns);
        self
    }
}

use crate::token::{AgentRole, SessionInfo};

async fn extract_session_info(
    tokens: &TokenRegistry,
    request: &Request<impl std::any::Any>,
) -> Option<SessionInfo> {
    let session_key = request.metadata().get(SESSION_KEY_HEADER)?.to_str().ok()?;
    tokens.resolve_info(session_key).await
}

/// RFC 002 §12.3: enforce sandbox access + workspace exclusivity + namespace permissions.
async fn check_access(
    info: Option<&SessionInfo>,
    uri: &str,
    is_write: bool,
    system: &logos_system::SystemModule,
) -> Result<(), Status> {
    // Sandbox isolation: workspace exclusivity
    if let Some(si) = info {
        if let Some(rest) = uri.strip_prefix("logos://sandbox/") {
            let uri_task = rest.split('/').next().unwrap_or("");
            if !uri_task.is_empty() && uri_task != "__system__" && uri_task != si.task_id {
                // Not our task — check if it's finished (non-owner can read finished tasks)
                if is_write {
                    return Err(Status::permission_denied(format!(
                        "task {} cannot write to sandbox of {uri_task}",
                        si.task_id
                    )));
                }
                // Read access: only if target task exists and is finished
                match system.get_task(uri_task).await {
                    Ok(Some(task_json)) => {
                        if let Ok(val) = serde_json::from_str::<serde_json::Value>(&task_json) {
                            if val["status"].as_str() != Some("finished") {
                                return Err(Status::permission_denied(format!(
                                    "task {} cannot read active sandbox of {uri_task}",
                                    si.task_id
                                )));
                            }
                        }
                    }
                    _ => {
                        return Err(Status::permission_denied(format!(
                            "task {} cannot access sandbox of unknown task {uri_task}",
                            si.task_id
                        )));
                    }
                }
            }
        }
    }

    // Namespace permission enforcement (RFC 002 §12.3)
    if is_write {
        if let Some(si) = info {
            if !si.role.is_admin() {
                // User agents: services, system, devices are read-only
                if uri.starts_with("logos://services/")
                    || uri.starts_with("logos://system/")
                    || uri.starts_with("logos://devices/")
                {
                    return Err(Status::permission_denied(format!(
                        "user agent cannot write to {}",
                        uri.split('/').take(4).collect::<Vec<_>>().join("/")
                    )));
                }
            }
        }
        // No session header = management interface = admin (allowed)
    }

    Ok(())
}

/// Auto-scope relative paths to the agent's sandbox.
/// Paths without `logos://` prefix are treated as relative to `logos://sandbox/{task_id}/`.
fn resolve_uri(uri: &str, info: Option<&SessionInfo>) -> String {
    if uri.starts_with("logos://") {
        return uri.to_string();
    }
    // Relative path — scope to agent's sandbox
    if let Some(si) = info {
        let trimmed = uri.trim_start_matches("./").trim_start_matches('/');
        return format!("logos://sandbox/{}/{}", si.task_id, trimmed);
    }
    // No session — pass through (will likely fail at routing)
    uri.to_string()
}

fn vfs_to_status(e: VfsError) -> Status {
    match &e {
        VfsError::InvalidUri(_) | VfsError::InvalidPath(_) | VfsError::InvalidJson(_) => {
            Status::invalid_argument(e.to_string())
        }
        VfsError::NotFound(_) | VfsError::NamespaceNotMounted(_) => {
            Status::not_found(e.to_string())
        }
        VfsError::NotReady => Status::unavailable(e.to_string()),
        _ => Status::internal(e.to_string()),
    }
}

#[tonic::async_trait]
impl Logos for LogosService {
    async fn read(&self, request: Request<ReadReq>) -> Result<Response<ReadRes>, Status> {
        let info = extract_session_info(&self.tokens, &request).await;
        let uri = resolve_uri(&request.into_inner().uri, info.as_ref());
        check_access(info.as_ref(), &uri, false, &self.system).await?;
        let content = self.table.read(&uri).await.map_err(vfs_to_status)?;
        Ok(Response::new(ReadRes { content }))
    }

    async fn write(&self, request: Request<WriteReq>) -> Result<Response<WriteRes>, Status> {
        let info = extract_session_info(&self.tokens, &request).await;
        let req = request.into_inner();
        let uri = resolve_uri(&req.uri, info.as_ref());
        check_access(info.as_ref(), &uri, true, &self.system).await?;
        self.table
            .write(&uri, &req.content)
            .await
            .map_err(vfs_to_status)?;
        Ok(Response::new(WriteRes {}))
    }

    async fn patch(&self, request: Request<PatchReq>) -> Result<Response<PatchRes>, Status> {
        let info = extract_session_info(&self.tokens, &request).await;
        let req = request.into_inner();
        let uri = resolve_uri(&req.uri, info.as_ref());
        check_access(info.as_ref(), &uri, true, &self.system).await?;
        self.table
            .patch(&uri, &req.partial)
            .await
            .map_err(vfs_to_status)?;
        Ok(Response::new(PatchRes {}))
    }

    async fn exec(&self, request: Request<ExecReq>) -> Result<Response<ExecRes>, Status> {
        let info = extract_session_info(&self.tokens, &request).await;
        let si = info.ok_or_else(|| Status::unauthenticated("exec requires a valid session"))?;
        let command = request.into_inner().command;

        if let Some(ref sns) = self.session_ns {
            let result = sns.exec(&si.task_id, &command).await.map_err(vfs_to_status)?;
            return Ok(Response::new(ExecRes {
                stdout: result.stdout,
                stderr: result.stderr,
                exit_code: result.exit_code,
            }));
        }

        #[cfg(feature = "sandbox")]
        {
            let sandbox = self.sandbox.as_ref()
                .ok_or_else(|| Status::unimplemented("sandbox not configured"))?;
            let result = sandbox.exec(&command, &si.agent_config_id, &si.task_id).await.map_err(vfs_to_status)?;
            return Ok(Response::new(ExecRes {
                stdout: result.stdout,
                stderr: result.stderr,
                exit_code: result.exit_code,
            }));
        }

        #[cfg(not(feature = "sandbox"))]
        Err(Status::unimplemented("exec requires sandbox or session namespace"))
    }

    async fn call(&self, request: Request<CallReq>) -> Result<Response<CallRes>, Status> {
        let info = extract_session_info(&self.tokens, &request).await;
        // No session = no call (management interface doesn't need proc tools)
        let _si = info.ok_or_else(|| Status::unauthenticated("call requires a valid session"))?;
        let req = request.into_inner();
        let result_json = self.proc_ns.call(&req.tool, &req.params_json)
            .await
            .map_err(vfs_to_status)?;
        Ok(Response::new(CallRes { result_json }))
    }

    // --- Kernel management interface ---

    async fn handshake(
        &self,
        request: Request<HandshakeReq>,
    ) -> Result<Response<HandshakeRes>, Status> {
        let token = request.into_inner().token;
        match self.tokens.consume(&token).await {
            Some((session_key, task_id, agent_config_id)) => {
                #[cfg(feature = "sandbox")]
                if !agent_config_id.is_empty() {
                    if let Some(ref sandbox) = self.sandbox {
                        if let Err(e) = sandbox.ensure_container_for(&agent_config_id, &task_id).await {
                            eprintln!("[logos] WARNING: pre-create container failed for {agent_config_id}: {e}");
                        }
                    }
                }

                let mut res = Response::new(HandshakeRes {
                    ok: true,
                    error: String::new(),
                });
                res.metadata_mut()
                    .insert(
                        SESSION_KEY_HEADER,
                        session_key.parse().map_err(|_| {
                            Status::internal("session key contains invalid characters")
                        })?,
                    );
                Ok(res)
            }
            None => Ok(Response::new(HandshakeRes {
                ok: false,
                error: "invalid or already consumed token".to_string(),
            })),
        }
    }

    async fn register_token(
        &self,
        request: Request<RegisterTokenReq>,
    ) -> Result<Response<RegisterTokenRes>, Status> {
        let req = request.into_inner();
        if req.token.is_empty() || req.task_id.is_empty() {
            return Err(Status::invalid_argument("token and task_id required"));
        }
        let role = AgentRole::from_str(&req.role);
        self.tokens.register(req.token, req.task_id, req.agent_config_id, role).await;
        Ok(Response::new(RegisterTokenRes {}))
    }

    async fn revoke_token(
        &self,
        request: Request<RevokeTokenReq>,
    ) -> Result<Response<RevokeTokenRes>, Status> {
        let token = request.into_inner().token;
        self.tokens.revoke(&token).await;
        Ok(Response::new(RevokeTokenRes {}))
    }

    async fn create_session(
        &self,
        request: Request<CreateSessionReq>,
    ) -> Result<Response<CreateSessionRes>, Status> {
        let req = request.into_inner();
        let sns = self.session_ns.as_ref()
            .ok_or_else(|| Status::unimplemented("session namespace not configured"))?;
        sns.create_session(&req.project_path, &req.session_id)
            .await
            .map_err(vfs_to_status)?;
        let workspace_path = format!("logos://session/{}/workspace", req.session_id);
        Ok(Response::new(CreateSessionRes {
            session_id: req.session_id,
            workspace_path,
        }))
    }

    async fn checkpoint(
        &self,
        request: Request<CheckpointReq>,
    ) -> Result<Response<CheckpointRes>, Status> {
        let req = request.into_inner();
        let sns = self.session_ns.as_ref()
            .ok_or_else(|| Status::unimplemented("session namespace not configured"))?;
        sns.checkpoint(&req.session_id, &req.checkpoint_id)
            .await
            .map_err(vfs_to_status)?;
        Ok(Response::new(CheckpointRes {
            checkpoint_id: req.checkpoint_id,
        }))
    }

    async fn rollback(
        &self,
        request: Request<RollbackReq>,
    ) -> Result<Response<RollbackRes>, Status> {
        let req = request.into_inner();
        let sns = self.session_ns.as_ref()
            .ok_or_else(|| Status::unimplemented("session namespace not configured"))?;
        sns.rollback(&req.session_id, &req.checkpoint_id)
            .await
            .map_err(vfs_to_status)?;
        Ok(Response::new(RollbackRes {}))
    }

    async fn fork(
        &self,
        request: Request<ForkReq>,
    ) -> Result<Response<ForkRes>, Status> {
        let req = request.into_inner();
        let sns = self.session_ns.as_ref()
            .ok_or_else(|| Status::unimplemented("session namespace not configured"))?;
        sns.fork(&req.from_session_id, &req.new_session_id)
            .await
            .map_err(vfs_to_status)?;
        Ok(Response::new(ForkRes {
            new_session_id: req.new_session_id,
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::token::{AgentRole, SessionInfo};

    async fn test_system() -> (logos_system::SystemModule, tempfile::TempDir) {
        let dir = tempfile::tempdir().unwrap();
        let sys = logos_system::SystemModule::init(dir.path().join("test.db"))
            .await
            .unwrap();
        (sys, dir)
    }

    fn make_session(task_id: &str, agent_config_id: &str, role: AgentRole) -> SessionInfo {
        SessionInfo {
            task_id: task_id.to_string(),
            agent_config_id: agent_config_id.to_string(),
            role,
        }
    }

    // --- Namespace permission tests ---

    #[tokio::test]
    async fn user_cannot_write_services() {
        let (sys, _dir) = test_system().await;
        let si = make_session("t-1", "a-1", AgentRole::User);
        let result = check_access(Some(&si), "logos://services/tts", true, &sys).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn user_cannot_write_system() {
        let (sys, _dir) = test_system().await;
        let si = make_session("t-1", "a-1", AgentRole::User);
        let result = check_access(Some(&si), "logos://system/tasks", true, &sys).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn admin_can_write_services() {
        let (sys, _dir) = test_system().await;
        let si = make_session("t-1", "a-1", AgentRole::Admin);
        let result = check_access(Some(&si), "logos://services/tts", true, &sys).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn user_can_read_services() {
        let (sys, _dir) = test_system().await;
        let si = make_session("t-1", "a-1", AgentRole::User);
        let result = check_access(Some(&si), "logos://services/tts", false, &sys).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn no_session_is_admin() {
        let (sys, _dir) = test_system().await;
        let result = check_access(None, "logos://system/tasks", true, &sys).await;
        assert!(result.is_ok());
    }

    // --- Sandbox ownership tests ---

    #[tokio::test]
    async fn owner_can_access_own_sandbox() {
        let (sys, _dir) = test_system().await;
        let si = make_session("t-1", "a-1", AgentRole::User);
        let result = check_access(Some(&si), "logos://sandbox/t-1/repo", true, &sys).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn non_owner_cannot_write_other_sandbox() {
        let (sys, _dir) = test_system().await;
        let si = make_session("t-1", "a-1", AgentRole::User);
        let result = check_access(Some(&si), "logos://sandbox/t-2/repo", true, &sys).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn non_owner_cannot_read_active_sandbox() {
        let (sys, _dir) = test_system().await;
        // Create task t-2 and make it active
        sys.create_task(r#"{"task_id":"t-2","description":"other task","chat_id":"c-1"}"#)
            .await.unwrap();
        sys.transition_task("t-2", "active").await.unwrap();

        let si = make_session("t-1", "a-1", AgentRole::User);
        let result = check_access(Some(&si), "logos://sandbox/t-2/repo", false, &sys).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn non_owner_can_read_finished_sandbox() {
        let (sys, _dir) = test_system().await;
        // Create task t-2 and finish it
        sys.create_task(r#"{"task_id":"t-2","description":"done task","chat_id":"c-1"}"#)
            .await.unwrap();
        sys.transition_task("t-2", "active").await.unwrap();
        sys.transition_task("t-2", "finished").await.unwrap();

        let si = make_session("t-1", "a-1", AgentRole::User);
        let result = check_access(Some(&si), "logos://sandbox/t-2/repo", false, &sys).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn system_sandbox_always_accessible() {
        let (sys, _dir) = test_system().await;
        let si = make_session("t-1", "a-1", AgentRole::User);
        let result = check_access(Some(&si), "logos://sandbox/__system__/proc/tool", false, &sys).await;
        assert!(result.is_ok());
    }
}
