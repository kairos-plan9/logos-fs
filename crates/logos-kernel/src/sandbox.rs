use std::collections::HashMap;
use std::path::PathBuf;

use async_trait::async_trait;
use logos_vfs::{Namespace, VfsError};
use tokio::sync::Mutex;

const CONTAINER_WORKDIR: &str = "/workspace";

pub struct ExecResult {
    pub stdout: String,
    pub stderr: String,
    pub exit_code: i32,
}

/// Sandbox executor trait — pluggable container backend.
/// Result of ensure_container: container ID + host path for filesystem operations.
#[derive(Clone)]
pub struct ContainerInfo {
    pub container_id: String,
    /// Host path where files can be read/written directly.
    /// For containerd: overlayfs upperdir. For Docker: bind-mount host dir.
    pub host_path: PathBuf,
}

#[async_trait]
pub trait SandboxExecutor: Send + Sync {
    /// Ensure a container is running for the given agent_config_id.
    async fn ensure_container(
        &self,
        agent_config_id: &str,
        sock_path: Option<&PathBuf>,
    ) -> Result<ContainerInfo, VfsError>;

    /// Execute a command inside the container. cwd = workspace path inside container.
    async fn exec_in_container(
        &self,
        container_id: &str,
        command: &str,
        cwd: &str,
    ) -> Result<ExecResult, VfsError>;

    /// Stop and remove a container.
    async fn remove_container(&self, container_id: &str) -> Result<(), VfsError>;
}

/// Sandbox namespace — `logos://sandbox/`.
///
/// Container is shared per agent_config_id. Each task gets its own subdirectory
/// under /workspace/{task_id}/ inside the container. URI routing ensures agents
/// can only access their own task's directory.
///
/// - **read/write/patch**: operate on per-task directory in overlayfs upperdir
/// - **exec**: runs commands in container, cwd = /workspace/{task_id}/
pub struct SandboxNs {
    executor: Box<dyn SandboxExecutor>,
    host_root: PathBuf,
    containers: Mutex<HashMap<String, ContainerInfo>>, // agent_config_id → info
    task_agent: Mutex<HashMap<String, String>>,        // task_id → agent_config_id
}

impl SandboxNs {
    pub async fn init(host_root: PathBuf, image: Option<String>) -> Result<Self, VfsError> {
        std::fs::create_dir_all(&host_root)
            .map_err(|e| VfsError::Io(format!("create sandbox dir: {e}")))?;

        let mode = std::env::var("SANDBOX_MODE").unwrap_or_default();
        let executor: Box<dyn SandboxExecutor> = if mode.eq_ignore_ascii_case("host") {
            println!("[logos] sandbox: using host executor (no container isolation)");
            Box::new(HostExecutor::new(host_root.clone()))
        } else {
            println!("[logos] sandbox: using containerd");
            Box::new(ContainerdExecutor::try_init(image).await?)
        };

        Ok(Self {
            executor,
            host_root,
            containers: Mutex::new(HashMap::new()),
            task_agent: Mutex::new(HashMap::new()),
        })
    }

    /// Ensure a container exists and is running for the given agent_config_id.
    /// Returns (container_id, host_path for filesystem ops).
    async fn ensure_container(&self, agent_config_id: &str) -> Result<ContainerInfo, VfsError> {
        {
            let containers = self.containers.lock().await;
            if let Some(info) = containers.get(agent_config_id) {
                return Ok(info.clone());
            }
        }

        let sock_path = self.host_root.join("logos.sock");
        let sock = if sock_path.exists() {
            Some(&sock_path)
        } else {
            None
        };

        let info = self
            .executor
            .ensure_container(agent_config_id, sock)
            .await?;

        let mut containers = self.containers.lock().await;
        containers.insert(agent_config_id.to_string(), info.clone());
        Ok(info)
    }

    /// Resolve a sandbox path to the per-task directory inside the correct container's overlayfs.
    /// path[0] = task_id, path[1..] = file path within that task's workspace.
    async fn resolve_path(&self, path: &[&str]) -> Result<PathBuf, VfsError> {
        if path.first().map(|s| *s) == Some("__system__") {
            return Ok(self.host_root.join(path.join("/")));
        }

        let task_id = *path
            .first()
            .ok_or_else(|| VfsError::InvalidPath("empty sandbox path".to_string()))?;

        // Look up which agent owns this task, then find that agent's container
        let agent_id = self
            .task_agent
            .lock()
            .await
            .get(task_id)
            .cloned()
            .ok_or_else(|| VfsError::Io(format!("no agent registered for task {task_id}")))?;

        let containers = self.containers.lock().await;
        let info = containers
            .get(&agent_id)
            .ok_or_else(|| VfsError::Io(format!("no container for agent {agent_id}")))?;

        let task_dir = info.host_path.join(task_id);
        std::fs::create_dir_all(&task_dir)
            .map_err(|e| VfsError::Io(format!("create task dir {task_id}: {e}")))?;

        let resolved = if path.len() > 1 {
            task_dir.join(path[1..].join("/"))
        } else {
            task_dir.clone()
        };

        // Canonicalize to prevent path traversal. The task_dir itself may not yet
        // contain subdirs, so we canonicalize the task_dir anchor separately.
        let canonical_task_dir = task_dir
            .canonicalize()
            .map_err(|e| VfsError::Io(format!("canonicalize task dir: {e}")))?;
        // For paths that don't exist yet, walk up until we find an existing prefix.
        let canonical_resolved = canonicalize_partial(&resolved)
            .map_err(|e| VfsError::Io(format!("canonicalize path: {e}")))?;
        if !canonical_resolved.starts_with(&canonical_task_dir) {
            return Err(VfsError::InvalidPath(format!(
                "path escapes task directory: {}",
                resolved.display()
            )));
        }
        Ok(resolved)
    }

    /// Pre-create a container and register task → agent mapping.
    /// Called at handshake time so the sandbox is ready before first read/write/exec.
    pub async fn ensure_container_for(
        &self,
        agent_config_id: &str,
        task_id: &str,
    ) -> Result<(), VfsError> {
        self.ensure_container(agent_config_id).await?;
        self.task_agent
            .lock()
            .await
            .insert(task_id.to_string(), agent_config_id.to_string());
        Ok(())
    }

    fn translate_service_uris(&self, command: &str) -> String {
        let host_root_str = self.host_root.to_string_lossy();
        command.replace(
            "logos://services/",
            &format!("{host_root_str}/__system__/svc/"),
        )
    }

    fn extract_task_id(command: &str) -> Option<String> {
        let prefix = "logos://sandbox/";
        if let Some(start) = command.find(prefix) {
            let rest = &command[start + prefix.len()..];
            let task_id = rest.split('/').next().unwrap_or("");
            if !task_id.is_empty() {
                return Some(task_id.to_string());
            }
        }
        None
    }

    pub async fn exec(
        &self,
        command: &str,
        agent_config_id: &str,
        task_id: &str,
    ) -> Result<ExecResult, VfsError> {
        let info = self.ensure_container(agent_config_id).await?;

        // Ensure per-task directory exists inside container
        let task_workspace = format!("{CONTAINER_WORKDIR}/{task_id}");
        // Create task dir on host side so it's visible in container
        let task_dir_host = info.host_path.join(task_id);
        std::fs::create_dir_all(&task_dir_host)
            .map_err(|e| VfsError::Io(format!("create task dir {task_id}: {e}")))?;

        let translated = command.replace(
            &format!("logos://sandbox/{task_id}/"),
            &format!("{task_workspace}/"),
        );
        let translated = translated.replace(&format!("logos://sandbox/{task_id}"), &task_workspace);
        let translated = self.translate_service_uris(&translated);

        self.executor
            .exec_in_container(&info.container_id, &translated, &task_workspace)
            .await
    }
}

#[async_trait]
impl Namespace for SandboxNs {
    fn name(&self) -> &str {
        "sandbox"
    }

    async fn read(&self, path: &[&str]) -> Result<String, VfsError> {
        if path.is_empty() {
            return Err(VfsError::InvalidPath("empty sandbox path".to_string()));
        }
        let file_path = self.resolve_path(path).await?;

        if file_path.is_dir() {
            let mut entries = Vec::new();
            let dir = std::fs::read_dir(&file_path)
                .map_err(|e| VfsError::Io(format!("read dir {}: {e}", file_path.display())))?;
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
                    VfsError::NotFound(format!("logos://sandbox/{}", path.join("/")))
                }
                _ => VfsError::Io(format!("read {}: {e}", file_path.display())),
            })
    }

    async fn write(&self, path: &[&str], content: &str) -> Result<(), VfsError> {
        if path.is_empty() {
            return Err(VfsError::InvalidPath("empty sandbox path".to_string()));
        }
        let file_path = self.resolve_path(path).await?;

        if let Some(parent) = file_path.parent() {
            tokio::fs::create_dir_all(parent)
                .await
                .map_err(|e| VfsError::Io(format!("mkdir {}: {e}", parent.display())))?;
        }

        tokio::fs::write(&file_path, content)
            .await
            .map_err(|e| VfsError::Io(format!("write {}: {e}", file_path.display())))
    }

    async fn patch(&self, path: &[&str], partial: &str) -> Result<(), VfsError> {
        if path.last().map(|s| *s) == Some("log") {
            let file_path = self.resolve_path(path).await?;
            if let Some(parent) = file_path.parent() {
                tokio::fs::create_dir_all(parent)
                    .await
                    .map_err(|e| VfsError::Io(format!("mkdir {}: {e}", parent.display())))?;
            }
            use tokio::io::AsyncWriteExt;
            let mut file = tokio::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(&file_path)
                .await
                .map_err(|e| VfsError::Io(format!("open log {}: {e}", file_path.display())))?;
            file.write_all(partial.as_bytes())
                .await
                .map_err(|e| VfsError::Io(format!("append log: {e}")))?;
            file.write_all(b"\n")
                .await
                .map_err(|e| VfsError::Io(format!("append newline: {e}")))?;
            return Ok(());
        }
        self.write(path, partial).await
    }
}

// =============================================================================
// Host executor — direct local execution, no container isolation.
// Enabled via SANDBOX_MODE=host. Useful for running inside an already-sandboxed
// environment (e.g. Harbor/Docker benchmark containers).
// =============================================================================

struct HostExecutor {
    root: PathBuf,
}

impl HostExecutor {
    fn new(root: PathBuf) -> Self {
        Self { root }
    }
}

#[async_trait]
impl SandboxExecutor for HostExecutor {
    async fn ensure_container(
        &self,
        agent_config_id: &str,
        _sock_path: Option<&PathBuf>,
    ) -> Result<ContainerInfo, VfsError> {
        let workspace = self.root.join(agent_config_id);
        std::fs::create_dir_all(&workspace)
            .map_err(|e| VfsError::Io(format!("create host workspace {agent_config_id}: {e}")))?;
        Ok(ContainerInfo {
            container_id: format!("host-{agent_config_id}"),
            host_path: workspace,
        })
    }

    async fn exec_in_container(
        &self,
        container_id: &str,
        command: &str,
        cwd: &str,
    ) -> Result<ExecResult, VfsError> {
        // container_id is "host-{agent_config_id}", resolve workspace from root
        let agent_id = container_id.strip_prefix("host-").unwrap_or(container_id);
        let workspace = self.root.join(agent_id);

        // Map /workspace/... paths to the actual host workspace directory
        let actual_cwd = if cwd.starts_with(CONTAINER_WORKDIR) {
            let relative = cwd.strip_prefix(CONTAINER_WORKDIR).unwrap_or("");
            let relative = relative.strip_prefix('/').unwrap_or(relative);
            if relative.is_empty() {
                workspace.clone()
            } else {
                workspace.join(relative)
            }
        } else {
            PathBuf::from(cwd)
        };

        std::fs::create_dir_all(&actual_cwd)
            .map_err(|e| VfsError::Io(format!("create cwd {}: {e}", actual_cwd.display())))?;

        let output = tokio::process::Command::new("sh")
            .arg("-c")
            .arg(command)
            .current_dir(&actual_cwd)
            .output()
            .await
            .map_err(|e| VfsError::Io(format!("exec host command: {e}")))?;

        Ok(ExecResult {
            stdout: String::from_utf8_lossy(&output.stdout).to_string(),
            stderr: String::from_utf8_lossy(&output.stderr).to_string(),
            exit_code: output.status.code().unwrap_or(-1),
        })
    }

    async fn remove_container(&self, _container_id: &str) -> Result<(), VfsError> {
        Ok(()) // noop — no container to remove
    }
}

// =============================================================================
// Containerd executor (default)
// =============================================================================

use containerd_client as ctrd;
use ctrd::services::v1::{
    Container, CreateContainerRequest, CreateTaskRequest, ExecProcessRequest, StartRequest,
    WaitRequest, container::Runtime, containers_client::ContainersClient,
    tasks_client::TasksClient,
};
use ctrd::with_namespace;
// with_namespace! macro requires tonic::Request in scope — use containerd-client's re-export
use ctrd::tonic::Request;
use ctrd::tonic::transport::Channel as CtrdChannel;
use prost_types::Any;

const CONTAINERD_NS: &str = "logos";

struct ContainerdExecutor {
    channel: CtrdChannel,
    image: String,
}

impl ContainerdExecutor {
    async fn try_init(image: Option<String>) -> Result<Self, VfsError> {
        // Try multiple socket paths
        let paths = [
            std::env::var("CONTAINERD_SOCKET").unwrap_or_default(),
            format!(
                "{}/.colima/default/containerd.sock",
                std::env::var("HOME").unwrap_or_else(|_| "/root".to_string())
            ),
            "/run/containerd/containerd.sock".to_string(),
        ];

        let mut last_err = String::new();
        for path in &paths {
            if path.is_empty() || !std::path::Path::new(path).exists() {
                continue;
            }
            match ctrd::connect(path).await {
                Ok(channel) => {
                    // Verify connection with version check
                    let mut vc =
                        ctrd::services::v1::version_client::VersionClient::new(channel.clone());
                    match vc.version(()).await {
                        Ok(_) => {
                            let image = image.unwrap_or_else(|| {
                                "docker.io/library/debian:stable-slim".to_string()
                            });
                            return Ok(Self { channel, image });
                        }
                        Err(e) => last_err = format!("version check failed on {path}: {e}"),
                    }
                }
                Err(e) => last_err = format!("connect {path}: {e}"),
            }
        }
        Err(VfsError::Io(format!(
            "no containerd socket found: {last_err}"
        )))
    }

    /// Generate OCI runtime spec.
    ///
    /// RFC 002 §5: sandbox paths are mapped via overlayfs, not bind mounts.
    /// The container's rootfs IS the workspace — files written to the overlayfs
    /// upperdir are visible inside the container at /workspace.
    fn make_spec(sock_path: Option<&PathBuf>) -> String {
        let mut mounts = vec![
            r#"{"destination":"/proc","type":"proc","source":"proc"}"#.to_string(),
            r#"{"destination":"/dev","type":"tmpfs","source":"tmpfs","options":["nosuid","strictatime","mode=755","size=65536k"]}"#.to_string(),
        ];

        if let Some(sock) = sock_path {
            let sock_str = sock
                .canonicalize()
                .unwrap_or(sock.clone())
                .to_string_lossy()
                .to_string();
            mounts.push(format!(
                r#"{{"destination":"/logos.sock","type":"bind","source":"{}","options":["rbind","rw"]}}"#,
                sock_str
            ));
        }

        let mounts_json = mounts.join(",");

        format!(
            r#"{{
            "ociVersion": "1.0.2",
            "process": {{
                "terminal": false,
                "user": {{"uid": 0, "gid": 0}},
                "args": ["sleep", "infinity"],
                "env": ["PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"],
                "cwd": "/workspace"
            }},
            "root": {{"path": "rootfs", "readonly": false}},
            "mounts": [{mounts_json}],
            "linux": {{
                "namespaces": [
                    {{"type": "pid"}},
                    {{"type": "ipc"}},
                    {{"type": "uts"}},
                    {{"type": "mount"}}
                ]
            }}
        }}"#
        )
    }

    /// Pull and unpack image if not already present.
    async fn ensure_image(&self) -> Result<(), VfsError> {
        let mut images = ctrd::services::v1::images_client::ImagesClient::new(self.channel.clone());
        let req = ctrd::services::v1::GetImageRequest {
            name: self.image.clone(),
        };
        let req = with_namespace!(req, CONTAINERD_NS);

        // Check if image exists and is already unpacked
        if images.get(req).await.is_ok() {
            // Image record exists — check if a valid snapshot parent can be found
            // (this means the image layers have been unpacked into the snapshotter)
            if self.find_image_snapshot_parent().await.is_ok() {
                return Ok(()); // already pulled and unpacked
            }
        }

        println!("[logos] pulling image {} via containerd...", self.image);

        let mut transfer =
            ctrd::services::v1::transfer_client::TransferClient::new(self.channel.clone());

        let source = ctrd::types::transfer::OciRegistry {
            reference: self.image.clone(),
            ..Default::default()
        };
        let dest = ctrd::types::transfer::ImageStore {
            name: self.image.clone(),
            unpacks: vec![ctrd::types::transfer::UnpackConfiguration {
                platform: Some(ctrd::types::Platform {
                    os: "linux".to_string(),
                    architecture: std::env::consts::ARCH
                        .replace("aarch64", "arm64")
                        .replace("x86_64", "amd64"),
                    ..Default::default()
                }),
                snapshotter: "overlayfs".to_string(),
            }],
            ..Default::default()
        };

        let req = ctrd::services::v1::TransferRequest {
            source: Some(ctrd::to_any(&source)),
            destination: Some(ctrd::to_any(&dest)),
            options: None,
        };
        let req = with_namespace!(req, CONTAINERD_NS);
        transfer
            .transfer(req)
            .await
            .map_err(|e| VfsError::Io(format!("pull image {}: {e}", self.image)))?;

        println!("[logos] image {} ready", self.image);
        Ok(())
    }

    /// Find the snapshot parent key for the unpacked image.
    ///
    /// After image unpack, containerd creates committed snapshots for each layer.
    /// The top layer's chain ID is the snapshot key we need as parent.
    /// For multi-arch images, we find the snapshot matching current platform by
    /// looking for the most recently created committed snapshot with a sha256: prefix.
    async fn find_image_snapshot_parent(&self) -> Result<String, VfsError> {
        let mut snapshots = ctrd::services::v1::snapshots::snapshots_client::SnapshotsClient::new(
            self.channel.clone(),
        );

        let req = ctrd::services::v1::snapshots::ListSnapshotsRequest {
            snapshotter: "overlayfs".to_string(),
            filters: vec![],
        };
        let req = with_namespace!(req, CONTAINERD_NS);
        let resp = snapshots
            .list(req)
            .await
            .map_err(|e| VfsError::Io(format!("list snapshots: {e}")))?;

        use futures_util::StreamExt;
        let mut stream = resp.into_inner();
        let mut best: Option<(String, Option<prost_types::Timestamp>)> = None;

        while let Some(Ok(resp)) = stream.next().await {
            for info in resp.info {
                // Committed snapshots (kind == 3) with sha256: prefix are image layers
                if info.kind == 3 && info.name.starts_with("sha256:") {
                    let is_newer = match (&best, &info.created_at) {
                        (None, _) => true,
                        (Some((_, Some(prev))), Some(curr)) => {
                            (curr.seconds, curr.nanos) > (prev.seconds, prev.nanos)
                        }
                        (Some((_, None)), Some(_)) => true,
                        _ => false,
                    };
                    if is_newer {
                        best = Some((info.name, info.created_at));
                    }
                }
            }
        }

        best.map(|(name, _)| name).ok_or_else(|| {
            VfsError::Io(format!(
                "no committed snapshot found for image {}. Is the image unpacked?",
                self.image
            ))
        })
    }

    /// Get the overlayfs upperdir/workspace for an existing snapshot.
    async fn get_upperdir(&self, snapshot_key: &str) -> Result<PathBuf, VfsError> {
        let mut snapshots = ctrd::services::v1::snapshots::snapshots_client::SnapshotsClient::new(
            self.channel.clone(),
        );
        let req = ctrd::services::v1::snapshots::MountsRequest {
            snapshotter: "overlayfs".to_string(),
            key: snapshot_key.to_string(),
        };
        let req = with_namespace!(req, CONTAINERD_NS);
        let resp = snapshots
            .mounts(req)
            .await
            .map_err(|e| VfsError::Io(format!("get snapshot mounts: {e}")))?;

        for mount in resp.into_inner().mounts {
            for opt in &mount.options {
                if let Some(upper) = opt.strip_prefix("upperdir=") {
                    return Ok(PathBuf::from(upper).join("workspace"));
                }
            }
        }
        Err(VfsError::Io("no upperdir found in snapshot".to_string()))
    }
}

#[async_trait]
impl SandboxExecutor for ContainerdExecutor {
    async fn ensure_container(
        &self,
        agent_config_id: &str,
        sock_path: Option<&PathBuf>,
    ) -> Result<ContainerInfo, VfsError> {
        let container_id = format!("logos-sandbox-{agent_config_id}");

        // Check if container already exists
        let mut containers = ContainersClient::new(self.channel.clone());
        let req = ctrd::services::v1::GetContainerRequest {
            id: container_id.clone(),
        };
        let req = with_namespace!(req, CONTAINERD_NS);
        if containers.get(req).await.is_ok() {
            let mut tasks = TasksClient::new(self.channel.clone());
            let req = ctrd::services::v1::GetRequest {
                container_id: container_id.clone(),
                exec_id: String::new(),
            };
            let req = with_namespace!(req, CONTAINERD_NS);
            if tasks.get(req).await.is_ok() {
                // Find upperdir from existing snapshot
                let snapshot_key = format!("logos-snap-{agent_config_id}");
                let host_path = self
                    .get_upperdir(&snapshot_key)
                    .await
                    .unwrap_or_else(|_| PathBuf::from("/tmp"));
                return Ok(ContainerInfo {
                    container_id,
                    host_path,
                });
            }
            let req = ctrd::services::v1::DeleteContainerRequest {
                id: container_id.clone(),
            };
            let req = with_namespace!(req, CONTAINERD_NS);
            let _ = containers.delete(req).await;
        }

        // Pull and unpack image if needed
        self.ensure_image().await?;

        // Find the image's top committed snapshot (from unpack)
        let parent_key = self.find_image_snapshot_parent().await?;

        // Prepare active snapshot for this agent (layered on top of image)
        let snapshot_key = format!("logos-snap-{agent_config_id}");
        let mut snapshots = ctrd::services::v1::snapshots::snapshots_client::SnapshotsClient::new(
            self.channel.clone(),
        );

        // Remove old snapshot if exists
        let req = ctrd::services::v1::snapshots::RemoveSnapshotRequest {
            snapshotter: "overlayfs".to_string(),
            key: snapshot_key.clone(),
        };
        let req = with_namespace!(req, CONTAINERD_NS);
        let _ = snapshots.remove(req).await;

        // Prepare new snapshot with image as parent
        let req = ctrd::services::v1::snapshots::PrepareSnapshotRequest {
            snapshotter: "overlayfs".to_string(),
            key: snapshot_key.clone(),
            parent: parent_key,
            ..Default::default()
        };
        let req = with_namespace!(req, CONTAINERD_NS);
        let mounts_resp = snapshots
            .prepare(req)
            .await
            .map_err(|e| VfsError::Io(format!("prepare snapshot: {e}")))?;
        let mounts = mounts_resp.into_inner().mounts;

        // Find the overlayfs upperdir — this is where host-side file ops go
        let mut upperdir = PathBuf::new();
        for mount in &mounts {
            for opt in &mount.options {
                if let Some(upper) = opt.strip_prefix("upperdir=") {
                    upperdir = PathBuf::from(upper);
                }
            }
        }
        // Pre-create /workspace in upperdir so runc can use it as cwd
        let workspace_dir = upperdir.join("workspace");
        std::fs::create_dir_all(&workspace_dir)
            .map_err(|e| VfsError::Io(format!("create workspace in upperdir: {e}")))?;

        // Create OCI spec (no bind mounts — overlayfs IS the filesystem)
        let spec_json = Self::make_spec(sock_path);
        let spec = Any {
            type_url: "types.containerd.io/opencontainers/runtime-spec/1/Spec".to_string(),
            value: spec_json.into_bytes(),
        };

        // Create container
        let container = Container {
            id: container_id.clone(),
            image: self.image.clone(),
            runtime: Some(Runtime {
                name: "io.containerd.runc.v2".to_string(),
                options: None,
            }),
            spec: Some(spec),
            snapshotter: "overlayfs".to_string(),
            snapshot_key: snapshot_key.clone(),
            ..Default::default()
        };

        let req = CreateContainerRequest {
            container: Some(container),
        };
        let req = with_namespace!(req, CONTAINERD_NS);
        containers
            .create(req)
            .await
            .map_err(|e| VfsError::Io(format!("create container: {e}")))?;

        // Create task (the running process) with FIFOs
        let tmp_dir = std::env::temp_dir().join(format!("logos-{agent_config_id}"));
        let _ = std::fs::remove_dir_all(&tmp_dir);
        std::fs::create_dir_all(&tmp_dir)
            .map_err(|e| VfsError::Io(format!("create tmp dir: {e}")))?;

        let stdin_path = tmp_dir.join("stdin");
        let stdout_path = tmp_dir.join("stdout");
        let stderr_path = tmp_dir.join("stderr");
        create_fifo(&stdin_path)?;
        create_fifo(&stdout_path)?;
        create_fifo(&stderr_path)?;

        // Spawn temporary FIFO helpers BEFORE creating task (prevents FIFO deadlock).
        // These tasks are aborted right after create/start to avoid leaking detached tasks.
        let so = stdout_path.clone();
        let se = stderr_path.clone();
        let si = stdin_path.clone();
        let stdout_reader = tokio::spawn(async move {
            let _ = tokio::fs::read_to_string(so).await;
        });
        let stderr_reader = tokio::spawn(async move {
            let _ = tokio::fs::read_to_string(se).await;
        });
        let stdin_writer = tokio::spawn(async move {
            let _ = tokio::fs::OpenOptions::new().write(true).open(si).await;
        });

        let mut tasks = TasksClient::new(self.channel.clone());
        let create_and_start = async {
            let req = CreateTaskRequest {
                container_id: container_id.clone(),
                rootfs: mounts,
                stdin: stdin_path.to_string_lossy().to_string(),
                stdout: stdout_path.to_string_lossy().to_string(),
                stderr: stderr_path.to_string_lossy().to_string(),
                ..Default::default()
            };
            let req = with_namespace!(req, CONTAINERD_NS);
            tasks
                .create(req)
                .await
                .map_err(|e| VfsError::Io(format!("create task: {e}")))?;

            // Start task
            let req = StartRequest {
                container_id: container_id.clone(),
                ..Default::default()
            };
            let req = with_namespace!(req, CONTAINERD_NS);
            tasks
                .start(req)
                .await
                .map_err(|e| VfsError::Io(format!("start task: {e}")))?;
            Ok::<(), VfsError>(())
        }
        .await;

        stdout_reader.abort();
        stderr_reader.abort();
        stdin_writer.abort();

        create_and_start?;

        Ok(ContainerInfo {
            container_id,
            host_path: workspace_dir,
        })
    }

    async fn exec_in_container(
        &self,
        container_id: &str,
        command: &str,
        cwd: &str,
    ) -> Result<ExecResult, VfsError> {
        let exec_id = format!(
            "exec-{}",
            chrono::Utc::now().timestamp_nanos_opt().unwrap_or(0)
        );

        let tmp_dir = std::env::temp_dir().join(format!("logos-exec-{exec_id}"));
        let _ = std::fs::remove_dir_all(&tmp_dir);
        std::fs::create_dir_all(&tmp_dir)
            .map_err(|e| VfsError::Io(format!("create exec tmp: {e}")))?;

        let stdin_path = tmp_dir.join("stdin");
        let stdout_path = tmp_dir.join("stdout");
        let stderr_path = tmp_dir.join("stderr");
        create_fifo(&stdin_path)?;
        create_fifo(&stdout_path)?;
        create_fifo(&stderr_path)?;

        // Spawn FIFO readers BEFORE exec (prevents FIFO deadlock).
        // Use incremental read instead of read_to_string — the latter blocks
        // until EOF, but containerd shim may not close the write end promptly.
        let so = stdout_path.clone();
        let se = stderr_path.clone();
        let si = stdin_path.clone();
        let stdout_reader = tokio::spawn(async move { read_fifo_incremental(so).await });
        let stderr_reader = tokio::spawn(async move { read_fifo_incremental(se).await });
        let stdin_writer = tokio::spawn(async move {
            let _ = tokio::fs::OpenOptions::new().write(true).open(si).await;
        });

        // Build exec process spec.
        // Use the OCI process `cwd` field instead of embedding cwd in the shell
        // command string, which would allow injection via a crafted task_id.
        let process_spec = serde_json::json!({
            "terminal": false,
            "user": {"uid": 0, "gid": 0},
            "args": ["sh", "-c", command],
            "env": ["PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"],
            "cwd": cwd
        });
        let spec = Any {
            type_url: "types.containerd.io/opencontainers/runtime-spec/1/Process".to_string(),
            value: serde_json::to_vec(&process_spec).unwrap_or_default(),
        };

        let mut tasks = TasksClient::new(self.channel.clone());

        // Exec
        let req = ExecProcessRequest {
            container_id: container_id.to_string(),
            exec_id: exec_id.clone(),
            stdin: stdin_path.to_string_lossy().to_string(),
            stdout: stdout_path.to_string_lossy().to_string(),
            stderr: stderr_path.to_string_lossy().to_string(),
            terminal: false,
            spec: Some(spec),
        };
        let req = with_namespace!(req, CONTAINERD_NS);
        tasks
            .exec(req)
            .await
            .map_err(|e| VfsError::Io(format!("exec: {e}")))?;

        // Start the exec process
        let req = StartRequest {
            container_id: container_id.to_string(),
            exec_id: exec_id.clone(),
        };
        let req = with_namespace!(req, CONTAINERD_NS);
        tasks
            .start(req)
            .await
            .map_err(|e| VfsError::Io(format!("start exec: {e}")))?;

        // Wait for completion
        let req = WaitRequest {
            container_id: container_id.to_string(),
            exec_id: exec_id.clone(),
        };
        let req = with_namespace!(req, CONTAINERD_NS);
        let wait_resp = tasks
            .wait(req)
            .await
            .map_err(|e| VfsError::Io(format!("wait exec: {e}")))?;
        let exit_code = wait_resp.into_inner().exit_status as i32;

        // Read stdout/stderr from FIFO readers with timeout.
        // After wait returns, give readers a few seconds to finish draining.
        let timeout = std::time::Duration::from_secs(5);
        let stdout = match tokio::time::timeout(timeout, stdout_reader).await {
            Ok(Ok(s)) => s,
            _ => String::new(),
        };
        let stderr = match tokio::time::timeout(timeout, stderr_reader).await {
            Ok(Ok(s)) => s,
            _ => String::new(),
        };
        stdin_writer.abort();

        // Cleanup
        let _ = std::fs::remove_dir_all(&tmp_dir);

        Ok(ExecResult {
            stdout,
            stderr,
            exit_code,
        })
    }

    async fn remove_container(&self, container_id: &str) -> Result<(), VfsError> {
        let mut tasks = TasksClient::new(self.channel.clone());

        // Kill and delete task
        let req = ctrd::services::v1::KillRequest {
            container_id: container_id.to_string(),
            exec_id: String::new(),
            signal: 9, // SIGKILL
            ..Default::default()
        };
        let req = with_namespace!(req, CONTAINERD_NS);
        let _ = tasks.kill(req).await;

        // Wait for exit
        let req = WaitRequest {
            container_id: container_id.to_string(),
            exec_id: String::new(),
        };
        let req = with_namespace!(req, CONTAINERD_NS);
        let _ = tasks.wait(req).await;

        // Delete task
        let req = ctrd::services::v1::DeleteTaskRequest {
            container_id: container_id.to_string(),
        };
        let req = with_namespace!(req, CONTAINERD_NS);
        let _ = tasks.delete(req).await;

        // Delete container
        let mut containers = ContainersClient::new(self.channel.clone());
        let req = ctrd::services::v1::DeleteContainerRequest {
            id: container_id.to_string(),
        };
        let req = with_namespace!(req, CONTAINERD_NS);
        let _ = containers.delete(req).await;

        Ok(())
    }
}

/// Create a FIFO (named pipe) at the given path. Required by containerd for stdio.
/// Read from a FIFO incrementally. Opens the FIFO, reads chunks as they arrive,
/// and returns all collected data. Unlike read_to_string, this doesn't block
/// waiting for EOF — it returns when the read would block (no more data).
async fn read_fifo_incremental(path: std::path::PathBuf) -> String {
    use tokio::io::AsyncReadExt;

    let Ok(mut file) = tokio::fs::File::open(&path).await else {
        return String::new();
    };

    let mut buf = Vec::with_capacity(8192);
    let mut tmp = [0u8; 4096];
    loop {
        match tokio::time::timeout(std::time::Duration::from_millis(500), file.read(&mut tmp)).await
        {
            Ok(Ok(0)) => break, // EOF — writer closed
            Ok(Ok(n)) => buf.extend_from_slice(&tmp[..n]),
            Ok(Err(_)) => break, // read error
            Err(_) => break,     // 500ms no new data — done
        }
    }
    String::from_utf8_lossy(&buf).to_string()
}

fn create_fifo(path: &std::path::Path) -> Result<(), VfsError> {
    let _ = std::fs::remove_file(path);
    let c_path = std::ffi::CString::new(path.to_string_lossy().as_bytes())
        .map_err(|e| VfsError::Io(format!("invalid fifo path: {e}")))?;
    let ret = unsafe { libc::mkfifo(c_path.as_ptr(), 0o644) };
    if ret != 0 {
        return Err(VfsError::Io(format!(
            "mkfifo {}: {}",
            path.display(),
            std::io::Error::last_os_error()
        )));
    }
    Ok(())
}

// =============================================================================
// Helpers
// =============================================================================

/// Canonicalize a path that may not fully exist yet.
/// Walks up the ancestor chain until we find an existing prefix, canonicalizes
/// that, then re-appends the remaining non-existent suffix.  This lets us check
/// containment even for paths that will be created later.
fn canonicalize_partial(path: &std::path::Path) -> std::io::Result<std::path::PathBuf> {
    // Try the full path first (cheap path: already exists).
    if let Ok(c) = path.canonicalize() {
        return Ok(c);
    }
    // Walk up until we find an existing ancestor.
    let mut existing = path.to_path_buf();
    let mut suffix = std::path::PathBuf::new();
    loop {
        if existing.exists() {
            let canonical = existing.canonicalize()?;
            return Ok(canonical.join(suffix));
        }
        match existing.file_name() {
            Some(name) => {
                let name = name.to_owned();
                existing.pop();
                // Prepend name to suffix
                suffix = std::path::Path::new(&name).join(&suffix);
            }
            None => return Err(std::io::Error::new(std::io::ErrorKind::NotFound, "no existing ancestor")),
        }
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    async fn test_sandbox(name: &str) -> (SandboxNs, PathBuf) {
        let home = std::env::var("HOME").expect("HOME not set");
        let dir = PathBuf::from(home).join(".logos-test").join(name);
        let _ = std::fs::remove_dir_all(&dir);
        let sandbox = SandboxNs::init(dir.clone(), None).await.unwrap();
        (sandbox, dir)
    }

    async fn cleanup(sandbox: &SandboxNs, dir: &PathBuf) {
        let containers = sandbox.containers.lock().await;
        for (_, info) in containers.iter() {
            let _ = sandbox.executor.remove_container(&info.container_id).await;
        }
        drop(containers);
        let _ = std::fs::remove_dir_all(dir);
    }

    #[tokio::test]
    #[ignore]
    async fn exec_basic() {
        let (sandbox, dir) = test_sandbox("exec-basic").await;

        let info = sandbox.ensure_container("test-exec").await.unwrap();
        assert!(!info.container_id.is_empty());

        let result = sandbox
            .exec("echo 'hello from sandbox'", "test-exec", "test-exec")
            .await
            .unwrap();
        assert_eq!(result.exit_code, 0);
        assert!(result.stdout.contains("hello from sandbox"));

        cleanup(&sandbox, &dir).await;
    }

    #[tokio::test]
    #[ignore]
    async fn file_visibility() {
        let (sandbox, dir) = test_sandbox("file-vis").await;

        // Create container first to get the overlayfs host path
        let info = sandbox.ensure_container("test-vis").await.unwrap();

        // Write file to the overlayfs upperdir (visible inside container as /workspace/hello.txt)
        let file_path = info.host_path.join("hello.txt");
        std::fs::create_dir_all(info.host_path.parent().unwrap_or(&info.host_path)).unwrap();
        std::fs::write(&file_path, "world").unwrap();

        // Exec inside container should see the file
        let result = sandbox
            .exec("cat /workspace/test-vis/hello.txt", "test-vis", "test-vis")
            .await
            .unwrap();
        assert_eq!(result.exit_code, 0, "stderr: {}", result.stderr);
        assert_eq!(result.stdout.trim(), "world");

        cleanup(&sandbox, &dir).await;
    }

    #[tokio::test]
    #[ignore]
    async fn uri_translation() {
        let (sandbox, dir) = test_sandbox("uri-trans").await;

        // Create container first
        let info = sandbox.ensure_container("test-uri").await.unwrap();

        // Write to overlayfs upperdir
        std::fs::write(info.host_path.join("data.txt"), "translated!").unwrap();

        // URI translation: logos://sandbox/test-uri/data.txt → /workspace/data.txt
        let result = sandbox
            .exec(
                "cat logos://sandbox/test-uri/data.txt",
                "test-uri",
                "test-uri",
            )
            .await
            .unwrap();
        assert_eq!(result.exit_code, 0, "stderr: {}", result.stderr);
        assert_eq!(result.stdout.trim(), "translated!");

        cleanup(&sandbox, &dir).await;
    }
}
