mod builtin_tools;
#[cfg(feature = "chat")]
mod consolidator;
#[cfg(feature = "chat")]
mod context;
mod cron;
#[cfg(feature = "chat")]
mod devices;
mod embedder;
mod grpc;
mod memory_store;
pub mod proc;
mod proc_store;
#[cfg(feature = "sandbox")]
mod sandbox;
mod services;
pub mod tmp;
mod token;
#[cfg(feature = "chat")]
pub mod users;

use std::path::PathBuf;
use std::sync::Arc;
use std::{io, net::SocketAddr};

use tokio::net::UnixListener;
use tokio_stream::wrappers::UnixListenerStream;
use tonic::transport::Server;

use logos_vfs::RoutingTable;

pub mod pb {
    tonic::include_proto!("logos.kernel.v1");
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    load_env();

    // --- Boot sequence ---
    println!("[logos] booting...");

    let mut table = RoutingTable::new();

    // Middleware: JSON validation on system/ and memory/ writes (RFC 002 §12.6)
    table.add_middleware(Box::new(logos_vfs::JsonValidator));

    // 1. users/ (chat feature)
    #[cfg(feature = "chat")]
    {
        let users_root = env_path("VFS_USERS_ROOT", "../../data/state/entities");
        let users_ns = users::UsersNs::init(users_root)?;
        table.mount(Box::new(users_ns));
        println!("[logos] mounted logos://users/");
    }

    // 2. memory/ (chat feature)
    #[cfg(feature = "chat")]
    let mm_arc;
    #[cfg(feature = "chat")]
    let sessions;
    #[cfg(feature = "chat")]
    let l2_disabled;
    #[cfg(feature = "chat")]
    {
        let memory_root = env_path("VFS_MEMORY_ROOT", "../../data/state/memory");
        let session_lance = memory_root.join("sessions.lance");
        let ollama_url = std::env::var("OLLAMA_URL").unwrap_or_else(|_| "http://localhost:11434".into());
        let embed_model = std::env::var("EMBED_MODEL").unwrap_or_else(|_| "qwen3-embedding:0.6b".into());
        let embed_dim: i32 = std::env::var("EMBED_DIM").ok().and_then(|v| v.parse().ok()).unwrap_or(1024);
        let (s, disabled) = match logos_mm::SessionStore::with_lancedb(
            &session_lance, &ollama_url, &embed_model, embed_dim, 64, 256,
        ).await {
            Ok(s) => (Arc::new(s), false),
            Err(e) => {
                eprintln!("[logos] WARNING: session L2 init failed ({e}), falling back to in-memory only");
                (Arc::new(logos_mm::SessionStore::new(64, 256)), true)
            }
        };
        sessions = s;
        l2_disabled = disabled;
        let memory_ns = logos_mm::MemoryModule::init(memory_root, Arc::clone(&sessions))?;
        mm_arc = Arc::new(memory_ns);
        table.mount(Box::new(MemoryNsRef(Arc::clone(&mm_arc))));
        println!("[logos] mounted logos://memory/");
    }

    // 3. system/
    let system_db = env_path("VFS_SYSTEM_DB", "../../data/state/system.db");
    let system_ns = logos_system::SystemModule::init(system_db).await?;
    let system_arc = Arc::new(system_ns);
    table.mount(Box::new(SystemNsRef(Arc::clone(&system_arc))));
    println!("[logos] mounted logos://system/");

    // 4. tmp/
    table.mount(Box::new(tmp::TmpNs::new()));
    println!("[logos] mounted logos://tmp/");

    // 5. sandbox/ (sandbox feature)
    #[cfg(feature = "sandbox")]
    let sandbox_arc;
    #[cfg(feature = "sandbox")]
    let sandbox_root_for_sock;
    #[cfg(feature = "sandbox")]
    {
        let sandbox_root = env_path("VFS_SANDBOX_ROOT", "../../data/state/sandbox");
        sandbox_root_for_sock = sandbox_root.clone();
        let sandbox_image = std::env::var("SANDBOX_IMAGE").ok();
        let sandbox_ns = sandbox::SandboxNs::init(sandbox_root, sandbox_image).await?;
        sandbox_arc = Arc::new(sandbox_ns);
    }

    // 6. proc/ — register built-in tools
    let mut proc_ns = proc::ProcNs::new();
    #[cfg(feature = "chat")]
    {
        proc_ns.register(Arc::new(builtin_tools::MemorySearchTool {
            mm: Arc::clone(&mm_arc),
        }));
        proc_ns.register(Arc::new(builtin_tools::MemoryRangeFetchTool {
            mm: Arc::clone(&mm_arc),
        }));
    }
    proc_ns.register(Arc::new(builtin_tools::SystemSearchTasksTool {
        system: Arc::clone(&system_arc),
    }));
    #[cfg(feature = "chat")]
    {
        proc_ns.register(Arc::new(builtin_tools::SystemGetContextTool {
            mm: Arc::clone(&mm_arc),
            sessions: Arc::clone(&sessions),
        }));
    }
    #[cfg(feature = "sandbox")]
    {
        proc_ns.register(Arc::new(builtin_tools::SystemCompleteTool {
            system: Arc::clone(&system_arc),
            sandbox: Arc::clone(&sandbox_arc),
        }));
    }
    proc_ns.register(Arc::new(builtin_tools::WebSearchTool::new()));
    proc_ns.register(Arc::new(builtin_tools::FetchUrlTool::new()));

    // 6b. pinchtab — optional browser control
    let mut pinchtab_child = builtin_tools::browse::spawn_pinchtab().await;
    if pinchtab_child.is_some() {
        proc_ns.register(Arc::new(builtin_tools::BrowseTool::new(
            "http://127.0.0.1:9867".to_string(),
        )));
        println!("[logos] registered browse tool (pinchtab)");
    }

    // 7. proc-store/
    let proc_store_root = env_path("VFS_PROC_STORE_ROOT", "../../data/state/proc-store");
    let proc_store_ns = proc_store::ProcStoreNs::init(proc_store_root)?;
    #[cfg(feature = "sandbox")]
    {
        let external_tools = proc_store_ns.restore_tools(&sandbox_arc).await.unwrap_or_default();
        for tool in external_tools {
            proc_ns.register(tool);
        }
    }
    table.mount(Box::new(proc_store_ns));
    println!("[logos] mounted logos://proc-store/");

    // 8b. execution memory (Milvus-lite sidecar + memory stores)
    let _milvus_sidecar = spawn_milvus_sidecar().await;
    if let Some(milvus_client) = connect_milvus().await {
        let milvus_arc = Arc::new(milvus_client);
        let ollama_embedder = Arc::new(embedder::OllamaEmbedder::from_env());

        let crystal_store = Arc::new(memory_store::CrystalMemoryStore::new(
            Arc::clone(&milvus_arc),
            Arc::clone(&ollama_embedder),
        ));
        let tool_mem_store = Arc::new(memory_store::ToolMemoryStore::new(
            Arc::clone(&milvus_arc),
            Arc::clone(&ollama_embedder),
        ));

        if let Err(e) = seed_crystals(&crystal_store).await {
            eprintln!("[logos] WARNING: crystal seed failed: {e}");
        }

        proc_ns.set_memory_stores(crystal_store, tool_mem_store);
        println!("[logos] execution memory ready (milvus + ollama)");
    } else {
        eprintln!("[logos] WARNING: milvus not available — execution memory disabled");
    }

    let proc_arc = Arc::new(proc_ns);
    table.mount(Box::new(ProcNsRef(Arc::clone(&proc_arc))));
    println!("[logos] mounted logos://proc/");

    // 8. services/
    let svc_store_root = env_path("VFS_SVC_STORE_ROOT", "../../data/state/svc-store");
    let services_ns = services::ServicesNs::init(svc_store_root.clone())?;
    let restored = services_ns.restore_from_store().await.unwrap_or(0);
    table.mount(Box::new(services_ns));
    println!("[logos] mounted logos://services/ ({restored} restored from svc-store)");

    // 9. svc-store/
    let svc_store_ns = services::SvcStoreNs::init(svc_store_root)?;
    table.mount(Box::new(svc_store_ns));
    println!("[logos] mounted logos://svc-store/");

    // 10. devices/ (chat feature)
    #[cfg(feature = "chat")]
    {
        let mut devices_ns = devices::DevicesNs::new();
        devices_ns.register(Arc::new(devices::MacSystemDriver));
        table.mount(Box::new(devices_ns));
        println!("[logos] mounted logos://devices/");
    }

    // 11. sandbox/ mount
    #[cfg(feature = "sandbox")]
    {
        table.mount(Box::new(SandboxNsRef(Arc::clone(&sandbox_arc))));
        println!("[logos] mounted logos://sandbox/");
    }

    // 12. session/ — TUI coding agent workspace (optional, enabled via VFS_SESSION_ROOT)
    let session_ns_arc = if let Ok(session_root) = std::env::var("VFS_SESSION_ROOT") {
        let sns = logos_session_ns::SessionNs::new(PathBuf::from(&session_root));
        let arc = Arc::new(sns);
        table.mount(Box::new(SessionNsRef(Arc::clone(&arc))));
        println!("[logos] mounted logos://session/ (root: {session_root})");
        Some(arc)
    } else {
        None
    };

    // --- Open ---
    table.open();
    println!(
        "[logos] kernel ready — {} namespace(s) mounted",
        table.mounted().len()
    );
    #[cfg(feature = "chat")]
    if l2_disabled {
        eprintln!("[logos] WARNING: session L2 disabled — sessions are NOT persisted across restarts");
    }

    // --- Consolidator cron jobs (chat feature) ---
    #[cfg(feature = "chat")]
    {
        let scheduler = Arc::new(cron::CronScheduler::new(Arc::clone(&system_arc)));
        consolidator::register_consolidator_jobs(&scheduler).await;
        scheduler.start();
    }

    // --- Serve ---
    let table = Arc::new(table);
    let tokens = token::TokenRegistry::new();
    let service = grpc::LogosService::new(
        Arc::clone(&table),
        system_arc,
        proc_arc,
        tokens,
    );
    let service = if let Some(sns) = session_ns_arc {
        service.with_session_ns(sns)
    } else {
        service
    };
    let grpc_service = pb::logos_server::LogosServer::new(service);

    // Listen
    #[cfg(feature = "sandbox")]
    let default_sock = format!("unix://{}/logos.sock", sandbox_root_for_sock.display());
    #[cfg(not(feature = "sandbox"))]
    let default_sock = {
        let sock_dir = env_path("VFS_SOCKET_DIR", "../../data/state");
        format!("unix://{}/logos.sock", sock_dir.display())
    };
    let listen = env_str("VFS_LISTEN", &default_sock);
    if let Some(socket_path) = parse_uds_path(&listen) {
        prepare_unix_socket(&socket_path)?;
        let listener = UnixListener::bind(&socket_path)?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let _ = std::fs::set_permissions(&socket_path, std::fs::Permissions::from_mode(0o777));
        }
        println!("[logos] listening on unix://{}", socket_path.display());
        Server::builder()
            .add_service(grpc_service)
            .serve_with_incoming(UnixListenerStream::new(listener))
            .await?;
    } else {
        let addr: SocketAddr = listen.parse()?;
        println!("[logos] listening on {addr}");
        Server::builder()
            .add_service(grpc_service)
            .serve(addr)
            .await?;
    }

    // Cleanup
    if let Some(ref mut child) = pinchtab_child {
        println!("[logos] stopping pinchtab...");
        let _ = child.kill().await;
    }

    Ok(())
}

// --- Namespace wrappers (Arc → Box<dyn Namespace>) ---

#[cfg(feature = "chat")]
struct MemoryNsRef(Arc<logos_mm::MemoryModule>);

#[cfg(feature = "chat")]
#[async_trait::async_trait]
impl logos_vfs::Namespace for MemoryNsRef {
    fn name(&self) -> &str { self.0.name() }
    async fn read(&self, path: &[&str]) -> Result<String, logos_vfs::VfsError> { self.0.read(path).await }
    async fn write(&self, path: &[&str], content: &str) -> Result<(), logos_vfs::VfsError> { self.0.write(path, content).await }
    async fn patch(&self, path: &[&str], partial: &str) -> Result<(), logos_vfs::VfsError> { self.0.patch(path, partial).await }
}

#[cfg(feature = "sandbox")]
struct SandboxNsRef(Arc<sandbox::SandboxNs>);

#[cfg(feature = "sandbox")]
#[async_trait::async_trait]
impl logos_vfs::Namespace for SandboxNsRef {
    fn name(&self) -> &str { self.0.name() }
    async fn read(&self, path: &[&str]) -> Result<String, logos_vfs::VfsError> { self.0.read(path).await }
    async fn write(&self, path: &[&str], content: &str) -> Result<(), logos_vfs::VfsError> { self.0.write(path, content).await }
    async fn patch(&self, path: &[&str], partial: &str) -> Result<(), logos_vfs::VfsError> { self.0.patch(path, partial).await }
}

struct SystemNsRef(Arc<logos_system::SystemModule>);

#[async_trait::async_trait]
impl logos_vfs::Namespace for SystemNsRef {
    fn name(&self) -> &str {
        self.0.name()
    }
    async fn read(&self, path: &[&str]) -> Result<String, logos_vfs::VfsError> {
        self.0.read(path).await
    }
    async fn write(&self, path: &[&str], content: &str) -> Result<(), logos_vfs::VfsError> {
        self.0.write(path, content).await
    }
    async fn patch(&self, path: &[&str], partial: &str) -> Result<(), logos_vfs::VfsError> {
        self.0.patch(path, partial).await
    }
}

struct ProcNsRef(Arc<proc::ProcNs>);

#[async_trait::async_trait]
impl logos_vfs::Namespace for ProcNsRef {
    fn name(&self) -> &str { self.0.name() }
    async fn read(&self, path: &[&str]) -> Result<String, logos_vfs::VfsError> { self.0.read(path).await }
    async fn write(&self, path: &[&str], content: &str) -> Result<(), logos_vfs::VfsError> { self.0.write(path, content).await }
    async fn patch(&self, path: &[&str], partial: &str) -> Result<(), logos_vfs::VfsError> { self.0.patch(path, partial).await }
}

struct SessionNsRef(Arc<logos_session_ns::SessionNs>);

#[async_trait::async_trait]
impl logos_vfs::Namespace for SessionNsRef {
    fn name(&self) -> &str { self.0.name() }
    async fn read(&self, path: &[&str]) -> Result<String, logos_vfs::VfsError> { self.0.read(path).await }
    async fn write(&self, path: &[&str], content: &str) -> Result<(), logos_vfs::VfsError> { self.0.write(path, content).await }
    async fn patch(&self, path: &[&str], partial: &str) -> Result<(), logos_vfs::VfsError> { self.0.patch(path, partial).await }
}

// --- Helpers ---

fn env_str(key: &str, default: &str) -> String {
    std::env::var(key).unwrap_or_else(|_| default.to_string())
}

fn env_path(key: &str, default: &str) -> PathBuf {
    if let Ok(val) = std::env::var(key) {
        return PathBuf::from(val);
    }
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(default);
    // Canonicalize to resolve ../../ — avoids exceeding macOS SUN_LEN (104) for UDS paths
    path.canonicalize().unwrap_or(path)
}

fn parse_uds_path(listen: &str) -> Option<PathBuf> {
    let path = listen
        .strip_prefix("unix://")
        .or_else(|| listen.strip_prefix("unix:"))?;
    if path.is_empty() {
        return None;
    }
    Some(PathBuf::from(path))
}

fn prepare_unix_socket(socket_path: &PathBuf) -> io::Result<()> {
    if let Some(parent) = socket_path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    match std::fs::remove_file(socket_path) {
        Ok(_) => {}
        Err(err) if err.kind() == io::ErrorKind::NotFound => {}
        Err(err) => return Err(err),
    }
    Ok(())
}

fn load_env() {
    let env_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join(".env");
    let _ = dotenvy::from_path(env_path);
}

async fn spawn_milvus_sidecar() -> Option<tokio::process::Child> {
    let ready_file = std::env::var("MILVUS_READY_FILE").unwrap_or_else(|_| "/tmp/milvus-ready".into());
    if std::path::Path::new(&ready_file).exists() {
        println!("[logos] milvus sidecar already running");
        return None;
    }

    let script = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../../scripts/milvus-sidecar.py");
    if !script.exists() {
        println!("[logos] milvus sidecar script not found at {}", script.display());
        return None;
    }

    let child = tokio::process::Command::new("python3")
        .arg(&script)
        .spawn()
        .ok();

    if child.is_some() {
        for _ in 0..30 {
            if std::path::Path::new(&ready_file).exists() {
                println!("[logos] milvus sidecar ready");
                return child;
            }
            tokio::time::sleep(std::time::Duration::from_secs(1)).await;
        }
        eprintln!("[logos] WARNING: milvus sidecar did not become ready in 30s");
    }
    child
}

async fn connect_milvus() -> Option<milvus::client::Client> {
    let port = std::env::var("MILVUS_PORT").unwrap_or_else(|_| "19530".into());
    let url: &'static str = Box::leak(format!("http://localhost:{port}").into_boxed_str());
    match milvus::client::Client::new(url).await {
        Ok(c) => Some(c),
        Err(e) => {
            eprintln!("[logos] milvus connect failed: {e}");
            None
        }
    }
}

async fn seed_crystals(store: &memory_store::CrystalMemoryStore) -> Result<(), logos_vfs::VfsError> {
    let crystals_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../../data/crystals");
    if !crystals_dir.exists() {
        println!("[logos] no crystals dir at {}, skipping seed", crystals_dir.display());
        return Ok(());
    }

    let mut all_seeds = Vec::new();
    let mut entries = tokio::fs::read_dir(&crystals_dir)
        .await
        .map_err(|e| logos_vfs::VfsError::Io(format!("read crystals dir: {e}")))?;

    while let Some(entry) = entries.next_entry().await.map_err(|e| logos_vfs::VfsError::Io(format!("{e}")))? {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) == Some("yaml") {
            let content = tokio::fs::read_to_string(&path)
                .await
                .map_err(|e| logos_vfs::VfsError::Io(format!("read {}: {e}", path.display())))?;
            let seeds: Vec<memory_store::CrystalSeed> = serde_yaml::from_str(&content)
                .map_err(|e| logos_vfs::VfsError::Io(format!("parse {}: {e}", path.display())))?;
            all_seeds.extend(seeds);
        }
    }

    if !all_seeds.is_empty() {
        println!("[logos] seeding {} crystal(s)...", all_seeds.len());
        store.seed(&all_seeds).await?;
    }
    Ok(())
}
