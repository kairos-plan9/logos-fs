mod builtin_tools;
mod consolidator;
mod context;
mod cron;
mod devices;
mod grpc;
pub mod proc;
mod proc_store;
mod sandbox;
mod services;
pub mod tmp;
mod token;
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

#[cfg(all(target_os = "linux", target_env = "gnu"))]
#[global_allocator]
static GLOBAL_ALLOCATOR: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    load_env();

    // --- Boot sequence ---
    println!("[logos] booting...");

    let mut table = RoutingTable::new();

    // Middleware: JSON validation on system/ and memory/ writes (RFC 002 §12.6)
    table.add_middleware(Box::new(logos_vfs::JsonValidator));

    // 1. users/
    let users_root = env_path("VFS_USERS_ROOT", "../../data/state/entities");
    let users_ns = users::UsersNs::init(users_root)?;
    table.mount(Box::new(users_ns));
    println!("[logos] mounted logos://users/");

    // 2. memory/
    let memory_root = env_path("VFS_MEMORY_ROOT", "../../data/state/memory");
    let session_lance = memory_root.join("sessions.lance");
    let ollama_url = std::env::var("OLLAMA_URL").unwrap_or_else(|_| "http://localhost:11434".into());
    let embed_model = std::env::var("EMBED_MODEL").unwrap_or_else(|_| "qwen3-embedding:0.6b".into());
    let embed_dim: i32 = std::env::var("EMBED_DIM").ok().and_then(|v| v.parse().ok()).unwrap_or(1024);
    let (sessions, l2_disabled) = match logos_mm::SessionStore::with_lancedb(
        &session_lance, &ollama_url, &embed_model, embed_dim, 64, 256,
    ).await {
        Ok(s) => (Arc::new(s), false),
        Err(e) => {
            eprintln!("[logos] WARNING: session L2 init failed ({e}), falling back to in-memory only");
            (Arc::new(logos_mm::SessionStore::new(64, 256)), true)
        }
    };
    let memory_ns = logos_mm::MemoryModule::init(memory_root, Arc::clone(&sessions))?;
    let mm_arc = Arc::new(memory_ns);
    table.mount(Box::new(MemoryNsRef(Arc::clone(&mm_arc))));
    println!("[logos] mounted logos://memory/");

    // 3. system/
    let system_db = env_path("VFS_SYSTEM_DB", "../../data/state/system.db");
    let system_ns = logos_system::SystemModule::init(system_db).await?;
    let system_arc = Arc::new(system_ns);
    // Mount system as Namespace via a wrapper that shares the Arc
    table.mount(Box::new(SystemNsRef(Arc::clone(&system_arc))));
    println!("[logos] mounted logos://system/");

    // 4. tmp/
    table.mount(Box::new(tmp::TmpNs::new()));
    println!("[logos] mounted logos://tmp/");

    // 5. Create sandbox early (needed by proc tool restore), but mount LAST
    let sandbox_root = env_path("VFS_SANDBOX_ROOT", "../../data/state/sandbox");
    let sandbox_root_for_sock = sandbox_root.clone();
    let sandbox_image = std::env::var("SANDBOX_IMAGE").ok();
    let sandbox_ns = sandbox::SandboxNs::init(sandbox_root, sandbox_image).await?;
    let sandbox_arc = Arc::new(sandbox_ns);

    // 6. proc/ — register built-in tools + restore external tools from proc-store
    let mut proc_ns = proc::ProcNs::new();
    proc_ns.register(Arc::new(builtin_tools::MemorySearchTool {
        mm: Arc::clone(&mm_arc),
    }));
    proc_ns.register(Arc::new(builtin_tools::MemoryRangeFetchTool {
        mm: Arc::clone(&mm_arc),
    }));
    proc_ns.register(Arc::new(builtin_tools::SystemSearchTasksTool {
        system: Arc::clone(&system_arc),
    }));
    proc_ns.register(Arc::new(builtin_tools::SystemGetContextTool {
        mm: Arc::clone(&mm_arc),
        sessions: Arc::clone(&sessions),
    }));
    proc_ns.register(Arc::new(builtin_tools::SystemCompleteTool {
        system: Arc::clone(&system_arc),
        sandbox: Arc::clone(&sandbox_arc),
    }));
    proc_ns.register(Arc::new(builtin_tools::WebSearchTool::new()));

    // 6b. pinchtab — optional browser control
    let mut pinchtab_child = builtin_tools::browse::spawn_pinchtab().await;
    if pinchtab_child.is_some() {
        proc_ns.register(Arc::new(builtin_tools::BrowseTool::new(
            "http://127.0.0.1:9867".to_string(),
        )));
        println!("[logos] registered browse tool (pinchtab)");
    }

    // 7. proc-store/ — persistent proc tool declarations
    let proc_store_root = env_path("VFS_PROC_STORE_ROOT", "../../data/state/proc-store");
    let proc_store_ns = proc_store::ProcStoreNs::init(proc_store_root)?;
    let external_tools = proc_store_ns.restore_tools(&sandbox_arc).await.unwrap_or_default();
    for tool in external_tools {
        proc_ns.register(tool);
    }
    table.mount(Box::new(proc_store_ns));
    println!("[logos] mounted logos://proc-store/");

    let proc_arc = Arc::new(proc_ns);
    table.mount(Box::new(ProcNsRef(Arc::clone(&proc_arc))));
    println!("[logos] mounted logos://proc/");

    // 8. services/ — runtime service registry
    let svc_store_root = env_path("VFS_SVC_STORE_ROOT", "../../data/state/svc-store");
    let services_ns = services::ServicesNs::init(svc_store_root.clone())?;
    let restored = services_ns.restore_from_store().await.unwrap_or(0);
    table.mount(Box::new(services_ns));
    println!("[logos] mounted logos://services/ ({restored} restored from svc-store)");

    // 9. svc-store/ — persistent service declarations
    let svc_store_ns = services::SvcStoreNs::init(svc_store_root)?;
    table.mount(Box::new(svc_store_ns));
    println!("[logos] mounted logos://svc-store/");

    // 10. devices/
    let mut devices_ns = devices::DevicesNs::new();
    devices_ns.register(Arc::new(devices::MacSystemDriver));
    table.mount(Box::new(devices_ns));
    println!("[logos] mounted logos://devices/");

    // 11. sandbox/ — LAST to mount (RFC 002 §8.2: sandbox signals kernel ready)
    table.mount(Box::new(SandboxNsRef(Arc::clone(&sandbox_arc))));
    println!("[logos] mounted logos://sandbox/");

    // --- Open (RFC 002 §8.2: sandbox registration signals kernel is complete) ---
    table.open();
    println!(
        "[logos] kernel ready — {} namespace(s) mounted",
        table.mounted().len()
    );
    if l2_disabled {
        eprintln!("[logos] WARNING: session L2 disabled — sessions are NOT persisted across restarts");
    }

    // --- Consolidator cron jobs (RFC 003 §4) ---
    let scheduler = Arc::new(cron::CronScheduler::new(Arc::clone(&system_arc)));
    consolidator::register_consolidator_jobs(&scheduler).await;
    scheduler.start();

    // --- Serve ---
    let table = Arc::new(table);
    let tokens = token::TokenRegistry::new();
    let service = grpc::LogosService::new(
        Arc::clone(&table),
        system_arc,
        sandbox_arc,
        proc_arc,
        tokens,
    );
    let grpc_service = pb::logos_server::LogosServer::new(service);

    // RFC 002 §11.1: socket at convention path inside sandbox root
    let default_sock = format!("unix://{}/logos.sock", sandbox_root_for_sock.display());
    let listen = env_str("VFS_LISTEN", &default_sock);
    if let Some(socket_path) = parse_uds_path(&listen) {
        prepare_unix_socket(&socket_path)?;
        let listener = UnixListener::bind(&socket_path)?;
        // Make socket accessible to non-root agents
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            // 0o660: owner+group read/write only; avoids world-accessible socket.
            let _ = std::fs::set_permissions(&socket_path, std::fs::Permissions::from_mode(0o660));
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

    // Cleanup: stop pinchtab if we spawned it
    if let Some(ref mut child) = pinchtab_child {
        println!("[logos] stopping pinchtab...");
        let _ = child.kill().await;
    }

    Ok(())
}

// --- SystemModule Namespace wrapper ---
// SystemModule implements Namespace, but we need to mount it as Box<dyn Namespace>
// while also keeping an Arc for the Complete handler. This wrapper delegates.
struct MemoryNsRef(Arc<logos_mm::MemoryModule>);

#[async_trait::async_trait]
impl logos_vfs::Namespace for MemoryNsRef {
    fn name(&self) -> &str { self.0.name() }
    async fn read(&self, path: &[&str]) -> Result<String, logos_vfs::VfsError> { self.0.read(path).await }
    async fn write(&self, path: &[&str], content: &str) -> Result<(), logos_vfs::VfsError> { self.0.write(path, content).await }
    async fn patch(&self, path: &[&str], partial: &str) -> Result<(), logos_vfs::VfsError> { self.0.patch(path, partial).await }
}

struct SandboxNsRef(Arc<sandbox::SandboxNs>);

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
