use std::path::{Path, PathBuf};

use async_trait::async_trait;
use logos_vfs::{Namespace, VfsError};

pub struct ExecResult {
    pub stdout: String,
    pub stderr: String,
    pub exit_code: i32,
}

pub struct SessionNs {
    mount_point: PathBuf,
}

impl SessionNs {
    pub fn new(mount_point: PathBuf) -> Self {
        Self { mount_point }
    }

    fn session_root(&self, sid: &str) -> PathBuf {
        self.mount_point.join("session").join(sid)
    }

    fn workspace_path(&self, sid: &str) -> PathBuf {
        self.session_root(sid).join("workspace")
    }

    fn resolve_path(&self, path: &[&str]) -> Result<PathBuf, VfsError> {
        if path.is_empty() {
            return Ok(self.mount_point.join("session"));
        }
        let sid = path[0];
        if path.len() == 1 {
            return Ok(self.session_root(sid));
        }
        Ok(self.session_root(sid).join(path[1..].join("/")))
    }

    fn translate_uris(&self, command: &str, sid: &str) -> String {
        let workspace = self.workspace_path(sid);
        let ws_str = workspace.to_string_lossy();
        let session_prefix = format!("logos://session/{sid}/workspace/");
        let session_prefix_no_slash = format!("logos://session/{sid}/workspace");
        command
            .replace(&session_prefix, &format!("{ws_str}/"))
            .replace(&session_prefix_no_slash, &ws_str.to_string())
    }

    pub async fn exec(&self, sid: &str, command: &str) -> Result<ExecResult, VfsError> {
        let cwd = self.workspace_path(sid);
        if !cwd.exists() {
            return Err(VfsError::NotFound(format!(
                "workspace not found: {}",
                cwd.display()
            )));
        }

        let translated = self.translate_uris(command, sid);

        let output = tokio::process::Command::new("sh")
            .arg("-c")
            .arg(&translated)
            .current_dir(&cwd)
            .output()
            .await
            .map_err(|e| VfsError::Io(format!("exec: {e}")))?;

        Ok(ExecResult {
            stdout: String::from_utf8_lossy(&output.stdout).to_string(),
            stderr: String::from_utf8_lossy(&output.stderr).to_string(),
            exit_code: output.status.code().unwrap_or(-1),
        })
    }

    async fn run_juicefs(&self, args: &[&str]) -> Result<String, VfsError> {
        let output = tokio::process::Command::new("juicefs")
            .args(args)
            .output()
            .await
            .map_err(|e| VfsError::Io(format!("juicefs: {e}")))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(VfsError::Io(format!(
                "juicefs {} failed (exit {}): {}",
                args.first().unwrap_or(&""),
                output.status.code().unwrap_or(-1),
                stderr.trim()
            )));
        }

        Ok(String::from_utf8_lossy(&output.stdout).to_string())
    }

    pub async fn create_session(&self, project_path: &str, sid: &str) -> Result<(), VfsError> {
        let project = Path::new(project_path);
        let session_root = self.session_root(sid);
        let workspace = self.workspace_path(sid);

        let projects_dir = self.mount_point.join("projects");
        std::fs::create_dir_all(&projects_dir)
            .map_err(|e| VfsError::Io(format!("mkdir projects: {e}")))?;

        let project_name = project
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("default");
        let jfs_project = projects_dir.join(project_name);

        if !jfs_project.exists() {
            let output = tokio::process::Command::new("rsync")
                .args([
                    "-a",
                    "--filter=:- .gitignore",
                    "--exclude=node_modules",
                    "--exclude=target",
                    "--exclude=__pycache__",
                    "--exclude=.venv",
                    "--exclude=venv",
                    "--exclude=.tox",
                    "--exclude=dist",
                    "--exclude=build",
                    &format!("{}/", project_path),
                    &format!("{}/", jfs_project.to_string_lossy()),
                ])
                .output()
                .await
                .map_err(|e| VfsError::Io(format!("rsync: {e}")))?;
            if !output.status.success() {
                let stderr = String::from_utf8_lossy(&output.stderr);
                return Err(VfsError::Io(format!("rsync failed: {}", stderr.trim())));
            }
        }

        std::fs::create_dir_all(&session_root)
            .map_err(|e| VfsError::Io(format!("mkdir session: {e}")))?;

        self.run_juicefs(&[
            "clone",
            &jfs_project.to_string_lossy(),
            &workspace.to_string_lossy(),
        ])
        .await?;

        std::fs::create_dir_all(session_root.join("state"))
            .map_err(|e| VfsError::Io(format!("mkdir state: {e}")))?;
        std::fs::create_dir_all(session_root.join("checkpoints"))
            .map_err(|e| VfsError::Io(format!("mkdir checkpoints: {e}")))?;

        Ok(())
    }

    pub async fn checkpoint(&self, sid: &str, checkpoint_id: &str) -> Result<(), VfsError> {
        let session_root = self.session_root(sid);
        let workspace = self.workspace_path(sid);
        let cp_dir = session_root.join("checkpoints").join(checkpoint_id);

        std::fs::create_dir_all(&cp_dir)
            .map_err(|e| VfsError::Io(format!("mkdir checkpoint: {e}")))?;

        let snapshot = cp_dir.join("snapshot");
        self.run_juicefs(&[
            "clone",
            &workspace.to_string_lossy(),
            &snapshot.to_string_lossy(),
        ])
        .await?;

        let state_src = session_root.join("state").join("PROJECT_STATE.md");
        let state_dst = cp_dir.join("PROJECT_STATE.md");
        if state_src.exists() {
            tokio::fs::copy(&state_src, &state_dst)
                .await
                .map_err(|e| VfsError::Io(format!("copy state: {e}")))?;
        }

        Ok(())
    }

    pub async fn rollback(&self, sid: &str, checkpoint_id: &str) -> Result<(), VfsError> {
        let session_root = self.session_root(sid);
        let workspace = self.workspace_path(sid);
        let cp_dir = session_root.join("checkpoints").join(checkpoint_id);
        let snapshot = cp_dir.join("snapshot");

        if !snapshot.exists() {
            return Err(VfsError::NotFound(format!(
                "checkpoint snapshot not found: {}",
                snapshot.display()
            )));
        }

        if workspace.exists() {
            tokio::fs::remove_dir_all(&workspace)
                .await
                .map_err(|e| VfsError::Io(format!("rm workspace: {e}")))?;
        }

        self.run_juicefs(&[
            "clone",
            &snapshot.to_string_lossy(),
            &workspace.to_string_lossy(),
        ])
        .await?;

        let state_backup = cp_dir.join("PROJECT_STATE.md");
        let state_dst = session_root.join("state").join("PROJECT_STATE.md");
        if state_backup.exists() {
            tokio::fs::copy(&state_backup, &state_dst)
                .await
                .map_err(|e| VfsError::Io(format!("restore state: {e}")))?;
        }

        Ok(())
    }

    pub async fn fork(&self, from_sid: &str, new_sid: &str) -> Result<(), VfsError> {
        let src = self.session_root(from_sid);
        let dst = self.session_root(new_sid);

        if !src.exists() {
            return Err(VfsError::NotFound(format!(
                "source session not found: {}",
                src.display()
            )));
        }

        self.run_juicefs(&[
            "clone",
            &src.to_string_lossy(),
            &dst.to_string_lossy(),
        ])
        .await?;

        Ok(())
    }

    pub async fn accept(
        &self,
        sid: &str,
        files: &[String],
        dest: &str,
    ) -> Result<(), VfsError> {
        let workspace = self.workspace_path(sid);

        for file in files {
            let src = workspace.join(file);
            let dst = Path::new(dest).join(file);
            if !src.exists() {
                continue;
            }
            if let Some(parent) = dst.parent() {
                tokio::fs::create_dir_all(parent)
                    .await
                    .map_err(|e| VfsError::Io(format!("mkdir {}: {e}", parent.display())))?;
            }
            tokio::fs::copy(&src, &dst)
                .await
                .map_err(|e| VfsError::Io(format!("copy {}: {e}", file)))?;
        }

        Ok(())
    }
}

#[async_trait]
impl Namespace for SessionNs {
    fn name(&self) -> &str {
        "session"
    }

    async fn read(&self, path: &[&str]) -> Result<String, VfsError> {
        let file_path = self.resolve_path(path)?;

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
                    VfsError::NotFound(format!("logos://session/{}", path.join("/")))
                }
                _ => VfsError::Io(format!("read {}: {e}", file_path.display())),
            })
    }

    async fn write(&self, path: &[&str], content: &str) -> Result<(), VfsError> {
        if path.len() < 2 {
            return Err(VfsError::InvalidPath("session write requires at least sid + subpath".to_string()));
        }
        let file_path = self.resolve_path(path)?;

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
        if path.last().map(|s| *s == "log" || s.ends_with(".log")).unwrap_or(false) {
            let file_path = self.resolve_path(path)?;
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

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn resolve_paths() {
        let ns = SessionNs::new(PathBuf::from("/mnt/kairos"));
        assert_eq!(
            ns.resolve_path(&["abc", "workspace", "src", "main.ts"]).unwrap(),
            PathBuf::from("/mnt/kairos/session/abc/workspace/src/main.ts")
        );
        assert_eq!(
            ns.resolve_path(&["abc", "state", "PROJECT_STATE.md"]).unwrap(),
            PathBuf::from("/mnt/kairos/session/abc/state/PROJECT_STATE.md")
        );
        assert_eq!(
            ns.resolve_path(&["abc"]).unwrap(),
            PathBuf::from("/mnt/kairos/session/abc")
        );
    }

    #[tokio::test]
    async fn uri_translation() {
        let ns = SessionNs::new(PathBuf::from("/mnt/kairos"));
        let translated = ns.translate_uris("cat logos://session/abc/workspace/src/main.ts", "abc");
        assert_eq!(translated, "cat /mnt/kairos/session/abc/workspace/src/main.ts");
    }

    #[tokio::test]
    async fn read_write_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let ns = SessionNs::new(dir.path().to_path_buf());

        let session_dir = dir.path().join("session").join("test-sid").join("workspace");
        std::fs::create_dir_all(&session_dir).unwrap();

        ns.write(&["test-sid", "workspace", "hello.txt"], "world").await.unwrap();
        let content = ns.read(&["test-sid", "workspace", "hello.txt"]).await.unwrap();
        assert_eq!(content, "world");
    }

    #[tokio::test]
    async fn read_dir_listing() {
        let dir = tempfile::tempdir().unwrap();
        let ns = SessionNs::new(dir.path().to_path_buf());

        let ws = dir.path().join("session").join("s1").join("workspace");
        std::fs::create_dir_all(&ws).unwrap();
        std::fs::write(ws.join("a.txt"), "a").unwrap();
        std::fs::write(ws.join("b.txt"), "b").unwrap();

        let listing = ns.read(&["s1", "workspace"]).await.unwrap();
        let entries: Vec<String> = serde_json::from_str(&listing).unwrap();
        assert!(entries.contains(&"a.txt".to_string()));
        assert!(entries.contains(&"b.txt".to_string()));
    }

    #[tokio::test]
    async fn patch_log_appends() {
        let dir = tempfile::tempdir().unwrap();
        let ns = SessionNs::new(dir.path().to_path_buf());

        let session_dir = dir.path().join("session").join("s1");
        std::fs::create_dir_all(&session_dir).unwrap();

        ns.patch(&["s1", "task.log"], "line 1").await.unwrap();
        ns.patch(&["s1", "task.log"], "line 2").await.unwrap();
        let content = ns.read(&["s1", "task.log"]).await.unwrap();
        assert!(content.contains("line 1"));
        assert!(content.contains("line 2"));
    }

    #[tokio::test]
    async fn exec_basic() {
        let dir = tempfile::tempdir().unwrap();
        let ns = SessionNs::new(dir.path().to_path_buf());

        let ws = dir.path().join("session").join("s1").join("workspace");
        std::fs::create_dir_all(&ws).unwrap();
        std::fs::write(ws.join("hello.txt"), "world").unwrap();

        let result = ns.exec("s1", "cat hello.txt").await.unwrap();
        assert_eq!(result.exit_code, 0);
        assert_eq!(result.stdout.trim(), "world");
    }

    #[tokio::test]
    async fn exec_cwd_is_workspace() {
        let dir = tempfile::tempdir().unwrap();
        let ns = SessionNs::new(dir.path().to_path_buf());

        let ws = dir.path().join("session").join("s1").join("workspace");
        std::fs::create_dir_all(&ws).unwrap();

        let result = ns.exec("s1", "pwd").await.unwrap();
        assert_eq!(result.exit_code, 0);
        assert_eq!(result.stdout.trim(), ws.to_string_lossy());
    }

    #[tokio::test]
    async fn exec_uri_translation() {
        let dir = tempfile::tempdir().unwrap();
        let ns = SessionNs::new(dir.path().to_path_buf());

        let ws = dir.path().join("session").join("s1").join("workspace");
        std::fs::create_dir_all(&ws).unwrap();
        std::fs::write(ws.join("data.txt"), "translated").unwrap();

        let result = ns.exec("s1", "cat logos://session/s1/workspace/data.txt").await.unwrap();
        assert_eq!(result.exit_code, 0);
        assert_eq!(result.stdout.trim(), "translated");
    }

    #[tokio::test]
    async fn exec_nonexistent_workspace() {
        let dir = tempfile::tempdir().unwrap();
        let ns = SessionNs::new(dir.path().to_path_buf());

        let result = ns.exec("nonexistent", "echo hi").await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn exec_exit_code() {
        let dir = tempfile::tempdir().unwrap();
        let ns = SessionNs::new(dir.path().to_path_buf());

        let ws = dir.path().join("session").join("s1").join("workspace");
        std::fs::create_dir_all(&ws).unwrap();

        let result = ns.exec("s1", "exit 42").await.unwrap();
        assert_eq!(result.exit_code, 42);
    }

    const JFS_MNT: &str = "/tmp/kairos-test/mnt";

    fn jfs_test_dir(name: &str) -> PathBuf {
        let p = PathBuf::from(JFS_MNT).join("test").join(name);
        let _ = std::fs::remove_dir_all(&p);
        std::fs::create_dir_all(&p).unwrap();
        p
    }

    #[tokio::test]
    #[ignore]
    async fn juicefs_create_session_git() {
        let base = jfs_test_dir("create-session");
        let ns = SessionNs::new(base.clone());

        let project = tempfile::tempdir().unwrap();
        std::fs::write(project.path().join("main.rs"), "fn main() {}").unwrap();
        std::fs::write(project.path().join("lib.rs"), "pub fn hello() {}").unwrap();

        ns.create_session(&project.path().to_string_lossy(), "test-session")
            .await
            .unwrap();

        let ws = base.join("session").join("test-session").join("workspace");
        assert!(ws.exists());
        assert!(ws.join("main.rs").exists());
        assert!(base.join("session").join("test-session").join("state").exists());
        assert!(base.join("session").join("test-session").join("checkpoints").exists());
    }

    #[tokio::test]
    #[ignore]
    async fn juicefs_checkpoint_and_rollback() {
        let base = jfs_test_dir("checkpoint-rollback");
        let ns = SessionNs::new(base.clone());

        let ws = base.join("session").join("s1").join("workspace");
        std::fs::create_dir_all(&ws).unwrap();
        std::fs::create_dir_all(base.join("session").join("s1").join("state")).unwrap();
        std::fs::create_dir_all(base.join("session").join("s1").join("checkpoints")).unwrap();
        std::fs::write(ws.join("file.txt"), "v1").unwrap();
        std::fs::write(
            base.join("session").join("s1").join("state").join("PROJECT_STATE.md"),
            "state v1",
        ).unwrap();

        ns.checkpoint("s1", "cp-1").await.unwrap();

        std::fs::write(ws.join("file.txt"), "v2").unwrap();
        std::fs::write(
            base.join("session").join("s1").join("state").join("PROJECT_STATE.md"),
            "state v2",
        ).unwrap();

        let content = std::fs::read_to_string(ws.join("file.txt")).unwrap();
        assert_eq!(content, "v2");

        ns.rollback("s1", "cp-1").await.unwrap();

        let content = std::fs::read_to_string(ws.join("file.txt")).unwrap();
        assert_eq!(content, "v1");
        let state = std::fs::read_to_string(
            base.join("session").join("s1").join("state").join("PROJECT_STATE.md"),
        ).unwrap();
        assert_eq!(state, "state v1");
    }

    #[tokio::test]
    #[ignore]
    async fn juicefs_fork() {
        let base = jfs_test_dir("fork");
        let ns = SessionNs::new(base.clone());

        let ws = base.join("session").join("s1").join("workspace");
        std::fs::create_dir_all(&ws).unwrap();
        std::fs::write(ws.join("file.txt"), "original").unwrap();

        ns.fork("s1", "s2").await.unwrap();

        let forked = base.join("session").join("s2").join("workspace").join("file.txt");
        assert!(forked.exists());
        assert_eq!(std::fs::read_to_string(&forked).unwrap(), "original");

        std::fs::write(ws.join("file.txt"), "modified in s1").unwrap();
        assert_eq!(std::fs::read_to_string(&forked).unwrap(), "original");
    }

    #[tokio::test]
    async fn accept_selective_files() {
        let dir = tempfile::tempdir().unwrap();
        let ns = SessionNs::new(dir.path().to_path_buf());

        let ws = dir.path().join("session").join("s1").join("workspace");
        std::fs::create_dir_all(&ws).unwrap();
        std::fs::write(ws.join("keep.txt"), "yes").unwrap();
        std::fs::write(ws.join("skip.txt"), "no").unwrap();

        let dest = tempfile::tempdir().unwrap();
        ns.accept("s1", &["keep.txt".to_string()], &dest.path().to_string_lossy())
            .await
            .unwrap();

        assert!(dest.path().join("keep.txt").exists());
        assert!(!dest.path().join("skip.txt").exists());
        assert_eq!(std::fs::read_to_string(dest.path().join("keep.txt")).unwrap(), "yes");
    }
}
