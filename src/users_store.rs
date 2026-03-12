use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use serde_json::{Map, Value};
use tokio::fs;
use tokio::sync::Mutex;

use crate::service::VfsError;

const LOGOS_SCHEME: &str = "logos://";
const MEM_SCHEME: &str = "mem://";

pub struct UsersStore {
    users_root: PathBuf,
    file_locks: Mutex<HashMap<PathBuf, Arc<Mutex<()>>>>,
}

#[derive(Debug, Clone)]
struct UsersFilePath {
    user_id: String,
    relative_path: String,
}

impl UsersStore {
    pub fn new(users_root: PathBuf) -> std::io::Result<Self> {
        std::fs::create_dir_all(&users_root)?;
        Ok(Self {
            users_root,
            file_locks: Mutex::new(HashMap::new()),
        })
    }

    pub fn users_root(&self) -> &Path {
        &self.users_root
    }

    pub async fn read(&self, raw_path: &str) -> Result<String, VfsError> {
        let path = parse_users_file_path(raw_path)?;
        let file_path = self.physical_path(&path);
        let content = fs::read_to_string(&file_path).await.map_err(|e| match e.kind() {
            std::io::ErrorKind::NotFound => {
                VfsError::NotFound(format!("file not found: {}", file_path.display()))
            }
            _ => VfsError::Io(format!("failed to read {}: {e}", file_path.display())),
        })?;

        if is_json_path(&path.relative_path) {
            let value: Value = serde_json::from_str(&content).map_err(|e| {
                VfsError::InvalidJson(format!("invalid json in {}: {e}", file_path.display()))
            })?;
            serde_json::to_string(&value)
                .map_err(|e| VfsError::InvalidJson(format!("failed to serialize json: {e}")))
        } else {
            Ok(content)
        }
    }

    pub async fn write(&self, raw_path: &str, content: &str) -> Result<(), VfsError> {
        let path = parse_users_file_path(raw_path)?;
        let file_path = self.physical_path(&path);

        let file_lock = self.get_or_create_file_lock(&file_path).await;
        let _guard = file_lock.lock().await;

        if is_json_path(&path.relative_path) {
            let parsed = parse_json_object(content)?;
            atomic_write_json(&file_path, &parsed).await
        } else {
            atomic_write(&file_path, content).await
        }
    }

    pub async fn patch(&self, raw_path: &str, partial_content: &str) -> Result<(), VfsError> {
        let path = parse_users_file_path(raw_path)?;
        if !is_json_path(&path.relative_path) {
            return Err(VfsError::InvalidRequest(
                "patch is only supported for .json files".to_string(),
            ));
        }

        let patch = parse_json_object(partial_content)?;
        let file_path = self.physical_path(&path);

        let file_lock = self.get_or_create_file_lock(&file_path).await;
        let _guard = file_lock.lock().await;

        let mut current = match fs::read_to_string(&file_path).await {
            Ok(content) => parse_json_object(&content)?,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Value::Object(Map::new()),
            Err(e) => {
                return Err(VfsError::Io(format!(
                    "failed to read current json for patch {}: {e}",
                    file_path.display()
                )))
            }
        };
        merge_json_object(&mut current, &patch);
        atomic_write_json(&file_path, &current).await
    }

    fn physical_path(&self, path: &UsersFilePath) -> PathBuf {
        self.users_root.join(&path.user_id).join(&path.relative_path)
    }

    async fn get_or_create_file_lock(&self, file_path: &Path) -> Arc<Mutex<()>> {
        let mut locks = self.file_locks.lock().await;
        locks
            .entry(file_path.to_path_buf())
            .or_insert_with(|| Arc::new(Mutex::new(())))
            .clone()
    }
}

fn parse_users_file_path(raw_path: &str) -> Result<UsersFilePath, VfsError> {
    let after_scheme = raw_path
        .strip_prefix(LOGOS_SCHEME)
        .or_else(|| raw_path.strip_prefix(MEM_SCHEME))
        .ok_or_else(|| {
            VfsError::InvalidPath(format!(
                "path must start with \"{LOGOS_SCHEME}\" or \"{MEM_SCHEME}\""
            ))
        })?;

    let segments: Vec<&str> = after_scheme.split('/').collect();
    if segments.len() < 3 {
        return Err(VfsError::InvalidPath(
            "path must match logos://users/{user_id}/{...path}".to_string(),
        ));
    }

    if segments[0] != "users" {
        return Err(VfsError::InvalidPath(format!(
            "expected namespace \"users\", got \"{}\"",
            segments[0]
        )));
    }

    let user_id = segments[1];
    if user_id.is_empty() {
        return Err(VfsError::InvalidPath("user_id cannot be empty".to_string()));
    }

    for seg in &segments[2..] {
        if *seg == ".." || seg.is_empty() {
            return Err(VfsError::InvalidPath(
                "path contains invalid segment".to_string(),
            ));
        }
    }

    let relative_path = segments[2..].join("/");

    Ok(UsersFilePath {
        user_id: user_id.to_string(),
        relative_path,
    })
}

fn is_json_path(path: &str) -> bool {
    path.ends_with(".json")
}

fn parse_json_object(content: &str) -> Result<Value, VfsError> {
    let value: Value = serde_json::from_str(content)
        .map_err(|e| VfsError::InvalidJson(format!("invalid json content: {e}")))?;
    if !value.is_object() {
        return Err(VfsError::InvalidJson(
            "json content must be an object".to_string(),
        ));
    }
    Ok(value)
}

fn merge_json_object(target: &mut Value, patch: &Value) {
    match (target, patch) {
        (Value::Object(target_map), Value::Object(patch_map)) => {
            for (key, patch_value) in patch_map {
                match target_map.get_mut(key) {
                    Some(existing_value) if existing_value.is_object() && patch_value.is_object() => {
                        merge_json_object(existing_value, patch_value);
                    }
                    _ => {
                        target_map.insert(key.clone(), patch_value.clone());
                    }
                }
            }
        }
        (target_value, patch_value) => {
            *target_value = patch_value.clone();
        }
    }
}

async fn atomic_write(target: &Path, content: &str) -> Result<(), VfsError> {
    if let Some(parent) = target.parent() {
        fs::create_dir_all(parent).await.map_err(|e| {
            VfsError::Io(format!(
                "failed to create parent directory {}: {e}",
                parent.display()
            ))
        })?;
    }

    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|e| VfsError::Io(format!("system clock error: {e}")))?
        .as_nanos();
    let temp_path = target.with_extension(format!("tmp.{nonce}"));

    fs::write(&temp_path, content).await.map_err(|e| {
        VfsError::Io(format!(
            "failed to write temp file {}: {e}",
            temp_path.display()
        ))
    })?;

    fs::rename(&temp_path, target).await.map_err(|e| {
        VfsError::Io(format!(
            "failed to replace file {} with temp {}: {e}",
            target.display(),
            temp_path.display()
        ))
    })?;
    Ok(())
}

async fn atomic_write_json(target: &Path, value: &Value) -> Result<(), VfsError> {
    let content = serde_json::to_string_pretty(value)
        .map_err(|e| VfsError::InvalidJson(format!("failed to serialize json: {e}")))?;
    atomic_write(target, &content).await
}
