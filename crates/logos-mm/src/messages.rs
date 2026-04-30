use std::collections::HashMap;
use std::path::PathBuf;

use sqlx::SqlitePool;
use sqlx::sqlite::SqlitePoolOptions;
use tokio::sync::RwLock;

use logos_vfs::VfsError;

const RANGE_FETCH_MAX_LIMIT: i64 = 200;
const FTS_MAX_LIMIT: i64 = 50;

/// Per-group SQLite message store.
///
/// Each group gets its own database file under `db_root/{chat_id}.db`.
/// Messages are append-only (the lossless raw archive).
///
/// Schema follows RFC 003 §2.1.
pub struct MessageDb {
    db_root: PathBuf,
    pools: RwLock<HashMap<String, SqlitePool>>,
}

impl MessageDb {
    pub fn new(db_root: PathBuf) -> Result<Self, VfsError> {
        std::fs::create_dir_all(&db_root)
            .map_err(|e| VfsError::Io(format!("create memory dir: {e}")))?;
        Ok(Self {
            db_root,
            pools: RwLock::new(HashMap::new()),
        })
    }

    /// Get or create a connection pool for a chat_id.
    pub(crate) async fn pool(&self, chat_id: &str) -> Result<SqlitePool, VfsError> {
        {
            let map = self.pools.read().await;
            if let Some(p) = map.get(chat_id) {
                return Ok(p.clone());
            }
        }

        let db_path = self.db_root.join(format!("{chat_id}.db"));
        let url = format!("sqlite:{}?mode=rwc", db_path.display());
        let pool = SqlitePoolOptions::new()
            .max_connections(4)
            .connect(&url)
            .await
            .map_err(|e| VfsError::Sqlite(format!("open {}: {e}", db_path.display())))?;
        init_schema(&pool).await?;

        let mut map = self.pools.write().await;
        if let Some(existing) = map.get(chat_id) {
            return Ok(existing.clone());
        }
        map.insert(chat_id.to_string(), pool.clone());
        Ok(pool)
    }

    /// Insert a message. Returns the new msg_id.
    ///
    /// `content` is a JSON object with fields:
    /// `ts`, `chat_id`, `actor_id|speaker`, `reply_to` (msg_id integer), `text`, `mentions`
    ///
    /// RFC 003 §2.1
    pub async fn insert(&self, chat_id: &str, content: &str) -> Result<i64, VfsError> {
        let val: serde_json::Value =
            serde_json::from_str(content).map_err(|e| VfsError::InvalidJson(e.to_string()))?;
        let pool = self.pool(chat_id).await?;

        let now = now_iso8601();
        let ts = val["ts"].as_str().unwrap_or(&now).to_string();
        let speaker = val["actor_id"]
            .as_str()
            .or_else(|| val["speaker"].as_str())
            .unwrap_or_default()
            .to_string();
        let text = val["text"].as_str().unwrap_or_default().to_string();
        let mentions = val["mentions"]
            .as_array()
            .map(|a| serde_json::to_string(a).unwrap_or_else(|_| "[]".to_string()))
            .unwrap_or_else(|| "[]".to_string());
        let meta = val
            .get("meta")
            .filter(|meta| !meta.is_null())
            .map(|meta| serde_json::to_string(meta).unwrap_or_else(|_| "null".to_string()));
        let reply_to: Option<i64> = val["reply_to"].as_i64();

        let user_msg_id: Option<i64> = val["msg_id"].as_i64();

        let result = sqlx::query(
            "INSERT INTO messages (msg_id, ts, chat_id, speaker, reply_to, text, mentions, meta)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
        )
        .bind(user_msg_id)
        .bind(&ts)
        .bind(chat_id)
        .bind(&speaker)
        .bind(reply_to)
        .bind(&text)
        .bind(&mentions)
        .bind(meta)
        .execute(&pool)
        .await
        .map_err(|e| VfsError::Sqlite(format!("insert: {e}")))?;

        let msg_id = user_msg_id.unwrap_or(result.last_insert_rowid());

        // Sync FTS
        sqlx::query("INSERT INTO messages_fts(rowid, text) VALUES (?1, ?2)")
            .bind(msg_id)
            .bind(&text)
            .execute(&pool)
            .await
            .map_err(|e| VfsError::Sqlite(format!("fts sync: {e}")))?;

        Ok(msg_id)
    }

    /// Get a single message by msg_id.
    pub async fn get_by_id(&self, chat_id: &str, msg_id: i64) -> Result<Option<String>, VfsError> {
        let pool = self.pool(chat_id).await?;
        let row = sqlx::query(
            "SELECT msg_id, ts, chat_id, speaker, reply_to, text, mentions, meta
             FROM messages WHERE msg_id = ?1",
        )
        .bind(msg_id)
        .fetch_optional(&pool)
        .await
        .map_err(|e| VfsError::Sqlite(e.to_string()))?;

        Ok(row.as_ref().map(msg_row_to_json).map(|v| v.to_string()))
    }

    /// FTS search. Returns JSON array of matching messages.
    /// RFC 003 §3.2 — memory.search
    pub async fn search_fts(
        &self,
        chat_id: &str,
        query: &str,
        limit: i64,
    ) -> Result<String, VfsError> {
        let pool = self.pool(chat_id).await?;
        let limit = limit.clamp(1, FTS_MAX_LIMIT);

        // Escape query for FTS5: wrap in double quotes to treat as phrase,
        // and escape any internal double quotes.
        let escaped = format!("\"{}\"", query.replace('"', "\"\""));

        let rows = sqlx::query(
            "SELECT m.msg_id, m.ts, m.chat_id, m.speaker, m.reply_to, m.text, m.mentions, m.meta
             FROM messages_fts f
             JOIN messages m ON m.msg_id = f.rowid
             WHERE messages_fts MATCH ?1
             ORDER BY f.rank
             LIMIT ?2",
        )
        .bind(&escaped)
        .bind(limit)
        .fetch_all(&pool)
        .await
        .map_err(|e| VfsError::Sqlite(e.to_string()))?;

        let results: Vec<serde_json::Value> = rows.iter().map(msg_row_to_json).collect();
        Ok(serde_json::to_string(&results).unwrap_or_else(|_| "[]".to_string()))
    }

    /// Range fetch: return messages within msg_id ranges.
    /// RFC 003 §3.2 — memory.range_fetch
    ///
    /// Input JSON: `{ "ranges": [[start, end], ...], "limit": N, "offset": N }`
    pub async fn range_fetch(&self, chat_id: &str, params: &str) -> Result<String, VfsError> {
        let val: serde_json::Value =
            serde_json::from_str(params).map_err(|e| VfsError::InvalidJson(e.to_string()))?;
        let ranges: Vec<(i64, i64)> = val["ranges"]
            .as_array()
            .unwrap_or(&vec![])
            .iter()
            .filter_map(|r| {
                let arr = r.as_array()?;
                Some((arr.first()?.as_i64()?, arr.get(1)?.as_i64()?))
            })
            .collect();
        if ranges.is_empty() {
            return Ok("[]".to_string());
        }
        let limit = val["limit"]
            .as_i64()
            .unwrap_or(50)
            .clamp(1, RANGE_FETCH_MAX_LIMIT);
        let offset = val["offset"].as_i64().unwrap_or(0).max(0);
        let pool = self.pool(chat_id).await?;

        let where_clauses: Vec<String> = ranges
            .iter()
            .map(|(s, e)| format!("(msg_id >= {s} AND msg_id <= {e})"))
            .collect();
        let sql = format!(
            "SELECT msg_id, ts, chat_id, speaker, reply_to, text, mentions, meta
             FROM messages WHERE ({}) ORDER BY msg_id ASC LIMIT {} OFFSET {}",
            where_clauses.join(" OR "),
            limit,
            offset
        );

        let rows = sqlx::query(&sql)
            .fetch_all(&pool)
            .await
            .map_err(|e| VfsError::Sqlite(e.to_string()))?;

        let results: Vec<serde_json::Value> = rows.iter().map(msg_row_to_json).collect();
        Ok(serde_json::to_string(&results).unwrap_or_else(|_| "[]".to_string()))
    }
}

fn msg_row_to_json(row: &sqlx::sqlite::SqliteRow) -> serde_json::Value {
    use sqlx::Row;
    serde_json::json!({
        "msg_id": row.get::<i64, _>("msg_id"),
        "ts": row.get::<String, _>("ts"),
        "chat_id": row.get::<String, _>("chat_id"),
        "actor_id": row.get::<String, _>("speaker"),
        "speaker": row.get::<String, _>("speaker"),
        "reply_to": row.get::<Option<i64>, _>("reply_to"),
        "text": row.get::<String, _>("text"),
        "mentions": row.get::<String, _>("mentions"),
        "meta": parse_json_column(row.get::<Option<String>, _>("meta")),
    })
}

/// RFC 003 §2.1 — messages table schema
async fn init_schema(pool: &SqlitePool) -> Result<(), VfsError> {
    sqlx::query(
        "CREATE TABLE IF NOT EXISTS messages (
            msg_id    INTEGER PRIMARY KEY,
            ts        TEXT NOT NULL,
            chat_id   TEXT NOT NULL,
            speaker   TEXT NOT NULL,
            reply_to  INTEGER,
            text      TEXT NOT NULL,
            mentions  TEXT,
            meta      TEXT
        )",
    )
    .execute(pool)
    .await
    .map_err(|e| VfsError::Sqlite(format!("init messages schema: {e}")))?;

    match sqlx::query("ALTER TABLE messages ADD COLUMN meta TEXT")
        .execute(pool)
        .await
    {
        Ok(_) => {}
        Err(error) => {
            let message = error.to_string();
            if !message.contains("duplicate column name: meta") {
                return Err(VfsError::Sqlite(format!("alter messages schema: {message}")));
            }
        }
    }

    sqlx::query("CREATE INDEX IF NOT EXISTS idx_messages_chat_ts ON messages(chat_id, ts)")
        .execute(pool)
        .await
        .map_err(|e| VfsError::Sqlite(format!("init messages index: {e}")))?;

    sqlx::query("CREATE INDEX IF NOT EXISTS idx_messages_reply_to ON messages(reply_to)")
        .execute(pool)
        .await
        .map_err(|e| VfsError::Sqlite(format!("init messages index: {e}")))?;

    sqlx::query(
        "CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(text, content='messages', content_rowid='msg_id')",
    )
    .execute(pool)
    .await
    .map_err(|e| VfsError::Sqlite(format!("init fts: {e}")))?;

    Ok(())
}

pub(crate) fn now_iso8601() -> String {
    chrono::Utc::now().format("%Y-%m-%dT%H:%M:%SZ").to_string()
}

fn parse_json_column(value: Option<String>) -> serde_json::Value {
    let Some(raw) = value else {
        return serde_json::Value::Null;
    };
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return serde_json::Value::Null;
    }
    serde_json::from_str(trimmed).unwrap_or(serde_json::Value::Null)
}
