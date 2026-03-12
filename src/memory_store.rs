use std::path::{Path, PathBuf};

use rusqlite::{params, Connection};
use rusqlite::types::Value as SqlValue;
use serde_json::Value;

use crate::service::VfsError;

const LOGOS_SCHEME: &str = "logos://";
const MEM_SCHEME: &str = "mem://";

const SCHEMA_SQL: &str = "
CREATE TABLE IF NOT EXISTS messages (
    msg_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    ts        TEXT NOT NULL,
    chat_id   TEXT NOT NULL,
    speaker   TEXT NOT NULL,
    reply_to  INTEGER REFERENCES messages(msg_id),
    text      TEXT NOT NULL,
    mentions  TEXT
);
CREATE INDEX IF NOT EXISTS idx_messages_chat_ts ON messages(chat_id, ts);
CREATE INDEX IF NOT EXISTS idx_messages_reply_to ON messages(reply_to);

CREATE TABLE IF NOT EXISTS summaries (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    chat_id       TEXT NOT NULL,
    layer         TEXT NOT NULL,
    period_start  TEXT NOT NULL,
    period_end    TEXT NOT NULL,
    msg_id_ranges TEXT NOT NULL,
    content       TEXT NOT NULL,
    generated_at  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_summaries_chat_layer ON summaries(chat_id, layer, period_start);

CREATE TABLE IF NOT EXISTS anchors (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    chat_id       TEXT NOT NULL,
    task_id       TEXT NOT NULL,
    summary       TEXT NOT NULL,
    facts         TEXT NOT NULL,
    source_msg_id INTEGER REFERENCES messages(msg_id),
    created_at    TEXT NOT NULL
);
";

pub struct MemoryStore {
    db_root: PathBuf,
}

#[derive(Debug)]
enum MemoryRead {
    Message { gid: String, msg_id: i64 },
    SummaryLatest { gid: String, layer: String },
    SummaryByDate { gid: String, layer: String, date: String },
    Anchor { gid: String, anchor_id: i64 },
}

#[derive(Debug)]
enum MemoryWrite {
    Message { gid: String },
    Summary { gid: String, layer: String },
    Anchor { gid: String },
}

impl MemoryStore {
    pub fn new(db_root: PathBuf) -> std::io::Result<Self> {
        std::fs::create_dir_all(&db_root)?;
        Ok(Self { db_root })
    }

    pub async fn read(&self, raw_path: &str) -> Result<String, VfsError> {
        let parsed = parse_memory_read_path(raw_path)?;
        let db_root = self.db_root.clone();

        tokio::task::spawn_blocking(move || match parsed {
            MemoryRead::Message { gid, msg_id } => {
                let conn = open_db(&db_root, &gid)?;
                read_message(&conn, msg_id)
            }
            MemoryRead::SummaryLatest { gid, layer } => {
                let conn = open_db(&db_root, &gid)?;
                read_summary_latest(&conn, &gid, &layer)
            }
            MemoryRead::SummaryByDate { gid, layer, date } => {
                let conn = open_db(&db_root, &gid)?;
                read_summary_by_date(&conn, &gid, &layer, &date)
            }
            MemoryRead::Anchor { gid, anchor_id } => {
                let conn = open_db(&db_root, &gid)?;
                read_anchor(&conn, anchor_id)
            }
        })
        .await
        .map_err(|e| VfsError::Io(format!("task join error: {e}")))?
    }

    pub async fn write(&self, raw_path: &str, content: &str) -> Result<(), VfsError> {
        let parsed = parse_memory_write_path(raw_path)?;
        let db_root = self.db_root.clone();
        let content = content.to_string();

        tokio::task::spawn_blocking(move || match parsed {
            MemoryWrite::Message { gid } => {
                let conn = open_db(&db_root, &gid)?;
                write_message(&conn, &content)
            }
            MemoryWrite::Summary { gid, layer } => {
                let conn = open_db(&db_root, &gid)?;
                write_summary(&conn, &layer, &content)
            }
            MemoryWrite::Anchor { gid } => {
                let conn = open_db(&db_root, &gid)?;
                write_anchor(&conn, &content)
            }
        })
        .await
        .map_err(|e| VfsError::Io(format!("task join error: {e}")))?
    }

    // -- proc tools ----------------------------------------------------------

    pub async fn range_fetch(&self, content: &str) -> Result<String, VfsError> {
        let v: Value = serde_json::from_str(content)
            .map_err(|e| VfsError::InvalidJson(format!("invalid json: {e}")))?;
        let chat_id = require_str(&v, "chat_id")?.to_string();
        let ranges = parse_ranges(&v)?;
        let limit = v["limit"].as_i64().unwrap_or(20);
        let offset = v["offset"].as_i64().unwrap_or(0);
        let db_root = self.db_root.clone();

        tokio::task::spawn_blocking(move || {
            let conn = open_db(&db_root, &chat_id)?;
            let rows = query_messages_by_ranges(&conn, &chat_id, &ranges, limit, offset)?;
            messages_to_json(&rows)
        })
        .await
        .map_err(|e| VfsError::Io(format!("task join error: {e}")))?
    }

    /// Concatenates message text truncated to `max_tokens` (rough estimate).
    /// LLM-based compression is a future enhancement.
    pub async fn range_summary(&self, content: &str) -> Result<String, VfsError> {
        let v: Value = serde_json::from_str(content)
            .map_err(|e| VfsError::InvalidJson(format!("invalid json: {e}")))?;
        let chat_id = require_str(&v, "chat_id")?.to_string();
        let ranges = parse_ranges(&v)?;
        let max_tokens = v["max_tokens"].as_i64().unwrap_or(500);
        let db_root = self.db_root.clone();

        tokio::task::spawn_blocking(move || {
            let conn = open_db(&db_root, &chat_id)?;
            let rows = query_messages_by_ranges(&conn, &chat_id, &ranges, i64::MAX, 0)?;
            let char_budget = (max_tokens * 4) as usize;
            let mut out = String::new();
            for row in &rows {
                let line = format!("[{}] {}: {}\n", row.ts, row.speaker, row.text);
                if out.len() + line.len() > char_budget {
                    out.push_str("... (truncated)\n");
                    break;
                }
                out.push_str(&line);
            }
            Ok(out)
        })
        .await
        .map_err(|e| VfsError::Io(format!("task join error: {e}")))?
    }

    pub async fn search_messages(&self, content: &str) -> Result<String, VfsError> {
        let v: Value = serde_json::from_str(content)
            .map_err(|e| VfsError::InvalidJson(format!("invalid json: {e}")))?;
        let chat_id = require_str(&v, "chat_id")?.to_string();
        let query = require_str(&v, "query")?.to_string();
        let limit = v["limit"].as_i64().unwrap_or(10);
        let db_root = self.db_root.clone();

        tokio::task::spawn_blocking(move || {
            let conn = open_db(&db_root, &chat_id)?;
            let rows = query_search_messages(&conn, &chat_id, &query, limit)?;
            messages_to_json(&rows)
        })
        .await
        .map_err(|e| VfsError::Io(format!("task join error: {e}")))?
    }
}

// ---------------------------------------------------------------------------
// Database
// ---------------------------------------------------------------------------

fn open_db(db_root: &Path, gid: &str) -> Result<Connection, VfsError> {
    let db_path = db_root.join(format!("{gid}.db"));
    if let Some(parent) = db_path.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| VfsError::Io(format!("failed to create db directory: {e}")))?;
    }
    let conn = Connection::open(&db_path)
        .map_err(|e| VfsError::Io(format!("failed to open database for group {gid}: {e}")))?;
    conn.execute_batch("PRAGMA journal_mode=WAL;")
        .map_err(|e| VfsError::Io(format!("failed to set WAL mode: {e}")))?;
    conn.execute_batch(SCHEMA_SQL)
        .map_err(|e| VfsError::Io(format!("failed to initialize schema: {e}")))?;
    Ok(conn)
}

// ---------------------------------------------------------------------------
// Path parsing
// ---------------------------------------------------------------------------

fn strip_memory_prefix(raw_path: &str) -> Result<&str, VfsError> {
    let after_scheme = raw_path
        .strip_prefix(LOGOS_SCHEME)
        .or_else(|| raw_path.strip_prefix(MEM_SCHEME))
        .ok_or_else(|| VfsError::InvalidPath("invalid scheme".into()))?;

    after_scheme
        .strip_prefix("memory/")
        .ok_or_else(|| VfsError::InvalidPath("expected memory namespace".into()))
}

fn parse_memory_read_path(raw_path: &str) -> Result<MemoryRead, VfsError> {
    let rest = strip_memory_prefix(raw_path)?;
    let seg: Vec<&str> = rest.split('/').collect();

    if seg.len() < 4 || seg[0] != "groups" || seg[1].is_empty() {
        return Err(VfsError::InvalidPath(
            "expected logos://memory/groups/{gid}/...".into(),
        ));
    }
    let gid = seg[1].to_string();

    match seg[2] {
        "messages" => {
            if seg.len() != 4 {
                return Err(VfsError::InvalidPath(
                    "expected logos://memory/groups/{gid}/messages/{msg_id}".into(),
                ));
            }
            let msg_id: i64 = seg[3]
                .parse()
                .map_err(|_| VfsError::InvalidPath("invalid msg_id".into()))?;
            Ok(MemoryRead::Message { gid, msg_id })
        }
        "summary" => {
            if seg.len() != 5 {
                return Err(VfsError::InvalidPath(
                    "expected logos://memory/groups/{gid}/summary/{layer}/{date|latest}".into(),
                ));
            }
            let layer = seg[3].to_string();
            validate_layer(&layer)?;
            if seg[4] == "latest" {
                Ok(MemoryRead::SummaryLatest { gid, layer })
            } else {
                Ok(MemoryRead::SummaryByDate {
                    gid,
                    layer,
                    date: seg[4].to_string(),
                })
            }
        }
        "anchors" => {
            if seg.len() != 4 {
                return Err(VfsError::InvalidPath(
                    "expected logos://memory/groups/{gid}/anchors/{anchor_id}".into(),
                ));
            }
            let anchor_id: i64 = seg[3]
                .parse()
                .map_err(|_| VfsError::InvalidPath("invalid anchor_id".into()))?;
            Ok(MemoryRead::Anchor { gid, anchor_id })
        }
        other => Err(VfsError::InvalidPath(format!(
            "unknown memory resource: \"{other}\""
        ))),
    }
}

fn parse_memory_write_path(raw_path: &str) -> Result<MemoryWrite, VfsError> {
    let rest = strip_memory_prefix(raw_path)?;
    let seg: Vec<&str> = rest.split('/').collect();

    if seg.len() < 3 || seg[0] != "groups" || seg[1].is_empty() {
        return Err(VfsError::InvalidPath(
            "expected logos://memory/groups/{gid}/...".into(),
        ));
    }
    let gid = seg[1].to_string();

    match seg[2] {
        "messages" if seg.len() == 3 => Ok(MemoryWrite::Message { gid }),
        "summary" if seg.len() == 4 => {
            let layer = seg[3].to_string();
            validate_layer(&layer)?;
            Ok(MemoryWrite::Summary { gid, layer })
        }
        "anchors" if seg.len() == 3 => Ok(MemoryWrite::Anchor { gid }),
        _ => Err(VfsError::InvalidPath(format!(
            "invalid write path: \"{raw_path}\""
        ))),
    }
}

fn validate_layer(layer: &str) -> Result<(), VfsError> {
    match layer {
        "short" | "mid" | "long" => Ok(()),
        _ => Err(VfsError::InvalidPath(format!(
            "invalid summary layer: \"{layer}\", expected short/mid/long"
        ))),
    }
}

// ---------------------------------------------------------------------------
// Read operations
// ---------------------------------------------------------------------------

fn read_message(conn: &Connection, msg_id: i64) -> Result<String, VfsError> {
    let mut stmt = conn
        .prepare(
            "SELECT msg_id, ts, chat_id, speaker, reply_to, text, mentions
             FROM messages WHERE msg_id = ?1",
        )
        .map_err(sql_err)?;

    let row = stmt
        .query_row(params![msg_id], |row| {
            Ok((
                row.get::<_, i64>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, String>(2)?,
                row.get::<_, String>(3)?,
                row.get::<_, Option<i64>>(4)?,
                row.get::<_, String>(5)?,
                row.get::<_, Option<String>>(6)?,
            ))
        })
        .map_err(|e| not_found_or_io(e, format!("message not found: msg_id={msg_id}")))?;

    let (msg_id, ts, chat_id, speaker, reply_to, text, mentions_raw) = row;
    let mentions = parse_json_column(&mentions_raw);

    to_json(&serde_json::json!({
        "msg_id": msg_id,
        "ts": ts,
        "chat_id": chat_id,
        "speaker": speaker,
        "reply_to": reply_to,
        "text": text,
        "mentions": mentions,
    }))
}

fn read_summary_latest(conn: &Connection, gid: &str, layer: &str) -> Result<String, VfsError> {
    let mut stmt = conn
        .prepare(
            "SELECT id, chat_id, layer, period_start, period_end,
                    msg_id_ranges, content, generated_at
             FROM summaries
             WHERE chat_id = ?1 AND layer = ?2
             ORDER BY period_start DESC LIMIT 1",
        )
        .map_err(sql_err)?;

    let row = stmt
        .query_row(params![gid, layer], summary_from_row)
        .map_err(|e| {
            not_found_or_io(e, format!("no {layer} summary found for group {gid}"))
        })?;

    summary_to_json(&row)
}

fn read_summary_by_date(
    conn: &Connection,
    gid: &str,
    layer: &str,
    date: &str,
) -> Result<String, VfsError> {
    let pattern = format!("{date}%");
    let mut stmt = conn
        .prepare(
            "SELECT id, chat_id, layer, period_start, period_end,
                    msg_id_ranges, content, generated_at
             FROM summaries
             WHERE chat_id = ?1 AND layer = ?2 AND period_start LIKE ?3
             ORDER BY period_start DESC LIMIT 1",
        )
        .map_err(sql_err)?;

    let row = stmt
        .query_row(params![gid, layer, pattern], summary_from_row)
        .map_err(|e| {
            not_found_or_io(
                e,
                format!("no {layer} summary found for group {gid} at {date}"),
            )
        })?;

    summary_to_json(&row)
}

fn read_anchor(conn: &Connection, anchor_id: i64) -> Result<String, VfsError> {
    let mut stmt = conn
        .prepare(
            "SELECT id, chat_id, task_id, summary, facts, source_msg_id, created_at
             FROM anchors WHERE id = ?1",
        )
        .map_err(sql_err)?;

    let row = stmt
        .query_row(params![anchor_id], |row| {
            Ok((
                row.get::<_, i64>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, String>(2)?,
                row.get::<_, String>(3)?,
                row.get::<_, String>(4)?,
                row.get::<_, Option<i64>>(5)?,
                row.get::<_, String>(6)?,
            ))
        })
        .map_err(|e| not_found_or_io(e, format!("anchor not found: id={anchor_id}")))?;

    let (id, chat_id, task_id, summary, facts_raw, source_msg_id, created_at) = row;
    let facts = parse_json_column(&Some(facts_raw));

    to_json(&serde_json::json!({
        "id": id,
        "chat_id": chat_id,
        "task_id": task_id,
        "summary": summary,
        "facts": facts,
        "source_msg_id": source_msg_id,
        "created_at": created_at,
    }))
}

// ---------------------------------------------------------------------------
// Write operations
// ---------------------------------------------------------------------------

fn write_message(conn: &Connection, content: &str) -> Result<(), VfsError> {
    let v: Value = serde_json::from_str(content)
        .map_err(|e| VfsError::InvalidJson(format!("invalid json: {e}")))?;

    let ts = require_str(&v, "ts")?;
    let chat_id = require_str(&v, "chat_id")?;
    let speaker = require_str(&v, "speaker")?;
    let reply_to = v["reply_to"].as_i64();
    let text = require_str(&v, "text")?;
    let mentions = optional_json_column(&v, "mentions");

    conn.execute(
        "INSERT INTO messages (ts, chat_id, speaker, reply_to, text, mentions)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
        params![ts, chat_id, speaker, reply_to, text, mentions],
    )
    .map_err(sql_err)?;

    Ok(())
}

fn write_summary(conn: &Connection, layer: &str, content: &str) -> Result<(), VfsError> {
    let v: Value = serde_json::from_str(content)
        .map_err(|e| VfsError::InvalidJson(format!("invalid json: {e}")))?;

    let chat_id = require_str(&v, "chat_id")?;
    let period_start = require_str(&v, "period_start")?;
    let period_end = require_str(&v, "period_end")?;
    let msg_id_ranges = v
        .get("msg_id_ranges")
        .map(|v| v.to_string())
        .ok_or_else(|| VfsError::InvalidRequest("missing field: msg_id_ranges".into()))?;
    let summary_content = require_str(&v, "content")?;
    let generated_at = require_str(&v, "generated_at")?;

    conn.execute(
        "INSERT INTO summaries (chat_id, layer, period_start, period_end,
                                msg_id_ranges, content, generated_at)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
        params![
            chat_id,
            layer,
            period_start,
            period_end,
            msg_id_ranges,
            summary_content,
            generated_at
        ],
    )
    .map_err(sql_err)?;

    Ok(())
}

fn write_anchor(conn: &Connection, content: &str) -> Result<(), VfsError> {
    let v: Value = serde_json::from_str(content)
        .map_err(|e| VfsError::InvalidJson(format!("invalid json: {e}")))?;

    let chat_id = require_str(&v, "chat_id")?;
    let task_id = require_str(&v, "task_id")?;
    let summary = require_str(&v, "summary")?;
    let facts = v
        .get("facts")
        .map(|v| v.to_string())
        .ok_or_else(|| VfsError::InvalidRequest("missing field: facts".into()))?;
    let source_msg_id = v["source_msg_id"].as_i64();
    let created_at = require_str(&v, "created_at")?;

    conn.execute(
        "INSERT INTO anchors (chat_id, task_id, summary, facts, source_msg_id, created_at)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
        params![chat_id, task_id, summary, facts, source_msg_id, created_at],
    )
    .map_err(sql_err)?;

    Ok(())
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

struct MessageRow {
    msg_id: i64,
    ts: String,
    #[allow(dead_code)]
    chat_id: String,
    speaker: String,
    reply_to: Option<i64>,
    text: String,
    mentions: Option<String>,
}

fn message_from_row(row: &rusqlite::Row) -> rusqlite::Result<MessageRow> {
    Ok(MessageRow {
        msg_id: row.get(0)?,
        ts: row.get(1)?,
        chat_id: row.get(2)?,
        speaker: row.get(3)?,
        reply_to: row.get(4)?,
        text: row.get(5)?,
        mentions: row.get(6)?,
    })
}

fn messages_to_json(rows: &[MessageRow]) -> Result<String, VfsError> {
    let array: Vec<Value> = rows
        .iter()
        .map(|r| {
            let mentions = parse_json_column(&r.mentions);
            serde_json::json!({
                "msg_id": r.msg_id,
                "ts": r.ts,
                "chat_id": r.chat_id,
                "speaker": r.speaker,
                "reply_to": r.reply_to,
                "text": r.text,
                "mentions": mentions,
            })
        })
        .collect();
    serde_json::to_string(&array)
        .map_err(|e| VfsError::InvalidJson(format!("serialize error: {e}")))
}

fn parse_ranges(v: &Value) -> Result<Vec<(i64, i64)>, VfsError> {
    let arr = v["ranges"]
        .as_array()
        .ok_or_else(|| VfsError::InvalidRequest("missing or invalid field: ranges".into()))?;
    let mut result = Vec::new();
    for item in arr {
        let pair = item.as_array().ok_or_else(|| {
            VfsError::InvalidRequest("each range must be a [start, end] pair".into())
        })?;
        if pair.len() != 2 {
            return Err(VfsError::InvalidRequest(
                "each range must be a [start, end] pair".into(),
            ));
        }
        let start = pair[0]
            .as_i64()
            .ok_or_else(|| VfsError::InvalidRequest("range start must be integer".into()))?;
        let end = pair[1]
            .as_i64()
            .ok_or_else(|| VfsError::InvalidRequest("range end must be integer".into()))?;
        result.push((start, end));
    }
    Ok(result)
}

fn query_messages_by_ranges(
    conn: &Connection,
    chat_id: &str,
    ranges: &[(i64, i64)],
    limit: i64,
    offset: i64,
) -> Result<Vec<MessageRow>, VfsError> {
    if ranges.is_empty() {
        return Ok(vec![]);
    }

    let range_clauses: Vec<String> = ranges
        .iter()
        .map(|_| "(msg_id BETWEEN ? AND ?)".to_string())
        .collect();
    let sql = format!(
        "SELECT msg_id, ts, chat_id, speaker, reply_to, text, mentions
         FROM messages WHERE chat_id = ? AND ({})
         ORDER BY msg_id LIMIT ? OFFSET ?",
        range_clauses.join(" OR ")
    );

    let mut sql_params: Vec<SqlValue> = vec![SqlValue::Text(chat_id.to_string())];
    for &(start, end) in ranges {
        sql_params.push(SqlValue::Integer(start));
        sql_params.push(SqlValue::Integer(end));
    }
    sql_params.push(SqlValue::Integer(limit));
    sql_params.push(SqlValue::Integer(offset));

    let mut stmt = conn.prepare(&sql).map_err(sql_err)?;
    let rows = stmt
        .query_map(rusqlite::params_from_iter(sql_params), message_from_row)
        .map_err(sql_err)?;
    rows.collect::<Result<Vec<_>, _>>().map_err(sql_err)
}

fn query_search_messages(
    conn: &Connection,
    chat_id: &str,
    query: &str,
    limit: i64,
) -> Result<Vec<MessageRow>, VfsError> {
    let pattern = format!("%{query}%");
    let mut stmt = conn
        .prepare(
            "SELECT msg_id, ts, chat_id, speaker, reply_to, text, mentions
             FROM messages WHERE chat_id = ?1 AND text LIKE ?2
             ORDER BY ts DESC LIMIT ?3",
        )
        .map_err(sql_err)?;
    let rows = stmt
        .query_map(params![chat_id, pattern, limit], message_from_row)
        .map_err(sql_err)?;
    rows.collect::<Result<Vec<_>, _>>().map_err(sql_err)
}

struct SummaryRow {
    id: i64,
    chat_id: String,
    layer: String,
    period_start: String,
    period_end: String,
    msg_id_ranges: String,
    content: String,
    generated_at: String,
}

fn summary_from_row(row: &rusqlite::Row) -> rusqlite::Result<SummaryRow> {
    Ok(SummaryRow {
        id: row.get(0)?,
        chat_id: row.get(1)?,
        layer: row.get(2)?,
        period_start: row.get(3)?,
        period_end: row.get(4)?,
        msg_id_ranges: row.get(5)?,
        content: row.get(6)?,
        generated_at: row.get(7)?,
    })
}

fn summary_to_json(row: &SummaryRow) -> Result<String, VfsError> {
    let ranges = parse_json_column(&Some(row.msg_id_ranges.clone()));
    to_json(&serde_json::json!({
        "id": row.id,
        "chat_id": row.chat_id,
        "layer": row.layer,
        "period_start": row.period_start,
        "period_end": row.period_end,
        "msg_id_ranges": ranges,
        "content": row.content,
        "generated_at": row.generated_at,
    }))
}

fn parse_json_column(raw: &Option<String>) -> Value {
    raw.as_deref()
        .and_then(|s| serde_json::from_str(s).ok())
        .unwrap_or(Value::Null)
}

fn optional_json_column(v: &Value, field: &str) -> Option<String> {
    v.get(field)
        .filter(|val| !val.is_null())
        .map(|val| val.to_string())
}

fn require_str<'a>(v: &'a Value, field: &str) -> Result<&'a str, VfsError> {
    v[field]
        .as_str()
        .ok_or_else(|| VfsError::InvalidRequest(format!("missing or non-string field: {field}")))
}

fn to_json(value: &Value) -> Result<String, VfsError> {
    serde_json::to_string(value)
        .map_err(|e| VfsError::InvalidJson(format!("serialize error: {e}")))
}

fn sql_err(e: rusqlite::Error) -> VfsError {
    VfsError::Io(format!("sqlite error: {e}"))
}

fn not_found_or_io(e: rusqlite::Error, not_found_msg: String) -> VfsError {
    match e {
        rusqlite::Error::QueryReturnedNoRows => VfsError::NotFound(not_found_msg),
        _ => VfsError::Io(format!("sqlite error: {e}")),
    }
}
