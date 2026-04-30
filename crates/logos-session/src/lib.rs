//! Session Clustering Module — RFC 003 §5
//!
//! Implements social-topology-first session clustering with a three-layer
//! LRU memory hierarchy (L0 active → L1 inactive → L2 archived to LanceDB).
//!
//! Core insight (RFC 003 §5.1): reply chains are hard bindings;
//! semantic similarity is only a fallback when topology is absent.
//!
//! L2 backend: LanceDB with vector embeddings via Ollama for semantic fallback.

use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap, HashSet};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, OnceLock};
use std::time::Instant;

use arrow_array::types::Float32Type;
use arrow_array::{
    Array, FixedSizeListArray, Float32Array, RecordBatch, RecordBatchIterator, StringArray,
};
use arrow_schema::{DataType, Field, Schema};
use futures::TryStreamExt;
use lancedb::index::Index;
use lancedb::query::{ExecutableQuery, QueryBase, QueryExecutionOptions, Select};
use lancedb::table::{Duration as LanceDuration, OptimizeAction};
use tokio::sync::{RwLock, Semaphore};

/// Embedding dimension — qwen3-embedding:0.6b produces 1024-dim vectors.
#[allow(dead_code)]
const DEFAULT_EMBED_DIM: i32 = 1024;
const LANCEDB_OPT_ENABLE_V2_MANIFEST_PATHS: &str = "new_table_enable_v2_manifest_paths";

/// Similarity threshold for semantic fallback.
/// Sessions with L2 distance below this are considered a match.
const SEMANTIC_THRESHOLD: f32 = 0.5;
const RECENT_CHAT_ATTACH_WINDOW_SECS: i64 = 8;
const LRU_ORDER_REBUILD_FACTOR: usize = 8;
const LRU_ORDER_REBUILD_MIN_EXTRA: usize = 256;

/// A reference to a message within a session.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct MsgRef {
    pub msg_id: i64,
    pub chat_id: String,
    pub reply_to: Option<i64>,
    pub text: String,
    pub speaker: String,
    pub ts: String,
}

/// A session — a cluster of topologically linked messages.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Session {
    pub session_id: String,
    pub chat_id: String,
    pub messages: Vec<MsgRef>,
    pub last_active: chrono::DateTime<chrono::Utc>,
}

impl Session {
    fn new(chat_id: &str, first_msg: MsgRef) -> Self {
        let session_id = format!("s-{}-{}", chat_id, first_msg.msg_id);
        Self {
            session_id,
            chat_id: chat_id.to_string(),
            messages: vec![first_msg],
            last_active: chrono::Utc::now(),
        }
    }

    fn touch(&mut self) {
        self.last_active = chrono::Utc::now();
    }

    fn add_message(&mut self, msg: MsgRef) {
        self.messages.push(msg);
        self.touch();
    }

    pub fn to_json(&self) -> serde_json::Value {
        serde_json::to_value(self).unwrap_or(serde_json::json!(null))
    }

    /// Concatenate all message texts for embedding.
    fn text_for_embed(&self) -> String {
        let joined = self.messages
            .iter()
            .map(|m| m.text.trim())
            .filter(|text| !text.is_empty())
            .collect::<Vec<_>>()
            .join("\n");
        if joined.is_empty() {
            "(empty)".to_string()
        } else {
            joined
        }
    }
}

struct Inner {
    msg_index: HashMap<i64, String>, // msg_id -> session_id
    session_msg_ids: HashMap<String, HashSet<i64>>, // session_id -> msg_ids for O(k) cleanup
    recent_chat_sid: HashMap<String, String>, // chat_id -> most recent session_id hint
    l0: LruLayer,
    l1: LruLayer,
    l0_capacity: usize,
    l1_capacity: usize,
    lru_ticket: u64,
}

impl Inner {
    fn next_ticket(&mut self) -> u64 {
        self.lru_ticket = self.lru_ticket.wrapping_add(1);
        self.lru_ticket
    }

    fn index_session_messages(&mut self, session: &Session) {
        let sid = session.session_id.clone();
        let entry = self.session_msg_ids.entry(sid.clone()).or_default();
        for m in &session.messages {
            self.msg_index.insert(m.msg_id, sid.clone());
            entry.insert(m.msg_id);
        }
    }

    fn index_message(&mut self, session_id: &str, msg_id: i64) {
        self.msg_index.insert(msg_id, session_id.to_string());
        self.session_msg_ids
            .entry(session_id.to_string())
            .or_default()
            .insert(msg_id);
    }

    fn remove_session_from_index(&mut self, session_id: &str) {
        if let Some(msg_ids) = self.session_msg_ids.remove(session_id) {
            for msg_id in msg_ids {
                self.msg_index.remove(&msg_id);
            }
        }
    }

    fn clear_recent_chat_sid_if_matches(&mut self, chat_id: &str, session_id: &str) {
        if self
            .recent_chat_sid
            .get(chat_id)
            .is_some_and(|current| current == session_id)
        {
            self.recent_chat_sid.remove(chat_id);
        }
    }

    fn insert_l0(&mut self, session: Session) {
        let sid = session.session_id.clone();
        let chat_id = session.chat_id.clone();
        let ticket = self.next_ticket();
        self.l0.insert(sid.clone(), session, ticket);
        self.recent_chat_sid.insert(chat_id, sid);
    }

    fn append_message_to_session(&mut self, sid: &str, msg: MsgRef) -> bool {
        let ticket = self.next_ticket();
        if self
            .l0
            .with_session_mut(sid, |session| session.add_message(msg.clone()), ticket)
        {
            if let Some(session) = self.l0.get(sid) {
                self.recent_chat_sid
                    .insert(session.chat_id.clone(), sid.to_string());
            }
            return true;
        }
        if let Some(mut session) = self.l1.remove(sid) {
            session.add_message(msg);
            let chat_id = session.chat_id.clone();
            self.l0.insert(sid.to_string(), session, ticket);
            self.recent_chat_sid.insert(chat_id, sid.to_string());
            return true;
        }
        false
    }

    fn has_session(&self, sid: &str) -> bool {
        self.l0.get(sid).is_some() || self.l1.get(sid).is_some()
    }

    fn cache_session_from_l2(&mut self, session: Session) {
        let sid = session.session_id.clone();
        if self.has_session(&sid) {
            return;
        }

        if self.l1.len() >= self.l1_capacity {
            if let Some((evicted_sid, evicted_session)) = self.l1.pop_oldest() {
                self.remove_session_from_index(&evicted_sid);
                self.clear_recent_chat_sid_if_matches(&evicted_session.chat_id, &evicted_sid);
            } else {
                return;
            }
        }

        self.index_session_messages(&session);
        let ticket = self.next_ticket();
        self.l1.insert(sid, session, ticket);
    }

    fn recent_session_id_for_chat(
        &self,
        chat_id: &str,
        window: chrono::Duration,
    ) -> Option<String> {
        let cutoff = chrono::Utc::now() - window;
        if let Some(sid) = self.recent_chat_sid.get(chat_id) {
            if let Some(session) = self.l0.get(sid).or_else(|| self.l1.get(sid)) {
                if session.last_active >= cutoff {
                    return Some(session.session_id.clone());
                }
            }
        }

        self.l0
            .values()
            .chain(self.l1.values())
            .filter(|s| s.chat_id == chat_id && s.last_active >= cutoff)
            .max_by_key(|s| s.last_active)
            .map(|s| s.session_id.clone())
    }
}

#[derive(Debug, Clone, Eq, PartialEq)]
struct LruEntry {
    last_active: chrono::DateTime<chrono::Utc>,
    ticket: u64,
    session_id: String,
}

impl Ord for LruEntry {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.last_active
            .cmp(&other.last_active)
            .then_with(|| self.ticket.cmp(&other.ticket))
            .then_with(|| self.session_id.cmp(&other.session_id))
    }
}

impl PartialOrd for LruEntry {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

struct LruSlot {
    session: Session,
    ticket: u64,
}

struct LruLayer {
    sessions: HashMap<String, LruSlot>,
    order: BinaryHeap<Reverse<LruEntry>>,
}

impl LruLayer {
    fn new() -> Self {
        Self {
            sessions: HashMap::new(),
            order: BinaryHeap::new(),
        }
    }

    fn len(&self) -> usize {
        self.sessions.len()
    }

    fn insert(&mut self, sid: String, session: Session, ticket: u64) {
        let last_active = session.last_active;
        self.sessions
            .insert(sid.clone(), LruSlot { session, ticket });
        self.order.push(Reverse(LruEntry {
            last_active,
            ticket,
            session_id: sid,
        }));
        self.maybe_compact_order();
    }

    fn remove(&mut self, sid: &str) -> Option<Session> {
        let removed = self.sessions.remove(sid).map(|slot| slot.session);
        if self.sessions.is_empty() {
            self.order.clear();
        }
        removed
    }

    fn get(&self, sid: &str) -> Option<&Session> {
        self.sessions.get(sid).map(|slot| &slot.session)
    }

    fn values(&self) -> impl Iterator<Item = &Session> {
        self.sessions.values().map(|slot| &slot.session)
    }

    fn with_session_mut<F>(&mut self, sid: &str, mutator: F, ticket: u64) -> bool
    where
        F: FnOnce(&mut Session),
    {
        let Some(slot) = self.sessions.get_mut(sid) else {
            return false;
        };
        mutator(&mut slot.session);
        slot.ticket = ticket;
        self.order.push(Reverse(LruEntry {
            last_active: slot.session.last_active,
            ticket,
            session_id: sid.to_string(),
        }));
        self.maybe_compact_order();
        true
    }

    fn pop_oldest(&mut self) -> Option<(String, Session)> {
        self.maybe_compact_order();
        while let Some(Reverse(entry)) = self.order.pop() {
            let Some((sid, slot)) = self.sessions.remove_entry(&entry.session_id) else {
                continue;
            };
            if slot.ticket != entry.ticket || slot.session.last_active != entry.last_active {
                self.sessions.insert(sid, slot);
                continue;
            }
            return Some((sid, slot.session));
        }
        None
    }

    fn maybe_compact_order(&mut self) {
        let active = self.sessions.len();
        if active == 0 {
            self.order.clear();
            return;
        }

        let heap_len = self.order.len();
        let max_len = active.saturating_mul(LRU_ORDER_REBUILD_FACTOR);
        let stale_estimate = heap_len.saturating_sub(active);
        if heap_len > max_len && stale_estimate > LRU_ORDER_REBUILD_MIN_EXTRA {
            self.rebuild_order();
        }
    }

    fn rebuild_order(&mut self) {
        let mut rebuilt = BinaryHeap::with_capacity(self.sessions.len());
        for (sid, slot) in &self.sessions {
            rebuilt.push(Reverse(LruEntry {
                last_active: slot.session.last_active,
                ticket: slot.ticket,
                session_id: sid.clone(),
            }));
        }
        self.order = rebuilt;
    }
}
const L2_OP_COUNT: usize = 5;
const L2_WAIT_BUCKET_UPPERS_MS: [u64; 12] = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024];

#[derive(Clone, Copy)]
enum L2Op {
    Persist,
    Delete,
    LoadById,
    LoadByMsg,
    SemanticSearch,
}

impl L2Op {
    fn idx(self) -> usize {
        match self {
            L2Op::Persist => 0,
            L2Op::Delete => 1,
            L2Op::LoadById => 2,
            L2Op::LoadByMsg => 3,
            L2Op::SemanticSearch => 4,
        }
    }
}

struct L2Metrics {
    enabled: bool,
    dump_every_ops: u64,
    ops_seen: AtomicU64,
    calls: [AtomicU64; L2_OP_COUNT],
    errors: [AtomicU64; L2_OP_COUNT],
    hits: [AtomicU64; L2_OP_COUNT],
    misses: [AtomicU64; L2_OP_COUNT],
    latency_ms_sum: [AtomicU64; L2_OP_COUNT],
    latency_ms_max: [AtomicU64; L2_OP_COUNT],
    sem_wait_count: AtomicU64,
    sem_wait_ms_sum: AtomicU64,
    sem_wait_ms_max: AtomicU64,
    sem_wait_buckets: Box<[AtomicU64]>,
}

impl L2Metrics {
    fn new() -> Self {
        let sem_wait_buckets = (0..(L2_WAIT_BUCKET_UPPERS_MS.len() + 1))
            .map(|_| AtomicU64::new(0))
            .collect::<Vec<_>>()
            .into_boxed_slice();

        Self {
            enabled: Self::metrics_enabled(),
            dump_every_ops: Self::metrics_dump_every_ops(),
            ops_seen: AtomicU64::new(0),
            calls: std::array::from_fn(|_| AtomicU64::new(0)),
            errors: std::array::from_fn(|_| AtomicU64::new(0)),
            hits: std::array::from_fn(|_| AtomicU64::new(0)),
            misses: std::array::from_fn(|_| AtomicU64::new(0)),
            latency_ms_sum: std::array::from_fn(|_| AtomicU64::new(0)),
            latency_ms_max: std::array::from_fn(|_| AtomicU64::new(0)),
            sem_wait_count: AtomicU64::new(0),
            sem_wait_ms_sum: AtomicU64::new(0),
            sem_wait_ms_max: AtomicU64::new(0),
            sem_wait_buckets,
        }
    }

    fn metrics_enabled() -> bool {
        static METRICS_ENABLED: OnceLock<bool> = OnceLock::new();
        *METRICS_ENABLED.get_or_init(|| {
            std::env::var("LOGOS_SESSION_METRICS")
                .ok()
                .map(|v| matches!(v.as_str(), "1" | "true" | "TRUE" | "yes" | "on"))
                .unwrap_or(false)
        })
    }

    fn metrics_dump_every_ops() -> u64 {
        static DUMP_EVERY_OPS: OnceLock<u64> = OnceLock::new();
        *DUMP_EVERY_OPS.get_or_init(|| {
            std::env::var("LOGOS_SESSION_METRICS_DUMP_EVERY")
                .ok()
                .and_then(|v| v.parse::<u64>().ok())
                .filter(|v| *v > 0)
                .unwrap_or(512)
        })
    }

    fn observe_sem_wait(&self, wait_ms: u64) {
        if !self.enabled {
            return;
        }
        self.sem_wait_count.fetch_add(1, Ordering::Relaxed);
        self.sem_wait_ms_sum.fetch_add(wait_ms, Ordering::Relaxed);
        self.sem_wait_ms_max.fetch_max(wait_ms, Ordering::Relaxed);

        let bucket_idx = L2_WAIT_BUCKET_UPPERS_MS
            .iter()
            .position(|upper| wait_ms <= *upper)
            .unwrap_or(L2_WAIT_BUCKET_UPPERS_MS.len());
        self.sem_wait_buckets[bucket_idx].fetch_add(1, Ordering::Relaxed);
    }

    fn observe_op_done<T>(
        &self,
        op: L2Op,
        elapsed_ms: u64,
        result: &Result<T, logos_vfs::VfsError>,
        hit: Option<bool>,
    ) {
        if !self.enabled {
            return;
        }

        let idx = op.idx();
        self.calls[idx].fetch_add(1, Ordering::Relaxed);
        self.latency_ms_sum[idx].fetch_add(elapsed_ms, Ordering::Relaxed);
        self.latency_ms_max[idx].fetch_max(elapsed_ms, Ordering::Relaxed);

        if result.is_err() {
            self.errors[idx].fetch_add(1, Ordering::Relaxed);
        } else if let Some(hit) = hit {
            if hit {
                self.hits[idx].fetch_add(1, Ordering::Relaxed);
            } else {
                self.misses[idx].fetch_add(1, Ordering::Relaxed);
            }
        }

        self.maybe_dump();
    }

    fn maybe_dump(&self) {
        if !self.enabled {
            return;
        }

        let seen = self.ops_seen.fetch_add(1, Ordering::Relaxed) + 1;
        if seen % self.dump_every_ops != 0 {
            return;
        }

        let load = |arr: &[AtomicU64; L2_OP_COUNT], op: L2Op| -> u64 {
            arr[op.idx()].load(Ordering::Relaxed)
        };

        let avg = |sum: u64, count: u64| -> f64 {
            if count == 0 {
                0.0
            } else {
                sum as f64 / count as f64
            }
        };

        let sem_wait_count = self.sem_wait_count.load(Ordering::Relaxed);
        let sem_wait_sum = self.sem_wait_ms_sum.load(Ordering::Relaxed);
        let sem_wait_max = self.sem_wait_ms_max.load(Ordering::Relaxed);
        let sem_wait_avg = avg(sem_wait_sum, sem_wait_count);
        let sem_wait_p50 = self.sem_wait_quantile_upper_ms(0.50);
        let sem_wait_p95 = self.sem_wait_quantile_upper_ms(0.95);

        let p_calls = load(&self.calls, L2Op::Persist);
        let d_calls = load(&self.calls, L2Op::Delete);
        let id_calls = load(&self.calls, L2Op::LoadById);
        let msg_calls = load(&self.calls, L2Op::LoadByMsg);
        let sem_calls = load(&self.calls, L2Op::SemanticSearch);

        let p_err = load(&self.errors, L2Op::Persist);
        let d_err = load(&self.errors, L2Op::Delete);
        let id_err = load(&self.errors, L2Op::LoadById);
        let msg_err = load(&self.errors, L2Op::LoadByMsg);
        let sem_err = load(&self.errors, L2Op::SemanticSearch);

        let id_hit = load(&self.hits, L2Op::LoadById);
        let id_miss = load(&self.misses, L2Op::LoadById);
        let msg_hit = load(&self.hits, L2Op::LoadByMsg);
        let msg_miss = load(&self.misses, L2Op::LoadByMsg);
        let sem_hit = load(&self.hits, L2Op::SemanticSearch);
        let sem_miss = load(&self.misses, L2Op::SemanticSearch);

        let p_sum = load(&self.latency_ms_sum, L2Op::Persist);
        let d_sum = load(&self.latency_ms_sum, L2Op::Delete);
        let id_sum = load(&self.latency_ms_sum, L2Op::LoadById);
        let msg_sum = load(&self.latency_ms_sum, L2Op::LoadByMsg);
        let sem_sum = load(&self.latency_ms_sum, L2Op::SemanticSearch);

        let p_max = load(&self.latency_ms_max, L2Op::Persist);
        let d_max = load(&self.latency_ms_max, L2Op::Delete);
        let id_max = load(&self.latency_ms_max, L2Op::LoadById);
        let msg_max = load(&self.latency_ms_max, L2Op::LoadByMsg);
        let sem_max = load(&self.latency_ms_max, L2Op::SemanticSearch);

        eprintln!(
            "[logos-session][metrics][lifetime] calls[persist={},delete={},id={},msg={},semantic={}] errs[persist={},delete={},id={},msg={},semantic={}] hits[id={}/{},msg={}/{},semantic={}/{}] avg_ms[persist={:.2},delete={:.2},id={:.2},msg={:.2},semantic={:.2}] max_ms[persist={},delete={},id={},msg={},semantic={}] sem_wait[count={},avg_ms={:.2},p50_upper_ms<={},p95_upper_ms<={},max_ms={}]",
            p_calls,
            d_calls,
            id_calls,
            msg_calls,
            sem_calls,
            p_err,
            d_err,
            id_err,
            msg_err,
            sem_err,
            id_hit,
            id_miss,
            msg_hit,
            msg_miss,
            sem_hit,
            sem_miss,
            avg(p_sum, p_calls),
            avg(d_sum, d_calls),
            avg(id_sum, id_calls),
            avg(msg_sum, msg_calls),
            avg(sem_sum, sem_calls),
            p_max,
            d_max,
            id_max,
            msg_max,
            sem_max,
            sem_wait_count,
            sem_wait_avg,
            sem_wait_p50,
            sem_wait_p95,
            sem_wait_max,
        );
    }

    fn sem_wait_quantile_upper_ms(&self, quantile: f64) -> u64 {
        let total = self.sem_wait_count.load(Ordering::Relaxed);
        if total == 0 {
            return 0;
        }
        let target = ((total as f64) * quantile).ceil() as u64;
        let mut cumulative = 0_u64;

        for (idx, bucket) in self.sem_wait_buckets.iter().enumerate() {
            cumulative = cumulative.saturating_add(bucket.load(Ordering::Relaxed));
            if cumulative >= target {
                return L2_WAIT_BUCKET_UPPERS_MS
                    .get(idx)
                    .copied()
                    .unwrap_or(*L2_WAIT_BUCKET_UPPERS_MS.last().unwrap_or(&0));
            }
        }

        *L2_WAIT_BUCKET_UPPERS_MS.last().unwrap_or(&0)
    }
}

// =============================================================================
// L2 LanceDB Store
// =============================================================================

/// LanceDB-backed L2 persistent store with Ollama embeddings.
struct L2Store {
    sessions_table: lancedb::Table,
    msg_index_table: lancedb::Table,
    sessions_schema: Arc<Schema>,
    msg_index_schema: Arc<Schema>,
    ollama_url: String,
    model: String,
    embed_dim: i32,
    http: reqwest::Client,
    l2_sem: Semaphore,
    metrics: L2Metrics,
    l2_prune_last_run_sec: AtomicU64,
}

impl L2Store {
    async fn new(
        db_path: &str,
        ollama_url: &str,
        model: &str,
        embed_dim: i32,
    ) -> Result<Self, logos_vfs::VfsError> {
        let db = lancedb::connect(db_path)
            .storage_option(LANCEDB_OPT_ENABLE_V2_MANIFEST_PATHS, "true")
            .execute()
            .await
            .map_err(|e| logos_vfs::VfsError::Io(format!("lancedb connect: {e}")))?;

        let sessions_schema = Self::build_sessions_schema(embed_dim);
        let msg_index_schema = Self::build_msg_index_schema();
        Self::ensure_tables(&db, &sessions_schema, &msg_index_schema).await?;

        // Keep table handles open for process lifetime to avoid repeated open_table metadata syscalls.
        let sessions_table = db
            .open_table("sessions")
            .execute()
            .await
            .map_err(|e| logos_vfs::VfsError::Io(format!("open sessions: {e}")))?;
        let msg_index_table = db
            .open_table("msg_index")
            .execute()
            .await
            .map_err(|e| logos_vfs::VfsError::Io(format!("open msg_index: {e}")))?;

        let l2_max_concurrency = Self::l2_max_concurrency();

        let store = Self {
            sessions_table,
            msg_index_table,
            sessions_schema,
            msg_index_schema,
            ollama_url: ollama_url.to_string(),
            model: model.to_string(),
            embed_dim,
            http: reqwest::Client::new(),
            l2_sem: Semaphore::new(l2_max_concurrency),
            metrics: L2Metrics::new(),
            l2_prune_last_run_sec: AtomicU64::new(0),
        };

        if let Err(e) = store.ensure_indexes().await {
            eprintln!("[logos-session] WARNING: ensure L2 indices failed: {e}");
        }
        if let Err(e) = store.prewarm_indexes().await {
            eprintln!("[logos-session] WARNING: prewarm L2 indices failed: {e}");
        }

        Ok(store)
    }

    fn sessions_schema(&self) -> Arc<Schema> {
        Arc::clone(&self.sessions_schema)
    }

    fn msg_index_schema(&self) -> Arc<Schema> {
        Arc::clone(&self.msg_index_schema)
    }

    fn build_sessions_schema(embed_dim: i32) -> Arc<Schema> {
        Arc::new(Schema::new(vec![
            Field::new("session_id", DataType::Utf8, false),
            Field::new("chat_id", DataType::Utf8, false),
            Field::new("data", DataType::Utf8, false),
            Field::new("last_active", DataType::Utf8, false),
            Field::new("text_for_embed", DataType::Utf8, false),
            Field::new(
                "vector",
                DataType::FixedSizeList(
                    Arc::new(Field::new("item", DataType::Float32, true)),
                    embed_dim,
                ),
                true,
            ),
        ]))
    }

    fn build_msg_index_schema() -> Arc<Schema> {
        Arc::new(Schema::new(vec![
            Field::new("msg_id", DataType::Utf8, false),
            Field::new("session_id", DataType::Utf8, false),
        ]))
    }

    async fn ensure_tables(
        db: &lancedb::Connection,
        sessions_schema: &Arc<Schema>,
        msg_index_schema: &Arc<Schema>,
    ) -> Result<(), logos_vfs::VfsError> {
        let table_names = db
            .table_names()
            .execute()
            .await
            .map_err(|e| logos_vfs::VfsError::Io(format!("list tables: {e}")))?;

        if !table_names.contains(&"sessions".to_string()) {
            db.create_empty_table("sessions", Arc::clone(sessions_schema))
                .execute()
                .await
                .map_err(|e| logos_vfs::VfsError::Io(format!("create sessions table: {e}")))?;
        }

        if !table_names.contains(&"msg_index".to_string()) {
            db.create_empty_table("msg_index", Arc::clone(msg_index_schema))
                .execute()
                .await
                .map_err(|e| logos_vfs::VfsError::Io(format!("create msg_index table: {e}")))?;
        }

        Ok(())
    }

    async fn ensure_indexes(&self) -> Result<(), logos_vfs::VfsError> {
        let specs: [(&lancedb::Table, &str, &[&str]); 4] = [
            (&self.sessions_table, "sessions", &["session_id"]),
            (&self.sessions_table, "sessions", &["chat_id"]),
            (&self.sessions_table, "sessions", &["vector"]),
            (&self.msg_index_table, "msg_index", &["msg_id"]),
        ];

        for (table, table_name, columns) in specs {
            if let Err(e) = self.ensure_index(table, table_name, columns).await {
                eprintln!(
                    "[logos-session] WARNING: index bootstrap failed for {}.{}: {e}",
                    table_name,
                    columns.join(",")
                );
            }
        }

        Ok(())
    }

    async fn ensure_index(
        &self,
        table: &lancedb::Table,
        table_name: &str,
        columns: &[&str],
    ) -> Result<(), logos_vfs::VfsError> {
        let existing = table
            .list_indices()
            .await
            .map_err(|e| logos_vfs::VfsError::Io(format!("list indices ({table_name}): {e}")))?;

        let wanted: Vec<String> = columns.iter().map(|c| c.to_string()).collect();
        let has_index = existing.iter().any(|idx| idx.columns == wanted);
        if has_index {
            return Ok(());
        }

        table
            .create_index(columns, Index::Auto)
            .replace(false)
            .execute()
            .await
            .map_err(|e| {
                logos_vfs::VfsError::Io(format!(
                    "create index ({table_name}.{}): {e}",
                    columns.join(",")
                ))
            })?;
        Ok(())
    }

    async fn prewarm_indexes(&self) -> Result<(), logos_vfs::VfsError> {
        for idx in self
            .sessions_table
            .list_indices()
            .await
            .map_err(|e| logos_vfs::VfsError::Io(format!("list sessions indices: {e}")))?
        {
            let _ = self.sessions_table.prewarm_index(&idx.name).await;
        }

        for idx in self
            .msg_index_table
            .list_indices()
            .await
            .map_err(|e| logos_vfs::VfsError::Io(format!("list msg_index indices: {e}")))?
        {
            let _ = self.msg_index_table.prewarm_index(&idx.name).await;
        }

        Ok(())
    }

    fn small_query_options() -> QueryExecutionOptions {
        let mut opts = QueryExecutionOptions::default();
        opts.max_batch_length = 64;
        opts
    }

    fn sql_quote(value: &str) -> String {
        format!("'{}'", value.replace('\'', "''"))
    }

    fn l2_max_concurrency() -> usize {
        static L2_MAX_CONCURRENCY: OnceLock<usize> = OnceLock::new();
        *L2_MAX_CONCURRENCY.get_or_init(|| {
            let auto = std::thread::available_parallelism()
                .map(|n| (n.get() / 2).max(1))
                .unwrap_or(2);
            std::env::var("LOGOS_SESSION_L2_MAX_CONCURRENCY")
                .ok()
                .and_then(|v| v.parse::<usize>().ok())
                .filter(|v| *v > 0)
                .unwrap_or(auto)
        })
    }

    fn parse_bool_env(name: &str) -> Option<bool> {
        std::env::var(name)
            .ok()
            .and_then(|value| match value.trim().to_ascii_lowercase().as_str() {
                "1" | "true" | "yes" | "on" => Some(true),
                "0" | "false" | "no" | "off" => Some(false),
                _ => None,
            })
    }

    fn l2_prune_enabled() -> bool {
        static ENABLED: OnceLock<bool> = OnceLock::new();
        *ENABLED
            .get_or_init(|| Self::parse_bool_env("LOGOS_SESSION_L2_PRUNE_ENABLED").unwrap_or(true))
    }

    fn l2_prune_interval_secs() -> u64 {
        static INTERVAL_SECS: OnceLock<u64> = OnceLock::new();
        *INTERVAL_SECS.get_or_init(|| {
            std::env::var("LOGOS_SESSION_L2_PRUNE_INTERVAL_SECS")
                .ok()
                .and_then(|v| v.parse::<u64>().ok())
                .filter(|v| *v >= 60)
                .unwrap_or(1800)
        })
    }

    fn l2_prune_retention_hours() -> i64 {
        static RETENTION_HOURS: OnceLock<i64> = OnceLock::new();
        *RETENTION_HOURS.get_or_init(|| {
            std::env::var("LOGOS_SESSION_L2_PRUNE_RETENTION_HOURS")
                .ok()
                .and_then(|v| v.parse::<i64>().ok())
                .filter(|v| *v > 0)
                .unwrap_or(24)
        })
    }

    fn l2_prune_delete_unverified() -> Option<bool> {
        static DELETE_UNVERIFIED: OnceLock<Option<bool>> = OnceLock::new();
        *DELETE_UNVERIFIED
            .get_or_init(|| Self::parse_bool_env("LOGOS_SESSION_L2_PRUNE_DELETE_UNVERIFIED"))
    }

    async fn maybe_prune_old_versions(&self) {
        if !Self::l2_prune_enabled() {
            return;
        }

        let now = chrono::Utc::now().timestamp();
        if now <= 0 {
            return;
        }

        let now = now as u64;
        let interval_secs = Self::l2_prune_interval_secs();
        let last_run = self.l2_prune_last_run_sec.load(Ordering::Relaxed);
        if now.saturating_sub(last_run) < interval_secs {
            return;
        }

        if self
            .l2_prune_last_run_sec
            .compare_exchange(last_run, now, Ordering::AcqRel, Ordering::Relaxed)
            .is_err()
        {
            return;
        }

        let retention_hours = Self::l2_prune_retention_hours();
        let delete_unverified = Self::l2_prune_delete_unverified();

        Self::prune_table_versions(
            &self.sessions_table,
            "sessions",
            retention_hours,
            delete_unverified,
        )
        .await;
        Self::prune_table_versions(
            &self.msg_index_table,
            "msg_index",
            retention_hours,
            delete_unverified,
        )
        .await;
    }

    async fn prune_table_versions(
        table: &lancedb::Table,
        table_name: &str,
        retention_hours: i64,
        delete_unverified: Option<bool>,
    ) {
        let Some(older_than) = LanceDuration::try_hours(retention_hours) else {
            eprintln!(
                "[logos-session] WARNING: invalid prune retention hours={retention_hours}"
            );
            return;
        };

        let action = OptimizeAction::Prune {
            older_than: Some(older_than),
            delete_unverified,
            error_if_tagged_old_versions: Some(false),
        };

        match table.optimize(action).await {
            Ok(stats) => {
                if let Some(prune) = stats.prune {
                    eprintln!(
                        "[logos-session] L2 prune table={} old_versions_removed={} bytes_removed={}",
                        table_name, prune.old_versions, prune.bytes_removed
                    );
                }
            }
            Err(e) => {
                eprintln!("[logos-session] WARNING: L2 prune failed table={table_name}: {e}");
            }
        }
    }

    async fn acquire_l2_permit(
        &self,
    ) -> Result<tokio::sync::SemaphorePermit<'_>, logos_vfs::VfsError> {
        let wait_started = Instant::now();
        let permit = self
            .l2_sem
            .acquire()
            .await
            .map_err(|e| logos_vfs::VfsError::Io(format!("l2 semaphore closed: {e}")))?;

        let wait_ms = wait_started.elapsed().as_millis() as u64;
        self.metrics.observe_sem_wait(wait_ms);
        if wait_ms >= Self::slow_l2_warn_ms() as u64 {
            eprintln!("[logos-session] slow L2 semaphore wait_ms={wait_ms}");
        }
        Ok(permit)
    }

    fn slow_l2_warn_ms() -> u128 {
        static SLOW_L2_WARN_MS: OnceLock<u128> = OnceLock::new();
        *SLOW_L2_WARN_MS.get_or_init(|| {
            std::env::var("LOGOS_SESSION_SLOW_L2_MS")
                .ok()
                .and_then(|v| v.parse::<u128>().ok())
                .unwrap_or(25)
        })
    }

    fn verify_l2_msg_index() -> bool {
        static VERIFY: OnceLock<bool> = OnceLock::new();
        *VERIFY.get_or_init(|| {
            std::env::var("LOGOS_SESSION_VERIFY_L2_MSG_INDEX")
                .ok()
                .map(|v| matches!(v.as_str(), "1" | "true" | "TRUE" | "yes" | "on"))
                .unwrap_or(false)
        })
    }

    fn warn_if_slow_l2(op: &str, started: Instant) {
        let elapsed_ms = started.elapsed().as_millis();
        if elapsed_ms >= Self::slow_l2_warn_ms() {
            eprintln!(
                "[logos-session] slow L2 op={} elapsed_ms={}",
                op, elapsed_ms
            );
        }
    }

    /// Call Ollama embedding API.
    async fn embed(&self, text: &str) -> Result<Vec<f32>, logos_vfs::VfsError> {
        let url = format!("{}/api/embed", self.ollama_url);
        let input = {
            let normalized = text.trim();
            if normalized.is_empty() {
                "(empty)"
            } else {
                normalized
            }
        };
        let body = serde_json::json!({
            "model": self.model,
            "input": input,
        });

        let resp = self
            .http
            .post(&url)
            .json(&body)
            .send()
            .await
            .map_err(|e| logos_vfs::VfsError::Io(format!("ollama request: {e}")))?;

        let status = resp.status();
        let raw = resp
            .text()
            .await
            .map_err(|e| logos_vfs::VfsError::Io(format!("ollama response body: {e}")))?;
        if !status.is_success() {
            return Err(logos_vfs::VfsError::Io(format!(
                "ollama embed failed status={} body={}",
                status,
                Self::truncate_for_log(&raw, 240)
            )));
        }

        let json: serde_json::Value = serde_json::from_str(&raw).map_err(|e| {
            logos_vfs::VfsError::Io(format!(
                "ollama response json parse failed: {e}; body={}",
                Self::truncate_for_log(&raw, 240)
            ))
        })?;

        let embedding = json
            .get("embeddings")
            .and_then(|value| value.as_array())
            .and_then(|rows| rows.first())
            .and_then(|first| first.as_array())
            .or_else(|| json.get("embedding").and_then(|value| value.as_array()))
            .ok_or_else(|| {
                logos_vfs::VfsError::Io(format!(
                    "ollama: no embeddings in response body={}",
                    Self::truncate_for_log(&raw, 240)
                ))
            })?;

        if embedding.is_empty() {
            return Err(logos_vfs::VfsError::Io(format!(
                "ollama: empty embedding in response body={}",
                Self::truncate_for_log(&raw, 240)
            )));
        }

        let vec = embedding
            .iter()
            .map(|v| v.as_f64().unwrap_or(0.0) as f32)
            .collect();

        Ok(vec)
    }

    fn truncate_for_log(text: &str, max_chars: usize) -> String {
        if text.chars().count() <= max_chars {
            return text.to_string();
        }
        let mut end = 0usize;
        for (idx, _) in text.char_indices().take(max_chars) {
            end = idx;
        }
        let prefix = if end == 0 { "" } else { &text[..end] };
        format!("{prefix}...(truncated)")
    }

    async fn persist(&self, session: &Session) -> Result<(), logos_vfs::VfsError> {
        let _permit = self.acquire_l2_permit().await?;
        let started = Instant::now();
        let result = async {
            let text = session.text_for_embed();
            let vector = self.embed(&text).await?;
            let data = serde_json::to_string(session)
                .map_err(|e| logos_vfs::VfsError::Io(format!("serialize: {e}")))?;
            let last_active = session.last_active.to_rfc3339();

            let dim = self.embed_dim as usize;
            if vector.len() != dim {
                return Err(logos_vfs::VfsError::Io(format!(
                    "embedding dim mismatch: expected {dim}, got {}",
                    vector.len()
                )));
            }

            let vector_array = FixedSizeListArray::from_iter_primitive::<Float32Type, _, _>(
                std::iter::once(Some(vector.into_iter().map(Some).collect::<Vec<_>>())),
                self.embed_dim,
            );

            let batch = RecordBatch::try_new(
                self.sessions_schema(),
                vec![
                    Arc::new(StringArray::from(vec![session.session_id.as_str()])),
                    Arc::new(StringArray::from(vec![session.chat_id.as_str()])),
                    Arc::new(StringArray::from(vec![data.as_str()])),
                    Arc::new(StringArray::from(vec![last_active.as_str()])),
                    Arc::new(StringArray::from(vec![text.as_str()])),
                    Arc::new(vector_array),
                ],
            )
            .map_err(|e| logos_vfs::VfsError::Io(format!("record batch: {e}")))?;

            let source =
                RecordBatchIterator::new(vec![Ok(batch)].into_iter(), self.sessions_schema());
            let mut merge = self.sessions_table.merge_insert(&["session_id"]);
            merge
                .when_matched_update_all(None)
                .when_not_matched_insert_all();
            merge
                .execute(Box::new(source))
                .await
                .map_err(|e| logos_vfs::VfsError::Io(format!("lancedb upsert: {e}")))?;

            // Update msg_index
            self.update_msg_index(session).await?;

            Ok(())
        }
        .await;
        drop(_permit);
        if result.is_ok() {
            self.maybe_prune_old_versions().await;
        }

        let elapsed_ms = started.elapsed().as_millis() as u64;
        self.metrics
            .observe_op_done(L2Op::Persist, elapsed_ms, &result, None);
        Self::warn_if_slow_l2("persist", started);
        result
    }

    async fn update_msg_index(&self, session: &Session) -> Result<(), logos_vfs::VfsError> {
        let msg_ids: Vec<i64> = session.messages.iter().map(|m| m.msg_id).collect();
        if msg_ids.is_empty() {
            return Ok(());
        }

        let msg_id_strings: Vec<String> = msg_ids.iter().map(|m| m.to_string()).collect();
        let msg_id_refs: Vec<&str> = msg_id_strings.iter().map(|s| s.as_str()).collect();
        let sid_repeated: Vec<&str> = msg_ids
            .iter()
            .map(|_| session.session_id.as_str())
            .collect();

        let idx_batch = RecordBatch::try_new(
            self.msg_index_schema(),
            vec![
                Arc::new(StringArray::from(msg_id_refs)),
                Arc::new(StringArray::from(sid_repeated)),
            ],
        )
        .map_err(|e| logos_vfs::VfsError::Io(format!("msg_index batch: {e}")))?;

        let source =
            RecordBatchIterator::new(vec![Ok(idx_batch)].into_iter(), self.msg_index_schema());
        let mut merge = self.msg_index_table.merge_insert(&["msg_id"]);
        merge
            .when_matched_update_all(None)
            .when_not_matched_insert_all();
        merge
            .execute(Box::new(source))
            .await
            .map_err(|e| logos_vfs::VfsError::Io(format!("msg_index upsert: {e}")))?;

        Ok(())
    }

    async fn delete(&self, session_id: &str) -> Result<(), logos_vfs::VfsError> {
        let _permit = self.acquire_l2_permit().await?;
        let started = Instant::now();
        let result = async {
            self.sessions_table
                .delete(&format!("session_id = {}", Self::sql_quote(session_id)))
                .await
                .map_err(|e| logos_vfs::VfsError::Io(format!("lancedb delete: {e}")))?;

            self.msg_index_table
                .delete(&format!("session_id = {}", Self::sql_quote(session_id)))
                .await
                .map_err(|e| logos_vfs::VfsError::Io(format!("msg_index delete: {e}")))?;

            Ok(())
        }
        .await;
        drop(_permit);
        if result.is_ok() {
            self.maybe_prune_old_versions().await;
        }

        let elapsed_ms = started.elapsed().as_millis() as u64;
        self.metrics
            .observe_op_done(L2Op::Delete, elapsed_ms, &result, None);
        Self::warn_if_slow_l2("delete", started);
        result
    }

    async fn load_by_id_unthrottled(
        &self,
        session_id: &str,
    ) -> Result<Option<Session>, logos_vfs::VfsError> {
        let started = Instant::now();
        let result = async {
            let mut stream = self
                .sessions_table
                .query()
                .select(Select::columns(&["data"]))
                .only_if(format!("session_id = {}", Self::sql_quote(session_id)))
                .limit(1)
                .execute_with_options(Self::small_query_options())
                .await
                .map_err(|e| logos_vfs::VfsError::Io(format!("query by session_id: {e}")))?;

            let batch = stream
                .try_next()
                .await
                .map_err(|e| logos_vfs::VfsError::Io(format!("query stream by session_id: {e}")))?;

            match batch {
                Some(batch) => Self::extract_session(std::slice::from_ref(&batch)),
                None => Ok(None),
            }
        }
        .await;

        let elapsed_ms = started.elapsed().as_millis() as u64;
        let hit = match &result {
            Ok(Some(_)) => Some(true),
            Ok(None) => Some(false),
            Err(_) => None,
        };
        self.metrics
            .observe_op_done(L2Op::LoadById, elapsed_ms, &result, hit);
        Self::warn_if_slow_l2("load_by_id", started);
        result
    }

    async fn load_by_id(&self, session_id: &str) -> Result<Option<Session>, logos_vfs::VfsError> {
        let _permit = self.acquire_l2_permit().await?;
        self.load_by_id_unthrottled(session_id).await
    }

    async fn load_by_msg(&self, msg_id: i64) -> Result<Option<Session>, logos_vfs::VfsError> {
        let _permit = self.acquire_l2_permit().await?;
        let started = Instant::now();
        let result = async {
            let mut stream = self
                .msg_index_table
                .query()
                .select(Select::columns(&["session_id"]))
                .only_if(format!("msg_id = {}", Self::sql_quote(&msg_id.to_string())))
                .limit(1)
                .execute_with_options(Self::small_query_options())
                .await
                .map_err(|e| logos_vfs::VfsError::Io(format!("idx query: {e}")))?;

            let batch = stream
                .try_next()
                .await
                .map_err(|e| logos_vfs::VfsError::Io(format!("idx stream: {e}")))?;

            if let Some(batch) = batch {
                if batch.num_rows() == 0 {
                    return Ok(None);
                }
                let sid_col = batch
                    .column_by_name("session_id")
                    .and_then(|c| c.as_any().downcast_ref::<StringArray>());
                if let Some(arr) = sid_col {
                    let sid = arr.value(0);
                    let loaded = self.load_by_id_unthrottled(sid).await?;
                    if !Self::verify_l2_msg_index() {
                        return Ok(loaded);
                    }
                    if let Some(ref session) = loaded {
                        if session.messages.iter().any(|msg| msg.msg_id == msg_id) {
                            return Ok(loaded);
                        }
                    }
                    return Ok(None);
                }
            }

            Ok(None)
        }
        .await;

        let elapsed_ms = started.elapsed().as_millis() as u64;
        let hit = match &result {
            Ok(Some(_)) => Some(true),
            Ok(None) => Some(false),
            Err(_) => None,
        };
        self.metrics
            .observe_op_done(L2Op::LoadByMsg, elapsed_ms, &result, hit);
        Self::warn_if_slow_l2("load_by_msg", started);
        result
    }

    /// Semantic fallback: find the most similar session by vector search.
    async fn semantic_search(
        &self,
        text: &str,
        chat_id: &str,
    ) -> Result<Option<Session>, logos_vfs::VfsError> {
        let _permit = self.acquire_l2_permit().await?;
        let started = Instant::now();
        let result = async {
            let vector = self.embed(text).await?;

            let mut stream = self
                .sessions_table
                .query()
                .limit(1)
                .select(Select::columns(&["data", "_distance"]))
                .only_if(format!("chat_id = {}", Self::sql_quote(chat_id)))
                .nearest_to(vector.as_slice())
                .map_err(|e| logos_vfs::VfsError::Io(format!("vector search: {e}")))?
                .execute_with_options(Self::small_query_options())
                .await
                .map_err(|e| logos_vfs::VfsError::Io(format!("search exec: {e}")))?;

            let batch = stream
                .try_next()
                .await
                .map_err(|e| logos_vfs::VfsError::Io(format!("search stream: {e}")))?;

            if let Some(batch) = batch {
                if batch.num_rows() == 0 {
                    return Ok(None);
                }
                // Check distance threshold
                if let Some(dist_col) = batch
                    .column_by_name("_distance")
                    .and_then(|c| c.as_any().downcast_ref::<Float32Array>())
                {
                    let dist = dist_col.value(0);
                    if dist > SEMANTIC_THRESHOLD {
                        return Ok(None);
                    }
                }
                Self::extract_session(std::slice::from_ref(&batch))
            } else {
                Ok(None)
            }
        }
        .await;

        let elapsed_ms = started.elapsed().as_millis() as u64;
        let hit = match &result {
            Ok(Some(_)) => Some(true),
            Ok(None) => Some(false),
            Err(_) => None,
        };
        self.metrics
            .observe_op_done(L2Op::SemanticSearch, elapsed_ms, &result, hit);
        Self::warn_if_slow_l2("semantic_search", started);
        result
    }

    fn extract_session(batches: &[RecordBatch]) -> Result<Option<Session>, logos_vfs::VfsError> {
        if let Some(batch) = batches.first() {
            if batch.num_rows() > 0 {
                let data_col = batch
                    .column_by_name("data")
                    .and_then(|c| c.as_any().downcast_ref::<StringArray>());
                if let Some(arr) = data_col {
                    let json_str = arr.value(0);
                    let session: Session = serde_json::from_str(json_str)
                        .map_err(|e| logos_vfs::VfsError::Io(format!("deserialize: {e}")))?;
                    return Ok(Some(session));
                }
            }
        }
        Ok(None)
    }
}

// =============================================================================
// SessionStore — public API
// =============================================================================

/// Session store with three-layer LRU: L0 (active) → L1 (inactive) → L2 (LanceDB).
pub struct SessionStore {
    inner: RwLock<Inner>,
    l2: Option<L2Store>,
}

impl SessionStore {
    /// Create an in-memory-only session store (no L2 persistence).
    pub fn new(l0_capacity: usize, l1_capacity: usize) -> Self {
        Self {
            inner: RwLock::new(Inner {
                msg_index: HashMap::new(),
                session_msg_ids: HashMap::new(),
                recent_chat_sid: HashMap::new(),
                l0: LruLayer::new(),
                l1: LruLayer::new(),
                l0_capacity,
                l1_capacity,
                lru_ticket: 0,
            }),
            l2: None,
        }
    }

    /// Create a session store with LanceDB L2 persistence and Ollama embeddings.
    pub async fn with_lancedb(
        db_path: impl AsRef<std::path::Path>,
        ollama_url: &str,
        model: &str,
        embed_dim: i32,
        l0_capacity: usize,
        l1_capacity: usize,
    ) -> Result<Self, logos_vfs::VfsError> {
        if let Some(parent) = db_path.as_ref().parent() {
            std::fs::create_dir_all(parent)
                .map_err(|e| logos_vfs::VfsError::Io(format!("create dir: {e}")))?;
        }

        let l2 = L2Store::new(
            db_path.as_ref().to_str().unwrap_or("sessions.lance"),
            ollama_url,
            model,
            embed_dim,
        )
        .await?;

        Ok(Self {
            inner: RwLock::new(Inner {
                msg_index: HashMap::new(),
                session_msg_ids: HashMap::new(),
                recent_chat_sid: HashMap::new(),
                l0: LruLayer::new(),
                l1: LruLayer::new(),
                l0_capacity,
                l1_capacity,
                lru_ticket: 0,
            }),
            l2: Some(l2),
        })
    }

    /// Observe a new message and update session topology.
    pub async fn observe(&self, msg: MsgRef) {
        let msg_id = msg.msg_id;
        let reply_to = msg.reply_to;

        // Page fault: load from L2 outside the lock
        let page_fault_session = if let Some(reply_to) = reply_to {
            let need_fault = {
                let inner = self.inner.read().await;
                inner.msg_index.get(&reply_to).is_none()
            };
            if need_fault {
                if let Some(ref l2) = self.l2 {
                    match l2.load_by_msg(reply_to).await {
                        Ok(s) => s,
                        Err(e) => {
                            eprintln!("[logos-session] L2 load_by_msg failed: {e}");
                            None
                        }
                    }
                } else {
                    None
                }
            } else {
                None
            }
        } else {
            None
        };

        // Delete from L2 outside the lock (if we loaded a session)
        if let Some(ref session) = page_fault_session {
            if let Some(ref l2) = self.l2 {
                if let Err(e) = l2.delete(&session.session_id).await {
                    eprintln!("[logos-session] L2 delete (page fault) failed: {e}");
                }
            }
        }

        // Hot-path fast join: if this chat has a recent in-memory session, append directly
        // and skip expensive semantic embedding search.
        let recent_chat_sid = if reply_to.is_none() && page_fault_session.is_none() {
            let inner = self.inner.read().await;
            inner.recent_session_id_for_chat(
                &msg.chat_id,
                chrono::Duration::seconds(RECENT_CHAT_ATTACH_WINDOW_SECS),
            )
        } else {
            None
        };

        // Semantic fallback: only when no reply_to, no page fault, and no recent in-memory hit.
        let semantic_match =
            if reply_to.is_none() && page_fault_session.is_none() && recent_chat_sid.is_none() {
                if let Some(ref l2) = self.l2 {
                    match l2.semantic_search(&msg.text, &msg.chat_id).await {
                        Ok(s) => s,
                        Err(e) => {
                            eprintln!("[logos-session] L2 semantic_search failed: {e}");
                            None
                        }
                    }
                } else {
                    None
                }
            } else {
                None
            };

        // If semantic match found, delete from L2 (we'll promote to L0)
        if let Some(ref session) = semantic_match {
            if let Some(ref l2) = self.l2 {
                if let Err(e) = l2.delete(&session.session_id).await {
                    eprintln!("[logos-session] L2 delete (semantic) failed: {e}");
                }
            }
        }

        // Now take the lock for in-memory operations only
        let evicted = {
            let mut inner = self.inner.write().await;

            if let Some(mut session) = page_fault_session {
                // Page fault from L2: restore to L0
                inner.index_session_messages(&session);
                session.add_message(msg.clone());
                inner.index_message(&session.session_id, msg_id);
                inner.insert_l0(session);
                Self::evict_collect(&mut inner)
            } else if let Some(reply_to) = reply_to {
                // Reply chain: join existing in-memory session
                if let Some(sid) = inner.msg_index.get(&reply_to).cloned() {
                    if inner.append_message_to_session(&sid, msg.clone()) {
                        inner.index_message(&sid, msg_id);
                        Self::evict_collect(&mut inner)
                    } else {
                        let chat_id = msg.chat_id.clone();
                        let session = Session::new(&chat_id, msg);
                        let sid = session.session_id.clone();
                        inner.index_message(&sid, msg_id);
                        inner.insert_l0(session);
                        Self::evict_collect(&mut inner)
                    }
                } else {
                    let chat_id = msg.chat_id.clone();
                    let session = Session::new(&chat_id, msg);
                    let sid = session.session_id.clone();
                    inner.index_message(&sid, msg_id);
                    inner.insert_l0(session);
                    Self::evict_collect(&mut inner)
                }
            } else if let Some(sid) = recent_chat_sid {
                // Same-chat burst: directly join recent active session to avoid per-message embed.
                if inner.append_message_to_session(&sid, msg.clone()) {
                    inner.index_message(&sid, msg_id);
                    Self::evict_collect(&mut inner)
                } else {
                    let chat_id = msg.chat_id.clone();
                    let session = Session::new(&chat_id, msg);
                    let sid = session.session_id.clone();
                    inner.index_message(&sid, msg_id);
                    inner.insert_l0(session);
                    Self::evict_collect(&mut inner)
                }
            } else if let Some(mut session) = semantic_match {
                // Semantic fallback: matched an L2 session by vector similarity
                inner.index_session_messages(&session);
                session.add_message(msg.clone());
                inner.index_message(&session.session_id, msg_id);
                inner.insert_l0(session);
                Self::evict_collect(&mut inner)
            } else {
                // No reply, no semantic match: create new session
                let chat_id = msg.chat_id.clone();
                let session = Session::new(&chat_id, msg);
                let sid = session.session_id.clone();
                inner.index_message(&sid, msg_id);
                inner.insert_l0(session);
                Self::evict_collect(&mut inner)
            }
        };

        // Persist evicted sessions to L2 outside the lock
        if let Some(ref l2) = self.l2 {
            for session in evicted {
                if let Err(e) = l2.persist(&session).await {
                    eprintln!(
                        "[logos-session] L2 persist failed for {}: {e}",
                        session.session_id
                    );
                }
            }
        }
    }

    fn evict_collect(inner: &mut Inner) -> Vec<Session> {
        while inner.l0.len() > inner.l0_capacity {
            if let Some((sid, session)) = inner.l0.pop_oldest() {
                let ticket = inner.next_ticket();
                inner.l1.insert(sid, session, ticket);
            } else {
                break;
            }
        }
        let mut evicted = Vec::new();
        while inner.l1.len() > inner.l1_capacity {
            if let Some((sid, session)) = inner.l1.pop_oldest() {
                inner.remove_session_from_index(&sid);
                inner.clear_recent_chat_sid_if_matches(&session.chat_id, &sid);
                evicted.push(session);
            } else {
                break;
            }
        }
        evicted
    }

    pub async fn get_session_for_msg(&self, msg_id: i64) -> Option<Session> {
        let inner = self.inner.read().await;
        if let Some(sid) = inner.msg_index.get(&msg_id) {
            if let Some(s) = inner.l0.get(sid) {
                return Some(s.clone());
            }
            if let Some(s) = inner.l1.get(sid) {
                return Some(s.clone());
            }
        }
        drop(inner);

        // Try L2 and opportunistically cache back to L1 to avoid repeated L2 scans.
        if let Some(ref l2) = self.l2 {
            match l2.load_by_msg(msg_id).await {
                Ok(Some(session)) => {
                    {
                        let mut inner = self.inner.write().await;
                        inner.cache_session_from_l2(session.clone());
                    }
                    return Some(session);
                }
                Ok(None) => return None,
                Err(e) => eprintln!("[logos-session] L2 load_by_msg failed: {e}"),
            }
        }
        None
    }

    pub async fn get_session(&self, session_id: &str) -> Option<Session> {
        let inner = self.inner.read().await;
        if let Some(s) = inner.l0.get(session_id) {
            return Some(s.clone());
        }
        if let Some(s) = inner.l1.get(session_id) {
            return Some(s.clone());
        }
        drop(inner);
        if let Some(ref l2) = self.l2 {
            match l2.load_by_id(session_id).await {
                Ok(Some(session)) => {
                    {
                        let mut inner = self.inner.write().await;
                        inner.cache_session_from_l2(session.clone());
                    }
                    return Some(session);
                }
                Ok(None) => return None,
                Err(e) => eprintln!("[logos-session] L2 load_by_id failed: {e}"),
            }
        }
        None
    }

    pub async fn get_active_session(&self, chat_id: &str) -> Option<Session> {
        let inner = self.inner.read().await;
        if let Some(sid) = inner.recent_chat_sid.get(chat_id) {
            if let Some(s) = inner.l0.get(sid) {
                return Some(s.clone());
            }
        }

        inner
            .l0
            .values()
            .filter(|s| s.chat_id == chat_id)
            .max_by_key(|s| s.last_active)
            .cloned()
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn make_msg(msg_id: i64, chat_id: &str, reply_to: Option<i64>) -> MsgRef {
        MsgRef {
            msg_id,
            chat_id: chat_id.to_string(),
            reply_to,
            text: format!("msg-{msg_id}"),
            speaker: "test".to_string(),
            ts: "2026-03-20T10:00:00Z".to_string(),
        }
    }

    // --- Pure in-memory tests (no L2, no Ollama) ---

    #[tokio::test]
    async fn new_message_creates_session() {
        let store = SessionStore::new(64, 256);
        store.observe(make_msg(1, "c1", None)).await;
        let s = store.get_session_for_msg(1).await.unwrap();
        assert_eq!(s.messages.len(), 1);
    }

    #[tokio::test]
    async fn reply_joins_existing_session() {
        let store = SessionStore::new(64, 256);
        store.observe(make_msg(1, "c1", None)).await;
        store.observe(make_msg(2, "c1", Some(1))).await;
        let s = store.get_session_for_msg(1).await.unwrap();
        assert_eq!(s.messages.len(), 2);
        assert_eq!(
            s.session_id,
            store.get_session_for_msg(2).await.unwrap().session_id
        );
    }

    #[tokio::test]
    async fn no_reply_within_window_joins_recent_session() {
        let store = SessionStore::new(64, 256);
        store.observe(make_msg(1, "c1", None)).await;
        store.observe(make_msg(2, "c1", None)).await;
        let s1 = store.get_session_for_msg(1).await.unwrap();
        let s2 = store.get_session_for_msg(2).await.unwrap();
        assert_eq!(s1.session_id, s2.session_id);
        assert_eq!(s1.messages.len(), 2);
    }

    #[tokio::test]
    async fn no_reply_outside_window_creates_separate_session() {
        let store = SessionStore::new(64, 256);
        store.observe(make_msg(1, "c1", None)).await;

        // Force session age beyond fast-join window.
        {
            let mut inner = store.inner.write().await;
            let sid = inner.msg_index.get(&1).unwrap().clone();
            if let Some(slot) = inner.l0.sessions.get_mut(&sid) {
                slot.session.last_active = chrono::Utc::now()
                    - chrono::Duration::seconds(RECENT_CHAT_ATTACH_WINDOW_SECS + 1);
            }
        }

        store.observe(make_msg(2, "c1", None)).await;
        let s1 = store.get_session_for_msg(1).await.unwrap();
        let s2 = store.get_session_for_msg(2).await.unwrap();
        assert_ne!(s1.session_id, s2.session_id);
    }

    #[tokio::test]
    async fn deep_reply_chain() {
        let store = SessionStore::new(64, 256);
        store.observe(make_msg(1, "c1", None)).await;
        store.observe(make_msg(2, "c1", Some(1))).await;
        store.observe(make_msg(3, "c1", Some(2))).await;
        let s = store.get_session_for_msg(3).await.unwrap();
        assert_eq!(s.messages.len(), 3);
    }

    #[tokio::test]
    async fn lru_eviction_l0_to_l1() {
        let store = SessionStore::new(2, 256);
        store.observe(make_msg(1, "c1", None)).await;
        store.observe(make_msg(2, "c1", None)).await;
        store.observe(make_msg(3, "c1", None)).await;
        assert!(store.get_session_for_msg(1).await.is_some());
        assert!(store.get_session_for_msg(2).await.is_some());
        assert!(store.get_session_for_msg(3).await.is_some());
    }

    #[tokio::test]
    async fn reply_promotes_from_l1() {
        let store = SessionStore::new(2, 256);
        // Keep messages in different chats so fast-join does not collapse them.
        store.observe(make_msg(1, "c1", None)).await;
        store.observe(make_msg(2, "c2", None)).await;
        store.observe(make_msg(3, "c3", None)).await;

        // msg 1 should now be in L1, and reply should promote that session to L0.
        store.observe(make_msg(4, "c1", Some(1))).await;
        let s = store.get_session_for_msg(1).await.unwrap();
        assert_eq!(s.chat_id, "c1");
        assert_eq!(s.messages.len(), 2);
        assert!(s.messages.iter().any(|m| m.msg_id == 1));
        assert!(s.messages.iter().any(|m| m.msg_id == 4));
    }

    // --- L2 LanceDB tests (require Ollama) ---

    #[tokio::test]
    #[ignore] // requires Ollama running with qwen3-embedding:0.6b
    async fn l2_persistence_and_page_fault() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("sessions.lance");
        let store = SessionStore::with_lancedb(
            &db_path,
            "http://localhost:11434",
            "qwen3-embedding:0.6b",
            DEFAULT_EMBED_DIM,
            2,
            2,
        )
        .await
        .unwrap();

        store.observe(make_msg(1, "c1", None)).await;
        store.observe(make_msg(2, "c1", None)).await;
        store.observe(make_msg(3, "c1", None)).await;
        store.observe(make_msg(4, "c1", None)).await;
        store.observe(make_msg(5, "c1", None)).await;

        // Reply to msg1 (should page fault from L2)
        store.observe(make_msg(6, "c1", Some(1))).await;
        let s = store.get_session_for_msg(1).await.unwrap();
        assert_eq!(s.messages.len(), 2);
    }

    #[tokio::test]
    #[ignore] // requires Ollama
    async fn l2_semantic_fallback() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("sessions.lance");
        let store = SessionStore::with_lancedb(
            &db_path,
            "http://localhost:11434",
            "qwen3-embedding:0.6b",
            DEFAULT_EMBED_DIM,
            2,
            2,
        )
        .await
        .unwrap();

        let mut msg1 = make_msg(1, "c1", None);
        msg1.text = "我在学习Rust编程语言，所有权系统很有趣".to_string();
        store.observe(msg1).await;

        // Force to L2
        store.observe(make_msg(10, "c1", None)).await;
        store.observe(make_msg(11, "c1", None)).await;
        store.observe(make_msg(12, "c1", None)).await;
        store.observe(make_msg(13, "c1", None)).await;

        // Semantically similar, no reply_to
        let mut msg2 = make_msg(20, "c1", None);
        msg2.text = "Rust的借用检查器太严格了".to_string();
        store.observe(msg2).await;

        let s = store.get_session_for_msg(20).await.unwrap();
        println!(
            "Semantic fallback: session has {} messages",
            s.messages.len()
        );
    }
}
