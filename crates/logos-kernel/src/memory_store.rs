use std::collections::HashMap;
use std::sync::Arc;

use logos_vfs::VfsError;
use milvus::client::Client;
use milvus::collection::SearchOption;
use milvus::data::FieldColumn;
use milvus::index::{IndexParams, IndexType, MetricType};
use milvus::schema::{CollectionSchemaBuilder, FieldSchema};
use milvus::value::Value;
use serde::{Deserialize, Serialize};
use tokio::sync::Mutex;

use crate::embedder::OllamaEmbedder;

const EMBED_DIM: i64 = 1024;

fn vfs_err(msg: impl Into<String>) -> VfsError {
    VfsError::Io(msg.into())
}

// ---------------------------------------------------------------------------
// Crystal Memory
// ---------------------------------------------------------------------------

const CRYSTAL_COLLECTION: &str = "crystal_memories";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrystalSeed {
    pub id: String,
    pub condition: String,
    pub label: String,
    pub bullets: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct Crystal {
    pub label: String,
    pub bullets: String,
    pub score: f32,
}

pub struct CrystalMemoryStore {
    client: Arc<Client>,
    embedder: Arc<OllamaEmbedder>,
    initialized: Mutex<bool>,
}

impl CrystalMemoryStore {
    pub fn new(client: Arc<Client>, embedder: Arc<OllamaEmbedder>) -> Self {
        Self { client, embedder, initialized: Mutex::new(false) }
    }

    async fn ensure_collection(&self) -> Result<(), VfsError> {
        let mut init = self.initialized.lock().await;
        if *init {
            return Ok(());
        }

        let has = self.client.has_collection(CRYSTAL_COLLECTION).await.map_err(|e| vfs_err(format!("milvus: {e}")))?;

        if !has {
            let schema = CollectionSchemaBuilder::new(CRYSTAL_COLLECTION, "behavioral heuristics")
                .add_field(FieldSchema::new_primary_int64("row_id", "", true))
                .add_field(FieldSchema::new_varchar("crystal_id", "", 256))
                .add_field(FieldSchema::new_varchar("condition", "", 4096))
                .add_field(FieldSchema::new_varchar("label", "", 512))
                .add_field(FieldSchema::new_varchar("bullets", "", 8192))
                .add_field(FieldSchema::new_float_vector("embedding", "", EMBED_DIM))
                .build()
                .map_err(|e| vfs_err(format!("schema: {e}")))?;

            let coll = self.client
                .create_collection(schema, None)
                .await
                .map_err(|e| vfs_err(format!("create collection: {e}")))?;

            let index_params = IndexParams::new(
                "crystal_idx".to_string(),
                IndexType::IvfFlat,
                MetricType::IP,
                HashMap::from([("nlist".to_string(), "128".to_string())]),
            );
            coll.create_index("embedding", index_params)
                .await
                .map_err(|e| vfs_err(format!("create index: {e}")))?;

            coll.load(1)
                .await
                .map_err(|e| vfs_err(format!("load collection: {e}")))?;
        }

        *init = true;
        Ok(())
    }

    async fn get_collection(&self) -> Result<milvus::collection::Collection, VfsError> {
        self.ensure_collection().await?;
        self.client.get_collection(CRYSTAL_COLLECTION).await.map_err(|e| vfs_err(format!("get collection: {e}")))
    }

    pub async fn seed(&self, crystals: &[CrystalSeed]) -> Result<(), VfsError> {
        if crystals.is_empty() {
            return Ok(());
        }

        let coll = self.get_collection().await?;

        let schema = CollectionSchemaBuilder::new(CRYSTAL_COLLECTION, "")
            .add_field(FieldSchema::new_primary_int64("row_id", "", true))
            .add_field(FieldSchema::new_varchar("crystal_id", "", 256))
            .add_field(FieldSchema::new_varchar("condition", "", 4096))
            .add_field(FieldSchema::new_varchar("label", "", 512))
            .add_field(FieldSchema::new_varchar("bullets", "", 8192))
            .add_field(FieldSchema::new_float_vector("embedding", "", EMBED_DIM))
            .build()
            .map_err(|e| vfs_err(format!("schema: {e}")))?;

        let mut ids = Vec::new();
        let mut conditions = Vec::new();
        let mut labels = Vec::new();
        let mut bullets_vec = Vec::new();
        let mut all_embeddings = Vec::new();

        for c in crystals {
            let emb = self.embedder.embed(&c.condition).await?;
            ids.push(c.id.clone());
            conditions.push(c.condition.clone());
            labels.push(c.label.clone());
            bullets_vec.push(c.bullets.clone());
            all_embeddings.extend(emb);
        }

        let columns = vec![
            FieldColumn::new(schema.get_field("crystal_id").unwrap(), ids),
            FieldColumn::new(schema.get_field("condition").unwrap(), conditions),
            FieldColumn::new(schema.get_field("label").unwrap(), labels),
            FieldColumn::new(schema.get_field("bullets").unwrap(), bullets_vec),
            FieldColumn::new(schema.get_field("embedding").unwrap(), all_embeddings),
        ];

        coll.insert(columns, None)
            .await
            .map_err(|e| vfs_err(format!("insert: {e}")))?;

        coll.flush()
            .await
            .map_err(|e| vfs_err(format!("flush: {e}")))?;

        Ok(())
    }

    pub async fn query(&self, context: &str, top_k: i32, threshold: f32) -> Result<Vec<Crystal>, VfsError> {
        let coll = self.get_collection().await?;
        let emb = self.embedder.embed(context).await?;
        let vector = Value::from(emb);

        let option = SearchOption::new();

        let results = coll
            .search(
                vec![vector],
                "embedding",
                top_k,
                MetricType::IP,
                vec!["label", "bullets"],
                &option,
            )
            .await
            .map_err(|e| vfs_err(format!("search: {e}")))?;

        let mut crystals = Vec::new();

        for result in &results {
            let labels_col = result.field.iter().find(|c| c.name == "label");
            let bullets_col = result.field.iter().find(|c| c.name == "bullets");

            if let (Some(lc), Some(bc)) = (labels_col, bullets_col) {
                let labels: Vec<String> = lc.value.clone().try_into().unwrap_or_default();
                let bullets: Vec<String> = bc.value.clone().try_into().unwrap_or_default();
                let scores: Vec<f32> = result.score.clone().try_into().unwrap_or_default();

                for ((label, bullet), score) in labels.into_iter().zip(bullets.into_iter()).zip(scores.into_iter()) {
                    if score >= threshold {
                        crystals.push(Crystal { label, bullets: bullet, score });
                    }
                }
            }
        }

        crystals.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        Ok(crystals)
    }
}

// ---------------------------------------------------------------------------
// Tool Memory
// ---------------------------------------------------------------------------

const TOOL_COLLECTION: &str = "tool_memories";

#[derive(Debug, Clone, Serialize)]
pub struct RankedTool {
    pub tool_name: String,
    pub score: f32,
    pub memories: Vec<String>,
}

pub struct ToolMemoryStore {
    client: Arc<Client>,
    embedder: Arc<OllamaEmbedder>,
    initialized: Mutex<bool>,
}

impl ToolMemoryStore {
    pub fn new(client: Arc<Client>, embedder: Arc<OllamaEmbedder>) -> Self {
        Self { client, embedder, initialized: Mutex::new(false) }
    }

    async fn ensure_collection(&self) -> Result<(), VfsError> {
        let mut init = self.initialized.lock().await;
        if *init {
            return Ok(());
        }

        let has = self.client.has_collection(TOOL_COLLECTION).await.map_err(|e| vfs_err(format!("milvus: {e}")))?;

        if !has {
            let schema = CollectionSchemaBuilder::new(TOOL_COLLECTION, "per-tool usage experience")
                .add_field(FieldSchema::new_primary_int64("row_id", "", true))
                .add_field(FieldSchema::new_varchar("tool_name", "", 256))
                .add_field(FieldSchema::new_varchar("condition", "", 4096))
                .add_field(FieldSchema::new_varchar("bullets", "", 8192))
                .add_field(FieldSchema::new_bool("success", ""))
                .add_field(FieldSchema::new_int64("created_at", ""))
                .add_field(FieldSchema::new_float_vector("embedding", "", EMBED_DIM))
                .build()
                .map_err(|e| vfs_err(format!("schema: {e}")))?;

            let coll = self.client
                .create_collection(schema, None)
                .await
                .map_err(|e| vfs_err(format!("create collection: {e}")))?;

            let index_params = IndexParams::new(
                "tool_idx".to_string(),
                IndexType::IvfFlat,
                MetricType::IP,
                HashMap::from([("nlist".to_string(), "128".to_string())]),
            );
            coll.create_index("embedding", index_params)
                .await
                .map_err(|e| vfs_err(format!("create index: {e}")))?;

            coll.load(1)
                .await
                .map_err(|e| vfs_err(format!("load collection: {e}")))?;
        }

        *init = true;
        Ok(())
    }

    async fn get_collection(&self) -> Result<milvus::collection::Collection, VfsError> {
        self.ensure_collection().await?;
        self.client.get_collection(TOOL_COLLECTION).await.map_err(|e| vfs_err(format!("get collection: {e}")))
    }

    pub async fn record(
        &self,
        tool_name: &str,
        condition: &str,
        bullets: &str,
        success: bool,
    ) -> Result<(), VfsError> {
        let coll = self.get_collection().await?;
        let emb = self.embedder.embed(condition).await?;
        let now = chrono::Utc::now().timestamp();

        let schema = CollectionSchemaBuilder::new(TOOL_COLLECTION, "")
            .add_field(FieldSchema::new_primary_int64("row_id", "", true))
            .add_field(FieldSchema::new_varchar("tool_name", "", 256))
            .add_field(FieldSchema::new_varchar("condition", "", 4096))
            .add_field(FieldSchema::new_varchar("bullets", "", 8192))
            .add_field(FieldSchema::new_bool("success", ""))
            .add_field(FieldSchema::new_int64("created_at", ""))
            .add_field(FieldSchema::new_float_vector("embedding", "", EMBED_DIM))
            .build()
            .map_err(|e| vfs_err(format!("schema: {e}")))?;

        let columns = vec![
            FieldColumn::new(schema.get_field("tool_name").unwrap(), vec![tool_name.to_string()]),
            FieldColumn::new(schema.get_field("condition").unwrap(), vec![condition.to_string()]),
            FieldColumn::new(schema.get_field("bullets").unwrap(), vec![bullets.to_string()]),
            FieldColumn::new(schema.get_field("success").unwrap(), vec![success]),
            FieldColumn::new(schema.get_field("created_at").unwrap(), vec![now]),
            FieldColumn::new(schema.get_field("embedding").unwrap(), emb),
        ];

        coll.insert(columns, None)
            .await
            .map_err(|e| vfs_err(format!("insert: {e}")))?;

        Ok(())
    }

    pub async fn rank_tools(
        &self,
        task: &str,
        all_tools: &[String],
        top_k: usize,
    ) -> Result<Vec<RankedTool>, VfsError> {
        let has = self.client.has_collection(TOOL_COLLECTION).await.map_err(|e| vfs_err(format!("milvus: {e}")))?;
        if !has {
            return Ok(all_tools.iter().map(|t| RankedTool {
                tool_name: t.clone(),
                score: 0.0,
                memories: vec![],
            }).collect());
        }

        let coll = self.get_collection().await?;
        let emb = self.embedder.embed(task).await?;
        let vector = Value::from(emb);

        let option = SearchOption::new();

        let results = coll
            .search(
                vec![vector],
                "embedding",
                50,
                MetricType::IP,
                vec!["tool_name", "bullets", "success"],
                &option,
            )
            .await
            .map_err(|e| vfs_err(format!("search: {e}")))?;

        let mut tool_scores: HashMap<String, (f32, Vec<String>)> = HashMap::new();

        for result in &results {
            let names_col = result.field.iter().find(|c| c.name == "tool_name");
            let bullets_col = result.field.iter().find(|c| c.name == "bullets");
            let success_col = result.field.iter().find(|c| c.name == "success");

            if let (Some(nc), Some(bc), Some(sc)) = (names_col, bullets_col, success_col) {
                let names: Vec<String> = nc.value.clone().try_into().unwrap_or_default();
                let bullets: Vec<String> = bc.value.clone().try_into().unwrap_or_default();
                let successes: Vec<bool> = sc.value.clone().try_into().unwrap_or_default();
                let scores: Vec<f32> = result.score.clone().try_into().unwrap_or_default();

                for (((name, bullet), success), score) in names.into_iter()
                    .zip(bullets.into_iter())
                    .zip(successes.into_iter())
                    .zip(scores.into_iter())
                {
                    let effective_score = if success { score } else { score * 0.5 };
                    let entry = tool_scores.entry(name).or_insert_with(|| (0.0, Vec::new()));
                    if effective_score > entry.0 {
                        entry.0 = effective_score;
                    }
                    if score > 0.5 && !bullet.is_empty() {
                        entry.1.push(bullet);
                    }
                }
            }
        }

        let mut ranked: Vec<RankedTool> = Vec::new();

        for tool in all_tools {
            if let Some((score, memories)) = tool_scores.remove(tool) {
                ranked.push(RankedTool {
                    tool_name: tool.clone(),
                    score,
                    memories,
                });
            } else {
                ranked.push(RankedTool {
                    tool_name: tool.clone(),
                    score: 0.0,
                    memories: vec![],
                });
            }
        }

        ranked.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        if top_k > 0 && ranked.len() > top_k {
            ranked.truncate(top_k);
        }
        Ok(ranked)
    }
}
