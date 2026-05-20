use logos_vfs::VfsError;

pub struct OllamaEmbedder {
    http: reqwest::Client,
    ollama_url: String,
    model: String,
}

impl OllamaEmbedder {
    pub fn new(ollama_url: &str, model: &str) -> Self {
        Self {
            http: reqwest::Client::new(),
            ollama_url: ollama_url.to_string(),
            model: model.to_string(),
        }
    }

    pub fn from_env() -> Self {
        let url = std::env::var("OLLAMA_URL").unwrap_or_else(|_| "http://127.0.0.1:11434".into());
        let model = std::env::var("EMBED_MODEL").unwrap_or_else(|_| "qwen3-embedding:0.6b".into());
        Self::new(&url, &model)
    }

    pub async fn embed(&self, text: &str) -> Result<Vec<f32>, VfsError> {
        let url = format!("{}/api/embed", self.ollama_url);
        let body = serde_json::json!({
            "model": self.model,
            "input": text,
        });

        let resp = self
            .http
            .post(&url)
            .json(&body)
            .send()
            .await
            .map_err(|e| VfsError::Io(format!("ollama request: {e}")))?;

        let json: serde_json::Value = resp
            .json()
            .await
            .map_err(|e| VfsError::Io(format!("ollama response: {e}")))?;

        let vec = json["embeddings"][0]
            .as_array()
            .ok_or_else(|| VfsError::Io("ollama: no embeddings in response".into()))?
            .iter()
            .map(|v| v.as_f64().unwrap_or(0.0) as f32)
            .collect();

        Ok(vec)
    }
}
