use async_trait::async_trait;
use logos_vfs::VfsError;

use crate::proc::ProcTool;

pub struct FetchUrlTool;

impl FetchUrlTool {
    pub fn new() -> Self {
        Self
    }
}

#[async_trait]
impl ProcTool for FetchUrlTool {
    fn name(&self) -> &str {
        "fetch_url"
    }

    fn schema(&self) -> serde_json::Value {
        serde_json::json!({
            "name": "fetch_url",
            "description": "Fetch a URL and return its content as plain text. Handles HTML (via lynx or builtin parser), PDF (via pdftotext), and plain text. Useful for reading documentation, API references, and web pages.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": { "type": "string", "description": "URL to fetch" }
                },
                "required": ["url"]
            }
        })
    }

    async fn call(&self, params: &str) -> Result<String, VfsError> {
        let val: serde_json::Value =
            serde_json::from_str(params).map_err(|e| VfsError::InvalidJson(e.to_string()))?;
        let url = val["url"]
            .as_str()
            .ok_or_else(|| VfsError::InvalidJson("missing 'url'".to_string()))?;

        let content_type = detect_content_type(url).await;

        let text = match content_type.as_deref() {
            Some(ct) if ct.contains("pdf") => fetch_pdf(url).await?,
            Some(ct) if ct.contains("json") => fetch_raw(url).await?,
            Some(ct) if ct.contains("text/plain") => fetch_raw(url).await?,
            _ => fetch_html(url).await?,
        };

        if text.trim().is_empty() {
            return Err(VfsError::Io(format!(
                "page returned empty content (content-type: {})",
                content_type.as_deref().unwrap_or("unknown")
            )));
        }

        Ok(text)
    }
}

async fn detect_content_type(url: &str) -> Option<String> {
    let output = tokio::process::Command::new("curl")
        .args([
            "-sI", "-L",
            "--max-time", "10",
            "-H", "User-Agent: Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
            url,
        ])
        .output()
        .await
        .ok()?;

    let headers = String::from_utf8_lossy(&output.stdout).to_lowercase();
    for line in headers.lines() {
        if line.starts_with("content-type:") {
            return Some(line.trim_start_matches("content-type:").trim().to_string());
        }
    }
    None
}

async fn fetch_pdf(url: &str) -> Result<String, VfsError> {
    let tmp_path = format!("/tmp/fetch_url_{}.pdf", std::process::id());

    let dl = tokio::process::Command::new("curl")
        .args([
            "-sL", "--max-time", "15",
            "-o", &tmp_path,
            "-H", "User-Agent: Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
            url,
        ])
        .output()
        .await
        .map_err(|e| VfsError::Io(format!("curl pdf: {e}")))?;

    if !dl.status.success() {
        let _ = tokio::fs::remove_file(&tmp_path).await;
        return Err(VfsError::Io(format!("curl pdf failed: {}", dl.status)));
    }

    let extract = tokio::process::Command::new("pdftotext")
        .args([&tmp_path, "-"])
        .output()
        .await;

    let _ = tokio::fs::remove_file(&tmp_path).await;

    match extract {
        Ok(out) if out.status.success() => {
            Ok(String::from_utf8_lossy(&out.stdout).to_string())
        }
        Ok(out) => Err(VfsError::Io(format!(
            "pdftotext failed: {}",
            String::from_utf8_lossy(&out.stderr)
        ))),
        Err(_) => Err(VfsError::Io(
            "pdftotext not installed (apt install poppler-utils)".to_string(),
        )),
    }
}

async fn fetch_raw(url: &str) -> Result<String, VfsError> {
    let output = tokio::process::Command::new("curl")
        .args([
            "-sL", "--max-time", "15",
            "-H", "User-Agent: Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
            url,
        ])
        .output()
        .await
        .map_err(|e| VfsError::Io(format!("curl: {e}")))?;

    if !output.status.success() {
        return Err(VfsError::Io(format!("curl failed: {}", output.status)));
    }

    Ok(String::from_utf8_lossy(&output.stdout).to_string())
}

fn clean_lynx_output(text: &str) -> String {
    text.lines()
        .filter(|line| {
            let trimmed = line.trim();
            if trimmed.is_empty() {
                return true;
            }
            let words: Vec<&str> = trimmed.split_whitespace().collect();
            if words.iter().all(|w| *w == "#alternate" || *w == "alternate") {
                return false;
            }
            true
        })
        .collect::<Vec<_>>()
        .join("\n")
}

async fn fetch_html(url: &str) -> Result<String, VfsError> {
    let lynx = tokio::process::Command::new("lynx")
        .args([
            "-dump", "-nolist", "-hiddenlinks=ignore", "-width=120",
            "-accept_all_cookies",
            "-useragent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
            url,
        ])
        .output()
        .await;

    if let Ok(output) = lynx {
        if output.status.success() {
            let text = String::from_utf8_lossy(&output.stdout).to_string();
            let text = clean_lynx_output(&text);
            if text.trim().len() > 50 {
                return Ok(text);
            }
        }
    }

    let output = tokio::process::Command::new("curl")
        .args([
            "-sL", "--max-time", "15",
            "-H", "User-Agent: Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "-H", "Accept: text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            url,
        ])
        .output()
        .await
        .map_err(|e| VfsError::Io(format!("curl: {e}")))?;

    if !output.status.success() {
        return Err(VfsError::Io(format!("curl failed: {}", output.status)));
    }

    Ok(html_to_text(&String::from_utf8_lossy(&output.stdout)))
}

fn html_to_text(html: &str) -> String {
    let mut result = String::with_capacity(html.len() / 2);
    let mut in_tag = false;
    let mut in_script = false;
    let mut in_style = false;
    let mut tag_name = String::new();
    let mut collecting_tag_name = false;
    let mut last_was_whitespace = false;

    for c in html.chars() {
        if c == '<' {
            in_tag = true;
            tag_name.clear();
            collecting_tag_name = true;
            continue;
        }

        if c == '>' {
            in_tag = false;
            collecting_tag_name = false;
            let lower = tag_name.to_lowercase();
            if lower == "script" {
                in_script = true;
            } else if lower == "/script" {
                in_script = false;
            } else if lower == "style" {
                in_style = true;
            } else if lower == "/style" {
                in_style = false;
            } else if matches!(
                lower.as_str(),
                "br" | "br/" | "p" | "/p" | "div" | "/div" | "h1" | "/h1"
                    | "h2" | "/h2" | "h3" | "/h3" | "h4" | "/h4" | "li" | "tr" | "/tr"
            ) {
                if !last_was_whitespace {
                    result.push('\n');
                    last_was_whitespace = true;
                }
            }
            continue;
        }

        if in_tag {
            if collecting_tag_name {
                if c.is_whitespace() {
                    collecting_tag_name = false;
                } else {
                    tag_name.push(c);
                }
            }
            continue;
        }

        if in_script || in_style {
            continue;
        }

        if c.is_whitespace() {
            if !last_was_whitespace {
                result.push(' ');
                last_was_whitespace = true;
            }
        } else {
            result.push(c);
            last_was_whitespace = false;
        }
    }

    result
        .replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", "\"")
        .replace("&#x27;", "'")
        .replace("&#39;", "'")
        .replace("&nbsp;", " ")
        .replace("&#x2F;", "/")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strips_script_and_style() {
        let html = "<html><head><style>body{color:red}</style></head><body><script>alert(1)</script><p>Hello world</p></body></html>";
        let text = html_to_text(html);
        assert!(!text.contains("color:red"));
        assert!(!text.contains("alert"));
        assert!(text.contains("Hello world"));
    }

    #[test]
    fn preserves_text_content() {
        let html = "<div><h1>Title</h1><p>Some paragraph with <b>bold</b> text.</p></div>";
        let text = html_to_text(html);
        assert!(text.contains("Title"));
        assert!(text.contains("Some paragraph with bold text."));
    }

    #[test]
    fn decodes_entities() {
        let html = "a &amp; b &lt; c &gt; d &quot;e&quot;";
        let text = html_to_text(html);
        assert_eq!(text, "a & b < c > d \"e\"");
    }

    #[test]
    fn collapses_whitespace() {
        let html = "<p>  lots   of    spaces  </p>";
        let text = html_to_text(html);
        assert!(!text.contains("  "));
    }
}
