use async_trait::async_trait;
use logos_vfs::VfsError;
use reqwest::Client;

use crate::proc::ProcTool;

/// web_search — multi-source web search (DDG + Wikipedia + StackOverflow)
pub struct WebSearchTool {
    http: Client,
}

impl WebSearchTool {
    pub fn new() -> Self {
        let http = Client::builder()
            .user_agent("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
            .timeout(std::time::Duration::from_secs(10))
            .build()
            .unwrap_or_default();
        Self { http }
    }
}

#[derive(Debug, serde::Serialize)]
struct SearchResult {
    source: &'static str,
    title: String,
    url: String,
    snippet: String,
}

#[async_trait]
impl ProcTool for WebSearchTool {
    fn name(&self) -> &str {
        "web_search"
    }

    fn schema(&self) -> serde_json::Value {
        serde_json::json!({
            "name": "web_search",
            "description": "Search the web using DuckDuckGo, Wikipedia, and StackOverflow. Returns results from all three sources.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": { "type": "string", "description": "Search query" },
                    "max_results": { "type": "integer", "default": 3, "description": "Max results per source (default 3)" }
                },
                "required": ["query"]
            }
        })
    }

    async fn call(&self, params: &str) -> Result<String, VfsError> {
        let val: serde_json::Value =
            serde_json::from_str(params).map_err(|e| VfsError::InvalidJson(e.to_string()))?;
        let query = val["query"]
            .as_str()
            .ok_or_else(|| VfsError::InvalidJson("missing 'query'".to_string()))?;
        let max = val["max_results"].as_u64().unwrap_or(3) as usize;

        let (ddg, wiki, so) = tokio::join!(
            search_ddg(&self.http, query, max),
            search_wikipedia(&self.http, query, max),
            search_stackoverflow(&self.http, query, max),
        );

        let mut results: Vec<SearchResult> = Vec::new();
        results.extend(ddg.unwrap_or_default());
        results.extend(wiki.unwrap_or_default());
        results.extend(so.unwrap_or_default());

        if results.is_empty() {
            return Err(VfsError::Io("all search sources failed".to_string()));
        }

        serde_json::to_string(&results)
            .map_err(|e| VfsError::Io(format!("serialize results: {e}")))
    }
}

fn truncate(s: &str, max_chars: usize) -> String {
    if s.len() <= max_chars {
        s.to_string()
    } else {
        let mut end = max_chars;
        while end > 0 && !s.is_char_boundary(end) {
            end -= 1;
        }
        format!("{}...", &s[..end])
    }
}

// --- DuckDuckGo HTML ---

async fn search_ddg(_http: &Client, query: &str, max: usize) -> Result<Vec<SearchResult>, String> {
    // Use system curl to bypass DDG TLS fingerprint detection.
    // reqwest (rustls/native-tls) gets blocked; curl with SecureTransport works.
    let output = tokio::process::Command::new("curl")
        .args([
            "-s", "-X", "POST",
            "https://lite.duckduckgo.com/lite/",
            "-H", "User-Agent: Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "-H", "Referer: https://lite.duckduckgo.com/",
            "-H", "Accept: text/html",
            "-d", &format!("q={}", query.replace(' ', "+")),
            "--max-time", "10",
        ])
        .output()
        .await
        .map_err(|e| format!("ddg curl: {e}"))?;

    if !output.status.success() {
        return Err(format!("ddg curl exit: {}", output.status));
    }

    let html = String::from_utf8_lossy(&output.stdout);
    Ok(parse_ddg_html(&html, max))
}

fn parse_ddg_html(html: &str, max: usize) -> Vec<SearchResult> {
    let mut results = Vec::new();

    // Lite format:
    //   <a ... class='result-link' href="URL">TITLE</a>
    //   <td class='result-snippet'>SNIPPET</td>
    let mut pos = 0;
    while results.len() < max {
        // Find next result-link
        let marker = "class='result-link'";
        let marker_pos = match html[pos..].find(marker) {
            Some(i) => pos + i,
            None => break,
        };

        // Find the opening '<a' for this tag
        let tag_start = match html[..marker_pos].rfind("<a ") {
            Some(i) => i,
            None => { pos = marker_pos + marker.len(); continue; }
        };

        // Extract href
        let tag_region = &html[tag_start..];
        let tag_end_pos = match tag_region.find('>') {
            Some(i) => i,
            None => { pos = marker_pos + marker.len(); continue; }
        };
        let tag_str = &tag_region[..tag_end_pos];

        let url = if let Some(hi) = tag_str.find("href=\"") {
            let href_start = hi + 6;
            let href_end = tag_str[href_start..].find('"').map(|i| href_start + i).unwrap_or(href_start);
            tag_str[href_start..href_end].to_string()
        } else {
            pos = marker_pos + marker.len();
            continue;
        };

        // Extract title
        let content_start = tag_start + tag_end_pos + 1;
        let title_end = match html[content_start..].find("</a>") {
            Some(i) => content_start + i,
            None => { pos = content_start; continue; }
        };
        let title = strip_html_tags(&html[content_start..title_end]);

        pos = title_end;

        // Extract snippet: next result-snippet td
        let snippet_marker = "class='result-snippet'";
        let snippet = if let Some(si) = html[pos..].find(snippet_marker) {
            let snippet_pos = pos + si;
            let s_content_start = html[snippet_pos..].find('>').map(|i| snippet_pos + i + 1);
            let s_end = s_content_start.and_then(|sc| html[sc..].find("</td>").map(|i| sc + i));
            match (s_content_start, s_end) {
                (Some(sc), Some(se)) => {
                    pos = se;
                    truncate(&strip_html_tags(&html[sc..se]), 500)
                }
                _ => String::new(),
            }
        } else {
            String::new()
        };

        if !url.is_empty() && !title.is_empty() {
            results.push(SearchResult {
                source: "web",
                title,
                url,
                snippet,
            });
        }
    }

    results
}

fn strip_html_tags(s: &str) -> String {
    let mut result = String::with_capacity(s.len());
    let mut in_tag = false;
    for c in s.chars() {
        if c == '<' {
            in_tag = true;
        } else if c == '>' {
            in_tag = false;
        } else if !in_tag {
            result.push(c);
        }
    }
    // Decode common HTML entities
    result
        .replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", "\"")
        .replace("&#x27;", "'")
        .replace("&nbsp;", " ")
        .trim()
        .to_string()
}

// --- Wikipedia ---

async fn search_wikipedia(
    http: &Client,
    query: &str,
    max: usize,
) -> Result<Vec<SearchResult>, String> {
    let resp = http
        .get("https://en.wikipedia.org/w/api.php")
        .query(&[
            ("action", "query"),
            ("list", "search"),
            ("srsearch", query),
            ("format", "json"),
            ("srlimit", &max.to_string()),
        ])
        .send()
        .await
        .map_err(|e| format!("wiki request: {e}"))?;

    let body: serde_json::Value = resp
        .json()
        .await
        .map_err(|e| format!("wiki json: {e}"))?;

    let items = body["query"]["search"]
        .as_array()
        .cloned()
        .unwrap_or_default();

    Ok(items
        .iter()
        .map(|item| {
            let title = item["title"].as_str().unwrap_or_default().to_string();
            let snippet = strip_html_tags(item["snippet"].as_str().unwrap_or_default());
            let page_title = title.replace(' ', "_");
            SearchResult {
                source: "wiki",
                url: format!("https://en.wikipedia.org/wiki/{page_title}"),
                title,
                snippet: truncate(&snippet, 500),
            }
        })
        .collect())
}

// --- StackOverflow ---

async fn search_stackoverflow(
    http: &Client,
    query: &str,
    max: usize,
) -> Result<Vec<SearchResult>, String> {
    let resp = http
        .get("https://api.stackexchange.com/2.3/search")
        .query(&[
            ("order", "desc"),
            ("sort", "relevance"),
            ("intitle", query),
            ("site", "stackoverflow"),
            ("pagesize", &max.to_string()),
        ])
        .send()
        .await
        .map_err(|e| format!("so request: {e}"))?;

    let body: serde_json::Value = resp
        .json()
        .await
        .map_err(|e| format!("so json: {e}"))?;

    let items = body["items"].as_array().cloned().unwrap_or_default();

    Ok(items
        .iter()
        .map(|item| {
            let title = strip_html_tags(item["title"].as_str().unwrap_or_default());
            let url = item["link"].as_str().unwrap_or_default().to_string();
            let score = item["score"].as_i64().unwrap_or(0);
            let answered = item["is_answered"].as_bool().unwrap_or(false);
            let tags = item["tags"]
                .as_array()
                .map(|arr| {
                    arr.iter()
                        .filter_map(|t| t.as_str())
                        .collect::<Vec<_>>()
                        .join(", ")
                })
                .unwrap_or_default();
            SearchResult {
                source: "so",
                title,
                url,
                snippet: format!("Score:{score} Answered:{answered} Tags:{tags}"),
            }
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_ddg_lite_html() {
        let html = r#"
            <td><a rel="nofollow" href="https://example.com/page" class='result-link'>Example Title</a></td>
            <td class='result-snippet'>This is a snippet about the result.</td>
        "#;
        let results = parse_ddg_html(html, 3);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].title, "Example Title");
        assert_eq!(results[0].url, "https://example.com/page");
        assert!(results[0].snippet.contains("snippet about the result"));
    }

    #[test]
    fn strip_tags() {
        assert_eq!(
            strip_html_tags("<b>hello</b> world &amp; more"),
            "hello world & more"
        );
    }

    #[test]
    fn truncate_long_string() {
        let s = "a".repeat(300);
        let t = truncate(&s, 200);
        assert!(t.len() <= 203); // 200 + "..."
        assert!(t.ends_with("..."));
    }

    #[test]
    fn truncate_short_string() {
        assert_eq!(truncate("hello", 200), "hello");
    }

    #[tokio::test]
    #[ignore]
    async fn ddg_only() {
        let http = Client::builder()
            .timeout(std::time::Duration::from_secs(10))
            .build()
            .unwrap();
        let results = search_ddg(&http, "rust programming", 3).await;
        println!("DDG results: {results:?}");
        let items = results.unwrap();
        assert!(!items.is_empty(), "DDG should return results");
        for r in &items {
            println!("  {} | {} | {}", r.title, r.url, r.snippet);
        }
    }

    /// End-to-end test hitting real network. Run with:
    /// cargo test --package logos-kernel -- web_search_e2e --ignored --nocapture
    #[tokio::test]
    #[ignore]
    async fn web_search_e2e() {
        let tool = WebSearchTool::new();
        let result = tool
            .call(r#"{"query":"rust programming language","max_results":2}"#)
            .await
            .unwrap();
        println!("=== web_search result ===");
        println!("{result}");

        let parsed: Vec<serde_json::Value> = serde_json::from_str(&result).unwrap();
        assert!(!parsed.is_empty(), "should have at least 1 result");

        let sources: Vec<&str> = parsed.iter().filter_map(|r| r["source"].as_str()).collect();
        println!("sources: {sources:?}");
        // At least one source should have returned results
        assert!(
            sources.contains(&"web") || sources.contains(&"wiki") || sources.contains(&"so"),
            "at least one source should succeed"
        );
    }
}
