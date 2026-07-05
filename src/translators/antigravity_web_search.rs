//! Shared Antigravity web search helpers.
//!
//! Ported from Go's `antigravity/claude/web_search.go` (500 lines).
//! Handles web search request building and response grounding processing.

use serde_json::{Value, json};

/// Web search grounding support — maps text segments to source URLs.
#[derive(Debug, Clone)]
pub struct WebSearchGroundingSupport {
    pub start_index: i64,
    pub end_index: i64,
    #[allow(dead_code)] // WHY: Go original writes text but never reads it — kept for struct compatibility
    pub text: String,
    pub chunk_urls: Vec<String>,
    pub chunk_title: String,
}

/// A text block with optional citation metadata from web search results.
#[derive(Debug, Clone)]
pub struct WebSearchCitedTextBlock {
    pub text: String,
    pub citations: Vec<Value>,
}

const ANTIGRAVITY_WEB_SEARCH_SYSTEM_INSTRUCTION: &str = "You are a search engine bot. You will be given a query from a user. \
     Your task is to search the web for relevant information that will help the user. \
     You MUST perform a web search. Do not respond or interact with the user, \
     please respond as if they typed the query into a search bar.";

// ── Tool type detection ────────────────────────────────────

pub fn is_claude_web_search_tool_type(tool_type: &str) -> bool {
    tool_type == "web_search_20250305" || tool_type == "web_search_20260209"
}

pub fn has_claude_web_search_tool(body: &Value) -> bool {
    if let Some(tools) = body.get("tools").and_then(|v| v.as_array()) {
        for tool in tools {
            if is_claude_web_search_tool_type(
                tool.get("type").and_then(|v| v.as_str()).unwrap_or(""),
            ) {
                return true;
            }
        }
    }
    false
}

pub fn has_only_claude_web_search_tools(body: &Value) -> bool {
    if let Some(tools) = body.get("tools").and_then(|v| v.as_array()) {
        let mut has_web = false;
        for tool in tools {
            let ttype = tool.get("type").and_then(|v| v.as_str()).unwrap_or("");
            if is_claude_web_search_tool_type(ttype) {
                has_web = true;
                continue;
            }
            return false;
        }
        return has_web;
    }
    false
}

pub fn allows_claude_web_search_tool_choice(body: &Value) -> bool {
    match body.get("tool_choice") {
        None => true,
        Some(Value::String(s)) => matches!(s.as_str(), "" | "auto" | "any"),
        Some(obj) if obj.is_object() => match obj.get("type").and_then(|v| v.as_str()) {
            Some("tool") => obj.get("name").and_then(|v| v.as_str()) == Some("web_search"),
            Some("" | "auto" | "any") => true,
            _ => false,
        },
        _ => false,
    }
}

// ── Request building ──────────────────────────────────────

pub fn should_build_antigravity_web_search_request(_model: &str, body: &Value) -> bool {
    // Simplified — skips registry lookup (antigravitySupportsNativeGoogleSearch)
    has_only_claude_web_search_tools(body) && allows_claude_web_search_tool_choice(body)
}

pub fn build_antigravity_web_search_request(model: &str, body: &Value) -> Value {
    let query = extract_claude_web_search_query(body);
    let max_result_count = extract_claude_web_search_max_uses(body);
    let included_domains = extract_claude_web_search_allowed_domains(body);

    let mut out = json!({
        "model": model,
        "requestType": "web_search",
        "request": {
            "contents": [{"role": "user", "parts": [{"text": query}]}],
            "systemInstruction": {"role": "user", "parts": [{"text": ANTIGRAVITY_WEB_SEARCH_SYSTEM_INSTRUCTION}]},
            "tools": [{"googleSearch": {"enhancedContent": {"imageSearch": {"maxResultCount": max_result_count}}}}],
            "generationConfig": {"candidateCount": 1},
        }
    });

    if !included_domains.is_empty() {
        out["request"]["tools"][0]["googleSearch"]["includedDomains"] = json!(included_domains);
    }

    out
}

// ── Extractors ────────────────────────────────────────────

pub fn extract_claude_web_search_max_uses(body: &Value) -> i64 {
    const DEFAULT: i64 = 5;
    if let Some(tools) = body.get("tools").and_then(|v| v.as_array()) {
        for tool in tools {
            if is_claude_web_search_tool_type(
                tool.get("type").and_then(|v| v.as_str()).unwrap_or(""),
            ) && let Some(max_uses) = tool.get("max_uses").and_then(|v| v.as_i64())
                && max_uses > 0
            {
                return max_uses;
            }
        }
    }
    DEFAULT
}

pub fn extract_claude_web_search_allowed_domains(body: &Value) -> Vec<String> {
    if let Some(tools) = body.get("tools").and_then(|v| v.as_array()) {
        for tool in tools {
            if is_claude_web_search_tool_type(
                tool.get("type").and_then(|v| v.as_str()).unwrap_or(""),
            ) && let Some(domains) = tool.get("allowed_domains").and_then(|v| v.as_array())
            {
                return domains
                    .iter()
                    .filter_map(|d| d.as_str())
                    .map(|s| s.trim().to_string())
                    .filter(|s| !s.is_empty())
                    .collect();
            }
        }
    }
    vec![]
}

pub fn extract_claude_web_search_query(body: &Value) -> String {
    if let Some(msgs) = body.get("messages").and_then(|v| v.as_array()) {
        for msg in msgs.iter().rev() {
            let role = msg.get("role").and_then(|v| v.as_str()).unwrap_or("");
            if !role.is_empty() && role != "user" {
                continue;
            }
            let query = extract_claude_text_content(msg.get("content"));
            if !query.is_empty() {
                return query;
            }
        }
    }
    String::new()
}

pub fn extract_claude_text_content(content: Option<&Value>) -> String {
    match content {
        Some(Value::String(s)) => s.trim().to_string(),
        Some(Value::Array(parts)) => parts
            .iter()
            .filter_map(|p| p.get("text").and_then(|v| v.as_str()))
            .map(|s| s.trim())
            .filter(|s| !s.is_empty())
            .collect::<Vec<_>>()
            .join("\n"),
        _ => String::new(),
    }
}

// ── Response grounding ────────────────────────────────────

pub fn has_antigravity_google_search_tool(body: &Value) -> bool {
    if let Some(tools) = body.pointer("/request/tools").and_then(|v| v.as_array()) {
        for tool in tools {
            if tool.get("googleSearch").is_some() {
                return true;
            }
        }
    }
    false
}

/// Check if the response should translate web search grounding.
pub fn should_translate_web_search_grounding(
    original_request: &Value,
    translated_request: &Value,
) -> bool {
    has_claude_web_search_tool(original_request)
        && has_antigravity_google_search_tool(translated_request)
}

/// Extract grounding metadata from a Gemini/Antigravity response.
pub fn antigravity_grounding_metadata(root: &Value) -> Option<Value> {
    root.pointer("/response/candidates/0/groundingMetadata")
        .or_else(|| root.pointer("/candidates/0/groundingMetadata"))
        .cloned()
}

/// Extract text content from a response's candidate parts.
pub fn antigravity_text_content(root: &Value) -> String {
    let parts = root
        .pointer("/response/candidates/0/content/parts")
        .or_else(|| root.pointer("/candidates/0/content/parts"))
        .and_then(|v| v.as_array());

    if let Some(part_results) = parts {
        part_results
            .iter()
            .filter_map(|p| p.get("text").and_then(|v| v.as_str()))
            .collect::<Vec<_>>()
            .join("")
    } else {
        String::new()
    }
}

/// Extract input/output tokens from usage metadata.
#[allow(dead_code)] // WHY: Go original defines this but it is unused in both translations — kept for parity
pub fn antigravity_usage_tokens(root: &Value) -> (i64, i64) {
    let usage = root
        .pointer("/response/usageMetadata")
        .or_else(|| root.get("usageMetadata"));

    if let Some(u) = usage {
        let input_tokens = u
            .get("promptTokenCount")
            .and_then(|v| v.as_i64())
            .unwrap_or(0);
        let output_tokens = u
            .get("candidatesTokenCount")
            .and_then(|v| v.as_i64())
            .unwrap_or(0)
            + u.get("thoughtsTokenCount")
                .and_then(|v| v.as_i64())
                .unwrap_or(0);
        if output_tokens == 0 {
            let total = u
                .get("totalTokenCount")
                .and_then(|v| v.as_i64())
                .unwrap_or(0);
            if total > 0 {
                let out = total - input_tokens;
                return (input_tokens, if out < 0 { 0 } else { out });
            }
        }
        (input_tokens, output_tokens)
    } else {
        (0, 0)
    }
}

/// Extract web search query from grounding metadata.
pub fn web_search_query_from_grounding(grounding_metadata: &Value) -> String {
    if let Some(queries) = grounding_metadata
        .get("webSearchQueries")
        .and_then(|v| v.as_array())
        && let Some(q) = queries.first().and_then(|v| v.as_str())
    {
        return q.to_string();
    }
    String::new()
}

/// Extract web search results from grounding chunks.
pub fn web_search_results_from_grounding(grounding_metadata: &Value) -> Vec<Value> {
    let mut results: Vec<Value> = Vec::new();
    let mut seen_urls = std::collections::HashSet::new();

    if let Some(chunks) = grounding_metadata
        .get("groundingChunks")
        .and_then(|v| v.as_array())
    {
        for chunk in chunks {
            let web = chunk.get("web");
            if web.is_none() {
                continue;
            }
            let web = web.unwrap();
            let uri = web
                .get("uri")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .trim()
                .to_string();
            if uri.is_empty() {
                continue;
            }
            if !seen_urls.insert(uri.clone()) {
                continue;
            }

            let mut result = json!({"type": "web_search_result", "url": uri, "page_age": null});
            if let Some(title) = web.get("title").and_then(|v| v.as_str()) {
                result["title"] = json!(title);
            }
            results.push(result);
        }
    }

    results
}

/// Parse grounding supports — maps text segments to source chunks.
pub fn parse_web_search_grounding_supports(
    grounding_metadata: &Value,
) -> Vec<WebSearchGroundingSupport> {
    let chunks = match grounding_metadata
        .get("groundingChunks")
        .and_then(|v| v.as_array())
    {
        Some(c) => c.clone(),
        None => return vec![],
    };

    let chunk_data: Vec<(String, String)> = chunks
        .iter()
        .map(|chunk| {
            if let Some(web) = chunk.get("web") {
                let uri = web
                    .get("uri")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                let title = web
                    .get("title")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                (uri, title)
            } else {
                (String::new(), String::new())
            }
        })
        .collect();

    let supports = match grounding_metadata
        .get("groundingSupports")
        .and_then(|v| v.as_array())
    {
        Some(s) => s,
        None => return vec![],
    };

    supports
        .iter()
        .filter_map(|support| {
            let segment = support.get("segment")?;
            let mut parsed = WebSearchGroundingSupport {
                start_index: segment
                    .get("startIndex")
                    .and_then(|v| v.as_i64())
                    .unwrap_or(0),
                end_index: segment
                    .get("endIndex")
                    .and_then(|v| v.as_i64())
                    .unwrap_or(0),
                text: segment
                    .get("text")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string(),
                chunk_urls: vec![],
                chunk_title: String::new(),
            };

            if let Some(indices) = support
                .get("groundingChunkIndices")
                .and_then(|v| v.as_array())
            {
                for idx_val in indices {
                    let ci = idx_val.as_u64().unwrap_or(0) as usize;
                    if ci < chunk_data.len() {
                        if !parsed.chunk_urls.contains(&chunk_data[ci].0) {
                            parsed.chunk_urls.push(chunk_data[ci].0.clone());
                        }
                        if parsed.chunk_title.is_empty() {
                            parsed.chunk_title = chunk_data[ci].1.clone();
                        }
                    }
                }
            }

            Some(parsed)
        })
        .collect()
}

/// Build cited text blocks from ground text content and support segments.
pub fn build_web_search_cited_text_blocks(
    text_content: &str,
    supports: Vec<WebSearchGroundingSupport>,
) -> Vec<WebSearchCitedTextBlock> {
    if supports.is_empty() {
        if text_content.is_empty() {
            return vec![];
        }
        return vec![WebSearchCitedTextBlock {
            text: text_content.to_string(),
            citations: vec![],
        }];
    }

    let text_bytes = text_content.as_bytes();
    let text_len = text_bytes.len() as i64;
    let mut blocks: Vec<WebSearchCitedTextBlock> = Vec::new();
    let mut last_end: i64 = 0;

    for support in supports {
        if support.end_index <= last_end {
            continue;
        }

        // Uncited segment before this citation
        if support.start_index > last_end {
            let start = last_end.min(text_len).max(0) as usize;
            let end = (support.start_index).min(text_len).max(0) as usize;
            if start < end {
                blocks.push(WebSearchCitedTextBlock {
                    text: String::from_utf8_lossy(&text_bytes[start..end]).to_string(),
                    citations: vec![],
                });
            }
        }

        // Cited segment
        let cited_start = support.start_index.max(last_end);
        let cited_end = support.end_index.min(text_len);
        if cited_start < cited_end {
            let cited_text =
                String::from_utf8_lossy(&text_bytes[cited_start as usize..cited_end as usize])
                    .to_string();

            if !cited_text.is_empty() && !support.chunk_urls.is_empty() {
                let citation = json!({
                    "type": "web_search_result_location",
                    "cited_text": &cited_text,
                    "url": support.chunk_urls[0],
                    "title": support.chunk_title,
                });
                blocks.push(WebSearchCitedTextBlock {
                    text: cited_text,
                    citations: vec![citation],
                });
            }
        }

        if support.end_index > last_end {
            last_end = support.end_index;
        }
    }

    // Trailing uncited segment
    if (last_end as usize) < text_bytes.len() {
        blocks.push(WebSearchCitedTextBlock {
            text: String::from_utf8_lossy(&text_bytes[last_end as usize..]).to_string(),
            citations: vec![],
        });
    }

    blocks
}

/// Build the Claude web search content array (server_tool_use + tool_result + cited text).
pub fn build_claude_web_search_content(
    tool_use_id: &str,
    text_content: &str,
    grounding_metadata: &Value,
) -> Vec<Value> {
    let mut content: Vec<Value> = Vec::new();

    // server_tool_use
    let mut server_tool_use = json!({
        "type": "server_tool_use",
        "id": tool_use_id,
        "name": "web_search",
        "input": {},
    });
    let query = web_search_query_from_grounding(grounding_metadata);
    if !query.is_empty() {
        server_tool_use["input"]["query"] = json!(query);
    }
    content.push(server_tool_use);

    // web_search_tool_result
    let results = web_search_results_from_grounding(grounding_metadata);
    content.push(json!({
        "type": "web_search_tool_result",
        "tool_use_id": tool_use_id,
        "content": results,
    }));

    // Cited text blocks
    let supports = parse_web_search_grounding_supports(grounding_metadata);
    for block in build_web_search_cited_text_blocks(text_content, supports) {
        if block.text.is_empty() {
            continue;
        }
        let mut text_block = json!({"type": "text", "text": block.text});
        if !block.citations.is_empty() {
            text_block["citations"] = json!(block.citations);
        }
        content.push(text_block);
    }

    content
}

/// Append web search stream blocks to an output list with proper SSE events.
/// Returns the new content index after appending.
pub fn append_claude_web_search_stream_blocks(
    out: &mut Vec<Value>,
    start_index: i32,
    tool_use_id: &str,
    text_content: &str,
    grounding_metadata: &Value,
) -> i32 {
    let mut idx = start_index;

    // server_tool_use block
    let mut server_start = json!({
        "type": "content_block_start",
        "index": idx,
        "content_block": {"type": "server_tool_use", "id": tool_use_id, "name": "web_search", "input": {}},
    });
    let query = web_search_query_from_grounding(grounding_metadata);
    if !query.is_empty() {
        server_start["content_block"]["input"]["query"] = json!(query);
    }
    out.push(json!({"__sse": true, "event": "content_block_start", "data": server_start}));
    out.push(json!({"__sse": true, "event": "content_block_stop", "data": {"type": "content_block_stop", "index": idx}}));
    idx += 1;

    // web_search_tool_result block
    let results = web_search_results_from_grounding(grounding_metadata);
    out.push(json!({
        "__sse": true, "event": "content_block_start",
        "data": {
            "type": "content_block_start",
            "index": idx,
            "content_block": {
                "type": "web_search_tool_result",
                "tool_use_id": tool_use_id,
                "content": results,
            },
        }
    }));
    out.push(json!({"__sse": true, "event": "content_block_stop", "data": {"type": "content_block_stop", "index": idx}}));
    idx += 1;

    // Cited text blocks
    let supports = parse_web_search_grounding_supports(grounding_metadata);
    for block in build_web_search_cited_text_blocks(text_content, supports) {
        if block.text.is_empty() {
            continue;
        }
        out.push(json!({
            "__sse": true, "event": "content_block_start",
            "data": {
                "type": "content_block_start",
                "index": idx,
                "content_block": {"type": "text", "text": ""},
            }
        }));
        for citation in &block.citations {
            out.push(json!({
                "__sse": true, "event": "content_block_delta",
                "data": {
                    "type": "content_block_delta",
                    "index": idx,
                    "delta": {"type": "citations_delta", "citation": citation},
                }
            }));
        }
        for chunk in split_runes_for_web_search(&block.text, 50) {
            out.push(json!({
                "__sse": true, "event": "content_block_delta",
                "data": {
                    "type": "content_block_delta",
                    "index": idx,
                    "delta": {"type": "text_delta", "text": chunk},
                }
            }));
        }
        out.push(json!({"__sse": true, "event": "content_block_stop", "data": {"type": "content_block_stop", "index": idx}}));
        idx += 1;
    }

    idx
}

/// Split text into chunks of at most `chunk_size` runes (for streaming).
pub fn split_runes_for_web_search(text: &str, chunk_size: usize) -> Vec<String> {
    if chunk_size == 0 || text.is_empty() {
        return vec![];
    }
    let chars: Vec<char> = text.chars().collect();
    chars
        .chunks(chunk_size)
        .map(|c| c.iter().collect::<String>())
        .collect()
}

/// Generate a unique web search tool use ID.
pub fn new_claude_web_search_tool_use_id() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    format!("srvtoolu_{}", nanos)
}
