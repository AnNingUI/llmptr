//! Claude Messages → Google Gemini request translation.
//!
//! Maps Claude Messages format to Gemini generateContent format.

use serde_json::{Value, json};
use llmptr_infra::signature;

/// Convert a Claude Messages request to Gemini generateContent format.
pub fn transform(model: &str, body: Value, _stream: bool) -> Value {
    let mut out = json!({
        "contents": [],
    });

    if !model.is_empty() {
        out["model"] = json!(model);
    }

    // ── generationConfig ──────────────────────────────────
    if let Some(temp) = body.get("temperature") {
        out["generationConfig"]["temperature"] = temp.clone();
    }
    if let Some(top_p) = body.get("top_p") {
        out["generationConfig"]["topP"] = top_p.clone();
    }
    if let Some(top_k) = body.get("top_k") {
        out["generationConfig"]["topK"] = top_k.clone();
    }

    // ── thinking config → generationConfig.thinkingConfig ──
    if let Some(thinking) = body.get("thinking").and_then(|v| v.as_object()) {
        let ttype = thinking.get("type").and_then(|v| v.as_str()).unwrap_or("");
        match ttype {
            "enabled" => {
                if let Some(budget) = thinking.get("budget_tokens").and_then(|v| v.as_u64()) {
                    out["generationConfig"]["thinkingConfig"] = json!({
                        "thinkingBudget": budget,
                        "includeThoughts": true,
                    });
                } else {
                    out["generationConfig"]["thinkingConfig"] = json!({
                        "includeThoughts": true,
                    });
                }
            }
            "adaptive" | "auto" => {
                if let Some(effort) = body
                    .pointer("/output_config/effort")
                    .and_then(|v| v.as_str())
                {
                    out["generationConfig"]["thinkingConfig"] = json!({
                        "thinkingLevel": effort,
                        "includeThoughts": true,
                    });
                } else {
                    // No explicit effort → default to high
                    out["generationConfig"]["thinkingConfig"] = json!({
                        "thinkingLevel": "high",
                        "includeThoughts": true,
                    });
                }
            }
            _ => {}
        }
    }

    // ── system_instruction (strip Claude Code attribution) ──
    if let Some(system) = body.get("system") {
        match system {
            Value::String(s) if !s.is_empty() => {
                let cleaned = strip_claude_attribution(s);
                if !cleaned.is_empty() {
                    out["system_instruction"] = json!({
                        "parts": [{"text": cleaned}]
                    });
                }
            }
            Value::Array(parts) => {
                let texts: Vec<String> = parts
                    .iter()
                    .filter_map(|p| p.get("text").and_then(|v| v.as_str()))
                    .filter(|s| !s.is_empty() && !is_claude_attribution(s))
                    .map(|s| s.to_string())
                    .collect();
                if !texts.is_empty() {
                    out["system_instruction"] = json!({
                        "role": "user",
                        "parts": texts.iter().map(|t| json!({"text": t})).collect::<Vec<_>>()
                    });
                }
            }
            _ => {}
        }
    }

    // ── messages → contents ──────────────────────────────
    let mut pending_tool_ids: Vec<String> = Vec::new();

    if let Some(msgs) = body.get("messages").and_then(|v| v.as_array()) {
        for msg in msgs {
            let role = msg.get("role").and_then(|v| v.as_str()).unwrap_or("user");
            let gemini_role = match role {
                "assistant" => "model",
                "system" => "user",
                r => r, // user, tool → user
            };

            let mut parts: Vec<Value> = Vec::new();
            let content = msg.get("content");

            if role == "system" {
                if let Some(reminder) = claude_system_reminder_text(content) {
                    out["contents"].as_array_mut().unwrap().push(json!({
                        "role": "user",
                        "parts": [{"text": reminder}],
                    }));
                }
                continue;
            }

            match content {
                Some(Value::Array(content_parts)) => {
                    for part in content_parts {
                        let ptype = part.get("type").and_then(|v| v.as_str()).unwrap_or("");
                        match ptype {
                            "text" => {
                                if let Some(t) = part.get("text").and_then(|v| v.as_str())
                                    && !t.is_empty()
                                {
                                    parts.push(json!({"text": t}));
                                }
                            }
                            "image" => {
                                if let Some(source) = part.get("source") {
                                    let media_type = source
                                        .get("media_type")
                                        .and_then(|v| v.as_str())
                                        .unwrap_or("image/png");
                                    let data =
                                        source.get("data").and_then(|v| v.as_str()).unwrap_or("");
                                    parts.push(json!({
                                        "inline_data": {
                                            "mime_type": media_type,
                                            "data": data,
                                        }
                                    }));
                                }
                            }
                            "tool_use" => {
                                if role == "assistant" {
                                    let mut name = part
                                        .get("name")
                                        .and_then(|v| v.as_str())
                                        .unwrap_or("")
                                        .to_string();
                                    if let Some(id) = part.get("id").and_then(|v| v.as_str())
                                        && let Some(derived) = tool_name_from_claude_tool_use_id(id)
                                    {
                                        name = derived;
                                    }
                                    let empty_obj = json!({});
                                    let args = part.get("input").unwrap_or(&empty_obj);
                                    let id = part.get("id").and_then(|v| v.as_str()).unwrap_or("");
                                    pending_tool_ids.push(id.to_string());
                                    let fn_name = sanitize_function_name(&name);
                                    parts.push(json!({
                                        "thoughtSignature": signature::GEMINI_SKIP_SENTINEL,
                                        "functionCall": {
                                            "name": fn_name,
                                            "args": args,
                                        }
                                    }));
                                }
                            }
                            "tool_result" => {
                                let tool_use_id = part
                                    .get("tool_use_id")
                                    .and_then(|v| v.as_str())
                                    .unwrap_or("");

                                // Match with pending tool IDs (FIFO)
                                let matched_id = if !tool_use_id.is_empty() {
                                    let idx =
                                        pending_tool_ids.iter().position(|id| id == tool_use_id);
                                    if let Some(i) = idx {
                                        let id = pending_tool_ids[i].clone();
                                        pending_tool_ids.remove(i);
                                        id
                                    } else {
                                        tool_use_id.to_string()
                                    }
                                } else if !pending_tool_ids.is_empty() {
                                    pending_tool_ids.remove(0)
                                } else {
                                    gen_tool_call_id()
                                };

                                let fn_name = tool_name_from_claude_tool_use_id(&matched_id)
                                    .unwrap_or_else(|| matched_id.clone());
                                let (result, images) =
                                    convert_claude_tool_result_content(part.get("content"));
                                parts.push(json!({
                                    "functionResponse": {
                                        "name": sanitize_function_name(&fn_name),
                                        "response": {
                                            "result": result,
                                        }
                                    }
                                }));
                                for image in images {
                                    parts.push(image);
                                }
                            }
                            "thinking" | "redacted_thinking" => {}
                            _ => {}
                        }
                    }
                }
                Some(Value::String(s)) if !s.is_empty() => {
                    parts.push(json!({"text": s}));
                }
                _ => {}
            }

            if !parts.is_empty() {
                out["contents"].as_array_mut().unwrap().push(json!({
                    "role": gemini_role,
                    "parts": parts,
                }));
            }
        }
    }

    // ── tools (including web_search) ──────────────────────
    let mut has_google_search = false;
    if let Some(tools) = body.get("tools").and_then(|v| v.as_array()) {
        let mut func_decls: Vec<Value> = Vec::new();
        for tool in tools {
            // Check for Claude web_search tool type → Gemini googleSearch
            let tool_type = tool.get("type").and_then(|v| v.as_str()).unwrap_or("");
            if tool_type == "web_search_20250305" {
                has_google_search = true;
                continue;
            }
            let raw_name = tool.get("name").and_then(|v| v.as_str()).unwrap_or("");
            let name = sanitize_function_name(raw_name);
            let desc = tool
                .get("description")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            let input_schema = tool.get("input_schema");

            let mut decl = json!({
                "name": name,
                "description": desc,
            });

            if let Some(schema) = input_schema {
                let mut clean = schema.clone();
                if let Some(obj) = clean.as_object_mut() {
                    obj.remove("cache_control");
                    obj.remove("defer_loading");
                    obj.remove("eager_input_streaming");
                }
                decl["parametersJsonSchema"] = clean;
            }

            func_decls.push(decl);
        }
        if !func_decls.is_empty() {
            out["tools"] = json!([{ "functionDeclarations": func_decls }]);
        }
        if has_google_search {
            if let Some(tools_arr) = out["tools"].as_array_mut() {
                tools_arr.push(json!({"googleSearch": {}}));
            } else {
                out["tools"] = json!([{"googleSearch": {}}]);
            }
        }
    }

    // ── tool_choice → tool_config ────────────────────────
    if let Some(tc) = body.get("tool_choice") {
        let ttype = tc.get("type").and_then(|v| v.as_str()).unwrap_or("auto");
        let mode = match ttype {
            "auto" => "AUTO",
            "any" => "ANY",
            "none" => "NONE",
            "tool" => "ANY",
            _ => "AUTO",
        };
        out["toolConfig"] = json!({
            "functionCallingConfig": {
                "mode": mode,
            }
        });
        if ttype == "tool"
            && let Some(name) = tc.get("name").and_then(|v| v.as_str())
            && !name.is_empty()
        {
            out["toolConfig"]["functionCallingConfig"]["allowedFunctionNames"] =
                json!([sanitize_function_name(name)]);
        }
    }

    // ── strip trailing model turn with unanswered function calls ──
    // Gemini returns empty responses when the last turn is a model
    // functionCall with no corresponding user functionResponse.
    if let Some(contents) = out.get("contents").and_then(|v| v.as_array())
        && let Some(last) = contents.last()
        && last.get("role").and_then(|v| v.as_str()) == Some("model")
    {
        let has_fc = last
            .get("parts")
            .and_then(|v| v.as_array())
            .map(|parts| parts.iter().any(|p| p.get("functionCall").is_some()))
            .unwrap_or(false);
        if has_fc {
            out["contents"].as_array_mut().unwrap().pop();
        }
    }

    // ── safetySettings (default: all OFF, CIVIC → BLOCK_NONE) ──
    if out.get("safetySettings").is_none() {
        out["safetySettings"] = json!([
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "OFF"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "OFF"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "OFF"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "OFF"},
            {"category": "HARM_CATEGORY_CIVIC_INTEGRITY", "threshold": "BLOCK_NONE"},
        ]);
    }

    out
}

// ── helpers ────────────────────────────────────────────────

/// Strip Claude Code attribution text from system messages.
fn strip_claude_attribution(s: &str) -> String {
    let attribution_phrases = ["The assistant is Claude", "由 Claude 开发"];
    for phrase in &attribution_phrases {
        if let Some(pos) = s.find(phrase) {
            let before = &s[..pos];
            let after = &s[pos + phrase.len()..];
            let trimmed = format!("{}{}", before.trim(), after.trim());
            return trimmed.trim().to_string();
        }
    }
    s.to_string()
}

/// Check if text is a Claude Code attribution string.
fn is_claude_attribution(s: &str) -> bool {
    s.contains("The assistant is Claude") || s.contains("由 Claude 开发")
}

/// Sanitize a function name for Gemini compatibility — replace non-alphanumeric
/// (except underscore, dash, dot) with underscores.
fn sanitize_function_name(name: &str) -> String {
    name.chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '_' || c == '-' || c == '.' {
                c
            } else {
                '_'
            }
        })
        .collect::<String>()
        .trim_end_matches('_')
        .to_string()
}

fn gen_tool_call_id() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    format!("toolu_{:x}", nanos)
}

fn tool_name_from_claude_tool_use_id(tool_use_id: &str) -> Option<String> {
    let parts: Vec<&str> = tool_use_id.split('-').collect();
    if parts.len() <= 1 {
        None
    } else {
        Some(parts[..parts.len() - 1].join("-"))
    }
}

fn claude_system_reminder_text(content: Option<&Value>) -> Option<String> {
    let parts: Vec<String> = match content {
        Some(Value::String(s)) if !s.is_empty() && !is_claude_attribution(s) => vec![s.clone()],
        Some(Value::Array(items)) => items
            .iter()
            .filter(|item| item.get("type").and_then(|v| v.as_str()) == Some("text"))
            .filter_map(|item| item.get("text").and_then(|v| v.as_str()))
            .filter(|text| !text.is_empty() && !is_claude_attribution(text))
            .map(ToString::to_string)
            .collect(),
        _ => Vec::new(),
    };

    if parts.is_empty() {
        return None;
    }

    let text = parts.join("\n");
    if text.trim().is_empty() {
        None
    } else {
        Some(format!("<system-reminder>\n{text}\n</system-reminder>"))
    }
}

fn convert_claude_tool_result_content(content: Option<&Value>) -> (Value, Vec<Value>) {
    match content {
        Some(Value::String(s)) => (json!(s), Vec::new()),
        Some(Value::Array(parts)) => {
            let mut text_parts: Vec<Value> = Vec::new();
            let mut images: Vec<Value> = Vec::new();

            for part in parts {
                match part.get("type").and_then(|v| v.as_str()) {
                    Some("text") => {
                        if let Some(text) = part.get("text").and_then(|v| v.as_str()) {
                            text_parts.push(json!({
                                "type": "text",
                                "text": text,
                            }));
                        }
                    }
                    Some("image") => {
                        if let Some(source) = part.get("source") {
                            let mime_type = source
                                .get("media_type")
                                .and_then(|v| v.as_str())
                                .unwrap_or("image/png");
                            let data = source.get("data").and_then(|v| v.as_str()).unwrap_or("");
                            if !data.is_empty() {
                                images.push(json!({
                                    "inline_data": {
                                        "mime_type": mime_type,
                                        "data": data,
                                    }
                                }));
                            }
                        }
                    }
                    _ => {}
                }
            }

            let result = match text_parts.len() {
                0 => json!(""),
                1 => text_parts.pop().unwrap(),
                _ => json!(text_parts),
            };
            (result, images)
        }
        Some(other) => (other.clone(), Vec::new()),
        None => (json!(""), Vec::new()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_claude_to_gemini_basic() {
        let body = json!({
            "model": "claude-sonnet-4-8",
            "max_tokens": 4096,
            "messages": [
                {"role": "user", "content": "Hello!"}
            ]
        });
        let result = transform("gemini-2.0-flash", body, false);
        assert_eq!(result["contents"][0]["role"], "user");
        assert_eq!(result["contents"][0]["parts"][0]["text"], "Hello!");
    }

    #[test]
    fn test_claude_to_gemini_with_system() {
        let body = json!({
            "system": "You are a helpful assistant.",
            "messages": [{"role": "user", "content": "Hi"}]
        });
        let result = transform("gemini-2.0-flash", body, false);
        assert_eq!(
            result["system_instruction"]["parts"][0]["text"],
            "You are a helpful assistant."
        );
    }

    #[test]
    fn test_claude_to_gemini_with_tools() {
        let body = json!({
            "tools": [{
                "name": "get_weather",
                "description": "Get weather",
                "input_schema": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"]
                }
            }],
            "messages": [{"role": "user", "content": "Weather?"}]
        });
        let result = transform("gemini-2.0-flash", body, false);
        assert!(result["tools"].is_array());
        assert_eq!(
            result["tools"][0]["functionDeclarations"][0]["name"],
            "get_weather"
        );
    }

    #[test]
    fn test_claude_to_gemini_max_tokens() {
        let body = json!({
            "max_tokens": 8192,
            "messages": [{"role": "user", "content": "Hi"}]
        });
        let result = transform("gemini-2.0-flash", body, false);
        assert!(result["generationConfig"]["maxOutputTokens"].is_null());
    }
}
