//! Google Gemini → Claude Messages request translation.

use serde_json::{Value, json};

/// Convert a Gemini generateContent request to Claude Messages format.
pub fn transform(model: &str, body: Value, stream: bool) -> Value {
    let mut out = json!({
        "model": model,
        "max_tokens": 32000,
        "messages": [],
        "metadata": {"user_id": "user_llmptr_account_default_session_default"},
        "stream": stream,
    });

    // ── safetySettings (if present, pass through) ──────────
    if let Some(safety) = body.get("safetySettings") {
        out["safetySettings"] = safety.clone();
    }

    // ── generationConfig → Claude top-level ─────────────────
    if let Some(gc) = body.get("generationConfig").and_then(|v| v.as_object()) {
        if let Some(max_tokens) = gc.get("maxOutputTokens").and_then(|v| v.as_u64()) {
            out["max_tokens"] = json!(max_tokens);
        }
        if let Some(temp) = gc.get("temperature") {
            out["temperature"] = temp.clone();
        } else if let Some(top_p) = gc.get("topP") {
            out["top_p"] = top_p.clone();
        }
        if let Some(stops) = gc.get("stopSequences").and_then(|v| v.as_array()) {
            let vals: Vec<Value> = stops
                .iter()
                .map(|s| json!(s.as_str().unwrap_or("")))
                .collect();
            if !vals.is_empty() {
                out["stop_sequences"] = json!(vals);
            }
        }

        // thinkingConfig → Claude thinking
        if let Some(tc) = gc.get("thinkingConfig").and_then(|v| v.as_object()) {
            if let Some(level) = tc.get("thinkingLevel").and_then(|v| v.as_str()) {
                match level {
                    "none" => {
                        out["thinking"] = json!({"type": "disabled"});
                    }
                    _ => {
                        out["thinking"] = json!({
                            "type": "adaptive",
                            "output_config": {"effort": level}
                        });
                    }
                }
            } else if let Some(budget) = tc.get("thinkingBudget").and_then(|v| v.as_u64()) {
                if budget == 0 {
                    out["thinking"] = json!({"type": "disabled"});
                } else {
                    out["thinking"] = json!({
                        "type": "enabled",
                        "budget_tokens": budget,
                    });
                }
            } else if tc.get("includeThoughts").and_then(|v| v.as_bool()) == Some(true) {
                out["thinking"] = json!({"type": "enabled"});
            }
        }
    }

    // ── system_instruction → Claude system field ───────────
    if let Some(si) = body.get("system_instruction").and_then(|v| v.as_object())
        && let Some(parts) = si.get("parts").and_then(|v| v.as_array())
    {
        let texts: Vec<String> = parts
            .iter()
            .filter_map(|p| p.get("text").and_then(|v| v.as_str()))
            .filter(|s| !s.is_empty())
            .map(|s| s.to_string())
            .collect();
        if !texts.is_empty() {
            out["messages"].as_array_mut().unwrap().push(json!({
                "role": "user",
                "content": [{"type": "text", "text": texts.join("\n")}],
            }));
        }
    }

    // ── contents → Claude messages ──────────────────────────
    let mut pending_tool_ids: Vec<String> = Vec::new();

    if let Some(contents) = body.get("contents").and_then(|v| v.as_array()) {
        for content in contents {
            let role = content
                .get("role")
                .and_then(|v| v.as_str())
                .unwrap_or("user");
            let claude_role = match role {
                "model" => "assistant",
                r => r,
            };

            let mut parts: Vec<Value> = Vec::new();

            if let Some(content_parts) = content.get("parts").and_then(|v| v.as_array()) {
                for part in content_parts {
                    // text
                    if let Some(text) = part.get("text").and_then(|v| v.as_str())
                        && !text.is_empty()
                    {
                        parts.push(json!({"type": "text", "text": text}));
                    }

                    // inline_data → image
                    if let Some(id) = part.get("inline_data").and_then(|v| v.as_object()) {
                        let media_type = id
                            .get("mime_type")
                            .and_then(|v| v.as_str())
                            .unwrap_or("image/png");
                        let data = id.get("data").and_then(|v| v.as_str()).unwrap_or("");
                        if !data.is_empty() {
                            parts.push(json!({
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": media_type,
                                    "data": data,
                                }
                            }));
                        }
                    }

                    // functionCall → tool_use
                    if let Some(fc) = part.get("functionCall").and_then(|v| v.as_object()) {
                        let name = fc.get("name").and_then(|v| v.as_str()).unwrap_or("");
                        let empty_obj = json!({});
                        let args = fc.get("args").unwrap_or(&empty_obj);
                        let tool_id = fc.get("id").and_then(|v| v.as_str()).unwrap_or("");
                        let id = if !tool_id.is_empty() {
                            tool_id.to_string()
                        } else {
                            format!("toolu_{:016x}", pending_tool_ids.len())
                        };
                        // Try to derive function name from tool ID if name is empty
                        let fn_name = if name.is_empty() {
                            tool_name_from_claude_tool_use_id(&id)
                        } else {
                            name.to_string()
                        };
                        pending_tool_ids.push(id.clone());

                        parts.push(json!({
                            "type": "tool_use",
                            "id": id,
                            "name": fn_name,
                            "input": args,
                        }));
                    }

                    // functionResponse → tool_result
                    if let Some(fr) = part.get("functionResponse").and_then(|v| v.as_object()) {
                        let custom_id = fr.get("id").and_then(|v| v.as_str()).unwrap_or("");
                        let matched_id = if !custom_id.is_empty() {
                            if let Some(pos) =
                                pending_tool_ids.iter().position(|id| id == custom_id)
                            {
                                let id = pending_tool_ids[pos].clone();
                                pending_tool_ids.remove(pos);
                                id
                            } else {
                                custom_id.to_string()
                            }
                        } else if !pending_tool_ids.is_empty() {
                            pending_tool_ids.remove(0)
                        } else {
                            format!("toolu_fallback_{:016x}", parts.len())
                        };

                        let empty_resp = json!({});
                        let response_val = fr.get("response").unwrap_or(&empty_resp);
                        let result_text = response_val
                            .get("result")
                            .and_then(|v| v.as_str())
                            .unwrap_or("");

                        parts.push(json!({
                            "type": "tool_result",
                            "tool_use_id": matched_id,
                            "content": result_text,
                        }));
                    }

                    // file_data → text reference
                    if let Some(fd) = part.get("file_data").and_then(|v| v.as_object()) {
                        let file_uri = fd.get("file_uri").and_then(|v| v.as_str()).unwrap_or("");
                        let mime = fd.get("mime_type").and_then(|v| v.as_str()).unwrap_or("");
                        let info = format!("File: {} (Type: {})", file_uri, mime);
                        parts.push(json!({"type": "text", "text": info}));
                    }
                }
            }

            if !parts.is_empty() {
                out["messages"].as_array_mut().unwrap().push(json!({
                    "role": claude_role,
                    "content": parts,
                }));
            }
        }
    }

    // ── tools → Claude tools ──────────────────────────────
    if let Some(tools) = body.get("tools").and_then(|v| v.as_array()) {
        let mut claude_tools: Vec<Value> = Vec::new();
        for tool_group in tools {
            if let Some(func_decls) = tool_group
                .get("functionDeclarations")
                .and_then(|v| v.as_array())
            {
                for decl in func_decls {
                    let name = decl.get("name").and_then(|v| v.as_str()).unwrap_or("");
                    let desc = decl
                        .get("description")
                        .and_then(|v| v.as_str())
                        .unwrap_or("");
                    let params = decl
                        .get("parameters")
                        .or_else(|| decl.get("parametersJsonSchema"));

                    let mut input_schema = params
                        .cloned()
                        .unwrap_or(json!({"type": "object", "properties": {}}));
                    lowercase_schema_types(&mut input_schema);
                    if let Some(obj) = input_schema.as_object_mut() {
                        obj.insert("additionalProperties".to_string(), json!(false));
                        obj.insert(
                            "$schema".to_string(),
                            json!("http://json-schema.org/draft-07/schema#"),
                        );
                    }

                    claude_tools.push(json!({
                        "name": name,
                        "description": desc,
                        "input_schema": input_schema,
                    }));
                }
            }
        }
        if !claude_tools.is_empty() {
            out["tools"] = json!(claude_tools);
        }
    }

    // ── tool_config → tool_choice ──────────────────────────
    if let Some(tc) = body.get("tool_config").and_then(|v| v.as_object())
        && let Some(fcc) = tc
            .get("function_calling_config")
            .and_then(|v| v.as_object())
        && let Some(mode) = fcc.get("mode").and_then(|v| v.as_str())
    {
        match mode {
            "AUTO" => {
                out["tool_choice"] = json!({"type": "auto"});
            }
            "NONE" => {
                out["tool_choice"] = json!({"type": "none"});
            }
            "ANY" => {
                out["tool_choice"] = json!({"type": "any"});
            }
            _ => {
                out["tool_choice"] = json!({"type": "auto"});
            }
        }
    }

    out
}

/// Derive a function name from a Claude tool use ID (e.g. "get_weather-abc123" → "get_weather").
fn tool_name_from_claude_tool_use_id(tool_use_id: &str) -> String {
    let parts: Vec<&str> = tool_use_id.splitn(2, '-').collect();
    if parts.len() > 1 && !parts[0].is_empty() {
        parts[0].to_string()
    } else {
        String::new()
    }
}

fn lowercase_schema_types(value: &mut Value) {
    match value {
        Value::Object(obj) => {
            if let Some(type_value) = obj.get_mut("type")
                && let Some(type_string) = type_value.as_str()
            {
                *type_value = json!(type_string.to_lowercase());
            }
            for child in obj.values_mut() {
                lowercase_schema_types(child);
            }
        }
        Value::Array(items) => {
            for item in items {
                lowercase_schema_types(item);
            }
        }
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gemini_to_claude_basic() {
        let body = json!({
            "contents": [{
                "role": "user",
                "parts": [{"text": "Hello!"}]
            }],
            "generationConfig": {"maxOutputTokens": 4096}
        });
        let result = transform("claude-sonnet-4-8", body, false);
        assert_eq!(result["messages"][0]["role"], "user");
        assert_eq!(result["messages"][0]["content"][0]["text"], "Hello!");
        assert_eq!(result["max_tokens"], 4096);
    }

    #[test]
    fn test_gemini_to_claude_with_system() {
        let body = json!({
            "system_instruction": {
                "parts": [{"text": "You are helpful."}]
            },
            "contents": [{"role": "user", "parts": [{"text": "Hi"}]}]
        });
        let result = transform("claude-sonnet-4-8", body, false);
        assert_eq!(result["messages"][0]["role"], "user");
        assert_eq!(
            result["messages"][0]["content"][0]["text"],
            "You are helpful."
        );
        assert_eq!(result["messages"][1]["content"][0]["text"], "Hi");
    }

    #[test]
    fn test_gemini_to_claude_with_tools() {
        let body = json!({
            "tools": [{
                "functionDeclarations": [{
                    "name": "get_weather",
                    "description": "Get weather",
                    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}
                }]
            }],
            "contents": [{"role": "user", "parts": [{"text": "Weather?"}]}]
        });
        let result = transform("claude-sonnet-4-8", body, false);
        assert_eq!(result["tools"][0]["name"], "get_weather");
        assert!(result["tools"][0]["input_schema"]["additionalProperties"] == json!(false));
    }

    #[test]
    fn test_gemini_to_claude_thinking() {
        let body = json!({
            "generationConfig": {
                "thinkingConfig": {"thinkingLevel": "high"}
            },
            "contents": [{"role": "user", "parts": [{"text": "Think"}]}]
        });
        let result = transform("claude-sonnet-4-8", body, false);
        assert_eq!(result["thinking"]["type"], "adaptive");
        assert_eq!(result["thinking"]["output_config"]["effort"], "high");
    }
}
