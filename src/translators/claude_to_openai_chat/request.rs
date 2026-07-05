//! Claude Messages → OpenAI ChatCompletions request translation.
//!
//! Maps:
//! - `system` field → `messages[{role:"system"}]`
//! - `messages[].role=assistant` + `thinking` blocks → `reasoning_content`
//! - `messages[].role=assistant` + `tool_use` blocks → `tool_calls`
//! - `messages[].role=user` + `tool_result` → role "tool"
//! - `tools[].input_schema` → `tools[].function.parameters`
//! - `thinking.budget_tokens` → `reasoning_effort`
//! - `stop_sequences` → `stop`
//! - `max_tokens`, `temperature`, `top_p` → direct

use serde_json::{Value, json};
use translator_infra::thinking;

/// Convert a Claude Messages API request to OpenAI ChatCompletions format.
pub fn transform(model: &str, body: Value, stream: bool) -> Value {
    let mut out = json!({
        "model": model,
        "messages": [],
    });

    // ── top-level scalar fields ──────────────────────────────────
    if let Some(max_tokens) = body.get("max_tokens").and_then(|v| v.as_u64()) {
        out["max_tokens"] = json!(max_tokens);
    }
    if let Some(temp) = body.get("temperature") {
        out["temperature"] = temp.clone();
    } else if let Some(top_p) = body.get("top_p") {
        out["top_p"] = top_p.clone();
    }

    // stop_sequences → OpenAI `stop`
    if let Some(stops) = body.get("stop_sequences").and_then(|v| v.as_array())
        && !stops.is_empty()
    {
        let vals: Vec<Value> = stops
            .iter()
            .map(|s| json!(s.as_str().unwrap_or("")))
            .collect();
        out["stop"] = if vals.len() == 1 {
            vals[0].clone()
        } else {
            json!(vals)
        };
    }

    out["stream"] = json!(stream);

    // ── thinking → reasoning_effort ───────────────────────────────
    if let Some(thinking) = body.get("thinking").and_then(|v| v.as_object()) {
        let ttype = thinking.get("type").and_then(|v| v.as_str()).unwrap_or("");
        match ttype {
            "enabled" => {
                let budget = thinking.get("budget_tokens").and_then(|v| v.as_u64());
                let (effort, _) = thinking::convert_budget_to_level(budget.unwrap_or(0) as i64);
                if !effort.is_empty() {
                    out["reasoning_effort"] = json!(effort);
                }
            }
            "adaptive" | "auto" => {
                let effort = body
                    .pointer("/output_config/effort")
                    .and_then(|v| v.as_str())
                    .unwrap_or("xhigh");
                out["reasoning_effort"] = json!(effort);
            }
            "disabled" => {
                let effort = thinking::convert_budget_to_level(0).0;
                if !effort.is_empty() {
                    out["reasoning_effort"] = json!(effort);
                }
            }
            _ => {}
        }
    }

    // ── build messages ───────────────────────────────────────────
    let mut messages: Vec<Value> = Vec::new();

    // 1) system prompt
    if let Some(system) = body.get("system") {
        let sys = claude_system_to_openai(system);
        if let Some(sys_msg) = sys {
            messages.push(sys_msg);
        }
    }

    // 2) conversation messages
    if let Some(msg_array) = body.get("messages").and_then(|v| v.as_array()) {
        for msg in msg_array {
            let role = msg.get("role").and_then(|v| v.as_str()).unwrap_or("user");

            if role == "system" {
                // Embedded system reminder → emit as a user message
                if let Some(content) = msg.get("content")
                    && let Some(reminder) = extract_system_reminder(content)
                {
                    messages.push(json!({
                        "role": "user",
                        "content": [{"type": "text", "text": reminder}]
                    }));
                }
                continue;
            }

            let content = msg.get("content");
            if content.is_none() {
                continue;
            }
            let content = content.unwrap();

            match content {
                Value::Array(parts) => {
                    let mut text_parts: Vec<Value> = Vec::new();
                    let mut reasoning_parts: Vec<String> = Vec::new();
                    let mut tool_calls: Vec<Value> = Vec::new();
                    let mut tool_results: Vec<Value> = Vec::new();

                    for part in parts {
                        let ptype = part.get("type").and_then(|v| v.as_str()).unwrap_or("");
                        match ptype {
                            "thinking" => {
                                if role == "assistant"
                                    && let Some(sig) =
                                        part.get("signature").and_then(|v| v.as_str())
                                    && !sig.is_empty()
                                    && let Some(t) = part.get("thinking").and_then(|v| v.as_str())
                                    && !t.trim().is_empty()
                                {
                                    reasoning_parts.push(t.to_string());
                                }
                            }
                            "redacted_thinking" => {
                                // Explicitly ignored for OpenAI compatibility
                            }
                            "text" | "image" => {
                                if let Some(converted) = claude_content_part_to_openai(part) {
                                    text_parts.push(converted);
                                }
                            }
                            "tool_use" => {
                                if role == "assistant" {
                                    let id = part.get("id").and_then(|v| v.as_str()).unwrap_or("");
                                    let name =
                                        part.get("name").and_then(|v| v.as_str()).unwrap_or("");
                                    let input = part.get("input").unwrap_or(&Value::Null);
                                    tool_calls.push(json!({
                                        "id": id,
                                        "type": "function",
                                        "function": {
                                            "name": name,
                                            "arguments": input.to_string(),
                                        }
                                    }));
                                }
                            }
                            "tool_result" => {
                                let tool_use_id = part
                                    .get("tool_use_id")
                                    .and_then(|v| v.as_str())
                                    .unwrap_or("");
                                let result_content =
                                    claude_tool_result_to_openai(part.get("content"));
                                tool_results.push(json!({
                                    "role": "tool",
                                    "tool_call_id": tool_use_id,
                                    "content": result_content,
                                }));
                            }
                            _ => {}
                        }
                    }

                    // Emit tool_results FIRST (they respond to previous assistant's tool_calls)
                    for tr in tool_results {
                        messages.push(tr);
                    }

                    if role == "assistant" {
                        let has_text = !text_parts.is_empty();
                        let has_reasoning = !reasoning_parts.is_empty();
                        let has_tool_calls = !tool_calls.is_empty();

                        if has_text || has_reasoning || has_tool_calls {
                            let mut assistant_msg = json!({"role": "assistant"});
                            if has_text {
                                assistant_msg["content"] = json!(text_parts);
                            } else {
                                assistant_msg["content"] = json!("");
                            }
                            if has_reasoning {
                                assistant_msg["reasoning_content"] =
                                    json!(reasoning_parts.join("\n\n"));
                            }
                            if has_tool_calls {
                                assistant_msg["tool_calls"] = json!(tool_calls);
                            }
                            messages.push(assistant_msg);
                        }
                    } else if !text_parts.is_empty() {
                        messages.push(json!({
                            "role": role,
                            "content": json!(text_parts),
                        }));
                    }
                }
                Value::String(text) => {
                    messages.push(json!({
                        "role": role,
                        "content": text,
                    }));
                }
                _ => {}
            }
        }
    }

    if !messages.is_empty() {
        out["messages"] = json!(messages);
    }

    // ── tools ────────────────────────────────────────────────────
    if let Some(tools) = body.get("tools").and_then(|v| v.as_array()) {
        let mut openai_tools: Vec<Value> = Vec::new();
        for tool in tools {
            let name = tool.get("name").and_then(|v| v.as_str()).unwrap_or("");
            let desc = tool
                .get("description")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            let input_schema = tool.get("input_schema");

            let mut fn_obj = json!({
                "name": name,
                "description": desc,
            });
            if let Some(schema) = input_schema {
                fn_obj["parameters"] = normalize_schema(schema.clone());
            } else {
                fn_obj["parameters"] = json!({"type": "object", "properties": {}});
            }

            openai_tools.push(json!({
                "type": "function",
                "function": fn_obj,
            }));
        }
        if !openai_tools.is_empty() {
            out["tools"] = json!(openai_tools);
        }
    }

    // ── tool_choice ──────────────────────────────────────────────
    if let Some(tc) = body.get("tool_choice") {
        let ttype = tc.get("type").and_then(|v| v.as_str()).unwrap_or("");
        match ttype {
            "auto" => out["tool_choice"] = json!("auto"),
            "any" => out["tool_choice"] = json!("required"),
            "tool" => {
                if let Some(name) = tc.get("name").and_then(|v| v.as_str()) {
                    out["tool_choice"] = json!({
                        "type": "function",
                        "function": { "name": name }
                    });
                }
            }
            _ => out["tool_choice"] = json!("auto"),
        }
    }

    // ── metadata passthrough ─────────────────────────────────────
    if let Some(user) = body.get("user").and_then(|v| v.as_str())
        && !user.is_empty()
    {
        out["user"] = json!(user);
    }

    out
}

// ── helper: Claude system → OpenAI system message ────────────

fn claude_system_to_openai(system: &Value) -> Option<Value> {
    match system {
        Value::String(s) => {
            if s.trim().is_empty() {
                return None;
            }
            Some(json!({"role": "system", "content": [{"type": "text", "text": s}]}))
        }
        Value::Array(arr) => {
            let mut parts: Vec<Value> = Vec::new();
            for item in arr {
                if let Some(converted) = claude_content_part_to_openai(item) {
                    parts.push(converted);
                }
            }
            if parts.is_empty() {
                None
            } else {
                Some(json!({"role": "system", "content": parts}))
            }
        }
        _ => None,
    }
}

// ── helper: single Claude content part → OpenAI format ────────

fn claude_content_part_to_openai(part: &Value) -> Option<Value> {
    let ptype = part.get("type").and_then(|v| v.as_str()).unwrap_or("");
    match ptype {
        "text" => {
            let text = part.get("text").and_then(|v| v.as_str()).unwrap_or("");
            if text.trim().is_empty() {
                return None;
            }
            Some(json!({"type": "text", "text": text}))
        }
        "image" => {
            let url = if let Some(source) = part.get("source") {
                let stype = source.get("type").and_then(|v| v.as_str()).unwrap_or("");
                match stype {
                    "base64" => {
                        let media_type = source
                            .get("media_type")
                            .and_then(|v| v.as_str())
                            .unwrap_or("application/octet-stream");
                        let data = source.get("data").and_then(|v| v.as_str()).unwrap_or("");
                        if data.is_empty() {
                            return None;
                        }
                        format!("data:{};base64,{}", media_type, data)
                    }
                    "url" => source
                        .get("url")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_string(),
                    _ => String::new(),
                }
            } else {
                part.get("url")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string()
            };

            if url.is_empty() {
                return None;
            }
            Some(json!({"type": "image_url", "image_url": {"url": url}}))
        }
        _ => None,
    }
}

// ── helper: tool_result content → OpenAI tool content ─────────

fn claude_tool_result_to_openai(content: Option<&Value>) -> Value {
    let content = match content {
        Some(c) => c,
        None => return json!(""),
    };

    match content {
        Value::String(s) => json!(s),
        Value::Array(parts) => {
            let mut text_parts: Vec<String> = Vec::new();
            let mut complex = false;
            for part in parts {
                let ptype = part.get("type").and_then(|v| v.as_str()).unwrap_or("");
                match ptype {
                    "text" => {
                        if let Some(t) = part.get("text").and_then(|v| v.as_str()) {
                            text_parts.push(t.to_string());
                        }
                    }
                    "image" => {
                        complex = true;
                    }
                    _ => {}
                }
            }
            if complex {
                // Stringify for simpler handling
                json!(content.to_string())
            } else {
                json!(text_parts.join("\n\n"))
            }
        }
        _ => json!(content.to_string()),
    }
}

// ── helper: extract system reminder from system-role messages ─

fn extract_system_reminder(content: &Value) -> Option<String> {
    if let Some(parts) = content.as_array() {
        for part in parts {
            if part.get("type").and_then(|v| v.as_str()) == Some("text") {
                let text = part.get("text").and_then(|v| v.as_str()).unwrap_or("");
                if !text.trim().is_empty() {
                    return Some(text.to_string());
                }
            }
        }
    }
    None
}

// ── helper: normalize JSON schema ─────────────────────────────

fn normalize_schema(schema: Value) -> Value {
    match schema {
        Value::Object(mut map) => {
            if map.get("type").and_then(|v| v.as_str()) == Some("object") {
                map.entry("properties").or_insert(json!({}));
            }
            for (_, v) in map.iter_mut() {
                *v = normalize_schema(v.take());
            }
            Value::Object(map)
        }
        Value::Array(arr) => Value::Array(arr.into_iter().map(normalize_schema).collect()),
        other => other,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_claude_to_openai_basic() {
        let body = json!({
            "model": "claude-sonnet-4-8",
            "max_tokens": 1024,
            "messages": [
                {"role": "user", "content": "Hello!"}
            ]
        });
        let result = transform("claude-sonnet-4-8", body, false);
        assert_eq!(result["model"], "claude-sonnet-4-8");
        assert_eq!(result["max_tokens"], 1024);
        assert_eq!(result["messages"][0]["role"], "user");
        assert_eq!(result["messages"][0]["content"], "Hello!");
    }

    #[test]
    fn test_claude_to_openai_with_system() {
        let body = json!({
            "model": "claude-sonnet-4-8",
            "system": "You are a helpful assistant.",
            "messages": [
                {"role": "user", "content": "Hi"}
            ]
        });
        let result = transform("claude-sonnet-4-8", body, false);
        assert_eq!(result["messages"][0]["role"], "system");
        assert_eq!(
            result["messages"][0]["content"][0]["text"],
            "You are a helpful assistant."
        );
    }

    #[test]
    fn test_claude_to_openai_with_thinking() {
        let body = json!({
            "model": "claude-sonnet-4-8",
            "thinking": {"type": "enabled", "budget_tokens": 1024},
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {"type": "thinking", "thinking": "Let me think...", "signature": "sig123"},
                        {"type": "text", "text": "The answer is 42."}
                    ]
                }
            ]
        });
        let result = transform("claude-sonnet-4-8", body, false);
        assert_eq!(result["reasoning_effort"], "low");
        assert_eq!(
            result["messages"][0]["reasoning_content"],
            "Let me think..."
        );
        assert_eq!(
            result["messages"][0]["content"][0]["text"],
            "The answer is 42."
        );
    }

    #[test]
    fn test_claude_to_openai_with_tools() {
        let body = json!({
            "model": "claude-sonnet-4-8",
            "tools": [{
                "name": "get_weather",
                "description": "Get weather for a city",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string"}
                    },
                    "required": ["city"]
                }
            }],
            "messages": [{"role": "user", "content": "Weather in Paris?"}]
        });
        let result = transform("claude-sonnet-4-8", body, false);
        assert_eq!(result["tools"][0]["function"]["name"], "get_weather");
        assert!(result["tools"][0]["function"]["parameters"]["properties"]["city"].is_object());
    }

    #[test]
    fn test_claude_to_openai_stop_sequences() {
        let body = json!({
            "model": "claude-sonnet-4-8",
            "stop_sequences": ["END", "STOP"],
            "messages": [{"role": "user", "content": "Hi"}]
        });
        let result = transform("claude-sonnet-4-8", body, false);
        assert!(result["stop"].is_array());
        assert_eq!(result["stop"][0], "END");
    }
}
