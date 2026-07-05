//! OpenAI Chat response -> Gemini response translation.

use serde_json::{Value, json};
use std::collections::HashMap;

#[derive(Debug, Default)]
struct StreamState {
    tool_calls: HashMap<i64, ToolCallAccumulator>,
    content: String,
    is_first_chunk: bool,
}

#[derive(Debug, Default)]
struct ToolCallAccumulator {
    id: String,
    name: String,
    arguments: String,
}

/// Transform an OpenAI non-streaming Chat Completions response into Gemini format.
pub fn transform_non_stream(
    _model: &str,
    _orig: &Value,
    _trans: &Value,
    response: Value,
    _param: Option<&mut Box<dyn std::any::Any>>,
) -> Value {
    let root = normalize_chunk_value(response);
    let mut out = json!({
        "candidates": [{
            "content": {"parts": [], "role": "model"},
            "index": 0,
        }]
    });
    apply_model(&mut out, &root);

    if let Some(choices) = root.get("choices").and_then(Value::as_array) {
        for choice in choices {
            let choice_idx = choice.get("index").and_then(Value::as_i64).unwrap_or(0);
            let message = choice.get("message").unwrap_or(&Value::Null);
            if message.get("role").and_then(Value::as_str) == Some("assistant") {
                out["candidates"][0]["content"]["role"] = json!("model");
            }

            let mut parts = Vec::new();
            if let Some(reasoning) = message.get("reasoning_content") {
                for text in extract_reasoning_texts(reasoning) {
                    if !text.is_empty() {
                        parts.push(json!({"thought": true, "text": text}));
                    }
                }
            }
            if let Some(content) = message.get("content").and_then(Value::as_str)
                && !content.is_empty()
            {
                parts.push(json!({"text": content}));
            }
            if let Some(tool_calls) = message.get("tool_calls").and_then(Value::as_array) {
                for tool_call in tool_calls {
                    if tool_call.get("type").and_then(Value::as_str) != Some("function") {
                        continue;
                    }
                    let name = tool_call
                        .pointer("/function/name")
                        .and_then(Value::as_str)
                        .unwrap_or("");
                    let args = tool_call
                        .pointer("/function/arguments")
                        .and_then(Value::as_str)
                        .map(parse_args_to_object)
                        .unwrap_or_else(|| json!({}));
                    parts.push(json!({"functionCall": {"name": name, "args": args}}));
                }
            }

            out["candidates"][0]["content"]["parts"] = Value::Array(parts);
            if choice.get("finish_reason").is_some() {
                out["candidates"][0]["finishReason"] =
                    json!(map_finish_reason(choice.get("finish_reason")));
            }
            out["candidates"][0]["index"] = json!(choice_idx);
        }
    }

    if let Some(usage) = root.get("usage") {
        out["usageMetadata"] = usage_metadata(usage);
    }

    out
}

/// Transform an OpenAI streaming Chat Completions chunk into Gemini format.
pub fn transform_stream(
    _model: &str,
    _orig: &Value,
    _trans: &Value,
    chunk: Value,
    param: Option<&mut Box<dyn std::any::Any>>,
) -> Vec<Value> {
    if chunk.as_str() == Some("[DONE]") {
        return vec![];
    }
    let root = normalize_chunk_value(chunk);
    if root.as_str() == Some("[DONE]") {
        return vec![];
    }

    let mut local_state = StreamState::default();
    let state = if let Some(param) = param {
        if !param.is::<StreamState>() {
            *param = Box::<StreamState>::default();
        }
        param.downcast_mut::<StreamState>().unwrap()
    } else {
        &mut local_state
    };

    let Some(choices) = root.get("choices").and_then(Value::as_array) else {
        return vec![];
    };

    if choices.is_empty() {
        if let Some(usage) = root.get("usage") {
            let mut out = json!({"candidates": [], "usageMetadata": usage_metadata(usage)});
            apply_model(&mut out, &root);
            return vec![out];
        }
        return vec![];
    }

    let mut results = Vec::new();
    for choice in choices {
        let delta = choice.get("delta").unwrap_or(&Value::Null);
        let mut template = stream_template(&root);

        if delta.get("role").is_some() && state.is_first_chunk {
            if delta.get("role").and_then(Value::as_str) == Some("assistant") {
                template["candidates"][0]["content"]["role"] = json!("model");
            }
            state.is_first_chunk = false;
            results.push(template);
            continue;
        }

        let mut chunk_outputs = Vec::new();
        if let Some(reasoning) = delta.get("reasoning_content") {
            for text in extract_reasoning_texts(reasoning) {
                if text.is_empty() {
                    continue;
                }
                let mut reasoning_template = stream_template(&root);
                reasoning_template["candidates"][0]["content"]["parts"] =
                    json!([{"thought": true, "text": text}]);
                chunk_outputs.push(reasoning_template);
            }
        }
        if let Some(content) = delta.get("content").and_then(Value::as_str)
            && !content.is_empty()
        {
            state.content.push_str(content);
            let mut content_template = stream_template(&root);
            content_template["candidates"][0]["content"]["parts"] = json!([{"text": content}]);
            chunk_outputs.push(content_template);
        }
        if !chunk_outputs.is_empty() {
            results.extend(chunk_outputs);
            continue;
        }

        if let Some(tool_calls) = delta.get("tool_calls").and_then(Value::as_array) {
            for tool_call in tool_calls {
                let tool_type = tool_call.get("type").and_then(Value::as_str).unwrap_or("");
                if !tool_type.is_empty() && tool_type != "function" {
                    continue;
                }
                let Some(function) = tool_call.get("function") else {
                    continue;
                };
                let index = tool_call.get("index").and_then(Value::as_i64).unwrap_or(0);
                let acc = state.tool_calls.entry(index).or_default();
                if let Some(id) = tool_call.get("id").and_then(Value::as_str)
                    && !id.is_empty()
                {
                    acc.id = id.to_string();
                }
                if let Some(name) = function.get("name").and_then(Value::as_str)
                    && !name.is_empty()
                {
                    acc.name = name.to_string();
                }
                if let Some(arguments) = function.get("arguments").and_then(Value::as_str)
                    && !arguments.is_empty()
                {
                    acc.arguments.push_str(arguments);
                }
            }
            continue;
        }

        if choice.get("finish_reason").is_some() {
            template["candidates"][0]["finishReason"] =
                json!(map_finish_reason(choice.get("finish_reason")));
            if !state.tool_calls.is_empty() {
                let mut parts = Vec::new();
                let mut indices: Vec<_> = state.tool_calls.keys().copied().collect();
                indices.sort_unstable();
                for index in indices {
                    if let Some(acc) = state.tool_calls.get(&index) {
                        parts.push(json!({
                            "functionCall": {
                                "name": acc.name,
                                "args": parse_args_to_object(&acc.arguments),
                            }
                        }));
                    }
                }
                template["candidates"][0]["content"]["parts"] = Value::Array(parts);
                state.tool_calls.clear();
            }
            results.push(template);
            continue;
        }

        if let Some(usage) = root.get("usage") {
            template["usageMetadata"] = usage_metadata(usage);
            results.push(template);
        }
    }

    results
}

fn normalize_chunk_value(value: Value) -> Value {
    let Some(raw) = value.as_str() else {
        return value;
    };
    if raw.trim() == "[DONE]" {
        return Value::String("[DONE]".to_string());
    }
    let payload = raw
        .trim()
        .strip_prefix("data:")
        .map(str::trim)
        .unwrap_or_else(|| raw.trim());
    serde_json::from_str(payload).unwrap_or(value)
}

fn stream_template(root: &Value) -> Value {
    let mut out = json!({
        "candidates": [{
            "content": {"parts": [], "role": "model"},
            "index": 0,
        }]
    });
    apply_model(&mut out, root);
    out
}

fn apply_model(out: &mut Value, root: &Value) {
    if let Some(model) = root.get("model") {
        out["model"] = json!(value_to_string(model));
    }
}

fn usage_metadata(usage: &Value) -> Value {
    let mut out = json!({
        "promptTokenCount": usage.get("prompt_tokens").and_then(Value::as_i64).unwrap_or(0),
        "candidatesTokenCount": usage.get("completion_tokens").and_then(Value::as_i64).unwrap_or(0),
        "totalTokenCount": usage.get("total_tokens").and_then(Value::as_i64).unwrap_or(0),
    });
    let reasoning_tokens = reasoning_tokens_from_usage(usage);
    if reasoning_tokens > 0 {
        out["thoughtsTokenCount"] = json!(reasoning_tokens);
    }
    out
}

fn reasoning_tokens_from_usage(usage: &Value) -> i64 {
    usage
        .pointer("/completion_tokens_details/reasoning_tokens")
        .and_then(Value::as_i64)
        .or_else(|| {
            usage
                .pointer("/output_tokens_details/reasoning_tokens")
                .and_then(Value::as_i64)
        })
        .unwrap_or(0)
}

fn map_finish_reason(reason: Option<&Value>) -> &'static str {
    match reason.and_then(Value::as_str).unwrap_or("") {
        "stop" => "STOP",
        "length" => "MAX_TOKENS",
        "tool_calls" => "STOP",
        "content_filter" => "SAFETY",
        _ => "STOP",
    }
}

fn extract_reasoning_texts(node: &Value) -> Vec<String> {
    match node {
        Value::String(text) => vec![text.clone()],
        Value::Array(items) => items.iter().flat_map(extract_reasoning_texts).collect(),
        Value::Object(obj) => obj
            .get("text")
            .and_then(Value::as_str)
            .map(|text| vec![text.to_string()])
            .unwrap_or_default(),
        _ => Vec::new(),
    }
}

fn parse_args_to_object(args: &str) -> Value {
    let trimmed = args.trim();
    if trimmed.is_empty() || trimmed == "{}" {
        return json!({});
    }

    if let Ok(value) = serde_json::from_str::<Value>(trimmed)
        && value.is_object()
    {
        return value;
    }

    tolerant_parse_object(trimmed).unwrap_or_else(|| json!({}))
}

fn tolerant_parse_object(input: &str) -> Option<Value> {
    let start = input.find('{')?;
    let end = input.rfind('}')?;
    if start >= end {
        return None;
    }
    let content = &input[start + 1..end];
    let mut obj = serde_json::Map::new();
    for pair in split_top_level(content, ',') {
        let Some(colon) = find_top_level_char(pair, ':') else {
            continue;
        };
        let key_raw = pair[..colon].trim();
        let value_raw = pair[colon + 1..].trim();
        let key = serde_json::from_str::<String>(key_raw).ok()?;
        obj.insert(key, parse_tolerant_value(value_raw));
    }
    if obj.is_empty() {
        None
    } else {
        Some(Value::Object(obj))
    }
}

fn parse_tolerant_value(raw: &str) -> Value {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return Value::String(String::new());
    }
    if let Ok(value) = serde_json::from_str::<Value>(trimmed) {
        return value;
    }
    if trimmed == "true" {
        return Value::Bool(true);
    }
    if trimmed == "false" {
        return Value::Bool(false);
    }
    if trimmed == "null" {
        return Value::Null;
    }
    if let Ok(value) = trimmed.parse::<i64>() {
        return json!(value);
    }
    if let Ok(value) = trimmed.parse::<u64>() {
        return json!(value);
    }
    if let Ok(value) = trimmed.parse::<f64>() {
        return json!(value);
    }
    Value::String(trimmed.to_string())
}

fn split_top_level(input: &str, delimiter: char) -> Vec<&str> {
    let mut parts = Vec::new();
    let mut start = 0;
    let mut depth = 0_i32;
    let mut in_string = false;
    let mut escaped = false;
    for (idx, ch) in input.char_indices() {
        if in_string {
            if ch == '\\' && !escaped {
                escaped = true;
                continue;
            }
            if ch == '"' && !escaped {
                in_string = false;
            }
            escaped = false;
            continue;
        }
        match ch {
            '"' => in_string = true,
            '{' | '[' => depth += 1,
            '}' | ']' => depth -= 1,
            c if c == delimiter && depth == 0 => {
                parts.push(input[start..idx].trim());
                start = idx + ch.len_utf8();
            }
            _ => {}
        }
    }
    parts.push(input[start..].trim());
    parts
}

fn find_top_level_char(input: &str, target: char) -> Option<usize> {
    let mut depth = 0_i32;
    let mut in_string = false;
    let mut escaped = false;
    for (idx, ch) in input.char_indices() {
        if in_string {
            if ch == '\\' && !escaped {
                escaped = true;
                continue;
            }
            if ch == '"' && !escaped {
                in_string = false;
            }
            escaped = false;
            continue;
        }
        match ch {
            '"' => in_string = true,
            '{' | '[' => depth += 1,
            '}' | ']' => depth -= 1,
            c if c == target && depth == 0 => return Some(idx),
            _ => {}
        }
    }
    None
}

fn value_to_string(value: &Value) -> String {
    match value {
        Value::String(value) => value.clone(),
        Value::Number(value) => value.to_string(),
        Value::Bool(value) => value.to_string(),
        Value::Null => String::new(),
        other => serde_json::to_string(other).unwrap_or_default(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn non_stream_basic() {
        let resp = json!({
            "model": "gpt-test",
            "choices": [{"index": 0, "message": {"role":"assistant","content": "Hello"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8}
        });
        let result = transform_non_stream("gemini-2.0-flash", &json!({}), &json!({}), resp, None);
        assert_eq!(result["model"], "gpt-test");
        assert_eq!(
            result["candidates"][0]["content"]["parts"][0]["text"],
            "Hello"
        );
        assert_eq!(result["candidates"][0]["finishReason"], "STOP");
        assert_eq!(result["usageMetadata"]["totalTokenCount"], 8);
    }

    #[test]
    fn stream_accumulates_tool_call_until_finish() {
        let mut param: Box<dyn std::any::Any> = Box::new(());
        let first = transform_stream(
            "gemini",
            &json!({}),
            &json!({}),
            json!({"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"search","arguments":"{\"q\":"}}]}}]}),
            Some(&mut param),
        );
        assert!(first.is_empty());
        let second = transform_stream(
            "gemini",
            &json!({}),
            &json!({}),
            json!({"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\"rust\"}"}}]}}]}),
            Some(&mut param),
        );
        assert!(second.is_empty());
        let done = transform_stream(
            "gemini",
            &json!({}),
            &json!({}),
            json!({"choices":[{"finish_reason":"tool_calls"}]}),
            Some(&mut param),
        );
        assert_eq!(
            done[0]["candidates"][0]["content"]["parts"][0]["functionCall"]["args"]["q"],
            "rust"
        );
    }
}
