//! Claude Messages response → Google Gemini response translation.
//!
//! Translates Claude SSE streaming events and non-streaming responses to Gemini format.

use serde_json::{Value, json};
use std::collections::HashMap;

/// Accumulated state for streaming Claude → Gemini response conversion.
#[derive(Debug, Clone, Default)]
pub struct GeminiStreamState {
    pub model: String,
    pub response_id: String,
    pub created_at: i64,
    /// function/tool name per Claude content block index
    pub tool_use_names: HashMap<usize, String>,
    /// accumulates partial_json across deltas per index
    pub tool_use_args: HashMap<usize, String>,
    /// tool use ID per block index
    pub tool_use_ids: HashMap<usize, String>,
    pub finish_reason: String,
}

/// Transform a Claude SSE streaming event to Gemini streaming format.
pub fn transform_stream(
    model: &str,
    _original_request: &Value,
    _translated_request: &Value,
    chunk: Value,
    param: Option<&mut Box<dyn std::any::Any>>,
) -> Vec<Value> {
    let state = if let Some(p) = param {
        if !p.is::<GeminiStreamState>() {
            *p = Box::<GeminiStreamState>::default();
        }
        p.downcast_mut::<GeminiStreamState>().unwrap()
    } else {
        return vec![chunk];
    };

    let event_type = chunk.get("type").and_then(|v| v.as_str()).unwrap_or("");
    let mut results: Vec<Value> = Vec::new();
    if state.model.is_empty() {
        state.model = model.to_string();
    }

    match event_type {
        "message_start" => {
            if let Some(msg) = chunk.get("message") {
                state.response_id = msg
                    .get("id")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                state.model = msg
                    .get("model")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
            }
            if state.created_at == 0 {
                state.created_at = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs() as i64;
            }
        }

        "content_block_start" => {
            if let Some(cb) = chunk.get("content_block")
                && cb.get("type").and_then(|v| v.as_str()) == Some("tool_use")
            {
                let idx = chunk.get("index").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
                if let Some(name) = cb.get("name").and_then(|v| v.as_str()) {
                    state.tool_use_names.insert(idx, name.to_string());
                }
                if let Some(id) = cb.get("id").and_then(|v| v.as_str())
                    && !id.is_empty()
                {
                    state.tool_use_ids.insert(idx, id.to_string());
                }
            }
        }

        "content_block_delta" => {
            if let Some(delta) = chunk.get("delta") {
                let dtype = delta.get("type").and_then(|v| v.as_str()).unwrap_or("");
                match dtype {
                    "text_delta" => {
                        if let Some(text) = delta.get("text").and_then(|v| v.as_str())
                            && !text.is_empty()
                        {
                            results.push(make_gemini_part(json!({"text": text}), state));
                        }
                    }
                    "thinking_delta" => {
                        if let Some(text) = delta.get("thinking").and_then(|v| v.as_str())
                            && !text.is_empty()
                        {
                            results.push(make_gemini_part(
                                json!({"thought": true, "text": text}),
                                state,
                            ));
                        }
                    }
                    "input_json_delta" => {
                        let idx = chunk.get("index").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
                        if let Some(pj) = delta.get("partial_json").and_then(|v| v.as_str()) {
                            state.tool_use_args.entry(idx).or_default().push_str(pj);
                        }
                    }
                    _ => {}
                }
            }
        }

        "content_block_stop" => {
            let idx = chunk.get("index").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
            let name = state.tool_use_names.remove(&idx);
            let args = state.tool_use_args.remove(&idx).unwrap_or_default();
            let tool_id = state.tool_use_ids.remove(&idx).unwrap_or_default();

            if name.is_some() || !args.is_empty() {
                let mut fc = json!({"functionCall": {"name": "", "args": {}}});
                if let Some(n) = name {
                    fc["functionCall"]["name"] = json!(n);
                }
                if !args.is_empty() {
                    if let Ok(parsed) = serde_json::from_str::<Value>(&args) {
                        fc["functionCall"]["args"] = parsed;
                    } else {
                        fc["functionCall"]["args"] = json!(args);
                    }
                }
                if !tool_id.is_empty() {
                    fc["functionCall"]["id"] = json!(tool_id);
                }
                let mut result = make_gemini_part(fc, state);
                result["candidates"][0]["finishReason"] = json!("STOP");
                results.push(result);
            }
        }

        "message_delta" => {
            if let Some(delta) = chunk.get("delta")
                && let Some(reason) = delta.get("stop_reason").and_then(|v| v.as_str())
            {
                state.finish_reason = map_claude_stop_reason_to_gemini(reason).to_string();
            }
            results = append_usage(results, &chunk, state);
        }

        "message_stop" => {}

        "error" => {
            let msg = chunk
                .pointer("/error/message")
                .and_then(|v| v.as_str())
                .unwrap_or("Unknown error");
            results.push(json!({
                "error": {"code": 400, "message": msg, "status": "INVALID_ARGUMENT"}
            }));
        }

        "ping" => {}

        _ => {}
    }

    results
}

/// Transform a non-streaming Claude response to Gemini format.
pub fn transform_non_stream(
    model: &str,
    _original_request: &Value,
    _translated_request: &Value,
    response: Value,
    _param: Option<&mut Box<dyn std::any::Any>>,
) -> Value {
    if let Some(raw_sse) = response.as_str() {
        return transform_sse_non_stream(model, raw_sse);
    }

    let mut out = json!({
        "candidates": [{
            "content": {"role": "model", "parts": []},
            "finishReason": "STOP",
        }],
        "usageMetadata": {
            "promptTokenCount": 0,
            "candidatesTokenCount": 0,
            "totalTokenCount": 0,
        },
        "modelVersion": model,
    });

    let content = response.get("content").and_then(|v| v.as_array());

    let mut parts: Vec<Value> = Vec::new();
    let mut saw_tool = false;

    if let Some(blocks) = content {
        for block in blocks {
            let btype = block.get("type").and_then(|v| v.as_str()).unwrap_or("");
            match btype {
                "text" => {
                    if let Some(t) = block.get("text").and_then(|v| v.as_str())
                        && !t.is_empty()
                    {
                        parts.push(json!({"text": t}));
                    }
                }
                "thinking" => {
                    if let Some(t) = block.get("thinking").and_then(|v| v.as_str())
                        && !t.is_empty()
                    {
                        parts.push(json!({"thought": true, "text": t}));
                    }
                }
                "tool_use" => {
                    saw_tool = true;
                    let name = block.get("name").and_then(|v| v.as_str()).unwrap_or("");
                    let empty_obj = json!({});
                    let args = block.get("input").unwrap_or(&empty_obj);
                    parts.push(json!({
                        "functionCall": {"name": name, "args": args}
                    }));
                }
                _ => {}
            }
        }
    }

    if !parts.is_empty() {
        out["candidates"][0]["content"]["parts"] = json!(consolidate_parts_raw(&parts));
    }
    if saw_tool {
        out["candidates"][0]["finishReason"] = json!("STOP");
    }

    // Stop reason
    if let Some(reason) = response.get("stop_reason").and_then(|v| v.as_str()) {
        out["candidates"][0]["finishReason"] = json!(map_claude_stop_reason_to_gemini(reason));
    }

    // Usage
    if let Some(usage) = response.get("usage") {
        let input_tokens = usage
            .get("input_tokens")
            .and_then(|v| v.as_u64())
            .unwrap_or(0);
        let output_tokens = usage
            .get("output_tokens")
            .and_then(|v| v.as_u64())
            .unwrap_or(0);
        out["usageMetadata"]["promptTokenCount"] = json!(input_tokens);
        out["usageMetadata"]["candidatesTokenCount"] = json!(output_tokens);
        out["usageMetadata"]["totalTokenCount"] = json!(input_tokens + output_tokens);
    }

    out
}

fn transform_sse_non_stream(model: &str, raw_sse: &str) -> Value {
    let mut parts = Vec::new();
    let mut response_id = String::new();
    let mut model_version = model.to_string();
    let mut final_usage = json!({"trafficType": "PROVISIONED_THROUGHPUT"});
    let mut tool_names: HashMap<usize, String> = HashMap::new();
    let mut tool_args: HashMap<usize, String> = HashMap::new();
    let mut tool_ids: HashMap<usize, String> = HashMap::new();

    for line in raw_sse.lines() {
        let trimmed = line.trim();
        let Some(payload) = trimmed.strip_prefix("data:") else {
            continue;
        };
        let Ok(event) = serde_json::from_str::<Value>(payload.trim()) else {
            continue;
        };
        match event.get("type").and_then(|v| v.as_str()).unwrap_or("") {
            "message_start" => {
                if let Some(message) = event.get("message") {
                    if let Some(id) = message.get("id").and_then(|v| v.as_str()) {
                        response_id = id.to_string();
                    }
                    if let Some(m) = message.get("model").and_then(|v| v.as_str()) {
                        model_version = m.to_string();
                    }
                }
            }
            "content_block_start" => {
                let idx = event.get("index").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
                if let Some(block) = event.get("content_block")
                    && block.get("type").and_then(|v| v.as_str()) == Some("tool_use")
                {
                    if let Some(name) = block.get("name").and_then(|v| v.as_str()) {
                        tool_names.insert(idx, name.to_string());
                    }
                    if let Some(id) = block.get("id").and_then(|v| v.as_str()) {
                        tool_ids.insert(idx, id.to_string());
                    }
                }
            }
            "content_block_delta" => {
                let idx = event.get("index").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
                if let Some(delta) = event.get("delta") {
                    match delta.get("type").and_then(|v| v.as_str()).unwrap_or("") {
                        "text_delta" => {
                            if let Some(text) = delta.get("text").and_then(|v| v.as_str())
                                && !text.is_empty()
                            {
                                parts.push(json!({"text": text}));
                            }
                        }
                        "thinking_delta" => {
                            if let Some(text) = delta.get("thinking").and_then(|v| v.as_str())
                                && !text.is_empty()
                            {
                                parts.push(json!({"thought": true, "text": text}));
                            }
                        }
                        "input_json_delta" => {
                            if let Some(partial) =
                                delta.get("partial_json").and_then(|v| v.as_str())
                            {
                                tool_args.entry(idx).or_default().push_str(partial);
                            }
                        }
                        _ => {}
                    }
                }
            }
            "content_block_stop" => {
                let idx = event.get("index").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
                let name = tool_names.remove(&idx).unwrap_or_default();
                let args = tool_args.remove(&idx).unwrap_or_default();
                let id = tool_ids.remove(&idx).unwrap_or_default();
                if !name.is_empty() || !args.trim().is_empty() {
                    let mut function_call = json!({"functionCall": {"name": name, "args": {}}});
                    if !args.trim().is_empty() {
                        function_call["functionCall"]["args"] =
                            serde_json::from_str(args.trim()).unwrap_or_else(|_| json!(args));
                    }
                    if !id.is_empty() {
                        function_call["functionCall"]["id"] = json!(id);
                    }
                    parts.push(function_call);
                }
            }
            "message_delta" => {
                if let Some(usage) = event.get("usage") {
                    let input = usage
                        .get("input_tokens")
                        .and_then(|v| v.as_i64())
                        .unwrap_or(0);
                    let output = usage
                        .get("output_tokens")
                        .and_then(|v| v.as_i64())
                        .unwrap_or(0);
                    let cache_creation = usage
                        .get("cache_creation_input_tokens")
                        .and_then(|v| v.as_i64())
                        .unwrap_or(0);
                    let cache_read = usage
                        .get("cache_read_input_tokens")
                        .and_then(|v| v.as_i64())
                        .unwrap_or(0);
                    final_usage = json!({
                        "promptTokenCount": input,
                        "candidatesTokenCount": output,
                        "totalTokenCount": input + output,
                        "trafficType": "PROVISIONED_THROUGHPUT",
                    });
                    if cache_creation > 0 || cache_read > 0 {
                        final_usage["cachedContentTokenCount"] = json!(cache_creation + cache_read);
                    }
                    if let Some(thinking) = usage.get("thinking_tokens").and_then(|v| v.as_i64()) {
                        final_usage["thoughtsTokenCount"] = json!(thinking);
                    }
                }
            }
            _ => {}
        }
    }

    json!({
        "candidates": [{
            "content": {"role": "model", "parts": consolidate_parts_raw(&parts)},
            "finishReason": "STOP",
        }],
        "usageMetadata": final_usage,
        "modelVersion": model_version,
        "createTime": "",
        "responseId": response_id,
    })
}

// ── helpers ──────────────────────────────────────────────────

fn make_gemini_part(part: Value, state: &GeminiStreamState) -> Value {
    json!({
        "candidates": [{
            "content": {"role": "model", "parts": [part]},
        }],
        "usageMetadata": {"trafficType": "PROVISIONED_THROUGHPUT"},
        "modelVersion": state.model,
        "createTime": "",
        "responseId": state.response_id,
    })
}

fn append_usage(mut results: Vec<Value>, chunk: &Value, state: &GeminiStreamState) -> Vec<Value> {
    if let Some(usage) = chunk.get("usage") {
        let input_tokens = usage
            .get("input_tokens")
            .and_then(|v| v.as_u64())
            .unwrap_or(0);
        let output_tokens = usage
            .get("output_tokens")
            .and_then(|v| v.as_u64())
            .unwrap_or(0);
        let cache_creation = usage
            .get("cache_creation_input_tokens")
            .and_then(|v| v.as_u64())
            .unwrap_or(0);
        let cache_read = usage
            .get("cache_read_input_tokens")
            .and_then(|v| v.as_u64())
            .unwrap_or(0);
        let reason = if state.finish_reason.is_empty() {
            "STOP"
        } else {
            &state.finish_reason
        };

        let mut usage_md = json!({
            "promptTokenCount": input_tokens,
            "candidatesTokenCount": output_tokens,
            "totalTokenCount": input_tokens + output_tokens,
            "trafficType": "PROVISIONED_THROUGHPUT",
        });
        if cache_creation > 0 || cache_read > 0 {
            usage_md["cachedContentTokenCount"] = json!(cache_creation + cache_read);
        }
        if let Some(thinking) = usage.get("thinking_tokens").and_then(|v| v.as_u64()) {
            usage_md["thoughtsTokenCount"] = json!(thinking);
        }

        results.push(json!({
            "candidates": [{
                "content": {"role": "model", "parts": []},
                "finishReason": reason,
            }],
            "usageMetadata": usage_md,
            "modelVersion": state.model,
            "createTime": "",
            "responseId": state.response_id,
        }));
    }
    results
}

fn consolidate_parts_raw(parts: &[Value]) -> Vec<Value> {
    let mut out: Vec<Value> = Vec::new();
    let mut text_buf = String::new();
    let mut thought_buf = String::new();

    let flush_text = |buf: &mut String, out: &mut Vec<Value>| {
        if !buf.is_empty() {
            out.push(json!({"text": buf.clone()}));
            buf.clear();
        }
    };
    let flush_thought = |buf: &mut String, out: &mut Vec<Value>| {
        if !buf.is_empty() {
            out.push(json!({"thought": true, "text": buf.clone()}));
            buf.clear();
        }
    };

    for part in parts {
        if part.get("thought") == Some(&json!(true)) {
            flush_text(&mut text_buf, &mut out);
            if let Some(t) = part.get("text").and_then(|v| v.as_str()) {
                thought_buf.push_str(t);
            }
        } else if part.get("text").is_some() {
            flush_thought(&mut thought_buf, &mut out);
            if let Some(t) = part.get("text").and_then(|v| v.as_str()) {
                text_buf.push_str(t);
            }
        } else {
            flush_text(&mut text_buf, &mut out);
            flush_thought(&mut thought_buf, &mut out);
            out.push(part.clone());
        }
    }
    flush_thought(&mut thought_buf, &mut out);
    flush_text(&mut text_buf, &mut out);
    out
}

fn map_claude_stop_reason_to_gemini(reason: &str) -> &'static str {
    match reason {
        "end_turn" => "STOP",
        "max_tokens" => "MAX_TOKENS",
        "tool_use" => "STOP",
        "stop_sequence" => "STOP",
        _ => "STOP",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gemini_to_claude_non_stream_basic() {
        let resp = json!({
            "id": "msg_123",
            "type": "message",
            "role": "assistant",
            "model": "gemini-2.0-flash",
            "content": [
                {"type": "text", "text": "Hello from Gemini!"}
            ],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 5, "output_tokens": 3}
        });

        let result = transform_non_stream("gemini-2.0-flash", &json!({}), &json!({}), resp, None);
        assert_eq!(
            result["candidates"][0]["content"]["parts"][0]["text"],
            "Hello from Gemini!"
        );
        assert_eq!(result["candidates"][0]["finishReason"], "STOP");
        assert_eq!(result["usageMetadata"]["promptTokenCount"], 5);
    }

    #[test]
    fn test_gemini_to_claude_non_stream_with_tools() {
        let resp = json!({
            "id": "msg_456",
            "type": "message",
            "content": [
                {"type": "tool_use", "id": "toolu_abc", "name": "get_weather", "input": {"city": "Tokyo"}}
            ],
            "stop_reason": "tool_use",
            "usage": {"input_tokens": 10, "output_tokens": 5}
        });

        let result = transform_non_stream("gemini-2.0-flash", &json!({}), &json!({}), resp, None);
        assert_eq!(
            result["candidates"][0]["content"]["parts"][0]["functionCall"]["name"],
            "get_weather"
        );
        assert_eq!(
            result["candidates"][0]["content"]["parts"][0]["functionCall"]["args"]["city"],
            "Tokyo"
        );
    }

    #[test]
    fn test_gemini_to_claude_non_stream_with_thinking() {
        let resp = json!({
            "id": "msg_789",
            "type": "message",
            "content": [
                {"type": "thinking", "thinking": "Let me reason..."},
                {"type": "text", "text": "Final answer."}
            ],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 8, "output_tokens": 12}
        });

        let result = transform_non_stream("gemini-2.0-flash", &json!({}), &json!({}), resp, None);
        assert!(result["candidates"][0]["content"]["parts"][0]["thought"] == json!(true));
        assert_eq!(
            result["candidates"][0]["content"]["parts"][1]["text"],
            "Final answer."
        );
    }
}
