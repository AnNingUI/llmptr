//! Claude Messages response -> OpenAI ChatCompletions response translation.

use std::collections::BTreeMap;

use serde_json::{Value, json};

#[derive(Debug, Clone, Default)]
pub struct ClaudeStreamState {
    created_at: i64,
    response_id: String,
    finish_reason: String,
    usage: UsageTokens,
    tool_calls: BTreeMap<usize, ToolCallAccumulator>,
}

#[derive(Debug, Clone, Default)]
struct UsageTokens {
    input_tokens: i64,
    output_tokens: i64,
    cache_creation_input_tokens: i64,
    cache_read_input_tokens: i64,
    has_usage: bool,
}

#[derive(Debug, Clone, Default)]
struct ToolCallAccumulator {
    id: String,
    name: String,
    arguments: String,
}

impl UsageTokens {
    fn merge(&mut self, usage: &Value) {
        if !usage.is_object() {
            return;
        }
        self.has_usage = true;
        if let Some(value) = usage.get("input_tokens").and_then(|v| v.as_i64()) {
            self.input_tokens = value;
        }
        if let Some(value) = usage.get("output_tokens").and_then(|v| v.as_i64()) {
            self.output_tokens = value;
        }
        if let Some(value) = usage
            .get("cache_creation_input_tokens")
            .and_then(|v| v.as_i64())
        {
            self.cache_creation_input_tokens = value;
        }
        if let Some(value) = usage
            .get("cache_read_input_tokens")
            .and_then(|v| v.as_i64())
        {
            self.cache_read_input_tokens = value;
        }
    }

    fn openai_usage(&self) -> (i64, i64, i64, i64) {
        let cached = self.cache_read_input_tokens;
        let prompt = self.input_tokens + self.cache_creation_input_tokens + cached;
        let completion = self.output_tokens;
        let total = prompt + completion;
        (prompt, completion, total, cached)
    }
}

pub fn transform_stream(
    model: &str,
    _original_request: &Value,
    _translated_request: &Value,
    chunk: Value,
    param: Option<&mut Box<dyn std::any::Any>>,
) -> Vec<Value> {
    let state = if let Some(param) = param {
        if !param.is::<ClaudeStreamState>() {
            *param = Box::<ClaudeStreamState>::default();
        }
        param.downcast_mut::<ClaudeStreamState>().unwrap()
    } else {
        return Vec::new();
    };

    let event_type = chunk.get("type").and_then(|v| v.as_str()).unwrap_or("");
    match event_type {
        "message_start" => {
            if let Some(message) = chunk.get("message") {
                state.response_id = message
                    .get("id")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                state.created_at = now_unix();
                if let Some(usage) = message.get("usage") {
                    state.usage.merge(usage);
                }
            }
            vec![openai_chunk(
                &state.response_id,
                state.created_at,
                model,
                json!({"role": "assistant"}),
                Value::Null,
                None,
            )]
        }
        "content_block_start" => {
            if chunk
                .pointer("/content_block/type")
                .and_then(|v| v.as_str())
                == Some("tool_use")
            {
                let index = chunk.get("index").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
                state.tool_calls.insert(
                    index,
                    ToolCallAccumulator {
                        id: chunk
                            .pointer("/content_block/id")
                            .and_then(|v| v.as_str())
                            .unwrap_or("")
                            .to_string(),
                        name: chunk
                            .pointer("/content_block/name")
                            .and_then(|v| v.as_str())
                            .unwrap_or("")
                            .to_string(),
                        arguments: String::new(),
                    },
                );
            }
            Vec::new()
        }
        "content_block_delta" => {
            let delta = chunk.get("delta").unwrap_or(&Value::Null);
            match delta.get("type").and_then(|v| v.as_str()).unwrap_or("") {
                "text_delta" => delta
                    .get("text")
                    .and_then(|v| v.as_str())
                    .map(|text| {
                        vec![openai_chunk(
                            &state.response_id,
                            state.created_at,
                            model,
                            json!({"content": text}),
                            Value::Null,
                            None,
                        )]
                    })
                    .unwrap_or_default(),
                "thinking_delta" => delta
                    .get("thinking")
                    .and_then(|v| v.as_str())
                    .map(|thinking| {
                        vec![openai_chunk(
                            &state.response_id,
                            state.created_at,
                            model,
                            json!({"reasoning_content": thinking}),
                            Value::Null,
                            None,
                        )]
                    })
                    .unwrap_or_default(),
                "input_json_delta" => {
                    let index = chunk.get("index").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
                    if let Some(partial) = delta.get("partial_json").and_then(|v| v.as_str())
                        && let Some(accumulator) = state.tool_calls.get_mut(&index)
                    {
                        accumulator.arguments.push_str(partial);
                    }
                    Vec::new()
                }
                _ => Vec::new(),
            }
        }
        "content_block_stop" => {
            let index = chunk.get("index").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
            let Some(accumulator) = state.tool_calls.remove(&index) else {
                return Vec::new();
            };
            let arguments = if accumulator.arguments.is_empty() {
                "{}".to_string()
            } else {
                accumulator.arguments
            };
            vec![openai_chunk(
                &state.response_id,
                state.created_at,
                model,
                json!({
                    "tool_calls": [{
                        "index": index,
                        "id": accumulator.id,
                        "type": "function",
                        "function": {
                            "name": accumulator.name,
                            "arguments": arguments,
                        }
                    }]
                }),
                Value::Null,
                None,
            )]
        }
        "message_delta" => {
            if let Some(reason) = chunk.pointer("/delta/stop_reason").and_then(|v| v.as_str()) {
                state.finish_reason = map_claude_stop_reason(reason).to_string();
            }
            if let Some(usage) = chunk.get("usage") {
                state.usage.merge(usage);
            }
            let usage = chunk.get("usage").map(|_| state.usage.openai_usage());
            vec![openai_chunk(
                &state.response_id,
                state.created_at,
                model,
                json!({}),
                json!(state.finish_reason),
                usage,
            )]
        }
        "error" => {
            let message = chunk
                .pointer("/error/message")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            let error_type = chunk
                .pointer("/error/type")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            vec![json!({"error": {"message": message, "type": error_type}})]
        }
        "message_stop" | "ping" => Vec::new(),
        _ => Vec::new(),
    }
}

pub fn transform_non_stream(
    _model: &str,
    _original_request: &Value,
    _translated_request: &Value,
    response: Value,
    _param: Option<&mut Box<dyn std::any::Any>>,
) -> Value {
    if let Some(raw_sse) = response.as_str() {
        return transform_sse_non_stream(raw_sse);
    }
    transform_claude_json_non_stream(response)
}

fn transform_sse_non_stream(raw_sse: &str) -> Value {
    let mut message_id = String::new();
    let mut model = String::new();
    let mut created_at = 0i64;
    let mut stop_reason = String::new();
    let mut content_parts = Vec::new();
    let mut reasoning_parts = Vec::new();
    let mut usage = UsageTokens::default();
    let mut tool_calls: BTreeMap<usize, ToolCallAccumulator> = BTreeMap::new();

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
                    message_id = message
                        .get("id")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_string();
                    model = message
                        .get("model")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_string();
                    created_at = now_unix();
                    if let Some(message_usage) = message.get("usage") {
                        usage.merge(message_usage);
                    }
                }
            }
            "content_block_start" => {
                if event
                    .pointer("/content_block/type")
                    .and_then(|v| v.as_str())
                    == Some("tool_use")
                {
                    let index = event.get("index").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
                    tool_calls.insert(
                        index,
                        ToolCallAccumulator {
                            id: event
                                .pointer("/content_block/id")
                                .and_then(|v| v.as_str())
                                .unwrap_or("")
                                .to_string(),
                            name: event
                                .pointer("/content_block/name")
                                .and_then(|v| v.as_str())
                                .unwrap_or("")
                                .to_string(),
                            arguments: String::new(),
                        },
                    );
                }
            }
            "content_block_delta" => {
                let delta = event.get("delta").unwrap_or(&Value::Null);
                match delta.get("type").and_then(|v| v.as_str()).unwrap_or("") {
                    "text_delta" => {
                        if let Some(text) = delta.get("text").and_then(|v| v.as_str()) {
                            content_parts.push(text.to_string());
                        }
                    }
                    "thinking_delta" => {
                        if let Some(thinking) = delta.get("thinking").and_then(|v| v.as_str()) {
                            reasoning_parts.push(thinking.to_string());
                        }
                    }
                    "input_json_delta" => {
                        let index =
                            event.get("index").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
                        if let Some(partial) = delta.get("partial_json").and_then(|v| v.as_str())
                            && let Some(accumulator) = tool_calls.get_mut(&index)
                        {
                            accumulator.arguments.push_str(partial);
                        }
                    }
                    _ => {}
                }
            }
            "content_block_stop" => {
                let index = event.get("index").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
                if let Some(accumulator) = tool_calls.get_mut(&index)
                    && accumulator.arguments.is_empty()
                {
                    accumulator.arguments.push_str("{}");
                }
            }
            "message_delta" => {
                if let Some(reason) = event.pointer("/delta/stop_reason").and_then(|v| v.as_str()) {
                    stop_reason = reason.to_string();
                }
                if let Some(event_usage) = event.get("usage") {
                    usage.merge(event_usage);
                }
            }
            _ => {}
        }
    }

    let mut out = json!({
        "id": message_id,
        "object": "chat.completion",
        "created": created_at,
        "model": model,
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": content_parts.join(""),
            },
            "finish_reason": "stop",
        }],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
    });

    if usage.has_usage {
        set_openai_usage(&mut out["usage"], usage.openai_usage());
    }
    if !reasoning_parts.is_empty() {
        out["choices"][0]["message"]["reasoning"] = json!(reasoning_parts.join(""));
    }

    let tool_calls_json = tool_calls_json(&tool_calls);
    if !tool_calls_json.is_empty() {
        out["choices"][0]["message"]["tool_calls"] = json!(tool_calls_json);
        out["choices"][0]["finish_reason"] = json!("tool_calls");
    } else {
        out["choices"][0]["finish_reason"] = json!(map_claude_stop_reason(&stop_reason));
    }

    out
}

fn transform_claude_json_non_stream(response: Value) -> Value {
    let mut content = String::new();
    let mut reasoning = Vec::new();
    let mut tool_calls = Vec::new();

    if let Some(blocks) = response.get("content").and_then(|v| v.as_array()) {
        for block in blocks {
            match block.get("type").and_then(|v| v.as_str()).unwrap_or("") {
                "text" => {
                    if let Some(text) = block.get("text").and_then(|v| v.as_str()) {
                        content.push_str(text);
                    }
                }
                "thinking" => {
                    if let Some(text) = block.get("thinking").and_then(|v| v.as_str()) {
                        reasoning.push(text.to_string());
                    }
                }
                "tool_use" => {
                    tool_calls.push(json!({
                        "id": block.get("id").and_then(|v| v.as_str()).unwrap_or(""),
                        "type": "function",
                        "function": {
                            "name": block.get("name").and_then(|v| v.as_str()).unwrap_or(""),
                            "arguments": block.get("input").map(Value::to_string).unwrap_or_else(|| "{}".to_string()),
                        }
                    }));
                }
                _ => {}
            }
        }
    }

    let mut out = json!({
        "id": response.get("id").and_then(|v| v.as_str()).unwrap_or(""),
        "object": "chat.completion",
        "created": now_unix(),
        "model": response.get("model").and_then(|v| v.as_str()).unwrap_or(""),
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": content},
            "finish_reason": map_claude_stop_reason(response.get("stop_reason").and_then(|v| v.as_str()).unwrap_or("")),
        }],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    });
    if !reasoning.is_empty() {
        out["choices"][0]["message"]["reasoning"] = json!(reasoning.join(""));
    }
    if !tool_calls.is_empty() {
        out["choices"][0]["message"]["tool_calls"] = json!(tool_calls);
        out["choices"][0]["finish_reason"] = json!("tool_calls");
    }
    if let Some(usage) = response.get("usage") {
        let mut usage_tokens = UsageTokens::default();
        usage_tokens.merge(usage);
        set_openai_usage(&mut out["usage"], usage_tokens.openai_usage());
    }
    out
}

fn openai_chunk(
    id: &str,
    created: i64,
    model: &str,
    delta: Value,
    finish_reason: Value,
    usage: Option<(i64, i64, i64, i64)>,
) -> Value {
    let mut chunk = json!({
        "id": id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{
            "index": 0,
            "delta": delta,
            "finish_reason": finish_reason,
        }]
    });
    if let Some(usage) = usage {
        chunk["usage"] = json!({});
        set_openai_usage(&mut chunk["usage"], usage);
    }
    chunk
}

fn set_openai_usage(target: &mut Value, usage: (i64, i64, i64, i64)) {
    let (prompt, completion, total, cached) = usage;
    *target = json!({
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": total,
        "prompt_tokens_details": {"cached_tokens": cached},
    });
}

fn tool_calls_json(tool_calls: &BTreeMap<usize, ToolCallAccumulator>) -> Vec<Value> {
    let mut out = Vec::new();
    for accumulator in tool_calls.values() {
        out.push(json!({
            "id": accumulator.id,
            "type": "function",
            "function": {
                "name": accumulator.name,
                "arguments": accumulator.arguments,
            }
        }));
    }
    out
}

fn map_claude_stop_reason(reason: &str) -> &'static str {
    match reason {
        "tool_use" => "tool_calls",
        "max_tokens" => "length",
        "end_turn" | "stop_sequence" => "stop",
        _ => "stop",
    }
}

fn now_unix() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_claude_json_non_stream() {
        let response = json!({
            "id": "msg_123",
            "model": "claude-test",
            "content": [
                {"type": "thinking", "thinking": "hidden"},
                {"type": "text", "text": "answer"}
            ],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 10, "output_tokens": 3}
        });

        let result = transform_non_stream("claude-test", &json!({}), &json!({}), response, None);
        assert_eq!(result["choices"][0]["message"]["content"], "answer");
        assert_eq!(result["choices"][0]["message"]["reasoning"], "hidden");
        assert_eq!(result["choices"][0]["finish_reason"], "stop");
    }
}
