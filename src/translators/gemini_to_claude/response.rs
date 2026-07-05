use serde_json::{Value, json};

#[derive(Debug, Clone, Default)]
pub struct GeminiToClaudeStreamState {
    pub has_first_response: bool,
    pub response_type: u8,
    pub response_index: usize,
    pub has_content: bool,
    pub saw_tool_call: bool,
    pub has_final_events: bool,
    pub tool_id_counter: usize,
}

pub fn transform_non_stream(
    model: &str,
    original_request: &Value,
    _translated_request: &Value,
    response: Value,
    _param: Option<&mut Box<dyn std::any::Any>>,
) -> Value {
    let mut out = json!({
        "id": response.get("responseId").and_then(|v| v.as_str()).unwrap_or(""),
        "type": "message",
        "role": "assistant",
        "model": response
            .get("modelVersion")
            .and_then(|v| v.as_str())
            .unwrap_or(model),
        "content": [],
        "stop_reason": null,
        "stop_sequence": null,
        "usage": {"input_tokens": 0, "output_tokens": 0},
    });

    let input_tokens = response
        .pointer("/usageMetadata/promptTokenCount")
        .and_then(|v| v.as_i64())
        .unwrap_or(0);
    let output_tokens = response
        .pointer("/usageMetadata/candidatesTokenCount")
        .and_then(|v| v.as_i64())
        .unwrap_or(0)
        + response
            .pointer("/usageMetadata/thoughtsTokenCount")
            .and_then(|v| v.as_i64())
            .unwrap_or(0);
    out["usage"]["input_tokens"] = json!(input_tokens);
    out["usage"]["output_tokens"] = json!(output_tokens);

    let tool_name_map = tool_name_map_from_claude_request(original_request);
    let sanitized_name_map = sanitized_tool_name_map(original_request);
    let mut content: Vec<Value> = Vec::new();
    let mut text_buffer = String::new();
    let mut thinking_buffer = String::new();
    let mut tool_id_counter = 0usize;
    let mut has_tool_call = false;

    if let Some(parts) = response
        .pointer("/candidates/0/content/parts")
        .and_then(|v| v.as_array())
    {
        for part in parts {
            if let Some(text) = part.get("text").and_then(|v| v.as_str())
                && !text.is_empty()
            {
                if part
                    .get("thought")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false)
                {
                    flush_text(&mut content, &mut text_buffer);
                    thinking_buffer.push_str(text);
                } else {
                    flush_thinking(&mut content, &mut thinking_buffer);
                    text_buffer.push_str(text);
                }
                continue;
            }

            if let Some(function_call) = part.get("functionCall") {
                flush_thinking(&mut content, &mut thinking_buffer);
                flush_text(&mut content, &mut text_buffer);
                has_tool_call = true;
                tool_id_counter += 1;

                let upstream_name = function_call
                    .get("name")
                    .and_then(|v| v.as_str())
                    .unwrap_or("");
                let restored_name = restore_sanitized_tool_name(&sanitized_name_map, upstream_name);
                let client_name = tool_name_map
                    .get(&restored_name)
                    .cloned()
                    .unwrap_or(restored_name.clone());
                let input = function_call
                    .get("args")
                    .filter(|v| v.is_object())
                    .cloned()
                    .unwrap_or_else(|| json!({}));

                content.push(json!({
                    "type": "tool_use",
                    "id": sanitize_claude_tool_id(&format!("{restored_name}-{tool_id_counter}")),
                    "name": client_name,
                    "input": input,
                }));
            }
        }
    }

    flush_thinking(&mut content, &mut thinking_buffer);
    flush_text(&mut content, &mut text_buffer);
    out["content"] = json!(content);

    let stop_reason = if has_tool_call {
        "tool_use"
    } else {
        match response
            .pointer("/candidates/0/finishReason")
            .and_then(|v| v.as_str())
        {
            Some("MAX_TOKENS") => "max_tokens",
            _ => "end_turn",
        }
    };
    out["stop_reason"] = json!(stop_reason);

    if response.get("usageMetadata").is_none() && input_tokens == 0 && output_tokens == 0 {
        out.as_object_mut().unwrap().remove("usage");
    }

    out
}

pub fn transform_stream(
    _model: &str,
    original_request: &Value,
    _translated_request: &Value,
    chunk: Value,
    param: Option<&mut Box<dyn std::any::Any>>,
) -> Vec<Value> {
    let mut local_state;
    let state = if let Some(p) = param {
        if !p.is::<GeminiToClaudeStreamState>() {
            *p = Box::<GeminiToClaudeStreamState>::default();
        }
        p.downcast_mut::<GeminiToClaudeStreamState>().unwrap()
    } else {
        local_state = GeminiToClaudeStreamState::default();
        &mut local_state
    };

    if chunk.as_str() == Some("[DONE]") {
        if state.has_content {
            return vec![Value::String(sse_event(
                "message_stop",
                &json!({"type":"message_stop"}),
            ))];
        }
        return Vec::new();
    }

    let mut output = String::new();

    if !state.has_first_response {
        let id = chunk
            .get("responseId")
            .and_then(|v| v.as_str())
            .unwrap_or("msg_1nZdL29xx5MUA1yADyHTEsnR8uuvGzszyY");
        let model = chunk
            .get("modelVersion")
            .and_then(|v| v.as_str())
            .unwrap_or("claude-3-5-sonnet-20241022");
        push_sse_payload(
            &mut output,
            "message_start",
            &format!(
                "{{\"type\":\"message_start\",\"message\":{{\"id\":{},\"type\":\"message\",\"role\":\"assistant\",\"content\":[],\"model\":{},\"stop_reason\":null,\"stop_sequence\":null,\"usage\":{{\"input_tokens\":0,\"output_tokens\":0}}}}}}",
                json_string(id),
                json_string(model)
            ),
        );
        state.has_first_response = true;
    }

    let tool_name_map = tool_name_map_from_claude_request(original_request);
    let sanitized_name_map = sanitized_tool_name_map(original_request);

    if let Some(parts) = chunk
        .pointer("/candidates/0/content/parts")
        .and_then(|v| v.as_array())
    {
        for part in parts {
            let text = part.get("text").and_then(|v| v.as_str());
            let function_call = part.get("functionCall");
            let thought_signature = part
                .get("thoughtSignature")
                .or_else(|| part.get("thought_signature"))
                .and_then(|v| v.as_str())
                .unwrap_or("");
            let has_thought_signature = !thought_signature.is_empty();

            if has_thought_signature && text.is_none() && function_call.is_none() {
                append_signature_delta(&mut output, state, thought_signature);
                continue;
            }

            if let Some(text) = text {
                if part
                    .get("thought")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false)
                    || has_thought_signature
                {
                    if has_thought_signature && text.is_empty() {
                        append_signature_delta(&mut output, state, thought_signature);
                        continue;
                    }
                    if state.response_type != 2 {
                        close_current_block(&mut output, state);
                        push_sse_payload(
                            &mut output,
                            "content_block_start",
                            &format!(
                                "{{\"type\":\"content_block_start\",\"index\":{},\"content_block\":{{\"type\":\"thinking\",\"thinking\":\"\"}}}}",
                                state.response_index
                            ),
                        );
                        state.response_type = 2;
                    }
                    push_sse_payload(
                        &mut output,
                        "content_block_delta",
                        &format!(
                            "{{\"type\":\"content_block_delta\",\"index\":{},\"delta\":{{\"type\":\"thinking_delta\",\"thinking\":{}}}}}",
                            state.response_index,
                            json_string(text)
                        ),
                    );
                    state.has_content = true;
                    append_signature_delta(&mut output, state, thought_signature);
                } else if !text.is_empty() {
                    if state.response_type != 1 {
                        close_current_block(&mut output, state);
                        push_sse_payload(
                            &mut output,
                            "content_block_start",
                            &format!(
                                "{{\"type\":\"content_block_start\",\"index\":{},\"content_block\":{{\"type\":\"text\",\"text\":\"\"}}}}",
                                state.response_index
                            ),
                        );
                        state.response_type = 1;
                    }
                    push_sse_payload(
                        &mut output,
                        "content_block_delta",
                        &format!(
                            "{{\"type\":\"content_block_delta\",\"index\":{},\"delta\":{{\"type\":\"text_delta\",\"text\":{}}}}}",
                            state.response_index,
                            json_string(text)
                        ),
                    );
                    state.has_content = true;
                }
            }

            if let Some(fc) = function_call {
                state.saw_tool_call = true;
                let upstream_name = fc.get("name").and_then(|v| v.as_str()).unwrap_or("");
                let restored_name = restore_sanitized_tool_name(&sanitized_name_map, upstream_name);

                if state.response_type == 3 && restored_name.is_empty() {
                    if let Some(args) = fc.get("args") {
                        push_input_json_delta(&mut output, state.response_index, args);
                    }
                    continue;
                }

                if state.response_type == 3 {
                    close_current_block(&mut output, state);
                }
                close_current_block(&mut output, state);

                let client_name = tool_name_map
                    .get(&restored_name)
                    .cloned()
                    .unwrap_or(restored_name.clone());
                state.tool_id_counter += 1;
                let tool_id =
                    sanitize_claude_tool_id(&format!("{restored_name}-{}", state.tool_id_counter));
                push_sse_payload(
                    &mut output,
                    "content_block_start",
                    &format!(
                        "{{\"type\":\"content_block_start\",\"index\":{},\"content_block\":{{\"type\":\"tool_use\",\"id\":{},\"name\":{},\"input\":{{}}}}}}",
                        state.response_index,
                        json_string(&tool_id),
                        json_string(&client_name)
                    ),
                );
                if let Some(args) = fc.get("args") {
                    push_input_json_delta(&mut output, state.response_index, args);
                }
                state.response_type = 3;
                state.has_content = true;
            }
        }
    }

    if chunk.get("usageMetadata").is_some()
        && chunk.pointer("/candidates/0/finishReason").is_some()
        && !state.has_final_events
        && state.has_content
    {
        close_current_block(&mut output, state);
        let usage = chunk.get("usageMetadata").unwrap();
        let output_tokens = usage
            .get("candidatesTokenCount")
            .and_then(|v| v.as_i64())
            .unwrap_or(0)
            + usage
                .get("thoughtsTokenCount")
                .and_then(|v| v.as_i64())
                .unwrap_or(0);
        let input_tokens = usage
            .get("promptTokenCount")
            .and_then(|v| v.as_i64())
            .unwrap_or(0);
        let stop_reason = if state.saw_tool_call {
            "tool_use"
        } else if chunk
            .pointer("/candidates/0/finishReason")
            .and_then(|v| v.as_str())
            == Some("MAX_TOKENS")
        {
            "max_tokens"
        } else {
            "end_turn"
        };
        push_sse_payload(
            &mut output,
            "message_delta",
            &format!(
                "{{\"type\":\"message_delta\",\"delta\":{{\"stop_reason\":{},\"stop_sequence\":null}},\"usage\":{{\"input_tokens\":{},\"output_tokens\":{}}}}}",
                json_string(stop_reason),
                input_tokens,
                output_tokens
            ),
        );
        state.has_final_events = true;
    }

    vec![Value::String(output)]
}

fn flush_text(content: &mut Vec<Value>, buffer: &mut String) {
    if !buffer.is_empty() {
        content.push(json!({"type": "text", "text": buffer}));
        buffer.clear();
    }
}

fn flush_thinking(content: &mut Vec<Value>, buffer: &mut String) {
    if !buffer.is_empty() {
        content.push(json!({"type": "thinking", "thinking": buffer}));
        buffer.clear();
    }
}

fn sse_event(event: &str, payload: &Value) -> String {
    let mut out = String::new();
    push_sse_event(&mut out, event, payload);
    out
}

fn push_sse_event(out: &mut String, event: &str, payload: &Value) {
    push_sse_payload(out, event, &serde_json::to_string(payload).unwrap());
}

fn push_sse_payload(out: &mut String, event: &str, payload: &str) {
    out.push_str("event: ");
    out.push_str(event);
    out.push('\n');
    out.push_str("data: ");
    out.push_str(payload);
    out.push_str("\n\n\n");
}

fn json_string(value: &str) -> String {
    serde_json::to_string(value).unwrap()
}

fn close_current_block(out: &mut String, state: &mut GeminiToClaudeStreamState) {
    if state.response_type != 0 {
        push_sse_payload(
            out,
            "content_block_stop",
            &format!(
                "{{\"type\":\"content_block_stop\",\"index\":{}}}",
                state.response_index
            ),
        );
        state.response_index += 1;
        state.response_type = 0;
    }
}

fn append_signature_delta(
    out: &mut String,
    state: &mut GeminiToClaudeStreamState,
    signature: &str,
) {
    if signature.is_empty() || state.response_type != 2 {
        return;
    }
    push_sse_payload(
        out,
        "content_block_delta",
        &format!(
            "{{\"type\":\"content_block_delta\",\"index\":{},\"delta\":{{\"type\":\"signature_delta\",\"signature\":{}}}}}",
            state.response_index,
            json_string(signature)
        ),
    );
    state.has_content = true;
}

fn push_input_json_delta(out: &mut String, index: usize, args: &Value) {
    let partial_json = serde_json::to_string(args).unwrap_or_else(|_| "{}".to_string());
    push_sse_payload(
        out,
        "content_block_delta",
        &format!(
            "{{\"type\":\"content_block_delta\",\"index\":{},\"delta\":{{\"type\":\"input_json_delta\",\"partial_json\":{}}}}}",
            index,
            json_string(&partial_json)
        ),
    );
}

fn tool_name_map_from_claude_request(request: &Value) -> std::collections::HashMap<String, String> {
    let mut map = std::collections::HashMap::new();
    if let Some(tools) = request.get("tools").and_then(|v| v.as_array()) {
        for tool in tools {
            if let Some(name) = tool.get("name").and_then(|v| v.as_str()) {
                map.insert(sanitize_function_name(name), name.to_string());
            }
        }
    }
    map
}

fn sanitized_tool_name_map(request: &Value) -> std::collections::HashMap<String, String> {
    let mut map = std::collections::HashMap::new();
    if let Some(tools) = request.get("tools").and_then(|v| v.as_array()) {
        for tool in tools {
            if let Some(name) = tool.get("name").and_then(|v| v.as_str()) {
                map.insert(sanitize_function_name(name), name.to_string());
            }
        }
    }
    map
}

fn restore_sanitized_tool_name(
    map: &std::collections::HashMap<String, String>,
    name: &str,
) -> String {
    map.get(name).cloned().unwrap_or_else(|| name.to_string())
}

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

fn sanitize_claude_tool_id(id: &str) -> String {
    id.chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '_' || c == '-' {
                c
            } else {
                '_'
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_claude_to_gemini_non_stream_basic() {
        let resp = json!({
            "responseId": "resp-test",
            "candidates": [{
                "content": {
                    "role": "model",
                    "parts": [{"text": "Hello from Claude!"}]
                },
                "finishReason": "STOP"
            }],
            "usageMetadata": {
                "promptTokenCount": 10,
                "candidatesTokenCount": 5,
                "totalTokenCount": 15
            },
            "modelVersion": "claude-sonnet-4-8"
        });

        let result = transform_non_stream("claude-sonnet-4-8", &json!({}), &json!({}), resp, None);
        assert_eq!(result["id"], "resp-test");
        assert_eq!(result["content"][0]["type"], "text");
        assert_eq!(result["content"][0]["text"], "Hello from Claude!");
        assert_eq!(result["usage"]["input_tokens"], 10);
    }

    #[test]
    fn test_claude_to_gemini_with_function_call() {
        let resp = json!({
            "responseId": "resp-test",
            "candidates": [{
                "content": {
                    "role": "model",
                    "parts": [{"functionCall": {"name": "get_weather", "args": {"city": "Tokyo"}}}]
                },
                "finishReason": "STOP"
            }],
            "usageMetadata": {"promptTokenCount": 5, "candidatesTokenCount": 3, "totalTokenCount": 8},
            "modelVersion": "claude-sonnet-4-8"
        });

        let result = transform_non_stream("claude-sonnet-4-8", &json!({}), &json!({}), resp, None);
        assert_eq!(result["content"][0]["type"], "tool_use");
        assert_eq!(result["content"][0]["name"], "get_weather");
        assert_eq!(result["stop_reason"], "tool_use");
    }
}
