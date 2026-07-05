//! OpenAI Responses → OpenAI Chat response translation.

use serde_json::{Value, json};

/// Transform a non-streaming OpenAI Responses format response to Chat Completions format.
pub fn transform_non_stream(
    _model: &str,
    _orig: &Value,
    _trans: &Value,
    response: Value,
    _param: Option<&mut Box<dyn std::any::Any>>,
) -> Value {
    let mut out = json!({
        "id": response.get("id").and_then(|v| v.as_str()).unwrap_or("resp_unknown"),
        "object": "chat.completion",
        "created": std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_secs() as i64,
        "model": response.get("model").and_then(|v| v.as_str()).unwrap_or(_model),
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": null},
            "finish_reason": "stop",
        }],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    });

    // Extract message from output[0].content[]
    if let Some(output) = response.get("output").and_then(|v| v.as_array()) {
        // Find the first message item
        for item in output {
            if item.get("type").and_then(|v| v.as_str()) == Some("message") {
                let mut msg = json!({"role": "assistant", "content": ""});
                let mut tool_calls: Vec<Value> = Vec::new();
                let mut text_content = String::new();

                if let Some(content) = item.get("content").and_then(|v| v.as_array()) {
                    for part in content {
                        match part.get("type").and_then(|v| v.as_str()).unwrap_or("") {
                            "output_text" => {
                                if let Some(t) = part.get("text").and_then(|v| v.as_str()) {
                                    text_content.push_str(t);
                                }
                            }
                            "reasoning" => {
                                if let Some(t) = part.get("text").and_then(|v| v.as_str()) {
                                    msg["reasoning_content"] = json!(t);
                                }
                            }
                            "tool_use" | "function_call" => {
                                let name = part.get("name").and_then(|v| v.as_str()).unwrap_or("");
                                let empty_obj = json!({});
                                let args = part
                                    .get("arguments")
                                    .or_else(|| part.get("input"))
                                    .unwrap_or(&empty_obj);
                                let id = part.get("id").and_then(|v| v.as_str()).unwrap_or("");
                                tool_calls.push(json!({
                                    "id": id,
                                    "type": "function",
                                    "function": {"name": name, "arguments": args.to_string()}
                                }));
                            }
                            _ => {}
                        }
                    }
                }

                msg["content"] = json!(text_content);
                if !tool_calls.is_empty() {
                    msg["tool_calls"] = json!(tool_calls);
                }
                out["choices"][0]["message"] = msg;

                // status → finish_reason
                if let Some(status) = item.get("status").and_then(|v| v.as_str()) {
                    out["choices"][0]["finish_reason"] = json!(match status {
                        "completed" =>
                            if tool_calls.is_empty() {
                                "stop"
                            } else {
                                "tool_calls"
                            },
                        "incomplete" => "length",
                        "failed" => "stop",
                        _ => "stop",
                    });
                }
                break;
            }
        }
    }

    // Usage
    if let Some(usage) = response.get("usage") {
        let pt = usage
            .get("input_tokens")
            .or_else(|| usage.get("prompt_tokens"))
            .and_then(|v| v.as_u64())
            .unwrap_or(0);
        let ct = usage
            .get("output_tokens")
            .or_else(|| usage.get("completion_tokens"))
            .and_then(|v| v.as_u64())
            .unwrap_or(0);
        out["usage"]["prompt_tokens"] = json!(pt);
        out["usage"]["completion_tokens"] = json!(ct);
        out["usage"]["total_tokens"] = json!(pt + ct);
    }

    out
}

/// Transform streaming Responses API SSE chunk to Chat Completions SSE.
pub fn transform_stream(
    _model: &str,
    _orig: &Value,
    _trans: &Value,
    chunk: Value,
    _param: Option<&mut Box<dyn std::any::Any>>,
) -> Vec<Value> {
    if chunk.as_str() == Some("[DONE]") {
        return vec![chunk.clone()];
    }

    let mut results: Vec<Value> = Vec::new();
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64;

    if let Some(delta) = chunk.get("delta") {
        let mut choices =
            json!([{"index": 0, "delta": {"role": "assistant"}, "finish_reason": null}]);

        if let Some(t) = delta.get("content").and_then(|v| v.as_str()) {
            choices[0]["delta"]["content"] = json!(t);
        }
        if let Some(t) = delta.get("text").and_then(|v| v.as_str()) {
            choices[0]["delta"]["content"] = json!(t);
        }
        if let Some(reasoning) = delta.get("reasoning").and_then(|v| v.as_str()) {
            choices[0]["delta"]["reasoning_content"] = json!(reasoning);
        }

        results.push(json!({
            "id": "chatcmpl-resp",
            "object": "chat.completion.chunk",
            "created": now,
            "model": _model,
            "choices": choices,
        }));
    }

    if let Some(status) = chunk.get("status").and_then(|v| v.as_str())
        && (status == "completed" || status == "incomplete")
    {
        results.push(json!({
                "id": "chatcmpl-resp",
                "object": "chat.completion.chunk",
                "created": now,
                "model": _model,
                "choices": [{"index": 0, "delta": {}, "finish_reason": if status == "completed" { "stop" } else { "length" }}],
            }));
    }

    results
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_response_non_stream_text() {
        let resp = json!({
            "id": "resp_123",
            "object": "response",
            "model": "gpt-4",
            "output": [{
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "Hello!"}],
                "status": "completed"
            }],
            "usage": {"input_tokens": 5, "output_tokens": 3}
        });
        let result = transform_non_stream("gpt-4", &json!({}), &json!({}), resp, None);
        assert_eq!(result["choices"][0]["message"]["content"], "Hello!");
        assert_eq!(result["choices"][0]["finish_reason"], "stop");
    }
}
