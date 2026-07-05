//! OpenAI Chat → OpenAI Responses API request translation.

use serde_json::{Value, json};

/// Convert an OpenAI Chat request to OpenAI Responses API format.
pub fn transform(model: &str, body: Value, _stream: bool) -> Value {
    let mut out = json!({
        "model": model,
        "input": [],
    });

    if let Some(instructions) = body.get("instructions") {
        out["instructions"] = instructions.clone();
    }

    // Extract system message as instructions
    let mut has_instructions = instructions_present(&body);

    if let Some(msgs) = body.get("messages").and_then(|v| v.as_array()) {
        for msg in msgs {
            let role = msg.get("role").and_then(|v| v.as_str()).unwrap_or("user");

            match role {
                "system" | "developer" => {
                    if !has_instructions && let Some(Value::String(s)) = msg.get("content") {
                        out["instructions"] = json!(s);
                        has_instructions = true;
                    }
                    // Skip system messages after the first
                }
                "user" | "assistant" => {
                    let mut item = json!({"role": role});

                    if let Some(content) = msg.get("content") {
                        match content {
                            Value::String(s) => {
                                item["content"] = json!([{"type": "input_text", "text": s}]);
                            }
                            Value::Array(parts) => {
                                let mut content_parts: Vec<Value> = Vec::new();
                                for p in parts {
                                    match p.get("type").and_then(|v| v.as_str()).unwrap_or("") {
                                        "text" => {
                                            if let Some(t) = p.get("text").and_then(|v| v.as_str())
                                            {
                                                content_parts
                                                    .push(json!({"type": "input_text", "text": t}));
                                            }
                                        }
                                        "image_url" => {
                                            if let Some(url) =
                                                p.pointer("/image_url/url").and_then(|v| v.as_str())
                                            {
                                                content_parts.push(json!({"type": "input_image", "image_url": url}));
                                            }
                                        }
                                        _ => {}
                                    }
                                }
                                item["content"] = json!(content_parts);
                            }
                            _ => {}
                        }
                    }

                    // tool_calls on assistant
                    if role == "assistant"
                        && let Some(tcs) = msg.get("tool_calls").and_then(|v| v.as_array())
                    {
                        let tool_calls: Vec<Value> = tcs
                            .iter()
                            .map(|tc| {
                                json!({
                                    "id": tc.get("id").and_then(|v| v.as_str()).unwrap_or(""),
                                    "type": "function_call",
                                    "function": tc.get("function"),
                                })
                            })
                            .collect();
                        if !tool_calls.is_empty() {
                            // Add tool_calls as content items
                            item["content"]
                                .as_array_mut()
                                .unwrap_or(&mut vec![])
                                .push(json!({"type": "tool_calls", "tool_calls": tool_calls}));
                        }
                    }

                    out["input"].as_array_mut().unwrap().push(item);
                }
                "tool" => {
                    let tool_call_id = msg
                        .get("tool_call_id")
                        .and_then(|v| v.as_str())
                        .unwrap_or("");
                    let content_text = msg.get("content").and_then(|v| v.as_str()).unwrap_or("");
                    out["input"].as_array_mut().unwrap().push(json!({
                        "type": "function_call_output",
                        "call_id": tool_call_id,
                        "output": content_text,
                    }));
                }
                _ => {}
            }
        }
    }

    // ── scalar passthrough ────────────────────────────────
    if let Some(max_tokens) = body.get("max_tokens").and_then(|v| v.as_u64()) {
        out["max_tokens"] = json!(max_tokens);
    }
    if let Some(temp) = body.get("temperature") {
        out["temperature"] = temp.clone();
    }
    if let Some(tools) = body.get("tools") {
        out["tools"] = tools.clone();
    }
    if let Some(tc) = body.get("tool_choice") {
        out["tool_choice"] = tc.clone();
    }

    out
}

fn instructions_present(body: &Value) -> bool {
    body.get("instructions").is_some()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chat_to_responses() {
        let body = json!({
            "model": "gpt-4",
            "messages": [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Hi!"}
            ]
        });
        let result = transform("gpt-4", body, false);
        assert_eq!(result["instructions"], "You are helpful.");
        assert_eq!(result["input"][0]["role"], "user");
        assert_eq!(result["input"][0]["content"][0]["text"], "Hi!");
    }

    #[test]
    fn test_chat_to_responses_tool_result() {
        let body = json!({
            "messages": [
                {"role": "assistant", "content": "", "tool_calls": [{"id":"c1","type":"function","function":{"name":"x","arguments":"{}"}}]},
                {"role": "tool", "tool_call_id": "c1", "content": "result"}
            ]
        });
        let result = transform("gpt-4", body, false);
        assert_eq!(result["input"][1]["type"], "function_call_output");
        assert_eq!(result["input"][1]["call_id"], "c1");
    }
}
