//! Codex SSE <->Gemini <->streaming state machine.
//!
//! Port of Go's `codex_gemini_response.go` `ConvertCodexResponseToGemini`
//! and `ConvertCodexResponseToGeminiNonStream`.

use serde_json::{Value, json};

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Default)]
struct State {
    model: String,
    created_at: i64,
    response_id: String,
    has_output_text_delta: bool,
}

// ---------------------------------------------------------------------------
// Stream
// ---------------------------------------------------------------------------

/// Transform a Codex streaming SSE chunk into zero or more Gemini JSON responses.
pub fn transform_stream(
    model: &str,
    _orig: &Value,
    _trans: &Value,
    chunk: Value,
    param: Option<&mut Box<dyn std::any::Any>>,
) -> Vec<Value> {
    let st = get_or_init_state(param, model);

    // Unwrap SSE "data:" prefix.
    let chunk = match chunk {
        Value::String(ref s) if s.starts_with("data:") => {
            let payload = s.trim_start_matches("data:").trim();
            match serde_json::from_str::<Value>(payload) {
                Ok(v) => v,
                Err(_) => return vec![],
            }
        }
        Value::String(_) => return vec![],
        other => other,
    };

    let event_type = chunk.get("type").and_then(|v| v.as_str()).unwrap_or("");

    match event_type {
        "response.created" => {
            if let Some(resp) = chunk.get("response") {
                if let Some(id) = resp.get("id").and_then(|v| v.as_str()) {
                    st.response_id = id.to_string();
                }
                if let Some(ca) = resp.get("created_at").and_then(|v| v.as_i64()) {
                    st.created_at = ca;
                }
            }
            vec![]
        }

        "response.output_text.delta" => {
            let delta = chunk.get("delta").and_then(|v| v.as_str()).unwrap_or("");
            if delta.is_empty() {
                return vec![];
            }
            st.has_output_text_delta = true;
            let mut t = make_template(st);
            push_part(&mut t, &json!({"text": delta}));
            vec![t]
        }

        "response.output_item.done" => {
            let item = match chunk.get("item") {
                Some(v) => v,
                None => return vec![],
            };
            let item_type = item.get("type").and_then(|v| v.as_str()).unwrap_or("");

            match item_type {
                "function_call" => {
                    let mut t = make_template(st);
                    let name = item
                        .get("name")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_string();
                    let args = item.get("arguments").cloned().unwrap_or(json!({}));
                    push_part(
                        &mut t,
                        &json!({"functionCall": {"name": name, "args": args}}),
                    );
                    t["candidates"][0]["finishReason"] = json!("STOP");
                    vec![t]
                }

                "web_search_call" => {
                    let id = item.get("id").and_then(|v| v.as_str()).unwrap_or("");
                    let tool_use_id = if id.is_empty() {
                        "web_search_0".to_string()
                    } else {
                        id.to_string()
                    };
                    let query = item.get("query").and_then(|v| v.as_str()).unwrap_or("");

                    let mut t1 = make_template(st);
                    push_part(
                        &mut t1,
                        &json!({
                            "type": "tool_use",
                            "id": tool_use_id,
                            "name": "web_search",
                            "input": {"query": query}
                        }),
                    );

                    let mut results = vec![];
                    if let Some(arr) = item.get("results").and_then(|v| v.as_array()) {
                        for r in arr {
                            let url = r.get("url").and_then(|v| v.as_str()).unwrap_or("");
                            let title = r.get("title").and_then(|v| v.as_str()).unwrap_or(url);
                            if !url.is_empty() {
                                results.push(json!({"type": "web_search_result", "title": title, "url": url}));
                            }
                        }
                    }
                    let mut t2 = make_template(st);
                    push_part(
                        &mut t2,
                        &json!({
                            "type": "tool_result",
                            "tool_use_id": tool_use_id,
                            "content": results,
                        }),
                    );
                    vec![t1, t2]
                }

                _ => vec![],
            }
        }

        "response.completed" => {
            if let Some(response) = chunk.get("response")
                && let Some(usage) = response.get("usage")
            {
                let mut t = make_template(st);
                let input = usage
                    .get("input_tokens")
                    .and_then(|v| v.as_i64())
                    .unwrap_or(0);
                let output = usage
                    .get("output_tokens")
                    .and_then(|v| v.as_i64())
                    .unwrap_or(0);
                t["usageMetadata"]["promptTokenCount"] = json!(input);
                t["usageMetadata"]["candidatesTokenCount"] = json!(output);
                t["usageMetadata"]["totalTokenCount"] = json!(input + output);
                return vec![t];
            }
            vec![]
        }

        _ => vec![],
    }
}

// ---------------------------------------------------------------------------
// Non-stream
// ---------------------------------------------------------------------------

/// Convert a completed Codex non-streaming response to Gemini format.
pub fn transform_non_stream(
    _model: &str,
    _orig: &Value,
    _trans: &Value,
    response: Value,
    _param: Option<&mut Box<dyn std::any::Any>>,
) -> Value {
    let resp = if response.get("type").and_then(|v| v.as_str()) == Some("response.completed") {
        response.get("response").cloned().unwrap_or(response)
    } else {
        response
    };

    let model_name = resp
        .get("model")
        .and_then(|v| v.as_str())
        .unwrap_or("gemini-2.5-pro");
    let mut out = json!({
        "candidates": [{
            "content": {"role": "model", "parts": []},
            "finishReason": "STOP"
        }],
        "usageMetadata": {"trafficType": "PROVISIONED_THROUGHPUT"},
        "modelVersion": model_name,
        "responseId": resp.get("id").and_then(|v| v.as_str()).unwrap_or(""),
        "createTime": ""
    });

    if let Some(output) = resp.get("output").and_then(|v| v.as_array()) {
        for item in output {
            let item_type = item.get("type").and_then(|v| v.as_str()).unwrap_or("");
            match item_type {
                "message" => {
                    if let Some(content) = item.get("content").and_then(|v| v.as_array()) {
                        for c in content {
                            if c.get("type").and_then(|v| v.as_str()) == Some("output_text")
                                && let Some(text) = c.get("text").and_then(|v| v.as_str())
                            {
                                push_part(&mut out, &json!({"text": text}));
                            }
                        }
                    }
                }
                "reasoning" => {
                    if let Some(summaries) = item.get("summary").and_then(|v| v.as_array()) {
                        for s in summaries {
                            if s.get("type").and_then(|v| v.as_str()) == Some("summary_text")
                                && let Some(text) = s.get("text").and_then(|v| v.as_str())
                            {
                                push_part(&mut out, &json!({"text": text, "thought": true}));
                            }
                        }
                    }
                }
                "function_call" => {
                    let name = item
                        .get("name")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_string();
                    let args = item.get("arguments").cloned().unwrap_or(json!({}));
                    push_part(
                        &mut out,
                        &json!({"functionCall": {"name": name, "args": args}}),
                    );
                }
                _ => {}
            }
        }
    }

    if let Some(usage) = resp.get("usage") {
        let input = usage
            .get("input_tokens")
            .and_then(|v| v.as_i64())
            .unwrap_or(0);
        let output = usage
            .get("output_tokens")
            .and_then(|v| v.as_i64())
            .unwrap_or(0);
        out["usageMetadata"]["promptTokenCount"] = json!(input);
        out["usageMetadata"]["candidatesTokenCount"] = json!(output);
        out["usageMetadata"]["totalTokenCount"] = json!(input + output);
    }

    out
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn get_or_init_state<'a>(
    param: Option<&'a mut Box<dyn std::any::Any>>,
    model: &str,
) -> &'a mut State {
    let p = param.unwrap();
    if !p.is::<State>() {
        *p = Box::new(State {
            model: model.to_string(),
            ..Default::default()
        });
    }
    p.downcast_mut::<State>().unwrap()
}

fn make_template(st: &State) -> Value {
    let mut t = json!({
        "candidates": [{"content": {"role": "model", "parts": []}}],
        "usageMetadata": {"trafficType": "PROVISIONED_THROUGHPUT"},
        "modelVersion": st.model,
        "createTime": "",
        "responseId": st.response_id
    });
    if st.created_at > 0 {
        t["createTime"] = json!(format!(
            "2025-01-01T00:{:02}:{:02}Z",
            (st.created_at / 60) % 60,
            st.created_at % 60
        ));
    }
    t
}

fn push_part(template: &mut Value, part: &Value) {
    if let Some(arr) = template["candidates"].as_array_mut()
        && let Some(c) = arr.first_mut()
        && let Some(parts) = c["content"]["parts"].as_array_mut()
    {
        parts.push(part.clone());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stream_text_delta() {
        let chunk = json!({"type": "response.output_text.delta", "delta": "Hello"});
        let mut p: Box<dyn std::any::Any> = Box::new(State::default());
        let results = transform_stream("gpt-4", &json!({}), &json!({}), chunk, Some(&mut p));
        assert!(!results.is_empty());
        let parts = results[0]["candidates"][0]["content"]["parts"]
            .as_array()
            .unwrap();
        assert_eq!(parts[0]["text"], "Hello");
    }

    #[test]
    fn test_stream_function_call_done() {
        let chunk = json!({
            "type": "response.output_item.done",
            "item": {"type": "function_call", "name": "get_weather", "arguments": {"city": "Paris"}}
        });
        let mut p: Box<dyn std::any::Any> = Box::new(State::default());
        let results = transform_stream("gpt-4", &json!({}), &json!({}), chunk, Some(&mut p));
        assert!(!results.is_empty());
        let parts = results[0]["candidates"][0]["content"]["parts"]
            .as_array()
            .unwrap();
        assert_eq!(parts[0]["functionCall"]["name"], "get_weather");
    }

    #[test]
    fn test_non_stream_basic() {
        let resp = json!({
            "type": "response.completed",
            "response": {
                "id": "resp_1", "model": "gpt-4",
                "output": [{"type": "message", "role": "assistant", "content": [
                    {"type": "output_text", "text": "Hi"}
                ]}],
                "usage": {"input_tokens": 5, "output_tokens": 3}
            }
        });
        let result = transform_non_stream("gpt-4", &json!({}), &json!({}), resp, None);
        let parts = result["candidates"][0]["content"]["parts"]
            .as_array()
            .unwrap();
        assert_eq!(parts[0]["text"], "Hi");
        assert_eq!(result["usageMetadata"]["promptTokenCount"], 5);
    }
}
