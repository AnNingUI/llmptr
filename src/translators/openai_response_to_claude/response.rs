//! OpenAI Responses → Claude response translation.

use serde_json::{Value, json};

/// Transform non-streaming OpenAI Responses response to Claude format.
pub fn transform_non_stream(
    _model: &str,
    _orig: &Value,
    _trans: &Value,
    response: Value,
    _param: Option<&mut Box<dyn std::any::Any>>,
) -> Value {
    let mut out = json!({
        "id": response.get("id").and_then(|v| v.as_str()).unwrap_or("msg_unknown"),
        "type": "message",
        "role": "assistant",
        "content": [],
        "stop_reason": "end_turn",
        "stop_sequence": null,
        "usage": {"input_tokens": 0, "output_tokens": 0},
    });

    if let Some(output) = response.get("output").and_then(|v| v.as_array()) {
        for item in output {
            match item.get("type").and_then(|v| v.as_str()).unwrap_or("") {
                "message" => {
                    if let Some(content) = item.get("content").and_then(|v| v.as_array()) {
                        for part in content {
                            if part.get("type").and_then(|v| v.as_str()) == Some("output_text")
                                && let Some(t) = part.get("text").and_then(|v| v.as_str())
                            {
                                out["content"]
                                    .as_array_mut()
                                    .unwrap()
                                    .push(json!({"type":"text","text":t}));
                            }
                        }
                    }
                }
                "reasoning" => {
                    if let Some(summary) = item.get("summary").and_then(|v| v.as_array()) {
                        for s in summary {
                            if let Some(t) = s.get("text").and_then(|v| v.as_str()) {
                                out["content"]
                                    .as_array_mut()
                                    .unwrap()
                                    .push(json!({"type":"thinking","thinking":t}));
                            }
                        }
                    }
                }
                "function_call" => {
                    let name = item.get("name").and_then(|v| v.as_str()).unwrap_or("");
                    let args_str = item
                        .get("arguments")
                        .and_then(|v| v.as_str())
                        .unwrap_or("{}");
                    let id = item.get("call_id").and_then(|v| v.as_str()).unwrap_or("");
                    let args = serde_json::from_str(args_str).unwrap_or(json!({}));
                    out["content"]
                        .as_array_mut()
                        .unwrap()
                        .push(json!({"type":"tool_use","id":id,"name":name,"input":args}));
                }
                _ => {}
            }
        }
    }

    if let Some(usage) = response.get("usage") {
        let it = usage
            .get("input_tokens")
            .and_then(|v| v.as_u64())
            .unwrap_or(0);
        let ot = usage
            .get("output_tokens")
            .and_then(|v| v.as_u64())
            .unwrap_or(0);
        out["usage"]["input_tokens"] = json!(it);
        out["usage"]["output_tokens"] = json!(ot);
    }

    out
}

/// Transform streaming Responses API event to Claude SSE.
pub fn transform_stream(
    _model: &str,
    _orig: &Value,
    _trans: &Value,
    chunk: Value,
    _param: Option<&mut Box<dyn std::any::Any>>,
) -> Vec<Value> {
    let ev = chunk.get("type").and_then(|v| v.as_str()).unwrap_or("");
    match ev {
        "response.created" | "response.in_progress" => vec![],
        "response.output_text.delta" => {
            if let Some(t) = chunk.get("delta").and_then(|v| v.as_str()) {
                vec![
                    json!({"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":t}}),
                ]
            } else {
                vec![]
            }
        }
        "response.completed" => {
            vec![
                json!({"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null}}),
                json!({"type":"message_stop"}),
            ]
        }
        _ => vec![],
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    #[test]
    fn test_basic() {
        let resp = json!({"id":"resp_1","output":[{"type":"message","content":[{"type":"output_text","text":"Hello"}]}],"usage":{"input_tokens":5,"output_tokens":3}});
        let r = transform_non_stream("claude", &json!({}), &json!({}), resp, None);
        assert_eq!(r["content"][0]["text"], "Hello");
    }
}
