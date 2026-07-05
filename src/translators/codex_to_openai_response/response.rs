//! Codex → OpenAI Responses response translation.
//!
//! Ported from Go's `codex/openai/responses/codex_openai-responses_response.go`.
//! Codex natively uses OpenAI Responses SSE format; this is essentially a passthrough
//! with minor normalization.

use serde_json::Value;

/// Convert streaming Codex response chunks — passthrough with SSE prefix normalization.
pub fn transform_stream(
    _model: &str,
    _original_request: &Value,
    _translated_request: &Value,
    chunk: Value,
    _param: Option<&mut Box<dyn std::any::Any>>,
) -> Vec<Value> {
    vec![chunk]
}

/// Convert a non-streaming Codex response.
///
/// Codex wraps the actual response in a `response.completed` envelope.
/// This extracts the inner `response` object, or returns an empty object
/// if the envelope is not present.
pub fn transform_non_stream(
    _model: &str,
    _original_request: &Value,
    _translated_request: &Value,
    response: Value,
    _param: Option<&mut Box<dyn std::any::Any>>,
) -> Value {
    // The response is expected to be a response.completed event;
    // extract the inner "response" field.
    if response.get("type").and_then(|v| v.as_str()) == Some("response.completed") {
        response.get("response").cloned().unwrap_or(response)
    } else {
        response
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_streaming_passthrough() {
        let chunk = json!({"type": "response.output_text.delta", "delta": "Hello"});
        let r = transform_stream("gpt-4", &json!({}), &json!({}), chunk.clone(), None);
        assert_eq!(r.len(), 1);
        assert_eq!(r[0], chunk);
    }

    #[test]
    fn test_non_stream_extracts_response() {
        let resp = json!({
            "type": "response.completed",
            "response": {
                "id": "resp_123",
                "output": [{"type": "message", "content": [{"text": "Hello"}]}]
            }
        });
        let r = transform_non_stream("gpt-4", &json!({}), &json!({}), resp, None);
        assert_eq!(r["id"], "resp_123");
        assert_eq!(r["output"][0]["content"][0]["text"], "Hello");
    }

    #[test]
    fn test_non_stream_passthrough_other() {
        let resp = json!({"id": "plain_resp", "object": "response"});
        let r = transform_non_stream("gpt-4", &json!({}), &json!({}), resp.clone(), None);
        assert_eq!(r, resp);
    }
}
