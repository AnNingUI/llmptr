//! Antigravity → OpenAI Responses response translation.
//!
//! Ported from Go's `antigravity/openai/responses/antigravity_openai-responses_response.go`.
//! Antigravity wraps responses under a `response` key and requests under a `request` key.
//! After unwrapping, delegates to the Gemini → OpenAI Responses handler.

use serde_json::Value;

/// Convert streaming Antigravity response chunks.
///
/// Unwraps the outer `response` envelope, then delegates to the Gemini stream handler.
pub fn transform_stream(
    model: &str,
    original_request: &Value,
    translated_request: &Value,
    chunk: Value,
    param: Option<&mut Box<dyn std::any::Any>>,
) -> Vec<Value> {
    // Antigravity wraps everything in a "response" envelope; extract it
    let inner = if let Some(resp) = chunk.get("response") {
        resp.clone()
    } else {
        chunk
    };
    crate::translators::gemini_to_openai_response::response::transform_stream(
        model,
        original_request,
        translated_request,
        inner,
        param,
    )
}

/// Convert a non-streaming Antigravity response.
///
/// Unwraps the outer `response` and `request` envelopes, then delegates
/// to the Gemini non-stream handler.
pub fn transform_non_stream(
    model: &str,
    original_request: &Value,
    translated_request: &Value,
    response: Value,
    param: Option<&mut Box<dyn std::any::Any>>,
) -> Value {
    // Unwrap the "response" envelope from Antigravity
    let inner = if let Some(resp) = response.get("response") {
        resp.clone()
    } else {
        response
    };

    // Unwrap the "request" envelope from both request objects
    let unwrapped_orig = original_request
        .get("request")
        .cloned()
        .unwrap_or_else(|| original_request.clone());
    let unwrapped_trans = translated_request
        .get("request")
        .cloned()
        .unwrap_or_else(|| translated_request.clone());

    crate::translators::gemini_to_openai_response::response::transform_non_stream(
        model,
        &unwrapped_orig,
        &unwrapped_trans,
        inner,
        param,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_streaming_unwraps_response() {
        let chunk = json!({
            "response": {
                "candidates": [{
                    "content": {
                        "parts": [{"text": "Hello"}]
                    }
                }]
            }
        });
        let r = transform_stream("gpt-4", &json!({}), &json!({}), chunk, None);
        // Should delegate and produce output text events
        assert!(!r.is_empty());
    }

    #[test]
    fn test_non_stream_unwraps_envelopes() {
        let resp = json!({
            "response": {
                "candidates": [{
                    "content": {
                        "parts": [{"text": "Hello"}]
                    }
                }]
            }
        });
        let orig = json!({"request": {"contents": []}});
        let trans = json!({"request": {"messages": []}});
        let r = transform_non_stream("gpt-4", &orig, &trans, resp, None);
        // Should have output text
        assert!(r.get("output").is_some() || r.get("id").is_some());
    }

    #[test]
    fn test_non_stream_handles_plain_response() {
        let resp = json!({"candidates": [{"content": {"parts": [{"text": "Hi"}]}}]});
        let r = transform_non_stream("gpt-4", &json!({}), &json!({}), resp, None);
        // Gemini handler always returns a standardized Responses envelope
        assert_eq!(r["object"], "response");
        assert!(r.get("output").is_some());
    }
}
