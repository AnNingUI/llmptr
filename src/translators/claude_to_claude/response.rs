//! Claude → Claude response passthrough.

use serde_json::Value;

pub fn passthrough_stream(
    _model: &str,
    _original_request: &Value,
    _translated_request: &Value,
    chunk: Value,
    _param: Option<&mut Box<dyn std::any::Any>>,
) -> Vec<Value> {
    let Some(raw) = chunk.as_str() else {
        return vec![chunk];
    };
    let payload = raw
        .trim()
        .strip_prefix("data:")
        .map(str::trim)
        .unwrap_or_else(|| raw.trim());
    if payload == "[DONE]" {
        return Vec::new();
    }
    if let Ok(parsed) = serde_json::from_str::<Value>(payload) {
        vec![parsed]
    } else {
        vec![Value::String(payload.to_string())]
    }
}

pub fn passthrough_non_stream(
    _model: &str,
    _original_request: &Value,
    _translated_request: &Value,
    response: Value,
    _param: Option<&mut Box<dyn std::any::Any>>,
) -> Value {
    response
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn non_stream_passthrough() {
        let response =
            json!({"id":"msg_1","type":"message","content":[{"type":"text","text":"Hi"}]});
        let result =
            passthrough_non_stream("claude", &json!({}), &json!({}), response.clone(), None);
        assert_eq!(result, response);
    }

    #[test]
    fn stream_passthrough() {
        let chunk = json!("data: {\"type\":\"content_block_delta\",\"delta\":{\"text\":\"Hi\"}}");
        let result = passthrough_stream("claude", &json!({}), &json!({}), chunk, None);
        assert!(!result.is_empty());
    }

    #[test]
    fn done_is_suppressed() {
        let result = passthrough_stream(
            "claude",
            &json!({}),
            &json!({}),
            json!("data: [DONE]"),
            None,
        );
        assert!(result.is_empty());
    }
}
