//! Codex SSE → OpenAI Responses SSE — passthrough with unwrapping.

use serde_json::Value;

/// Stream: Codex SSE is already in OpenAI Responses format — just pass through.
pub fn transform_stream(
    _model: &str,
    _orig: &Value,
    _trans: &Value,
    chunk: Value,
    _param: Option<&mut Box<dyn std::any::Any>>,
) -> Vec<Value> {
    vec![chunk]
}

/// Non-stream: extract the `response` field from `response.completed` event.
///
/// Port of Go's `ConvertCodexResponseToOpenAIResponsesNonStream`.
pub fn transform_non_stream(
    _model: &str,
    _orig: &Value,
    _trans: &Value,
    response: Value,
    _param: Option<&mut Box<dyn std::any::Any>>,
) -> Value {
    // Only process response.completed events.
    if response.get("type").and_then(|v| v.as_str()) != Some("response.completed") {
        return Value::Null;
    }
    response.get("response").cloned().unwrap_or(Value::Null)
}
