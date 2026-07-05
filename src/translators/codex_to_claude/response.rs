//! Codex → Claude response translation.
//! Converts Claude server response back to Codex (OpenAI Responses) format.
//! Currently passthrough — Codex responses are OpenAI Responses format.

use serde_json::Value;

pub fn passthrough_stream(
    _model: &str,
    _original_request: &Value,
    _translated_request: &Value,
    chunk: Value,
    _param: Option<&mut Box<dyn std::any::Any>>,
) -> Vec<Value> {
    vec![chunk]
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
