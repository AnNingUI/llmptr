//! Gemini → Gemini response passthrough.

use serde_json::Value;

pub fn passthrough_stream(
    _model: &str,
    _orig: &Value,
    _trans: &Value,
    chunk: Value,
    _param: Option<&mut Box<dyn std::any::Any>>,
) -> Vec<Value> {
    vec![chunk]
}

pub fn passthrough_non_stream(
    _model: &str,
    _orig: &Value,
    _trans: &Value,
    response: Value,
    _param: Option<&mut Box<dyn std::any::Any>>,
) -> Value {
    response
}
