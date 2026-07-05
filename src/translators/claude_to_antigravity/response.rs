//! Claude Messages → Antigravity (Gemini AI Studio) response translation.
//! Delegates to Gemini→Claude response translation since the response format is identical.

use serde_json::Value;

pub fn transform_non_stream(
    m: &str,
    a: &Value,
    b: &Value,
    c: Value,
    d: Option<&mut Box<dyn std::any::Any>>,
) -> Value {
    crate::translators::antigravity_to_claude::response::transform_non_stream(m, a, b, c, d)
}

pub fn transform_stream(
    m: &str,
    a: &Value,
    b: &Value,
    c: Value,
    d: Option<&mut Box<dyn std::any::Any>>,
) -> Vec<Value> {
    crate::translators::antigravity_to_claude::response::transform_stream(m, a, b, c, d)
}
