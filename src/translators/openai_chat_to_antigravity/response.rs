//! OpenAI Chat → Antigravity response translation.
//! Delegates to Gemini response translator since format is identical.

use serde_json::Value;

pub fn transform_non_stream(
    model: &str,
    a: &Value,
    b: &Value,
    c: Value,
    d: Option<&mut Box<dyn std::any::Any>>,
) -> Value {
    crate::translators::openai_chat_to_gemini::response::transform_non_stream(model, a, b, c, d)
}

pub fn transform_stream(
    model: &str,
    a: &Value,
    b: &Value,
    c: Value,
    d: Option<&mut Box<dyn std::any::Any>>,
) -> Vec<Value> {
    crate::translators::openai_chat_to_gemini::response::transform_stream(model, a, b, c, d)
}
