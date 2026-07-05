//! Antigravity → OpenAI Chat response translation.
//! Delegates to Gemini→OpenAI Chat.

use serde_json::Value;

pub fn transform_non_stream(
    m: &str,
    a: &Value,
    b: &Value,
    c: Value,
    d: Option<&mut Box<dyn std::any::Any>>,
) -> Value {
    crate::translators::gemini_to_openai_chat::response::transform_non_stream(m, a, b, c, d)
}
pub fn transform_stream(
    m: &str,
    a: &Value,
    b: &Value,
    c: Value,
    d: Option<&mut Box<dyn std::any::Any>>,
) -> Vec<Value> {
    crate::translators::gemini_to_openai_chat::response::transform_stream(m, a, b, c, d)
}
