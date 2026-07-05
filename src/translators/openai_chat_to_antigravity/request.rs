//! OpenAI Chat → Antigravity request translation.
//! Antigravity format is identical to Gemini, so delegates to the Gemini translator.

use serde_json::Value;

pub fn transform(model: &str, body: Value, stream: bool) -> Value {
    // Antigravity uses the same format as Gemini
    crate::translators::openai_chat_to_gemini::request::transform(model, body, stream)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    #[test]
    fn test_basic() {
        let b = json!({"messages":[{"role":"user","content":"Hello"}]});
        let r = transform("antigravity-model", b, false);
        assert_eq!(r["contents"][0]["parts"][0]["text"], "Hello");
    }
}
