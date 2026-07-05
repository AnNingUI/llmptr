//! OpenAI Responses → Antigravity request translation.
//! Delegates to OpenAIResponse→Gemini translator.

use serde_json::Value;

pub fn transform(model: &str, body: Value, stream: bool) -> Value {
    crate::translators::openai_response_to_gemini::request::transform(model, body, stream)
}

pub fn normalize(model: &str, body: Value, stream: bool) -> Value {
    crate::translators::openai_response_to_gemini::request::normalize(model, body, stream)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    #[test]
    fn test_basic() {
        let b = json!({"input":"Hello","model":"gpt-4"});
        let r = transform("gemini-pro", b, false);
        assert_eq!(r["contents"][0]["parts"][0]["text"], "Hello");
    }
}
