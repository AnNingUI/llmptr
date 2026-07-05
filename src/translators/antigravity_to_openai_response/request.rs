//! Antigravity → OpenAI Responses request translation.
//! Antigravity uses Gemini format, delegates to Gemini→Responses translator.

use serde_json::Value;

pub fn transform(model: &str, body: Value, stream: bool) -> Value {
    crate::translators::gemini_to_openai_response::request::transform(model, body, stream)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    #[test]
    fn test_basic() {
        let b = json!({"contents":[{"role":"user","parts":[{"text":"Hi"}]}]});
        let r = transform("gpt-4", b, false);
        assert_eq!(r["input"][0]["content"][0]["text"], "Hi");
    }
}
