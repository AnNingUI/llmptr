//! Antigravity → OpenAI Chat request translation.
//! Antigravity format is identical to Gemini, so delegates to Gemini→OpenAI Chat translator.

use serde_json::Value;

pub fn transform(model: &str, body: Value, stream: bool) -> Value {
    crate::translators::gemini_to_openai_chat::request::transform(model, body, stream)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    #[test]
    fn test_basic() {
        let b = json!({"contents":[{"role":"user","parts":[{"text":"Hello"}]}]});
        let r = transform("gpt-4", b, false);
        assert_eq!(r["messages"][0]["role"], "user");
    }
}
