//! Gemini → Codex request translation.
//!
//! Port of Go''s `ConvertGeminiRequestToCodex`. Composes Gemini → OpenAI Chat →
//! Codex. Common Codex template defaults (instructions, stream, type:message)
//! are now applied by `openai_response_to_codex::normalize`.

use serde_json::Value;

pub fn transform(model: &str, body: Value, _stream: bool) -> Value {
    let chat = crate::translators::gemini_to_openai_chat::request::transform(model, body, false);
    crate::translators::openai_chat_to_codex::request::transform(model, chat, true)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_gemini_to_codex() {
        let body = json!({"contents":[{"role":"user","parts":[{"text":"Hello"}]}]});
        let r = transform("gpt-4", body, false);
        assert!(r.get("model").is_some());
        assert_eq!(r["instructions"], "");
        assert_eq!(r["stream"], true);
    }

    #[test]
    fn test_input_items_have_type_message() {
        let body = json!({"contents":[{"role":"user","parts":[{"text":"Hello"}]}]});
        let r = transform("gpt-4", body, false);
        let input_item = &r["input"][0];
        assert_eq!(input_item["type"], "message");
    }
}
