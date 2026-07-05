//! Codex → Gemini request translation.
//! Codex uses OpenAI Responses format. First convert to Chat format, then to Gemini.

use serde_json::Value;

pub fn transform(model: &str, body: Value, _stream: bool) -> Value {
    // First convert Codex (Responses) to OpenAI Chat, then to Gemini
    let chat =
        crate::translators::openai_response_to_openai_chat::request::transform(model, body, false);
    crate::translators::openai_chat_to_gemini::request::transform(model, chat, false)
}

pub fn normalize(model: &str, body: Value, stream: bool) -> Value {
    // Codex→Gemini also normalizes roles
    let gemini = transform(model, body, stream);
    crate::translators::gemini_to_gemini::request::normalize(model, gemini, stream)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    #[test]
    fn test_codex_to_gemini() {
        let body = json!({"input":[{"role":"user","content":[{"type":"input_text","text":"Hi"}]}],"model":"gpt-4"});
        let r = transform("gemini-pro", body, false);
        assert!(r.get("contents").is_some());
    }
}
