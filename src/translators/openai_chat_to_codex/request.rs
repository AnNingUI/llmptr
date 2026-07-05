//! OpenAI Chat Completions → Codex (OpenAI Responses format) request translation.

use serde_json::Value;

/// Convert an OpenAI Chat request to Codex (Responses API) format.
pub fn transform(model: &str, body: Value, _stream: bool) -> Value {
    // First convert Chat to Responses, then normalize
    let resp =
        crate::translators::openai_chat_to_openai_response::request::transform(model, body, true);
    crate::translators::openai_response_to_codex::request::normalize(model, resp, true)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    #[test]
    fn test_chat_to_codex_basic() {
        let b = json!({"messages":[{"role":"system","content":"You are helpful."},{"role":"user","content":"Hi"}],"model":"gpt-4"});
        let r = transform("gpt-4", b, false);
        assert_eq!(r["model"], "gpt-4");
    }
}
