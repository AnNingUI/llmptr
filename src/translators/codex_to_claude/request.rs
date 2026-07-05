//! Codex → Claude request translation.
//! Codex uses OpenAI Responses format. Convert to Claude Messages format.

use serde_json::Value;

pub fn transform(model: &str, body: Value, stream: bool) -> Value {
    crate::translators::openai_response_to_claude::request::transform(model, body, stream)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_codex_to_claude() {
        let body = json!({"input":[{"role":"user","content":[{"type":"input_text","text":"Hi"}]}],"model":"gpt-4"});
        let r = transform("claude-sonnet-4-20250514", body, false);
        assert!(r.get("messages").is_some());
    }
}
