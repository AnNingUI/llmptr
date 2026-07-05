use serde_json::Value;

/// Transform Codex (Responses format) request to OpenAI Chat format.
pub fn transform(model: &str, body: Value, stream: bool) -> Value {
    crate::translators::openai_response_to_openai_chat::request::transform(model, body, stream)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    #[test]
    fn test_basic() {
        let b = json!({"input":"Hello","model":"gpt-4"});
        let r = transform("gpt-4", b, false);
        assert_eq!(r["messages"][0]["content"], "Hello");
    }
}
