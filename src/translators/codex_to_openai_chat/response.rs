//! Codex → OpenAI Chat response translation.

use serde_json::Value;

/// Transform a non-streaming Codex response to OpenAI Chat format.
pub fn transform_non_stream(
    model: &str,
    _orig: &Value,
    _trans: &Value,
    response: Value,
    _param: Option<&mut Box<dyn std::any::Any>>,
) -> Value {
    crate::translators::openai_response_to_openai_chat::response::transform_non_stream(
        model, _orig, _trans, response, _param,
    )
}

/// Transform streaming Codex SSE to OpenAI Chat SSE.
pub fn transform_stream(
    model: &str,
    _orig: &Value,
    _trans: &Value,
    chunk: Value,
    _param: Option<&mut Box<dyn std::any::Any>>,
) -> Vec<Value> {
    crate::translators::openai_response_to_openai_chat::response::transform_stream(
        model, _orig, _trans, chunk, _param,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    #[test]
    fn test_basic() {
        let resp = json!({"id":"resp_1","model":"gpt-4","output":[{"type":"message","role":"assistant","content":[{"type":"output_text","text":"Hi"}],"status":"completed"}],"usage":{"input_tokens":5,"output_tokens":3}});
        let r = transform_non_stream("gpt-4", &json!({}), &json!({}), resp, None);
        assert_eq!(r["choices"][0]["message"]["content"], "Hi");
    }
}
