//! Codex → OpenAI Responses request translation.
//! Codex natively uses the OpenAI Responses API format, so this is largely passthrough
//! with format normalization.

use serde_json::{Value, json};

pub fn transform(model: &str, body: Value, _stream: bool) -> Value {
    // Codex is already in Responses API format - just normalize
    let mut out = body;
    if out
        .get("model")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .is_empty()
    {
        out["model"] = json!(model);
    }
    if out.get("input").is_none() {
        out["input"] = json!([]);
    }
    out
}

pub fn normalize(model: &str, body: Value, stream: bool) -> Value {
    transform(model, body, stream)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    #[test]
    fn test_codex_to_responses() {
        let body = json!({"input":"Hello","model":"gpt-4"});
        let r = transform("gpt-4", body, false);
        assert_eq!(r["model"], "gpt-4");
        assert_eq!(r["input"], "Hello");
    }
    #[test]
    fn test_codex_to_responses_empty_model() {
        let body = json!({"input":"Hi"});
        let r = transform("gpt-4", body, false);
        assert_eq!(r["model"], "gpt-4");
    }
}
