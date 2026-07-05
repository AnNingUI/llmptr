//! OpenAI Response → OpenAI Response (self-normalizer).

use serde_json::Value;

pub fn normalize(model: &str, body: Value, _stream: bool) -> Value {
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

// Use json import
use serde_json::json;

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_normalize() {
        let b = json!({"input": "hi"});
        let r = normalize("gpt-4", b, false);
        assert_eq!(r["model"], "gpt-4");
    }
}
