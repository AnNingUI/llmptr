//! Claude → Claude request normalization.
//!
//! Ensures minimum valid shape for Claude Messages API requests:
//! - model field is set
//! - messages array exists
//! - max_tokens has a sane default

use serde_json::{Value, json};

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

    if out.get("messages").is_none() {
        out["messages"] = json!([]);
    }

    if out.get("max_tokens").is_none() {
        out["max_tokens"] = json!(4096);
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sets_missing_model() {
        let body = json!({"messages": [{"role": "user", "content": "Hi"}]});
        let result = normalize("claude-sonnet-4-20250514", body, false);
        assert_eq!(result["model"], "claude-sonnet-4-20250514");
    }

    #[test]
    fn sets_missing_messages() {
        let body = json!({"model": "claude-sonnet"});
        let result = normalize("claude-sonnet-4-20250514", body, false);
        assert!(result.get("messages").is_some());
    }

    #[test]
    fn sets_missing_max_tokens() {
        let body = json!({"model": "claude-sonnet", "messages": []});
        let result = normalize("claude-sonnet-4-20250514", body, false);
        assert_eq!(result["max_tokens"], 4096);
    }

    #[test]
    fn preserves_existing_fields() {
        let body = json!({"model": "claude-opus", "messages": [{"role": "user", "content": "Hi"}], "max_tokens": 1024, "stream": true});
        let result = normalize("claude-sonnet-4-20250514", body, true);
        assert_eq!(result["max_tokens"], 1024);
        assert_eq!(result["stream"], true);
    }
}
