//! OpenAI Chat -> OpenAI Chat request normalization.

use serde_json::{Value, json};

pub fn normalize(model: &str, body: Value, _stream: bool) -> Value {
    let mut out = body;
    if out.is_object() {
        out["model"] = json!(model);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sets_missing_model() {
        let body = json!({"messages": [{"role": "user", "content": "Hi"}]});
        let result = normalize("gpt-4", body, false);
        assert_eq!(result["model"], "gpt-4");
    }

    #[test]
    fn preserves_existing_fields() {
        let body = json!({"messages": [{"role": "user", "content": "Hi"}], "stream": true});
        let result = normalize("gpt-4", body, true);
        assert_eq!(result["stream"], true);
        assert!(result.get("messages").is_some());
    }

    #[test]
    fn overwrites_existing_model_without_adding_defaults() {
        let body = json!({"model": "old-model", "foo": true});
        let result = normalize("gpt-4", body, false);
        assert_eq!(result, json!({"model": "gpt-4", "foo": true}));
    }
}
