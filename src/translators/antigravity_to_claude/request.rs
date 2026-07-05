//! Antigravity → Claude request translation.
//! Antigravity uses Gemini format with an outer `request` wrapper.
//! Unwraps request.contents → contents before delegating to Gemini→Claude.

use serde_json::Value;

pub fn transform(model: &str, body: Value, stream: bool) -> Value {
    let unwrapped = if body.get("request").is_some() {
        let request_fields = body.get("request").cloned().unwrap_or_default();
        let mut out = body;
        if let Some(obj) = request_fields.as_object() {
            for (k, v) in obj {
                out[k] = v.clone();
            }
        }
        out
    } else {
        body
    };
    crate::translators::gemini_to_claude::request::transform(model, unwrapped, stream)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    #[test]
    fn test_basic() {
        let b = json!({"contents":[{"role":"user","parts":[{"text":"Hello"}]}]});
        let r = transform("claude-sonnet", b, false);
        assert_eq!(r["messages"][0]["role"], "user");
    }
}
