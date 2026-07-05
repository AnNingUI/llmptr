//! Claude Messages → Codex (OpenAI Responses format) request translation.
//!
//! Port of Go's `ConvertClaudeRequestToCodex` in `codex_claude_request.go`.
//! Composes through `claude_to_openai_response` + `openai_response_to_codex`
//! (which now applies all Codex template defaults), then strips Claude-specific
//! fields that Go doesn't forward (metadata, max_output_tokens).

use serde_json::Value;

/// Transform a Claude Messages API request to Codex format.
pub fn transform(model: &str, body: Value, _stream: bool) -> Value {
    // Compose Claude → OpenAI Responses → Codex normalisation.
    let mut out =
        crate::translators::claude_to_openai_response::request::transform(model, body, true);
    out = crate::translators::openai_response_to_codex::request::normalize(model, out, true);

    // ---- Claude-specific stripping (Go doesn't forward these) ----

    // Go does NOT forward `metadata` from Claude.
    if let Some(obj) = out.as_object_mut() {
        obj.remove("metadata");
    }

    // Go does NOT forward `max_tokens` → `max_output_tokens` in the Claude→Codex path.
    if let Some(obj) = out.as_object_mut() {
        obj.remove("max_output_tokens");
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_claude_to_codex_basic() {
        let b = json!({"messages":[{"role":"user","content":"Hello"}],"model":"claude"});
        let r = transform("gpt-4", b, false);
        assert_eq!(r["model"], "gpt-4");
        assert_eq!(r["instructions"], "");
    }

    #[test]
    fn test_no_max_output_tokens_forwarded() {
        let b = json!({"messages":[{"role":"user","content":"Hi"}],"max_tokens":32000});
        let r = transform("gpt-4", b, false);
        assert!(r.get("max_output_tokens").is_none());
    }

    #[test]
    fn test_no_metadata_forwarded() {
        let b = json!({"messages":[{"role":"user","content":"Hi"}],"metadata":{"user_id":"u1"}});
        let r = transform("gpt-4", b, false);
        assert!(r.get("metadata").is_none());
    }

    #[test]
    fn test_instructions_is_empty_without_system() {
        let b = json!({"messages":[{"role":"user","content":"Hello"}]});
        let r = transform("gpt-4", b, false);
        assert_eq!(r["instructions"], "");
    }
}
