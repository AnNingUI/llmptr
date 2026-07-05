//! OpenAI Responses → Codex request translation.
//! Applies Codex-specific name shortening (64-char limit) to function names
//! and template defaults (instructions, stream, type:message).

use serde_json::{Value, json};
use llmptr_infra::util::{build_short_name_map, shorten_name_if_needed};

pub fn transform(model: &str, body: Value, stream: bool) -> Value {
    normalize(model, body, stream)
}

pub fn normalize(model: &str, body: Value, _stream: bool) -> Value {
    let mut out = body;

    // Normalize model
    if out
        .get("model")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .is_empty()
    {
        out["model"] = json!(model);
    }

    // Codex-specific defaults (matching Go's base template behaviour).
    out["parallel_tool_calls"] = json!(true);
    out["reasoning"] = json!({
        "summary": "auto",
        "effort": out.get("reasoning")
            .and_then(|r| r.get("effort"))
            .and_then(|v| v.as_str())
            .unwrap_or("medium"),
    });
    out["include"] = json!(["reasoning.encrypted_content"]);
    out["store"] = json!(false);

    // Go always ships `"instructions": ""` and `"stream": true` in Codex template.
    if !out
        .as_object()
        .map(|o| o.contains_key("instructions"))
        .unwrap_or(false)
    {
        out["instructions"] = json!("");
    }
    if !out
        .as_object()
        .map(|o| o.contains_key("stream"))
        .unwrap_or(false)
    {
        out["stream"] = json!(true);
    }

    // Go adds `"type": "message"` to each input item that has a role.
    if let Some(input) = out.get_mut("input").and_then(|v| v.as_array_mut()) {
        for item in input.iter_mut() {
            if !item
                .as_object()
                .map(|o| o.contains_key("type"))
                .unwrap_or(false)
                && item.get("role").is_some()
            {
                item["type"] = json!("message");
            }
        }
    }

    // Apply name shortening to tools
    if let Some(tools) = out.get("tools").and_then(|v| v.as_array()) {
        let mut orig_names: Vec<String> = Vec::new();
        for tool in tools {
            if let Some(name) = tool.get("name").and_then(|v| v.as_str()) {
                orig_names.push(name.to_string());
            }
        }
        if !orig_names.is_empty() {
            let name_map = build_short_name_map(&orig_names);
            if let Some(tools_mut) = out.get_mut("tools").and_then(|v| v.as_array_mut()) {
                for tool in tools_mut.iter_mut() {
                    if let Some(name) = tool.get("name").and_then(|v| v.as_str()) {
                        if let Some(short) = name_map.get(name) {
                            tool["name"] = json!(short);
                        } else {
                            let s = shorten_name_if_needed(name);
                            tool["name"] = json!(s);
                        }
                    }
                }
            }
        }
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    #[test]
    fn test_resp_to_codex() {
        let body = json!({"input":"Hello"});
        let r = transform("gpt-4", body, false);
        assert_eq!(r["model"], "gpt-4");
        assert_eq!(r["instructions"], "");
        assert_eq!(r["stream"], true);
    }
}
