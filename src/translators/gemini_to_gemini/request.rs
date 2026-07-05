//! Gemini → Gemini (self-normalizer).
//! Normalizes role values, ensures contents array, adds default generationConfig.

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
    if out.get("contents").is_none() {
        out["contents"] = json!([]);
    }
    if out.get("generationConfig").is_none() {
        out["generationConfig"] = json!({});
    }

    // Normalize roles and backfill empty functionResponse names
    let contents_len = out["contents"].as_array().map(|a| a.len()).unwrap_or(0);
    if contents_len > 0 {
        // First pass: collect model-turn functionCall names per content index
        let mut model_call_names: Vec<Vec<String>> = Vec::new();
        for ci in 0..contents_len {
            let role = out["contents"][ci]["role"]
                .as_str()
                .unwrap_or("user")
                .to_lowercase();
            if role == "assistant" {
                let parts_len = out["contents"][ci]["parts"]
                    .as_array()
                    .map(|a| a.len())
                    .unwrap_or(0);
                let mut names = Vec::new();
                for pi in 0..parts_len {
                    if let Some(fc) = out["contents"][ci]["parts"][pi]
                        .get("functionCall")
                        .and_then(|v| v.as_object())
                        && let Some(name) = fc.get("name").and_then(|v| v.as_str())
                    {
                        names.push(name.to_string());
                    }
                }
                model_call_names.push(names);
            } else {
                model_call_names.push(Vec::new());
            }
        }

        // Second pass: normalize roles and backfill
        let mut pending_call_names: Vec<String> = Vec::new();
        for ci in 0..contents_len {
            let role = out["contents"][ci]["role"]
                .as_str()
                .unwrap_or("user")
                .to_lowercase();
            let normalized = match role.as_str() {
                "assistant" => "model",
                _ => "user",
            };
            out["contents"][ci]["role"] = json!(normalized);

            if normalized == "model"
                && ci < model_call_names.len()
                && !model_call_names[ci].is_empty()
            {
                pending_call_names = model_call_names[ci].clone();
            }

            if normalized == "user" && !pending_call_names.is_empty() {
                let parts_len = out["contents"][ci]["parts"]
                    .as_array()
                    .map(|a| a.len())
                    .unwrap_or(0);
                for pi in 0..parts_len {
                    let fr_name = out["contents"][ci]["parts"][pi]
                        .pointer("/functionResponse/name")
                        .and_then(|v| v.as_str())
                        .unwrap_or("");
                    if fr_name.trim().is_empty()
                        && out["contents"][ci]["parts"][pi]
                            .pointer("/functionResponse")
                            .is_some()
                        && pi < pending_call_names.len()
                    {
                        out["contents"][ci]["parts"][pi]["functionResponse"]["name"] =
                            json!(pending_call_names[pi].clone());
                    }
                }
                pending_call_names.clear();
            }
        }
    }

    out
}

pub fn passthrough_stream(
    _model: &str,
    _orig: &Value,
    _trans: &Value,
    chunk: Value,
    _param: Option<&mut Box<dyn std::any::Any>>,
) -> Vec<Value> {
    vec![chunk]
}

pub fn passthrough_non_stream(
    _model: &str,
    _orig: &Value,
    _trans: &Value,
    response: Value,
    _param: Option<&mut Box<dyn std::any::Any>>,
) -> Value {
    response
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gemini_normalize_roles() {
        let body = json!({
            "contents": [
                {"role": "assistant", "parts": [{"text": "Hi"}]},
                {"role": "user", "parts": [{"text": "Hello"}]}
            ]
        });
        let result = normalize("gemini-2.0-flash", body, false);
        assert_eq!(result["contents"][0]["role"], "model");
        assert_eq!(result["contents"][1]["role"], "user");
    }
}
