//! Antigravity 鈫?Gemini request translation.
//! Antigravity shares the same format as Gemini. But we need to extract from
//! `request.contents` (Antigravity wrapper) rather than `contents`.

use serde_json::{Value, json};

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
    crate::translators::gemini_to_gemini::request::normalize(model, unwrapped, stream)
}

/// Wrap a Gemini request in the Antigravity request envelope.
pub fn wrap_gemini_request(model: &str, body: Value, stream: bool) -> Value {
    let mut request = crate::translators::gemini_to_gemini::request::normalize(model, body, stream);

    if let Some(system_instruction) = request.get("system_instruction").cloned() {
        request["systemInstruction"] = system_instruction;
        if let Some(obj) = request.as_object_mut() {
            obj.remove("system_instruction");
        }
    }
    if let Some(obj) = request.as_object_mut() {
        obj.remove("model");
    }

    // Attach default safety settings (matching Go's AttachDefaultSafetySettings).
    if !request
        .as_object()
        .map(|o| o.contains_key("safetySettings"))
        .unwrap_or(false)
    {
        request["safetySettings"] = json!([
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "OFF"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "OFF"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "OFF"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "OFF"},
            {"category": "HARM_CATEGORY_CIVIC_INTEGRITY", "threshold": "BLOCK_NONE"},
        ]);
    }

    json!({
        "project": "",
        "model": model,
        "request": request,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn wraps_gemini_request_for_antigravity() {
        let body = json!({
            "model": "client-model",
            "system_instruction": {"parts": [{"text": "sys"}]},
            "contents": [{"parts": [{"text": "Hi"}]}]
        });

        let result = wrap_gemini_request("antigravity-model", body, false);

        assert_eq!(result["model"], "antigravity-model");
        assert!(result["request"]["model"].is_null());
        assert_eq!(
            result["request"]["systemInstruction"]["parts"][0]["text"],
            "sys"
        );
        assert_eq!(result["request"]["contents"][0]["role"], "user");
    }
}
