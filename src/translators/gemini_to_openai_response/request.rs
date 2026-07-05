//! Gemini → OpenAI Responses request translation.

use serde_json::{Value, json};

pub fn transform(model: &str, body: Value, _stream: bool) -> Value {
    let mut out = json!({
        "model": model,
        "input": [],
    });

    if let Some(si) = body.get("system_instruction").and_then(|v| v.as_object())
        && let Some(parts) = si.get("parts").and_then(|v| v.as_array())
    {
        let texts: Vec<String> = parts
            .iter()
            .filter_map(|p| p.get("text").and_then(|v| v.as_str()))
            .filter(|s| !s.is_empty())
            .map(|s| s.to_string())
            .collect();
        if !texts.is_empty() {
            out["instructions"] = json!(texts.join("\n"));
        }
    }

    if let Some(contents) = body.get("contents").and_then(|v| v.as_array()) {
        for content in contents {
            let role = content
                .get("role")
                .and_then(|v| v.as_str())
                .unwrap_or("user");
            let resp_role = match role {
                "model" => "assistant",
                r => r,
            };

            if let Some(parts) = content.get("parts").and_then(|v| v.as_array()) {
                let mut content_arr: Vec<Value> = Vec::new();
                for part in parts {
                    if let Some(t) = part.get("text").and_then(|v| v.as_str()) {
                        content_arr.push(json!({"type": "input_text", "text": t}));
                    }
                    if let Some(id) = part.get("inline_data").and_then(|v| v.as_object()) {
                        let media = id.get("mime_type").and_then(|v| v.as_str()).unwrap_or("");
                        let data = id.get("data").and_then(|v| v.as_str()).unwrap_or("");
                        if !data.is_empty() {
                            content_arr.push(json!({
                                "type": "input_image",
                                "image_url": format!("data:{};base64,{}", media, data),
                            }));
                        }
                    }
                }

                if !content_arr.is_empty() {
                    let mut item = json!({"role": resp_role, "content": content_arr});
                    // Check for functionCall parts
                    for part in parts {
                        if let Some(fc) = part.get("functionCall").and_then(|v| v.as_object()) {
                            let name = fc.get("name").and_then(|v| v.as_str()).unwrap_or("");
                            let empty_obj = json!({});
                            let args = fc.get("args").unwrap_or(&empty_obj);
                            item["tool_calls"] = json!([{
                                "id": fc.get("id").and_then(|v| v.as_str()).unwrap_or(""),
                                "type": "function",
                                "function": {"name": name, "arguments": args.to_string()},
                            }]);
                        }
                    }
                    out["input"].as_array_mut().unwrap().push(item);
                }
            }
        }
    }

    if let Some(gc) = body.get("generationConfig").and_then(|v| v.as_object())
        && let Some(t) = gc.get("maxOutputTokens").and_then(|v| v.as_u64())
    {
        out["max_tokens"] = json!(t);
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_gemini_to_resp() {
        let body = json!({"contents": [{"role": "user", "parts": [{"text": "Hi"}]}]});
        let r = transform("gpt-4", body, false);
        assert_eq!(r["input"][0]["content"][0]["text"], "Hi");
    }
}
