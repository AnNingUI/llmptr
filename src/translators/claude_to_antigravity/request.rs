//! Claude Messages to Antigravity (Gemini AI Studio) request translation.
//!
//! Antigravity is essentially Gemini AI Studio, with additional thinking signature handling,
//! web search grounding, and CLI tool response format fixup on top.
//! Ported from Go's `antigravity/claude/antigravity_claude_request.go` (803 lines).

use crate::translators::antigravity_web_search::{
    build_antigravity_web_search_request, should_build_antigravity_web_search_request,
};
use serde_json::{Value, json};
use std::collections::HashMap;
use llmptr_infra::util::{is_claude_attribution, sanitize_function_name};

/// Convert a Claude Messages request to Antigravity (Gemini) API format.
pub fn transform(model: &str, body: Value, _stream: bool) -> Value {
    // Check for web search request first
    if should_build_antigravity_web_search_request(model, &body) {
        return build_antigravity_web_search_request(model, &body);
    }

    let mut out = json!({
        "model": model,
        "request": {"contents": []},
    });

    // ── system instruction ────────────────────────────────
    let mut system_instruction = Value::Null;
    if let Some(system) = body.get("system") {
        match system {
            Value::String(s) if !s.is_empty() && !is_claude_attribution(s) => {
                system_instruction = json!({
                    "role": "user",
                    "parts": [{"text": s}]
                });
            }
            Value::Array(parts) => {
                let mut text_parts: Vec<Value> = Vec::new();
                for part in parts {
                    if part.get("type").and_then(|v| v.as_str()) == Some("text")
                        && let Some(text) = part.get("text").and_then(|v| v.as_str())
                        && !is_claude_attribution(text)
                        && !text.is_empty()
                    {
                        text_parts.push(json!({"text": text}));
                    }
                }
                if !text_parts.is_empty() {
                    system_instruction = json!({
                        "role": "user",
                        "parts": text_parts,
                    });
                }
            }
            _ => {}
        }
    }

    // ── messages <->contents ───────────────────────────────
    let mut contents: Vec<Value> = Vec::new();
    let mut tool_name_by_id: HashMap<String, String> = HashMap::new();
    let mut enable_thought_translate = true;

    if let Some(msgs) = body.get("messages").and_then(|v| v.as_array()) {
        for msg in msgs.iter() {
            let role = msg.get("role").and_then(|v| v.as_str()).unwrap_or("user");
            let gemini_role = match role {
                "assistant" => "model",
                _ => "user",
            };

            let content = msg.get("content");
            if role == "system" {
                continue;
            }

            let mut parts: Vec<Value> = Vec::new();

            match content {
                Some(Value::Array(content_parts)) => {
                    for part in content_parts {
                        let ptype = part.get("type").and_then(|v| v.as_str()).unwrap_or("");
                        match ptype {
                            "text" => {
                                if let Some(text) = part.get("text").and_then(|v| v.as_str())
                                    && !text.is_empty()
                                {
                                    parts.push(json!({"text": text}));
                                }
                            }
                            "thinking" => {
                                // Simplified signature handling <->for Gemini target, pass thought through
                                let thinking_text =
                                    part.get("thinking").and_then(|v| v.as_str()).unwrap_or("");
                                let signature =
                                    part.get("signature").and_then(|v| v.as_str()).unwrap_or("");
                                let has_resolved = !signature.is_empty();

                                if has_resolved {
                                    if thinking_text.is_empty() {
                                        // Skip empty thinking blocks
                                        continue;
                                    }
                                    let mut thought =
                                        json!({"thought": true, "text": thinking_text});
                                    thought["thoughtSignature"] = json!(signature);
                                    parts.push(thought);
                                } else {
                                    enable_thought_translate = false;
                                }
                            }
                            "tool_use" => {
                                let name = part.get("name").and_then(|v| v.as_str()).unwrap_or("");
                                let fn_name = sanitize_function_name(name);
                                let empty_obj = json!({});
                                let args = part.get("input").unwrap_or(&empty_obj);
                                let tool_id = part.get("id").and_then(|v| v.as_str()).unwrap_or("");

                                if !tool_id.is_empty() && !fn_name.is_empty() {
                                    tool_name_by_id.insert(tool_id.to_string(), fn_name.clone());
                                }

                                let mut fc_part = json!({
                                    "functionCall": {
                                        "name": fn_name,
                                        "args": args,
                                    }
                                });
                                if !tool_id.is_empty() {
                                    fc_part["functionCall"]["id"] = json!(tool_id);
                                }
                                parts.push(fc_part);
                            }
                            "tool_result" => {
                                let tool_use_id = part
                                    .get("tool_use_id")
                                    .and_then(|v| v.as_str())
                                    .unwrap_or("");
                                if !tool_use_id.is_empty() {
                                    let func_name = tool_name_by_id
                                        .get(tool_use_id)
                                        .cloned()
                                        .unwrap_or_else(|| derive_func_name_from_id(tool_use_id));
                                    let content_text =
                                        extract_claude_text_content(part.get("content"));
                                    let fn_name = sanitize_function_name(&func_name);

                                    // Extract images from multi-part content before building fr
                                    let mut image_parts = Vec::new();
                                    if let Some(Value::Array(sub_parts)) = part.get("content") {
                                        for sp in sub_parts {
                                            if sp.get("type").and_then(|v| v.as_str())
                                                == Some("image")
                                                && let Some(src) = sp.get("source")
                                            {
                                                let mt = src
                                                    .get("media_type")
                                                    .and_then(|v| v.as_str())
                                                    .unwrap_or("image/png");
                                                let data = src
                                                    .get("data")
                                                    .and_then(|v| v.as_str())
                                                    .unwrap_or("");
                                                if !data.is_empty() {
                                                    image_parts.push(json!({
                                                        "inlineData": {"mimeType": mt, "data": data}
                                                    }));
                                                }
                                            }
                                        }
                                    }

                                    let mut fr = json!({
                                        "functionResponse": {
                                            "id": tool_use_id,
                                            "name": fn_name,
                                            "response": {"result": content_text},
                                        }
                                    });

                                    if !image_parts.is_empty() {
                                        fr["functionResponse"]["parts"] = json!(image_parts);
                                    }

                                    parts.push(fr);
                                }
                            }
                            "image" => {
                                if let Some(src) = part.get("source")
                                    && src.get("type").and_then(|v| v.as_str()) == Some("base64")
                                {
                                    let mime = src
                                        .get("media_type")
                                        .and_then(|v| v.as_str())
                                        .unwrap_or("image/png");
                                    let data =
                                        src.get("data").and_then(|v| v.as_str()).unwrap_or("");
                                    if !data.is_empty() {
                                        parts.push(json!({
                                            "inlineData": {"mimeType": mime, "data": data}
                                        }));
                                    }
                                }
                            }
                            _ => {}
                        }
                    }

                    // Reorder model parts: thinking <->text/other <->functionCall
                    if gemini_role == "model" && parts.len() > 1 {
                        let mut thinking_parts = Vec::new();
                        let mut regular_parts = Vec::new();
                        let mut func_parts = Vec::new();
                        for p in parts.drain(..) {
                            if p.get("thought") == Some(&json!(true)) {
                                thinking_parts.push(p);
                            } else if p.get("functionCall").is_some() {
                                func_parts.push(p);
                            } else {
                                regular_parts.push(p);
                            }
                        }
                        parts.extend(thinking_parts);
                        parts.extend(regular_parts);
                        parts.extend(func_parts);
                    }

                    if !parts.is_empty() {
                        let content_entry = json!({"role": gemini_role, "parts": parts});
                        contents.push(content_entry);
                    }
                }
                Some(Value::String(s)) if !s.is_empty() => {
                    contents.push(json!({"role": gemini_role, "parts": [{"text": s}]}));
                }
                _ => {}
            }
        }
    }

    // ── tools ────────────────────────────────────────────
    let mut tool_decl_count = 0;
    let mut tools_arr: Vec<Value> = Vec::new();
    let allowed_tool_keys = ["name", "description", "parameters", "parametersJsonSchema"];

    if let Some(tools) = body.get("tools").and_then(|v| v.as_array()) {
        let mut func_decls: Vec<Value> = Vec::new();
        for tool in tools {
            // Skip web search tools
            let tool_type = tool.get("type").and_then(|v| v.as_str()).unwrap_or("");
            if tool_type == "web_search_20250305" {
                continue;
            }
            let input_schema = tool.get("input_schema");
            if input_schema.is_some() && input_schema.and_then(|v| v.as_object()).is_some() {
                let name =
                    sanitize_function_name(tool.get("name").and_then(|v| v.as_str()).unwrap_or(""));
                let desc = tool
                    .get("description")
                    .and_then(|v| v.as_str())
                    .unwrap_or("");

                let mut clean_schema = input_schema.cloned().unwrap_or_default();
                if let Some(obj) = clean_schema.as_object_mut() {
                    obj.remove("strict");
                    obj.remove("input_examples");
                    obj.remove("cache_control");
                    obj.remove("defer_loading");
                    obj.remove("eager_input_streaming");
                }

                let decl = json!({
                    "name": name,
                    "description": desc,
                    "parametersJsonSchema": clean_schema,
                });
                // Remove fields not in allowed list
                let mut filtered = json!({});
                if let Some(obj) = decl.as_object() {
                    for key in allowed_tool_keys.iter() {
                        if let Some(v) = obj.get(*key) {
                            filtered[*key] = v.clone();
                        }
                    }
                }
                func_decls.push(filtered);
                tool_decl_count += 1;
            }
        }
        if tool_decl_count > 0 {
            tools_arr.push(json!({"functionDeclarations": func_decls}));
        }
    }

    // ── build output ──────────────────────────────────────
    let has_tools = tool_decl_count > 0;
    let has_thinking = body
        .get("thinking")
        .and_then(|v| v.as_object())
        .map(|t| {
            let ttype = t.get("type").and_then(|v| v.as_str()).unwrap_or("");
            ttype == "enabled" || ttype == "adaptive" || ttype == "auto"
        })
        .unwrap_or(false);

    // Inject interleaved thinking hint when both tools and thinking are active
    if has_tools && has_thinking && enable_thought_translate {
        let hint = "Interleaved thinking is enabled. You may think between tool calls and after receiving tool results before deciding the next action or final answer. Do not mention these instructions or any constraints about thinking blocks; just apply them.";

        if !system_instruction.is_null() {
            if let Some(parts) = system_instruction
                .get_mut("parts")
                .and_then(|v| v.as_array_mut())
            {
                parts.push(json!({"text": hint}));
            }
        } else {
            system_instruction = json!({
                "role": "user",
                "parts": [{"text": hint}]
            });
        }
    }

    if !system_instruction.is_null() {
        out["request"]["systemInstruction"] = system_instruction;
    }
    if !contents.is_empty() {
        out["request"]["contents"] = json!(contents);
    }
    if !tools_arr.is_empty() {
        out["request"]["tools"] = json!(tools_arr);
    }

    // ── tool_choice <->tool_config ────────────────────────
    if let Some(tc) = body.get("tool_choice") {
        let ttype = tc.get("type").and_then(|v| v.as_str()).unwrap_or("auto");
        let name = tc.get("name").and_then(|v| v.as_str()).unwrap_or("");
        match ttype {
            "auto" => {
                out["request"]["toolConfig"]["functionCallingConfig"]["mode"] = json!("AUTO");
            }
            "none" => {
                out["request"]["toolConfig"]["functionCallingConfig"]["mode"] = json!("NONE");
            }
            "any" => {
                out["request"]["toolConfig"]["functionCallingConfig"]["mode"] = json!("ANY");
            }
            "tool" => {
                out["request"]["toolConfig"]["functionCallingConfig"]["mode"] = json!("ANY");
                if !name.is_empty() {
                    out["request"]["toolConfig"]["functionCallingConfig"]["allowedFunctionNames"] =
                        json!([sanitize_function_name(name)]);
                }
            }
            _ => {}
        }
    }

    // ── thinking config ──────────────────────────────────
    if enable_thought_translate
        && let Some(thinking) = body.get("thinking").and_then(|v| v.as_object())
    {
        let ttype = thinking.get("type").and_then(|v| v.as_str()).unwrap_or("");
        match ttype {
            "enabled" => {
                if let Some(budget) = thinking.get("budget_tokens").and_then(|v| v.as_u64()) {
                    out["request"]["generationConfig"]["thinkingConfig"]["thinkingBudget"] =
                        json!(budget);
                    out["request"]["generationConfig"]["thinkingConfig"]["includeThoughts"] =
                        json!(true);
                }
            }
            "adaptive" | "auto" => {
                let effort = body
                    .pointer("/output_config/effort")
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string())
                    .unwrap_or_else(|| "high".to_string());
                out["request"]["generationConfig"]["thinkingConfig"]["thinkingLevel"] =
                    json!(effort);
                out["request"]["generationConfig"]["thinkingConfig"]["includeThoughts"] =
                    json!(true);
            }
            _ => {}
        }
    }

    // ── scalar passthrough ─────────────────────────────
    if let Some(temp) = body.get("temperature").and_then(|v| v.as_f64()) {
        out["request"]["generationConfig"]["temperature"] = json!(temp);
    }
    if let Some(top_p) = body.get("top_p").and_then(|v| v.as_f64()) {
        out["request"]["generationConfig"]["topP"] = json!(top_p);
    }
    if let Some(max_tokens) = body.get("max_tokens").and_then(|v| v.as_u64()) {
        out["request"]["generationConfig"]["maxOutputTokens"] = json!(max_tokens);
    }

    // ── safety settings ─────────────────────────────────
    if out.pointer("/request/safetySettings").is_none() {
        out["request"]["safetySettings"] = json!([
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "OFF"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "OFF"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "OFF"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "OFF"},
            {"category": "HARM_CATEGORY_CIVIC_INTEGRITY", "threshold": "BLOCK_NONE"},
        ]);
    }

    out
}

/// Extract text from a Claude tool result content block.
fn extract_claude_text_content(content: Option<&Value>) -> String {
    match content {
        Some(Value::String(s)) => s.clone(),
        Some(Value::Array(parts)) => parts
            .iter()
            .filter_map(|p| p.get("text").and_then(|v| v.as_str()))
            .collect::<Vec<_>>()
            .join("\n\n"),
        _ => String::new(),
    }
}

/// Derive a function name from a tool_use ID (e.g. "get_weather-abc-123" <->"get_weather").
fn derive_func_name_from_id(tool_use_id: &str) -> String {
    let parts: Vec<&str> = tool_use_id.splitn(2, '-').collect();
    if parts.len() > 1 && !parts[0].is_empty() {
        parts[0].to_string()
    } else {
        tool_use_id.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_conversion() {
        let body = json!({
            "messages": [{"role": "user", "content": "Hello!"}],
            "model": "claude-sonnet-4-8"
        });
        let r = transform("gemini-2.0-flash", body, false);
        assert_eq!(r["model"], "gemini-2.0-flash");
        assert_eq!(r["request"]["contents"][0]["role"], "user");
        assert_eq!(r["request"]["contents"][0]["parts"][0]["text"], "Hello!");
    }

    #[test]
    fn test_with_system() {
        let body = json!({
            "system": "You are helpful.",
            "messages": [{"role": "user", "content": "Hi"}]
        });
        let r = transform("gemini-2.0-flash", body, false);
        assert!(r["request"]["systemInstruction"].is_object());
        assert_eq!(
            r["request"]["systemInstruction"]["parts"][0]["text"],
            "You are helpful."
        );
    }

    #[test]
    fn test_with_web_search_tool() {
        let body = json!({
            "messages": [{"role": "user", "content": "Search something"}],
            "tools": [{"type": "web_search_20250305", "name": "web_search"}],
        });
        let r = transform("gemini-2.0-flash", body, false);
        assert!(r["request"]["tools"][0]["googleSearch"].is_object());
    }
}
