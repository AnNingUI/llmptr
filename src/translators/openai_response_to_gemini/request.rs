//! OpenAI Responses -> Gemini request translation.

use serde_json::{Value, json};
use translator_infra::{
    signature::{self, SignatureBlockKind, SignatureProvider},
    util,
};

const GEMINI_RESPONSES_THOUGHT_SIGNATURE: &str = signature::GEMINI_SKIP_SENTINEL;

pub fn transform(model: &str, body: Value, _stream: bool) -> Value {
    let use_native_reasoning =
        signature::provider_from_model_name(model) == SignatureProvider::Gemini;
    let mut out = json!({"contents": []});

    if let Some(instructions) = body.get("instructions") {
        push_system_part(&mut out, value_to_string(instructions));
    }

    if let Some(input) = body.get("input") {
        match input {
            Value::Array(items) => append_input_items(&mut out, &body, items, use_native_reasoning),
            Value::String(text) => {
                out["contents"]
                    .as_array_mut()
                    .unwrap()
                    .push(json!({"role": "user", "parts": [{"text": text}]}));
            }
            _ => {}
        }
    }

    strip_trailing_model_prefill(&mut out);
    append_tools(&mut out, &body);
    apply_generation_config(&mut out, &body);
    apply_text_format(&mut out, &body);
    apply_reasoning_effort(&mut out, &body);
    attach_default_safety_settings(&mut out);
    out
}

pub fn normalize(_model: &str, body: Value, _stream: bool) -> Value {
    let mut out = body;
    if out
        .get("model")
        .and_then(Value::as_str)
        .unwrap_or("")
        .is_empty()
    {
        out["model"] = json!("unknown");
    }
    if out.get("input").is_none() {
        out["input"] = json!([]);
    }
    out
}

fn append_input_items(out: &mut Value, root: &Value, items: &[Value], use_native_reasoning: bool) {
    let normalized = normalize_function_call_order(items);
    let mut i = 0usize;
    while i < normalized.len() {
        let item = &normalized[i];
        let item_type = responses_item_type(item);
        match item_type.as_str() {
            "message" => {
                let role = item.get("role").and_then(Value::as_str).unwrap_or("");
                if role.eq_ignore_ascii_case("system") || role.eq_ignore_ascii_case("developer") {
                    append_message_to_system_instruction(out, item);
                } else {
                    append_message_contents(out, item);
                }
            }
            "function_call" => append_function_call(out, item),
            "function_call_output" => append_function_call_output(out, root, item),
            "reasoning" => {
                let thought_text = item
                    .pointer("/summary/0/text")
                    .and_then(Value::as_str)
                    .unwrap_or("");
                let signature = openai_responses_gemini_thought_signature(
                    item.get("encrypted_content")
                        .and_then(Value::as_str)
                        .unwrap_or(""),
                );
                let mut visible_text = String::new();
                if use_native_reasoning
                    && i + 1 < normalized.len()
                    && let Some(text) = assistant_visible_text(&normalized[i + 1])
                {
                    visible_text = text;
                    i += 1;
                }
                out["contents"]
                    .as_array_mut()
                    .unwrap()
                    .push(build_reasoning_model_content(
                        thought_text,
                        &visible_text,
                        &signature,
                        use_native_reasoning,
                    ));
            }
            _ => {}
        }
        i += 1;
    }
}

fn normalize_function_call_order(items: &[Value]) -> Vec<Value> {
    let mut normalized = Vec::with_capacity(items.len());
    let mut i = 0usize;
    while i < items.len() {
        let item_type = responses_item_type(&items[i]);
        if item_type == "function_call" {
            let mut calls = Vec::new();
            let mut outputs = Vec::new();
            while i < items.len() && responses_item_type(&items[i]) == "function_call" {
                calls.push(items[i].clone());
                i += 1;
            }
            while i < items.len() && responses_item_type(&items[i]) == "function_call_output" {
                outputs.push(items[i].clone());
                i += 1;
            }
            let mut used = vec![false; outputs.len()];
            for call in &calls {
                normalized.push(call.clone());
                let call_id = call.get("call_id").and_then(Value::as_str).unwrap_or("");
                if let Some((idx, output)) = outputs.iter().enumerate().find(|(idx, output)| {
                    !used[*idx]
                        && output.get("call_id").and_then(Value::as_str).unwrap_or("") == call_id
                }) {
                    normalized.push(output.clone());
                    used[idx] = true;
                }
            }
            for (idx, output) in outputs.into_iter().enumerate() {
                if !used[idx] {
                    normalized.push(output);
                }
            }
            continue;
        }
        normalized.push(items[i].clone());
        i += 1;
    }
    normalized
}

fn append_message_to_system_instruction(out: &mut Value, item: &Value) {
    let Some(content) = item.get("content") else {
        return;
    };
    match content {
        Value::Array(items) => {
            for content_item in items {
                push_system_part(
                    out,
                    content_item
                        .get("text")
                        .map(value_to_string)
                        .unwrap_or_default(),
                );
            }
        }
        Value::String(text) => push_system_part(out, text.clone()),
        _ => {}
    }
}

fn append_message_contents(out: &mut Value, item: &Value) {
    let Some(content) = item.get("content") else {
        return;
    };
    if let Some(content_items) = content.as_array() {
        let mut current_role = String::new();
        let mut current_parts: Vec<Value> = Vec::new();

        let flush = |out: &mut Value, role: &mut String, parts: &mut Vec<Value>| {
            if !role.is_empty() && !parts.is_empty() {
                out["contents"].as_array_mut().unwrap().push(json!({
                    "role": role,
                    "parts": std::mem::take(parts),
                }));
            } else {
                parts.clear();
            }
            role.clear();
        };

        for content_item in content_items {
            let content_type = content_item
                .get("type")
                .and_then(Value::as_str)
                .unwrap_or("input_text");
            let effective_role = effective_gemini_role(item, content_type);
            if !current_role.is_empty() && current_role != effective_role {
                flush(out, &mut current_role, &mut current_parts);
            }
            if current_role.is_empty() {
                current_role = effective_role;
            }

            if let Some(part) = responses_content_item_to_gemini_part(content_item, content_type) {
                current_parts.push(part);
            }
        }
        flush(out, &mut current_role, &mut current_parts);
    } else if let Some(text) = content.as_str() {
        out["contents"].as_array_mut().unwrap().push(json!({
            "role": effective_gemini_role_for_role(item.get("role").and_then(Value::as_str).unwrap_or("")),
            "parts": [{"text": text}],
        }));
    }
}

fn append_function_call(out: &mut Value, item: &Value) {
    let name = util::sanitize_function_name(item.get("name").and_then(Value::as_str).unwrap_or(""));
    let args = item
        .get("arguments")
        .and_then(Value::as_str)
        .and_then(|raw| serde_json::from_str::<Value>(raw).ok())
        .unwrap_or_else(|| json!({}));
    out["contents"].as_array_mut().unwrap().push(json!({
        "role": "model",
        "parts": [{
            "functionCall": {
                "name": name,
                "args": args,
                "id": item.get("call_id").and_then(Value::as_str).unwrap_or(""),
            },
            "thoughtSignature": GEMINI_RESPONSES_THOUGHT_SIGNATURE,
        }],
    }));
}

fn append_function_call_output(out: &mut Value, root: &Value, item: &Value) {
    let call_id = item.get("call_id").and_then(Value::as_str).unwrap_or("");
    let function_name = find_function_call_name(root, call_id);
    let mut response = json!({});
    if let Some(output) = item.get("output") {
        let output_raw = match output {
            Value::String(text) => text.clone(),
            other => serde_json::to_string(other).unwrap_or_default(),
        };
        if !output_raw.is_empty() && output_raw != "null" {
            if let Ok(parsed) = serde_json::from_str::<Value>(&output_raw) {
                response["result"] = parsed;
            } else {
                response["result"] = json!(output_raw);
            }
        }
    }
    out["contents"].as_array_mut().unwrap().push(json!({
        "role": "function",
        "parts": [{
            "functionResponse": {
                "name": util::sanitize_function_name(&function_name),
                "id": call_id,
                "response": response,
            }
        }],
    }));
}

fn append_tools(out: &mut Value, root: &Value) {
    let Some(tools) = root.get("tools").and_then(Value::as_array) else {
        return;
    };
    let mut declarations = Vec::new();
    for tool in tools {
        if tool.get("type").and_then(Value::as_str) != Some("function") {
            continue;
        }
        let mut declaration = json!({
            "name": util::sanitize_function_name(tool.get("name").and_then(Value::as_str).unwrap_or("")),
            "description": tool.get("description").and_then(Value::as_str).unwrap_or(""),
            "parametersJsonSchema": {},
        });
        if let Some(parameters) = tool.get("parameters") {
            declaration["parametersJsonSchema"] = clean_schema_for_gemini(parameters.clone());
        }
        declarations.push(declaration);
    }
    if !declarations.is_empty() {
        out["tools"] = json!([{"functionDeclarations": declarations}]);
    }
}

fn apply_generation_config(out: &mut Value, root: &Value) {
    if let Some(max_output_tokens) = root.get("max_output_tokens").and_then(Value::as_i64) {
        set_path(
            out,
            &["generationConfig", "maxOutputTokens"],
            json!(max_output_tokens),
        );
    }
    if let Some(temperature) = root.get("temperature").and_then(Value::as_f64) {
        set_path(
            out,
            &["generationConfig", "temperature"],
            json!(temperature),
        );
    }
    if let Some(top_p) = root.get("top_p").and_then(Value::as_f64) {
        set_path(out, &["generationConfig", "topP"], json!(top_p));
    }
    if let Some(stop_sequences) = root.get("stop_sequences").and_then(Value::as_array) {
        let sequences: Vec<Value> = stop_sequences
            .iter()
            .map(|value| json!(value_to_string(value)))
            .collect();
        set_path(
            out,
            &["generationConfig", "stopSequences"],
            Value::Array(sequences),
        );
    }
}

fn apply_text_format(out: &mut Value, root: &Value) {
    let Some(text_format) = root.pointer("/text/format") else {
        return;
    };
    let format_type = text_format
        .get("type")
        .and_then(Value::as_str)
        .unwrap_or("")
        .trim()
        .to_lowercase();
    match format_type.as_str() {
        "json_object" => {
            set_path(
                out,
                &["generationConfig", "responseMimeType"],
                json!("application/json"),
            );
        }
        "json_schema" => {
            set_path(
                out,
                &["generationConfig", "responseMimeType"],
                json!("application/json"),
            );
            if let Some(schema) = text_format
                .get("schema")
                .or_else(|| text_format.pointer("/json_schema/schema"))
            {
                set_path(
                    out,
                    &["generationConfig", "responseJsonSchema"],
                    schema.clone(),
                );
            }
        }
        _ => {}
    }
}

fn apply_reasoning_effort(out: &mut Value, root: &Value) {
    let Some(effort) = root.pointer("/reasoning/effort").and_then(Value::as_str) else {
        return;
    };
    let effort = effort.trim().to_lowercase();
    if effort.is_empty() {
        return;
    }
    if effort == "auto" {
        set_path(
            out,
            &["generationConfig", "thinkingConfig", "thinkingBudget"],
            json!(-1),
        );
        set_path(
            out,
            &["generationConfig", "thinkingConfig", "includeThoughts"],
            json!(true),
        );
    } else {
        set_path(
            out,
            &["generationConfig", "thinkingConfig", "thinkingLevel"],
            json!(effort),
        );
        set_path(
            out,
            &["generationConfig", "thinkingConfig", "includeThoughts"],
            json!(effort != "none"),
        );
    }
}

fn responses_content_item_to_gemini_part(
    content_item: &Value,
    content_type: &str,
) -> Option<Value> {
    match content_type {
        "input_text" | "output_text" => content_item
            .get("text")
            .map(|text| json!({"text": value_to_string(text)})),
        "input_image" => {
            let image_url = content_item
                .get("image_url")
                .or_else(|| content_item.get("url"))
                .and_then(Value::as_str)
                .unwrap_or("");
            data_url_to_inline_data(image_url)
        }
        "input_audio" => {
            let data = content_item
                .get("data")
                .and_then(Value::as_str)
                .unwrap_or("");
            if data.is_empty() {
                return None;
            }
            let format = content_item
                .get("format")
                .and_then(Value::as_str)
                .unwrap_or("");
            Some(json!({
                "inline_data": {
                    "mime_type": input_audio_mime_type(format),
                    "data": data,
                }
            }))
        }
        _ => None,
    }
}

fn data_url_to_inline_data(image_url: &str) -> Option<Value> {
    if image_url.is_empty() || !image_url.starts_with("data:") {
        return None;
    }
    let trimmed = image_url.strip_prefix("data:")?;
    let mut mime_type = "application/octet-stream";
    let mut data = "";
    if let Some((media, payload)) = trimmed.split_once(";base64,") {
        if !media.is_empty() {
            mime_type = media;
        }
        data = payload;
    } else if let Some((media, payload)) = trimmed.split_once(',') {
        if !media.is_empty() {
            mime_type = media;
        }
        data = payload;
    }
    if data.is_empty() {
        return None;
    }
    Some(json!({"inline_data": {"mime_type": mime_type, "data": data}}))
}

fn build_reasoning_model_content(
    thought_text: &str,
    visible_text: &str,
    signature: &str,
    use_native_reasoning: bool,
) -> Value {
    if use_native_reasoning {
        json!({
            "role": "model",
            "parts": [
                {"text": thought_text, "thought": true},
                {"text": visible_text, "thoughtSignature": signature},
            ],
        })
    } else {
        json!({
            "role": "model",
            "parts": [{
                "text": thought_text,
                "thoughtSignature": signature,
                "thought": true,
            }],
        })
    }
}

fn assistant_visible_text(item: &Value) -> Option<String> {
    if responses_item_type(item) != "message" {
        return None;
    }
    let content = item.get("content")?;
    if let Some(text) = content.as_str() {
        let role = item.get("role").and_then(Value::as_str).unwrap_or("");
        if role.eq_ignore_ascii_case("assistant") || role.eq_ignore_ascii_case("model") {
            return Some(text.to_string());
        }
        return None;
    }
    let content_items = content.as_array()?;
    let mut parts = Vec::new();
    let mut has_output_text = false;
    for content_item in content_items {
        let content_type = content_item
            .get("type")
            .and_then(Value::as_str)
            .unwrap_or("input_text");
        if content_type == "output_text" {
            has_output_text = true;
            parts.push(
                content_item
                    .get("text")
                    .map(value_to_string)
                    .unwrap_or_default(),
            );
        }
    }
    has_output_text.then(|| parts.join("\n"))
}

fn strip_trailing_model_prefill(out: &mut Value) {
    let Some(contents) = out.get_mut("contents").and_then(Value::as_array_mut) else {
        return;
    };
    let Some(last) = contents.last() else {
        return;
    };
    if last.get("role").and_then(Value::as_str) != Some("model") {
        return;
    }
    let has_thought = last
        .get("parts")
        .and_then(Value::as_array)
        .map(|parts| {
            parts.iter().any(|part| {
                part.get("thought")
                    .and_then(Value::as_bool)
                    .unwrap_or(false)
            })
        })
        .unwrap_or(false);
    if !has_thought {
        contents.pop();
    }
}

fn push_system_part(out: &mut Value, text: String) {
    if out.get("systemInstruction").is_none() {
        out["systemInstruction"] = json!({"parts": []});
    }
    out["systemInstruction"]["parts"]
        .as_array_mut()
        .expect("systemInstruction.parts is array")
        .push(json!({"text": text}));
}

fn attach_default_safety_settings(out: &mut Value) {
    if out.get("safetySettings").is_some() {
        return;
    }
    out["safetySettings"] = json!([
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "OFF"},
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "OFF"},
        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "OFF"},
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "OFF"},
        {"category": "HARM_CATEGORY_CIVIC_INTEGRITY", "threshold": "BLOCK_NONE"},
    ]);
}

fn clean_schema_for_gemini(schema: Value) -> Value {
    let raw = serde_json::to_string(&schema).unwrap_or_else(|_| "{}".to_string());
    serde_json::from_str(&util::clean_json_schema_for_gemini(&raw))
        .unwrap_or_else(|_| json!({"type": "object", "properties": {}}))
}

fn openai_responses_gemini_thought_signature(raw: &str) -> String {
    signature::gemini_replay_sig(raw, SignatureBlockKind::GeminiModelPart)
}

fn find_function_call_name(root: &Value, call_id: &str) -> String {
    root.get("input")
        .and_then(Value::as_array)
        .and_then(|items| {
            items.iter().find_map(|item| {
                (item.get("type").and_then(Value::as_str) == Some("function_call")
                    && item.get("call_id").and_then(Value::as_str).unwrap_or("") == call_id)
                    .then(|| {
                        item.get("name")
                            .and_then(Value::as_str)
                            .unwrap_or("")
                            .to_string()
                    })
            })
        })
        .unwrap_or_else(|| "unknown".to_string())
}

fn responses_item_type(item: &Value) -> String {
    let explicit = item.get("type").and_then(Value::as_str).unwrap_or("");
    if !explicit.is_empty() {
        return explicit.to_string();
    }
    if item.get("role").and_then(Value::as_str).is_some() {
        return "message".to_string();
    }
    String::new()
}

fn effective_gemini_role(item: &Value, content_type: &str) -> String {
    if content_type == "output_text" {
        return "model".to_string();
    }
    effective_gemini_role_for_role(item.get("role").and_then(Value::as_str).unwrap_or(""))
}

fn effective_gemini_role_for_role(role: &str) -> String {
    match role.to_lowercase().as_str() {
        "assistant" | "model" => "model".to_string(),
        "" => "user".to_string(),
        other => other.to_string(),
    }
}

fn input_audio_mime_type(format: &str) -> String {
    match format {
        "mp3" => "audio/mpeg".to_string(),
        "" | "wav" => "audio/wav".to_string(),
        "ogg" => "audio/ogg".to_string(),
        "flac" => "audio/flac".to_string(),
        "aac" => "audio/aac".to_string(),
        "webm" => "audio/webm".to_string(),
        "pcm16" => "audio/pcm".to_string(),
        "g711_ulaw" | "g711_alaw" => "audio/basic".to_string(),
        other => format!("audio/{other}"),
    }
}

fn set_path(root: &mut Value, path: &[&str], value: Value) {
    let mut cursor = root;
    for key in &path[..path.len() - 1] {
        if !cursor.is_object() {
            *cursor = json!({});
        }
        cursor = cursor
            .as_object_mut()
            .unwrap()
            .entry((*key).to_string())
            .or_insert_with(|| json!({}));
    }
    if !cursor.is_object() {
        *cursor = json!({});
    }
    cursor
        .as_object_mut()
        .unwrap()
        .insert(path[path.len() - 1].to_string(), value);
}

fn value_to_string(value: &Value) -> String {
    match value {
        Value::String(value) => value.clone(),
        Value::Number(value) => value.to_string(),
        Value::Bool(value) => value.to_string(),
        Value::Null => String::new(),
        other => serde_json::to_string(other).unwrap_or_default(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn simple_string_input_maps_to_user_content() {
        let body = json!({"input": "Hello"});
        let out = transform("gemini-test", body, false);
        assert_eq!(out["contents"][0]["role"], "user");
        assert_eq!(out["contents"][0]["parts"][0]["text"], "Hello");
    }

    #[test]
    fn trailing_plain_model_prefill_is_stripped() {
        let body = json!({
            "input": [
                {"type":"message","role":"user","content":[{"type":"input_text","text":"hello"}]},
                {"type":"message","role":"assistant","content":[{"type":"output_text","text":"old answer"}]}
            ]
        });
        let out = transform("gemini-test", body, false);
        assert_eq!(out["contents"].as_array().unwrap().len(), 1);
    }
}
