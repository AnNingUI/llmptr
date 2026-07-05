use serde_json::{Value, json};
use std::collections::HashMap;
use translator_infra::{
    signature::{self, SignatureBlockKind},
    util,
};

const GEMINI_FUNCTION_THOUGHT_SIGNATURE: &str = signature::GEMINI_SKIP_SENTINEL;

/// Convert an OpenAI Chat Completions request to Gemini generateContent format.
pub fn transform(model: &str, body: Value, _stream: bool) -> Value {
    let mut out = json!({
        "contents": [],
        "model": model,
    });

    if let Some(generation_config) = body.get("generationConfig") {
        set_path(&mut out, &["generationConfig"], generation_config.clone());
    }
    apply_generation_config(&mut out, &body);
    append_messages(&mut out, &body);
    append_tools(&mut out, &body);
    attach_default_safety_settings(&mut out);
    out
}

fn apply_generation_config(out: &mut Value, body: &Value) {
    if let Some(effort) = body.get("reasoning_effort").and_then(Value::as_str) {
        let effort = effort.trim().to_lowercase();
        if !effort.is_empty() {
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
    }

    if let Some(temperature) = body.get("temperature").and_then(Value::as_f64) {
        set_path(
            out,
            &["generationConfig", "temperature"],
            json!(temperature),
        );
    }
    if let Some(top_p) = body.get("top_p").and_then(Value::as_f64) {
        set_path(out, &["generationConfig", "topP"], json!(top_p));
    }
    if let Some(top_k) = body.get("top_k").and_then(Value::as_i64) {
        set_path(out, &["generationConfig", "topK"], json!(top_k));
    }
    if let Some(n) = body.get("n").and_then(Value::as_i64)
        && n > 1
    {
        set_path(out, &["generationConfig", "candidateCount"], json!(n));
    }

    if let Some(modalities) = body.get("modalities").and_then(Value::as_array) {
        let response_modalities: Vec<Value> = modalities
            .iter()
            .filter_map(Value::as_str)
            .filter_map(|value| match value.to_lowercase().as_str() {
                "text" => Some(json!("TEXT")),
                "image" => Some(json!("IMAGE")),
                _ => None,
            })
            .collect();
        if !response_modalities.is_empty() {
            set_path(
                out,
                &["generationConfig", "responseModalities"],
                Value::Array(response_modalities),
            );
        }
    }

    if let Some(image_config) = body.get("image_config").and_then(Value::as_object) {
        if let Some(aspect_ratio) = image_config.get("aspect_ratio").and_then(Value::as_str) {
            set_path(
                out,
                &["generationConfig", "imageConfig", "aspectRatio"],
                json!(aspect_ratio),
            );
        }
        if let Some(image_size) = image_config.get("image_size").and_then(Value::as_str) {
            set_path(
                out,
                &["generationConfig", "imageConfig", "imageSize"],
                json!(image_size),
            );
        }
    }
}

fn append_messages(out: &mut Value, body: &Value) {
    let Some(messages) = body.get("messages").and_then(Value::as_array) else {
        return;
    };

    let mut tool_call_names = HashMap::new();
    let mut tool_responses = HashMap::new();
    for message in messages {
        if message.get("role").and_then(Value::as_str) == Some("assistant")
            && let Some(tool_calls) = message.get("tool_calls").and_then(Value::as_array)
        {
            for tool_call in tool_calls {
                if tool_call.get("type").and_then(Value::as_str) == Some("function")
                    && let (Some(id), Some(name)) = (
                        tool_call.get("id").and_then(Value::as_str),
                        tool_call.pointer("/function/name").and_then(Value::as_str),
                    )
                    && !id.is_empty()
                    && !name.is_empty()
                {
                    tool_call_names.insert(id.to_string(), name.to_string());
                }
            }
        }
        if message.get("role").and_then(Value::as_str) == Some("tool")
            && let Some(id) = message.get("tool_call_id").and_then(Value::as_str)
            && !id.is_empty()
        {
            let raw = message
                .get("content")
                .map(json_raw_string)
                .unwrap_or_default();
            tool_responses.insert(id.to_string(), raw);
        }
    }

    let mut contents = Vec::new();
    let mut system_parts = Vec::new();
    for message in messages {
        let role = message.get("role").and_then(Value::as_str).unwrap_or("");
        let content = message.get("content").unwrap_or(&Value::Null);
        if (role == "system" || role == "developer") && messages.len() > 1 {
            append_system_parts(&mut system_parts, content);
        } else if role == "user"
            || ((role == "system" || role == "developer") && messages.len() == 1)
        {
            contents.push(json!({
                "role": "user",
                "parts": user_message_parts(content),
            }));
        } else if role == "assistant" {
            append_assistant_message(
                &mut contents,
                message,
                content,
                &tool_call_names,
                &tool_responses,
            );
        }
    }

    if contents
        .last()
        .and_then(|last| last.get("role"))
        .and_then(Value::as_str)
        == Some("model")
    {
        contents.pop();
    }

    set_path(out, &["contents"], Value::Array(contents));
    if !system_parts.is_empty() {
        set_path(
            out,
            &["systemInstruction"],
            json!({"role": "user", "parts": system_parts}),
        );
    }
}

fn append_system_parts(parts: &mut Vec<Value>, content: &Value) {
    match content {
        Value::String(text) => parts.push(json!({"text": text})),
        Value::Object(obj) if obj.get("type").and_then(Value::as_str) == Some("text") => {
            parts.push(json!({
                "text": obj.get("text").and_then(Value::as_str).unwrap_or("")
            }));
        }
        Value::Array(items) if !items.is_empty() => {
            for item in items {
                parts.push(json!({
                    "text": item.get("text").and_then(Value::as_str).unwrap_or("")
                }));
            }
        }
        _ => {}
    }
}

fn user_message_parts(content: &Value) -> Vec<Value> {
    let mut parts = Vec::new();
    match content {
        Value::String(text) => parts.push(json!({"text": text})),
        Value::Array(items) => {
            for item in items {
                match item.get("type").and_then(Value::as_str).unwrap_or("") {
                    "text" => {
                        if let Some(text) = item.get("text").and_then(Value::as_str)
                            && !text.is_empty()
                        {
                            parts.push(json!({"text": text}));
                        }
                    }
                    "image_url" => {
                        if let Some(part) = openai_media_url_to_inline_data(
                            item.pointer("/image_url/url")
                                .and_then(Value::as_str)
                                .unwrap_or(""),
                            true,
                        ) {
                            parts.push(part);
                        }
                    }
                    "video_url" => {
                        if let Some(part) = openai_media_url_to_inline_data(
                            item.pointer("/video_url/url")
                                .and_then(Value::as_str)
                                .unwrap_or(""),
                            false,
                        ) {
                            parts.push(part);
                        }
                    }
                    "file" => {
                        if let Some(part) = openai_file_to_inline_data(item) {
                            parts.push(part);
                        }
                    }
                    "input_audio" => {
                        if let Some(data) =
                            item.pointer("/input_audio/data").and_then(Value::as_str)
                            && !data.is_empty()
                        {
                            let format = item
                                .pointer("/input_audio/format")
                                .and_then(Value::as_str)
                                .unwrap_or("");
                            parts.push(json!({
                                "inlineData": {
                                    "mime_type": openai_input_audio_mime_type(format),
                                    "data": data,
                                }
                            }));
                        }
                    }
                    _ => {}
                }
            }
        }
        _ => {}
    }
    parts
}

fn append_assistant_message(
    contents: &mut Vec<Value>,
    message: &Value,
    content: &Value,
    tool_call_names: &HashMap<String, String>,
    tool_responses: &HashMap<String, String>,
) {
    let mut parts = assistant_message_parts(content);
    let mut function_ids = Vec::new();
    let tool_calls = message.get("tool_calls").and_then(Value::as_array);

    if let Some(tool_calls) = tool_calls {
        for tool_call in tool_calls {
            if tool_call.get("type").and_then(Value::as_str) != Some("function") {
                continue;
            }
            let id = tool_call.get("id").and_then(Value::as_str).unwrap_or("");
            let name = util::sanitize_function_name(
                tool_call
                    .pointer("/function/name")
                    .and_then(Value::as_str)
                    .unwrap_or(""),
            );
            let args = tool_call
                .pointer("/function/arguments")
                .and_then(Value::as_str)
                .and_then(|raw| serde_json::from_str::<Value>(raw).ok())
                .unwrap_or_else(|| json!({}));
            parts.push(json!({
                "functionCall": {"name": name, "args": args},
                "thoughtSignature": openai_tool_call_gemini_thought_signature(tool_call),
            }));
            if !id.is_empty() {
                function_ids.push(id.to_string());
            }
        }

        contents.push(json!({"role": "model", "parts": parts}));

        let mut response_parts = Vec::new();
        for id in function_ids {
            if let Some(name) = tool_call_names.get(&id) {
                let response = tool_responses
                    .get(&id)
                    .cloned()
                    .unwrap_or_else(|| "{}".to_string());
                response_parts.push(json!({
                    "functionResponse": {
                        "name": util::sanitize_function_name(name),
                        "response": {"result": response},
                    }
                }));
            }
        }
        if !response_parts.is_empty() {
            contents.push(json!({"role": "user", "parts": response_parts}));
        }
    } else {
        contents.push(json!({"role": "model", "parts": parts}));
    }
}

fn assistant_message_parts(content: &Value) -> Vec<Value> {
    let mut parts = Vec::new();
    match content {
        Value::String(text) => parts.push(json!({"text": text})),
        Value::Array(items) => {
            for item in items {
                match item.get("type").and_then(Value::as_str).unwrap_or("") {
                    "text" => {
                        if let Some(text) = item.get("text").and_then(Value::as_str)
                            && !text.is_empty()
                        {
                            parts.push(json!({"text": text}));
                        }
                    }
                    "image_url" => {
                        if let Some(part) = openai_media_url_to_inline_data(
                            item.pointer("/image_url/url")
                                .and_then(Value::as_str)
                                .unwrap_or(""),
                            true,
                        ) {
                            parts.push(part);
                        }
                    }
                    _ => {}
                }
            }
        }
        _ => {}
    }
    parts
}

fn append_tools(out: &mut Value, body: &Value) {
    let Some(tools) = body.get("tools").and_then(Value::as_array) else {
        return;
    };
    if tools.is_empty() {
        return;
    }

    let mut function_declarations = Vec::new();
    let mut passthrough_tools = Vec::new();

    for tool in tools {
        if tool.get("type").and_then(Value::as_str) == Some("function")
            && let Some(function) = tool.get("function").and_then(Value::as_object)
        {
            let mut declaration = Value::Object(function.clone());
            let name = declaration
                .get("name")
                .and_then(Value::as_str)
                .map(util::sanitize_function_name)
                .unwrap_or_default();
            declaration["name"] = json!(name);
            let parameters = declaration
                .get("parameters")
                .cloned()
                .unwrap_or_else(|| json!({"type": "object", "properties": {}}));
            declaration["parametersJsonSchema"] = clean_schema_for_gemini(parameters);
            if let Some(obj) = declaration.as_object_mut() {
                obj.remove("parameters");
                obj.remove("strict");
            }
            function_declarations.push(declaration);
        }
        if let Some(google_search) = tool.get("google_search") {
            passthrough_tools.push(json!({"googleSearch": google_search}));
        }
        if let Some(code_execution) = tool.get("code_execution") {
            passthrough_tools.push(json!({"codeExecution": code_execution}));
        }
        if let Some(url_context) = tool.get("url_context") {
            passthrough_tools.push(json!({"urlContext": url_context}));
        }
    }

    let mut out_tools = Vec::new();
    if !function_declarations.is_empty() {
        out_tools.push(json!({"functionDeclarations": function_declarations}));
    }
    out_tools.extend(passthrough_tools);
    if !out_tools.is_empty() {
        set_path(out, &["tools"], Value::Array(out_tools));
    }
}

fn clean_schema_for_gemini(schema: Value) -> Value {
    let raw = serde_json::to_string(&schema).unwrap_or_else(|_| "{}".to_string());
    serde_json::from_str(&util::clean_json_schema_for_gemini(&raw))
        .unwrap_or_else(|_| json!({"type": "object", "properties": {}}))
}

fn attach_default_safety_settings(out: &mut Value) {
    if out.get("safetySettings").is_some() {
        return;
    }
    set_path(
        out,
        &["safetySettings"],
        json!([
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "OFF"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "OFF"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "OFF"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "OFF"},
            {"category": "HARM_CATEGORY_CIVIC_INTEGRITY", "threshold": "BLOCK_NONE"},
        ]),
    );
}

fn openai_media_url_to_inline_data(url: &str, add_image_signature: bool) -> Option<Value> {
    let rest = url.strip_prefix("data:")?;
    let (mime, encoded) = rest.split_once(';')?;
    let data = encoded.strip_prefix("base64,")?;
    let mut out = json!({
        "inlineData": {"mime_type": mime, "data": data},
    });
    if add_image_signature {
        out["thoughtSignature"] = json!(GEMINI_FUNCTION_THOUGHT_SIGNATURE);
    }
    Some(out)
}

fn openai_file_to_inline_data(item: &Value) -> Option<Value> {
    let filename = item
        .pointer("/file/filename")
        .and_then(Value::as_str)
        .unwrap_or("");
    let data = item
        .pointer("/file/file_data")
        .and_then(Value::as_str)
        .unwrap_or("");
    let ext = filename.rsplit('.').next().unwrap_or("");
    let mime = mime_type_for_extension(ext)?;
    Some(json!({"inlineData": {"mime_type": mime, "data": data}}))
}

fn mime_type_for_extension(ext: &str) -> Option<&'static str> {
    match ext.to_lowercase().as_str() {
        "txt" => Some("text/plain"),
        "md" => Some("text/markdown"),
        "pdf" => Some("application/pdf"),
        "png" => Some("image/png"),
        "jpg" | "jpeg" => Some("image/jpeg"),
        "gif" => Some("image/gif"),
        "webp" => Some("image/webp"),
        "csv" => Some("text/csv"),
        "json" => Some("application/json"),
        _ => None,
    }
}

fn openai_input_audio_mime_type(format: &str) -> String {
    match format {
        "" | "wav" => "audio/wav".to_string(),
        "mp3" => "audio/mpeg".to_string(),
        "ogg" => "audio/ogg".to_string(),
        "flac" => "audio/flac".to_string(),
        "aac" => "audio/aac".to_string(),
        "webm" => "audio/webm".to_string(),
        "pcm16" => "audio/pcm".to_string(),
        "g711_ulaw" | "g711_alaw" => "audio/basic".to_string(),
        other => format!("audio/{other}"),
    }
}

fn openai_tool_call_gemini_thought_signature(tool_call: &Value) -> String {
    for pointer in [
        "/extra_content/google/thought_signature",
        "/function/extra_content/google/thought_signature",
        "/thoughtSignature",
        "/thought_signature",
    ] {
        if let Some(signature) = tool_call.pointer(pointer).and_then(Value::as_str) {
            return signature::gemini_replay_sig(signature, SignatureBlockKind::GeminiFunctionCall);
        }
    }
    GEMINI_FUNCTION_THOUGHT_SIGNATURE.to_string()
}

fn json_raw_string(value: &Value) -> String {
    serde_json::to_string(value).unwrap_or_default()
}

fn set_path(root: &mut Value, path: &[&str], value: Value) {
    if path.is_empty() {
        *root = value;
        return;
    }

    let mut cursor = root;
    for key in &path[..path.len() - 1] {
        if !cursor.is_object() {
            *cursor = json!({});
        }
        cursor = cursor
            .as_object_mut()
            .expect("cursor is object")
            .entry((*key).to_string())
            .or_insert_with(|| json!({}));
    }

    if !cursor.is_object() {
        *cursor = json!({});
    }
    cursor
        .as_object_mut()
        .expect("cursor is object")
        .insert(path[path.len() - 1].to_string(), value);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_request_maps_to_gemini() {
        let body = json!({
            "messages": [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Hello!"}
            ],
            "temperature": 0.2
        });
        let out = transform("gemini-test", body, false);
        assert_eq!(out["model"], "gemini-test");
        assert_eq!(
            out["systemInstruction"]["parts"][0]["text"],
            "You are helpful."
        );
        assert_eq!(out["contents"][0]["parts"][0]["text"], "Hello!");
        assert_eq!(out["generationConfig"]["temperature"], 0.2);
    }
}
