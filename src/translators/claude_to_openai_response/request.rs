use serde_json::{Value, json};
use translator_infra::{thinking, util};

/// Transform a Claude Messages request to the OpenAI Responses API format.
///
/// CLIProxyAPI does not currently register this request direction directly.
/// This implementation keeps the Rust matrix usable for the direct pair while
/// following the Responses item shapes used by the reverse Go translator.
pub fn transform(model: &str, body: Value, stream: bool) -> Value {
    let mut out = json!({
        "model": model,
        "input": [],
        "stream": stream,
    });

    if let Some(system) = body.get("system")
        && let Some(instructions) = claude_system_to_instructions(system)
    {
        out["instructions"] = json!(instructions);
    }

    if let Some(messages) = body.get("messages").and_then(Value::as_array) {
        for message in messages {
            append_message_items(&mut out["input"], message);
        }
    }

    if let Some(tools) = body.get("tools").and_then(Value::as_array) {
        let converted: Vec<Value> = tools.iter().filter_map(convert_tool).collect();
        if !converted.is_empty() {
            out["tools"] = Value::Array(converted);
        }
    }

    if let Some(tool_choice) = body.get("tool_choice").and_then(convert_tool_choice) {
        out["tool_choice"] = tool_choice;
    }

    if let Some(reasoning) = convert_thinking_config(&body) {
        out["reasoning"] = reasoning;
    }

    if let Some(max_tokens) = body.get("max_tokens").and_then(Value::as_u64) {
        out["max_output_tokens"] = json!(max_tokens);
    }
    if let Some(temperature) = body.get("temperature") {
        out["temperature"] = temperature.clone();
    }
    if let Some(top_p) = body.get("top_p") {
        out["top_p"] = top_p.clone();
    }
    if let Some(metadata) = body.get("metadata") {
        out["metadata"] = metadata.clone();
    }

    out
}

fn append_message_items(input: &mut Value, message: &Value) {
    let role = message
        .get("role")
        .and_then(Value::as_str)
        .unwrap_or("user");
    let Some(content) = message.get("content") else {
        return;
    };

    if role == "tool" {
        let call_id = message
            .get("tool_call_id")
            .and_then(Value::as_str)
            .unwrap_or("");
        push_input_item(
            input,
            json!({
                "type": "function_call_output",
                "call_id": call_id,
                "output": tool_result_output(Some(content)),
            }),
        );
        return;
    }

    let responses_role = match role {
        "assistant" => "assistant",
        "system" => "user",
        _ => "user",
    };

    match content {
        Value::String(text) => {
            if text.is_empty() || util::is_claude_attribution(text) {
                return;
            }
            push_message(
                input,
                responses_role,
                vec![text_part_for_role(responses_role, text)],
            );
        }
        Value::Array(parts) => append_content_parts(input, responses_role, parts),
        _ => {}
    }
}

fn append_content_parts(input: &mut Value, role: &str, parts: &[Value]) {
    let mut message_parts = Vec::new();
    let flush_message = |input: &mut Value, message_parts: &mut Vec<Value>| {
        if !message_parts.is_empty() {
            push_message(input, role, std::mem::take(message_parts));
        }
    };

    for part in parts {
        match part.get("type").and_then(Value::as_str).unwrap_or("") {
            "text" => {
                if let Some(text) = part.get("text").and_then(Value::as_str)
                    && !text.is_empty()
                    && !util::is_claude_attribution(text)
                {
                    message_parts.push(text_part_for_role(role, text));
                }
            }
            "image" if role == "user" => {
                if let Some(image) = claude_image_to_responses(part) {
                    message_parts.push(image);
                }
            }
            "document" if role == "user" => {
                if let Some(file) = claude_document_to_responses(part) {
                    message_parts.push(file);
                }
            }
            "thinking" if role == "assistant" => {
                flush_message(input, &mut message_parts);
                if let Some(reasoning) = claude_thinking_to_reasoning_item(part) {
                    push_input_item(input, reasoning);
                }
            }
            "redacted_thinking" => {}
            "tool_use" if role == "assistant" => {
                flush_message(input, &mut message_parts);
                push_input_item(input, claude_tool_use_to_function_call(part));
            }
            "tool_result" => {
                flush_message(input, &mut message_parts);
                let call_id = part
                    .get("tool_use_id")
                    .and_then(Value::as_str)
                    .unwrap_or("");
                push_input_item(
                    input,
                    json!({
                        "type": "function_call_output",
                        "call_id": call_id,
                        "output": tool_result_output(part.get("content")),
                    }),
                );
            }
            _ => {}
        }
    }

    flush_message(input, &mut message_parts);
}

fn push_message(input: &mut Value, role: &str, content: Vec<Value>) {
    push_input_item(
        input,
        json!({
            "type": "message",
            "role": role,
            "content": content,
        }),
    );
}

fn push_input_item(input: &mut Value, item: Value) {
    if let Some(items) = input.as_array_mut() {
        items.push(item);
    }
}

fn text_part_for_role(role: &str, text: &str) -> Value {
    let part_type = if role == "assistant" {
        "output_text"
    } else {
        "input_text"
    };
    json!({"type": part_type, "text": text})
}

fn claude_system_to_instructions(system: &Value) -> Option<String> {
    match system {
        Value::String(text) => {
            let text = text.trim();
            if text.is_empty() || util::is_claude_attribution(text) {
                None
            } else {
                Some(text.to_string())
            }
        }
        Value::Array(parts) => {
            let texts: Vec<_> = parts
                .iter()
                .filter_map(|part| part.get("text").and_then(Value::as_str))
                .filter(|text| !text.trim().is_empty() && !util::is_claude_attribution(text))
                .map(str::to_string)
                .collect();
            if texts.is_empty() {
                None
            } else {
                Some(texts.join("\n"))
            }
        }
        _ => None,
    }
}

fn claude_image_to_responses(part: &Value) -> Option<Value> {
    let source = part.get("source")?;
    let url = match source.get("type").and_then(Value::as_str).unwrap_or("") {
        "base64" => {
            let media_type = source
                .get("media_type")
                .and_then(Value::as_str)
                .unwrap_or("application/octet-stream");
            let data = source.get("data").and_then(Value::as_str).unwrap_or("");
            if data.is_empty() {
                return None;
            }
            format!("data:{media_type};base64,{data}")
        }
        "url" => source
            .get("url")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string(),
        _ => return None,
    };
    if url.is_empty() {
        None
    } else {
        Some(json!({"type": "input_image", "image_url": url}))
    }
}

fn claude_document_to_responses(part: &Value) -> Option<Value> {
    let source = part.get("source")?;
    if source.get("type").and_then(Value::as_str) != Some("base64") {
        return None;
    }
    let media_type = source
        .get("media_type")
        .and_then(Value::as_str)
        .unwrap_or("application/octet-stream");
    let data = source.get("data").and_then(Value::as_str).unwrap_or("");
    if data.is_empty() {
        return None;
    }
    let mut file = json!({
        "type": "input_file",
        "file_data": format!("data:{media_type};base64,{data}"),
    });
    if let Some(filename) = part.get("title").or_else(|| part.get("filename")) {
        file["filename"] = filename.clone();
    }
    Some(file)
}

fn claude_thinking_to_reasoning_item(part: &Value) -> Option<Value> {
    let thinking = part.get("thinking").and_then(Value::as_str).unwrap_or("");
    let signature = part.get("signature").and_then(Value::as_str).unwrap_or("");
    if thinking.trim().is_empty() && signature.is_empty() {
        return None;
    }
    let mut item = json!({
        "type": "reasoning",
        "summary": [],
    });
    if !thinking.trim().is_empty() {
        item["summary"] = json!([{"type": "summary_text", "text": thinking}]);
    }
    if !signature.is_empty() {
        item["encrypted_content"] = json!(signature);
    }
    Some(item)
}

fn claude_tool_use_to_function_call(part: &Value) -> Value {
    let call_id = part.get("id").and_then(Value::as_str).unwrap_or("");
    let name = part.get("name").and_then(Value::as_str).unwrap_or("");
    let input = part.get("input").cloned().unwrap_or_else(|| json!({}));
    json!({
        "type": "function_call",
        "call_id": call_id,
        "name": name,
        "arguments": input.to_string(),
    })
}

fn tool_result_output(content: Option<&Value>) -> String {
    let Some(content) = content else {
        return String::new();
    };
    match content {
        Value::String(text) => text.clone(),
        Value::Array(parts) => {
            let mut texts = Vec::new();
            let mut complex = false;
            for part in parts {
                match part.get("type").and_then(Value::as_str).unwrap_or("") {
                    "text" => {
                        if let Some(text) = part.get("text").and_then(Value::as_str) {
                            texts.push(text.to_string());
                        }
                    }
                    "image" | "document" => complex = true,
                    _ => {}
                }
            }
            if complex {
                content.to_string()
            } else {
                texts.join("\n\n")
            }
        }
        _ => content.to_string(),
    }
}

fn convert_tool(tool: &Value) -> Option<Value> {
    let tool_type = tool.get("type").and_then(Value::as_str).unwrap_or("");
    if tool_type.starts_with("web_search") {
        let mut out = json!({
            "type": "web_search",
            "name": tool.get("name").and_then(Value::as_str).unwrap_or("web_search"),
        });
        if let Some(max_uses) = tool.get("max_uses") {
            out["max_uses"] = max_uses.clone();
        }
        return Some(out);
    }

    let name = tool.get("name").and_then(Value::as_str).unwrap_or("");
    if name.is_empty() {
        return None;
    }
    let mut out = json!({
        "type": "function",
        "name": name,
        "parameters": normalize_tool_schema(
            tool.get("input_schema")
                .cloned()
                .unwrap_or_else(|| json!({"type": "object", "properties": {}}))
        ),
    });
    if let Some(description) = tool.get("description").and_then(Value::as_str)
        && !description.is_empty()
    {
        out["description"] = json!(description);
    }
    Some(out)
}

fn convert_tool_choice(tool_choice: &Value) -> Option<Value> {
    match tool_choice {
        Value::String(value) => Some(json!(value)),
        Value::Object(obj) => match obj.get("type").and_then(Value::as_str).unwrap_or("") {
            "auto" => Some(json!("auto")),
            "any" => Some(json!("required")),
            "tool" => obj.get("name").and_then(Value::as_str).map(|name| {
                json!({
                    "type": "function",
                    "name": name,
                })
            }),
            _ => None,
        },
        _ => None,
    }
}

fn convert_thinking_config(body: &Value) -> Option<Value> {
    let thinking_config = body.get("thinking")?.as_object()?;
    match thinking_config
        .get("type")
        .and_then(Value::as_str)
        .unwrap_or("")
    {
        "enabled" => {
            let budget = thinking_config
                .get("budget_tokens")
                .and_then(Value::as_i64)
                .unwrap_or(-1);
            let (effort, _) = thinking::convert_budget_to_level(budget);
            if effort.is_empty() {
                None
            } else {
                Some(json!({"effort": effort}))
            }
        }
        "adaptive" | "auto" => {
            let effort = body
                .pointer("/output_config/effort")
                .and_then(Value::as_str)
                .unwrap_or("xhigh");
            Some(json!({"effort": effort}))
        }
        "disabled" => {
            let (effort, _) = thinking::convert_budget_to_level(0);
            if effort.is_empty() {
                None
            } else {
                Some(json!({"effort": effort}))
            }
        }
        _ => None,
    }
}

fn normalize_tool_schema(schema: Value) -> Value {
    let mut schema = normalize_schema(schema);
    if let Value::Object(map) = &mut schema {
        if !map.contains_key("type") {
            map.insert("type".to_string(), json!("object"));
        }
        if map.get("type").and_then(Value::as_str) == Some("object") {
            map.entry("properties".to_string())
                .or_insert_with(|| json!({}));
        }
    }
    schema
}

fn normalize_schema(schema: Value) -> Value {
    match schema {
        Value::Object(mut map) => {
            if map.get("type").and_then(Value::as_str) == Some("object") {
                map.entry("properties".to_string())
                    .or_insert_with(|| json!({}));
            }
            for value in map.values_mut() {
                *value = normalize_schema(value.take());
            }
            Value::Object(map)
        }
        Value::Array(items) => Value::Array(items.into_iter().map(normalize_schema).collect()),
        other => other,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_request_maps_to_responses_input() {
        let body = json!({
            "system": "Be helpful",
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": "Hello"}]
        });
        let out = transform("gpt-4.1", body, true);
        assert_eq!(out["model"], "gpt-4.1");
        assert_eq!(out["stream"], true);
        assert_eq!(out["instructions"], "Be helpful");
        assert_eq!(out["max_output_tokens"], 1024);
        assert_eq!(out["input"][0]["type"], "message");
        assert_eq!(out["input"][0]["content"][0]["type"], "input_text");
    }

    #[test]
    fn assistant_thinking_and_tool_use_are_separate_items() {
        let body = json!({
            "thinking": {"type": "enabled", "budget_tokens": 8192},
            "messages": [{
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "work", "signature": "sig"},
                    {"type": "tool_use", "id": "call_1", "name": "search", "input": {"q": "rust"}},
                    {"type": "text", "text": "done"}
                ]
            }]
        });
        let out = transform("gpt-4.1", body, false);
        assert_eq!(out["reasoning"]["effort"], "medium");
        assert_eq!(out["input"][0]["type"], "reasoning");
        assert_eq!(out["input"][1]["type"], "function_call");
        assert_eq!(out["input"][2]["content"][0]["type"], "output_text");
    }
}
