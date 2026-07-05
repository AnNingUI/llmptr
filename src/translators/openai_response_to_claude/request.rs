use serde_json::{Value, json};
use std::collections::{HashMap, HashSet};
use translator_infra::{
    signature,
    thinking::{self, ThinkingConfig, ThinkingMode},
    util,
};

#[derive(Debug, Clone)]
struct PendingToolUse {
    call_id: String,
    message: Value,
}

/// Transform an OpenAI Responses API request to Claude Messages format.
pub fn transform(model: &str, body: Value, stream: bool) -> Value {
    let mut out = json!({
        "model": model,
        "max_tokens": 32000,
        "messages": [],
        "metadata": {"user_id": "user_llmptr_account_static_session_static"},
        "stream": stream,
    });

    if let Some(reasoning) = convert_reasoning_config(&body, model) {
        merge_object(&mut out, reasoning);
    }

    if let Some(max_output_tokens) = body.get("max_output_tokens").and_then(Value::as_i64) {
        out["max_tokens"] = json!(max_output_tokens);
    }

    let mut extracted_system = false;
    if let Some(instructions) = body.get("instructions").and_then(Value::as_str) {
        if !instructions.is_empty() {
            push_message(&mut out, json!({"role": "user", "content": instructions}));
        }
    } else if let Some(system_text) = extract_first_system_input_text(&body)
        && !system_text.is_empty()
    {
        push_message(&mut out, json!({"role": "user", "content": system_text}));
        extracted_system = true;
    }

    let mut pending_reasoning: Vec<Value> = Vec::new();
    let mut pending_tool_uses: Vec<PendingToolUse> = Vec::new();

    if let Some(input) = body.get("input").and_then(Value::as_array) {
        for item in input {
            if extracted_system
                && item
                    .get("role")
                    .and_then(Value::as_str)
                    .is_some_and(|role| role.eq_ignore_ascii_case("system"))
            {
                continue;
            }

            let item_type = item
                .get("type")
                .and_then(Value::as_str)
                .filter(|value| !value.is_empty())
                .unwrap_or_else(|| {
                    if item.get("role").and_then(Value::as_str).is_some() {
                        "message"
                    } else {
                        ""
                    }
                });

            match item_type {
                "message" => append_message_item(
                    &mut out,
                    item,
                    &mut pending_reasoning,
                    &mut pending_tool_uses,
                ),
                "reasoning" => {
                    if let Some(thinking) = convert_reasoning_item(item) {
                        pending_reasoning.push(thinking);
                    }
                }
                "function_call" => {
                    append_function_call_item(item, &mut pending_reasoning, &mut pending_tool_uses);
                }
                "function_call_output" => {
                    flush_pending_reasoning(&mut out, &mut pending_reasoning);
                    let call_id = util::sanitize_claude_tool_id(
                        item.get("call_id").and_then(Value::as_str).unwrap_or(""),
                    );
                    flush_pending_tool_use_for(&mut out, &mut pending_tool_uses, &call_id);
                    let output = item.get("output").and_then(Value::as_str).unwrap_or("");
                    push_message(
                        &mut out,
                        json!({
                            "role": "user",
                            "content": [{
                                "type": "tool_result",
                                "tool_use_id": call_id,
                                "content": output,
                            }],
                        }),
                    );
                }
                _ => {}
            }
        }
    }
    flush_pending_reasoning(&mut out, &mut pending_reasoning);
    flush_pending_tool_uses(&mut out, &mut pending_tool_uses);

    append_tools_and_choice(&mut out, &body);
    out
}

fn convert_reasoning_config(body: &Value, model: &str) -> Option<Value> {
    let effort = body.pointer("/reasoning/effort").and_then(Value::as_str)?;
    let effort = effort.trim().to_lowercase();
    if effort.is_empty() {
        return None;
    }
    let config = match effort.as_str() {
        "none" => ThinkingConfig {
            mode: ThinkingMode::None,
            budget: 0,
            level: effort,
        },
        "auto" => ThinkingConfig {
            mode: ThinkingMode::Auto,
            budget: -1,
            level: effort,
        },
        _ => ThinkingConfig {
            mode: ThinkingMode::Level,
            budget: 0,
            level: effort,
        },
    };
    Some(thinking::apply_claude_thinking(json!({}), &config, model))
}

fn merge_object(target: &mut Value, patch: Value) {
    let Some(target_obj) = target.as_object_mut() else {
        return;
    };
    if let Value::Object(patch_obj) = patch {
        for (key, value) in patch_obj {
            target_obj.insert(key, value);
        }
    }
}

fn extract_first_system_input_text(body: &Value) -> Option<String> {
    let input = body.get("input")?.as_array()?;
    for item in input {
        if !item
            .get("role")
            .and_then(Value::as_str)
            .is_some_and(|role| role.eq_ignore_ascii_case("system"))
        {
            continue;
        }
        let Some(content) = item.get("content") else {
            continue;
        };
        let text = match content {
            Value::String(text) => text.clone(),
            Value::Array(parts) => parts
                .iter()
                .filter_map(|part| part.get("text").and_then(Value::as_str))
                .filter(|text| !text.is_empty())
                .collect::<Vec<_>>()
                .join("\n"),
            _ => String::new(),
        };
        if !text.is_empty() {
            return Some(text);
        }
    }
    None
}

fn append_message_item(
    out: &mut Value,
    item: &Value,
    pending_reasoning: &mut Vec<Value>,
    pending_tool_uses: &mut Vec<PendingToolUse>,
) {
    let mut role = String::new();
    let mut text_aggregate = String::new();
    let mut parts = Vec::new();
    let mut has_image = false;
    let mut has_file = false;

    match item.get("content") {
        Some(Value::Array(content)) => {
            for part in content {
                match part.get("type").and_then(Value::as_str).unwrap_or("") {
                    "input_text" | "output_text" => {
                        if let Some(text) = part.get("text").and_then(Value::as_str) {
                            text_aggregate.push_str(text);
                            parts.push(json!({"type": "text", "text": text}));
                        }
                        role = if part.get("type").and_then(Value::as_str) == Some("output_text") {
                            "assistant".to_string()
                        } else {
                            "user".to_string()
                        };
                    }
                    "input_image" => {
                        let url = part
                            .get("image_url")
                            .or_else(|| part.get("url"))
                            .and_then(Value::as_str)
                            .unwrap_or("");
                        if let Some(image) = convert_input_image(url) {
                            parts.push(image);
                            if role.is_empty() {
                                role = "user".to_string();
                            }
                            has_image = true;
                        }
                    }
                    "input_file" => {
                        if let Some(file_data) = part.get("file_data").and_then(Value::as_str)
                            && let Some(document) = convert_input_file(file_data)
                        {
                            parts.push(document);
                            if role.is_empty() {
                                role = "user".to_string();
                            }
                            has_file = true;
                        }
                    }
                    _ => {}
                }
            }
        }
        Some(Value::String(text)) => {
            text_aggregate.push_str(text);
        }
        _ => {}
    }

    if role.is_empty() {
        role = match item.get("role").and_then(Value::as_str).unwrap_or("user") {
            "user" | "assistant" | "system" => item
                .get("role")
                .and_then(Value::as_str)
                .unwrap_or("user")
                .to_string(),
            _ => "user".to_string(),
        };
    }

    if role != "assistant" {
        flush_pending_tool_uses(out, pending_tool_uses);
    }

    let mut has_reasoning = false;
    if !pending_reasoning.is_empty() {
        if role == "assistant" {
            if parts.is_empty() && !text_aggregate.is_empty() {
                parts.push(json!({"type": "text", "text": text_aggregate}));
            }
            let mut prefixed = std::mem::take(pending_reasoning);
            prefixed.extend(parts);
            parts = prefixed;
            has_reasoning = true;
        } else {
            flush_pending_reasoning(out, pending_reasoning);
        }
    }

    if !parts.is_empty() {
        let content = if parts.len() == 1 && !has_image && !has_file && !has_reasoning {
            parts[0]
                .get("text")
                .and_then(Value::as_str)
                .map(|text| json!(text))
                .unwrap_or_else(|| Value::Array(parts))
        } else {
            Value::Array(parts)
        };
        push_message(out, json!({"role": role, "content": content}));
    } else if !text_aggregate.is_empty() || role == "system" {
        push_message(out, json!({"role": role, "content": text_aggregate}));
    }
}

fn convert_reasoning_item(item: &Value) -> Option<Value> {
    let encrypted = item.get("encrypted_content").and_then(Value::as_str)?;
    let signature = signature::normalize_claude_native_sig(encrypted, false)?;
    let thinking = reasoning_summary_text(item);
    Some(json!({
        "type": "thinking",
        "thinking": thinking,
        "signature": signature,
    }))
}

fn reasoning_summary_text(item: &Value) -> String {
    let Some(summary) = item.get("summary").and_then(Value::as_array) else {
        return String::new();
    };
    let mut out = String::new();
    for part in summary {
        if let Some(text) = part.get("text").and_then(Value::as_str) {
            out.push_str(text);
        } else if let Some(text) = part.as_str() {
            out.push_str(text);
        }
    }
    out
}

fn append_function_call_item(
    item: &Value,
    pending_reasoning: &mut Vec<Value>,
    pending_tool_uses: &mut Vec<PendingToolUse>,
) {
    let call_id_raw = item
        .get("call_id")
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty())
        .unwrap_or("toolu_generated");
    let call_id = util::sanitize_claude_tool_id(call_id_raw);
    let name = item.get("name").and_then(Value::as_str).unwrap_or("");
    let args = item
        .get("arguments")
        .and_then(Value::as_str)
        .and_then(|raw| serde_json::from_str::<Value>(raw).ok())
        .filter(Value::is_object)
        .unwrap_or_else(|| json!({}));

    let mut content = std::mem::take(pending_reasoning);
    content.push(json!({
        "type": "tool_use",
        "id": call_id,
        "name": name,
        "input": args,
    }));
    pending_tool_uses.push(PendingToolUse {
        call_id,
        message: json!({"role": "assistant", "content": content}),
    });
}

fn convert_input_image(url: &str) -> Option<Value> {
    if url.is_empty() {
        return None;
    }
    if let Some((media_type, data)) = parse_data_url(url) {
        return Some(json!({
            "type": "image",
            "source": {"type": "base64", "media_type": media_type, "data": data},
        }));
    }
    Some(json!({
        "type": "image",
        "source": {"type": "url", "url": url},
    }))
}

fn convert_input_file(file_data: &str) -> Option<Value> {
    if file_data.is_empty() {
        return None;
    }
    let (media_type, data) = parse_data_url(file_data).unwrap_or_else(|| {
        (
            "application/octet-stream".to_string(),
            file_data.to_string(),
        )
    });
    Some(json!({
        "type": "document",
        "source": {"type": "base64", "media_type": media_type, "data": data},
    }))
}

fn parse_data_url(value: &str) -> Option<(String, String)> {
    let trimmed = value.strip_prefix("data:")?;
    let (media_type, data) = trimmed.split_once(";base64,")?;
    if data.is_empty() {
        return None;
    }
    let media_type = if media_type.is_empty() {
        "application/octet-stream"
    } else {
        media_type
    };
    Some((media_type.to_string(), data.to_string()))
}

fn push_message(out: &mut Value, message: Value) {
    if let Some(messages) = out.get_mut("messages").and_then(Value::as_array_mut) {
        messages.push(message);
    }
}

fn flush_pending_reasoning(out: &mut Value, pending_reasoning: &mut Vec<Value>) {
    if pending_reasoning.is_empty() {
        return;
    }
    push_message(
        out,
        json!({"role": "assistant", "content": std::mem::take(pending_reasoning)}),
    );
}

fn flush_pending_tool_uses(out: &mut Value, pending_tool_uses: &mut Vec<PendingToolUse>) {
    for pending in std::mem::take(pending_tool_uses) {
        push_message(out, pending.message);
    }
}

fn flush_pending_tool_use_for(
    out: &mut Value,
    pending_tool_uses: &mut Vec<PendingToolUse>,
    call_id: &str,
) {
    if let Some(index) = pending_tool_uses
        .iter()
        .position(|pending| pending.call_id == call_id)
    {
        let pending = pending_tool_uses.remove(index);
        push_message(out, pending.message);
    } else {
        flush_pending_tool_uses(out, pending_tool_uses);
    }
}

fn append_tools_and_choice(out: &mut Value, body: &Value) {
    let mut included_tool_names = HashSet::new();
    let mut tool_name_map = HashMap::new();

    if let Some(tools) = body.get("tools").and_then(Value::as_array) {
        let mut converted = Vec::new();
        for tool in tools {
            for value in convert_responses_tool_to_claude_tools(tool, &mut tool_name_map) {
                if let Some(name) = value.get("name").and_then(Value::as_str)
                    && !name.is_empty()
                {
                    included_tool_names.insert(name.to_string());
                }
                converted.push(value);
            }
        }
        if !converted.is_empty() {
            out["tools"] = Value::Array(converted);
        }
    }

    if let Some(tool_choice) = body.get("tool_choice")
        && let Some(converted) =
            convert_tool_choice(tool_choice, &tool_name_map, &included_tool_names)
    {
        out["tool_choice"] = converted;
    }
}

fn convert_responses_tool_to_claude_tools(
    tool: &Value,
    tool_name_map: &mut HashMap<String, String>,
) -> Vec<Value> {
    match tool
        .get("type")
        .and_then(Value::as_str)
        .unwrap_or("")
        .trim()
    {
        "" | "function" => convert_responses_function_tool(tool, "")
            .into_iter()
            .collect(),
        "namespace" => convert_responses_namespace_tool(tool, tool_name_map),
        "web_search" => convert_responses_web_search_tool(tool, tool_name_map)
            .into_iter()
            .collect(),
        "custom" if tool.get("name").and_then(Value::as_str) == Some("apply_patch") => Vec::new(),
        tool_type if is_unsupported_builtin_tool(tool_type) => Vec::new(),
        _ => tool
            .get("name")
            .and_then(Value::as_str)
            .filter(|name| !name.is_empty())
            .map(|_| vec![tool.clone()])
            .unwrap_or_default(),
    }
}

fn convert_responses_namespace_tool(
    tool: &Value,
    tool_name_map: &mut HashMap<String, String>,
) -> Vec<Value> {
    let namespace_name = tool
        .get("name")
        .and_then(Value::as_str)
        .unwrap_or("")
        .trim();
    let Some(children) = tool.get("tools").and_then(Value::as_array) else {
        return Vec::new();
    };
    let mut out = Vec::new();
    for child in children {
        let child_name = responses_tool_name(child);
        let qualified = qualify_namespace_tool_name(namespace_name, child_name);
        if let Some(converted) = convert_responses_function_tool(child, &qualified) {
            tool_name_map.insert(qualified.clone(), qualified.clone());
            if !child_name.is_empty() {
                tool_name_map.insert(child_name.to_string(), qualified.clone());
            }
            out.push(converted);
        }
    }
    out
}

fn convert_responses_function_tool(tool: &Value, override_name: &str) -> Option<Value> {
    let name = if override_name.is_empty() {
        responses_tool_name(tool)
    } else {
        override_name
    };
    if name.is_empty() {
        return None;
    }
    let mut out = json!({
        "name": name,
        "description": "",
        "input_schema": normalize_claude_tool_input_schema(responses_tool_parameters(tool)),
    });
    if let Some(description) = responses_tool_description(tool).filter(|value| !value.is_empty()) {
        out["description"] = json!(description);
    }
    Some(out)
}

fn convert_responses_web_search_tool(
    tool: &Value,
    tool_name_map: &mut HashMap<String, String>,
) -> Option<Value> {
    if tool
        .get("external_web_access")
        .and_then(Value::as_bool)
        .is_some_and(|value| !value)
    {
        return None;
    }
    let name = tool
        .get("name")
        .and_then(Value::as_str)
        .filter(|value| !value.trim().is_empty())
        .unwrap_or("web_search");
    let mut out = json!({"type": "web_search_20250305", "name": name});
    if let Some(max_uses) = tool.get("max_uses").and_then(Value::as_i64) {
        out["max_uses"] = json!(max_uses);
    }
    if let Some(allowed_domains) = tool.pointer("/filters/allowed_domains")
        && allowed_domains.is_array()
    {
        out["allowed_domains"] = allowed_domains.clone();
    }
    if let Some(user_location) = tool.get("user_location")
        && user_location.is_object()
    {
        out["user_location"] = user_location.clone();
    }
    tool_name_map.insert(name.to_string(), name.to_string());
    Some(out)
}

fn convert_tool_choice(
    tool_choice: &Value,
    tool_name_map: &HashMap<String, String>,
    included_tool_names: &HashSet<String>,
) -> Option<Value> {
    match tool_choice {
        Value::String(value) => match value.as_str() {
            "auto" => Some(json!({"type": "auto"})),
            "required" if !included_tool_names.is_empty() => Some(json!({"type": "any"})),
            "none" => None,
            _ => None,
        },
        Value::Object(obj) if obj.get("type").and_then(Value::as_str) == Some("function") => {
            let mut name = obj
                .get("function")
                .and_then(|function| function.get("name"))
                .or_else(|| obj.get("name"))
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_string();
            if let Some(mapped) = tool_name_map.get(&name) {
                name = mapped.clone();
            }
            if included_tool_names.contains(&name) {
                Some(json!({"name": name, "type": "tool"}))
            } else {
                None
            }
        }
        _ => None,
    }
}

fn responses_tool_name(tool: &Value) -> &str {
    tool.get("name")
        .and_then(Value::as_str)
        .filter(|name| !name.trim().is_empty())
        .or_else(|| tool.pointer("/function/name").and_then(Value::as_str))
        .unwrap_or("")
}

fn responses_tool_description(tool: &Value) -> Option<&str> {
    tool.get("description").and_then(Value::as_str).or_else(|| {
        tool.pointer("/function/description")
            .and_then(Value::as_str)
    })
}

fn responses_tool_parameters(tool: &Value) -> Option<Value> {
    [
        "/parameters",
        "/parametersJsonSchema",
        "/input_schema",
        "/function/parameters",
        "/function/parametersJsonSchema",
    ]
    .into_iter()
    .find_map(|path| tool.pointer(path).cloned())
}

fn normalize_claude_tool_input_schema(parameters: Option<Value>) -> Value {
    let Some(Value::Object(mut map)) = parameters else {
        return json!({"type": "object", "properties": {}});
    };
    if !map.contains_key("type") {
        map.insert("type".to_string(), json!("object"));
    }
    if map.get("type").and_then(Value::as_str) == Some("object") && !map.contains_key("properties")
    {
        map.insert("properties".to_string(), json!({}));
    }
    Value::Object(map)
}

fn qualify_namespace_tool_name(namespace_name: &str, child_name: &str) -> String {
    let child = child_name.trim();
    if child.is_empty() || namespace_name.is_empty() || child.starts_with("mcp__") {
        return child.to_string();
    }
    if child.starts_with(namespace_name) {
        return child.to_string();
    }
    if namespace_name.ends_with("__") {
        return format!("{namespace_name}{child}");
    }
    format!("{namespace_name}__{child}")
}

fn is_unsupported_builtin_tool(tool_type: &str) -> bool {
    matches!(
        tool_type,
        "image_generation" | "file_search" | "code_interpreter" | "computer_use_preview"
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_request_maps_to_claude() {
        let body = json!({
            "input": [{"role":"user","content":[{"type":"input_text","text":"Hello"}]}],
            "model":"gpt-4"
        });
        let out = transform("claude", body, false);
        assert_eq!(out["stream"], false);
        assert_eq!(out["messages"][0]["role"], "user");
        assert_eq!(out["messages"][0]["content"], "Hello");
    }

    #[test]
    fn function_call_ids_are_sanitized() {
        let body = json!({
            "input": [
                {"type":"function_call","call_id":"call.with space:1","name":"Read","arguments":"{\"path\":\"README.md\"}"},
                {"type":"function_call_output","call_id":"call.with space:1","output":"ok"}
            ]
        });
        let out = transform("claude", body, false);
        assert_eq!(out["messages"][0]["content"][0]["id"], "call_with_space_1");
        assert_eq!(
            out["messages"][1]["content"][0]["tool_use_id"],
            "call_with_space_1"
        );
    }
}
