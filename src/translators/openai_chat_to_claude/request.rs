//! OpenAI ChatCompletions -> Claude Messages request translation.

use serde_json::{Value, json};
use translator_infra::{thinking, util};

pub fn transform(model: &str, body: Value, stream: bool) -> Value {
    let mut out = json!({
        "model": model,
        "max_tokens": 32000,
        "messages": [],
        "metadata": {"user_id": "user_llmptr_account_default_session_default"},
    });

    if let Some(effort) = body.get("reasoning_effort").and_then(|v| v.as_str()) {
        let effort = effort.trim().to_lowercase();
        if !effort.is_empty() {
            let (budget, ok) = thinking::convert_level_to_budget(&effort);
            if ok {
                match budget {
                    0 => out["thinking"] = json!({"type": "disabled"}),
                    -1 => out["thinking"] = json!({"type": "enabled"}),
                    n if n > 0 => {
                        out["thinking"] = json!({"type": "enabled", "budget_tokens": n});
                    }
                    _ => {}
                }
            }
        }
    }

    if let Some(max_tokens) = body.get("max_tokens").and_then(|v| v.as_i64()) {
        out["max_tokens"] = json!(max_tokens);
    }
    if let Some(temp) = body.get("temperature") {
        out["temperature"] = temp.clone();
    } else if let Some(top_p) = body.get("top_p") {
        out["top_p"] = top_p.clone();
    }
    if let Some(stop) = body.get("stop") {
        match stop {
            Value::Array(items) => out["stop_sequences"] = json!(items),
            Value::String(s) => out["stop_sequences"] = json!([s]),
            other => out["stop_sequences"] = json!([value_to_string(other)]),
        }
    }
    out["stream"] = json!(stream);

    let mut message_count = 0usize;
    if let Some(messages) = body.get("messages").and_then(|v| v.as_array()) {
        for message in messages {
            let role = message.get("role").and_then(|v| v.as_str()).unwrap_or("");
            let content = message.get("content").unwrap_or(&Value::Null);

            match role {
                "system" => append_system_content(&mut out, content),
                "user" | "assistant" => {
                    let mut parts = Vec::new();
                    append_openai_content(&mut parts, content);

                    if role == "assistant" {
                        append_tool_calls(&mut parts, message.get("tool_calls"));
                    }

                    out["messages"].as_array_mut().unwrap().push(json!({
                        "role": role,
                        "content": parts,
                    }));
                    message_count += 1;
                }
                "tool" => {
                    let tool_call_id = util::sanitize_claude_tool_id(
                        message
                            .get("tool_call_id")
                            .and_then(|v| v.as_str())
                            .unwrap_or(""),
                    );
                    let converted = convert_openai_tool_result_content(content);
                    out["messages"].as_array_mut().unwrap().push(json!({
                        "role": "user",
                        "content": [{
                            "type": "tool_result",
                            "tool_use_id": tool_call_id,
                            "content": converted,
                        }]
                    }));
                    message_count += 1;
                }
                _ => {}
            }
        }
    }

    if message_count == 0
        && out
            .get("system")
            .and_then(|v| v.as_array())
            .map(|items| !items.is_empty())
            .unwrap_or(false)
    {
        out["messages"].as_array_mut().unwrap().push(json!({
            "role": "user",
            "content": [{"type": "text", "text": ""}],
        }));
    }

    if let Some(tools) = body.get("tools").and_then(|v| v.as_array()) {
        for tool in tools {
            if tool.get("type").and_then(|v| v.as_str()) != Some("function") {
                continue;
            }
            let Some(function) = tool.get("function") else {
                continue;
            };
            let mut claude_tool = json!({
                "name": function.get("name").and_then(|v| v.as_str()).unwrap_or(""),
                "description": function.get("description").and_then(|v| v.as_str()).unwrap_or(""),
            });
            if let Some(parameters) = function
                .get("parameters")
                .or_else(|| function.get("parametersJsonSchema"))
            {
                claude_tool["input_schema"] = parameters.clone();
            }
            out.as_object_mut()
                .unwrap()
                .entry("tools")
                .or_insert_with(|| json!([]))
                .as_array_mut()
                .unwrap()
                .push(claude_tool);
        }
        if out
            .get("tools")
            .and_then(|v| v.as_array())
            .map(|items| items.is_empty())
            .unwrap_or(false)
        {
            out.as_object_mut().unwrap().remove("tools");
        }
    }

    if let Some(tool_choice) = body.get("tool_choice") {
        match tool_choice {
            Value::String(choice) => match choice.as_str() {
                "auto" => out["tool_choice"] = json!({"type": "auto"}),
                "required" => out["tool_choice"] = json!({"type": "any"}),
                _ => {}
            },
            Value::Object(obj) => {
                if obj.get("type").and_then(|v| v.as_str()) == Some("function") {
                    let name = obj
                        .get("function")
                        .and_then(|v| v.get("name"))
                        .and_then(|v| v.as_str())
                        .unwrap_or("");
                    out["tool_choice"] = json!({"type": "tool", "name": name});
                }
            }
            _ => {}
        }
    }

    out
}

fn append_system_content(out: &mut Value, content: &Value) {
    match content {
        Value::String(text) if !text.is_empty() => {
            push_system_part(out, json!({"type": "text", "text": text}))
        }
        Value::Array(parts) => {
            for part in parts {
                if part.get("type").and_then(|v| v.as_str()) == Some("text") {
                    push_system_part(
                        out,
                        json!({"type": "text", "text": part.get("text").and_then(|v| v.as_str()).unwrap_or("")}),
                    );
                }
            }
        }
        _ => {}
    }
}

fn push_system_part(out: &mut Value, part: Value) {
    out.as_object_mut()
        .unwrap()
        .entry("system")
        .or_insert_with(|| json!([]))
        .as_array_mut()
        .unwrap()
        .push(part);
}

fn append_openai_content(parts: &mut Vec<Value>, content: &Value) {
    match content {
        Value::String(text) if !text.is_empty() => {
            parts.push(json!({"type": "text", "text": text}));
        }
        Value::Array(items) => {
            for item in items {
                if let Some(converted) = openai_content_part_to_claude(item) {
                    parts.push(converted);
                }
            }
        }
        _ => {}
    }
}

fn append_tool_calls(parts: &mut Vec<Value>, tool_calls: Option<&Value>) {
    let Some(tool_calls) = tool_calls.and_then(|v| v.as_array()) else {
        return;
    };
    for tool_call in tool_calls {
        if tool_call.get("type").and_then(|v| v.as_str()) != Some("function") {
            continue;
        }
        let function = tool_call.get("function").unwrap_or(&Value::Null);
        let id = util::sanitize_claude_tool_id(
            tool_call.get("id").and_then(|v| v.as_str()).unwrap_or(""),
        );
        let args = function.get("arguments");
        let input = match args {
            Some(Value::String(raw)) if !raw.is_empty() => serde_json::from_str::<Value>(raw)
                .ok()
                .filter(Value::is_object)
                .unwrap_or_else(|| json!({})),
            Some(Value::Object(_)) => args.cloned().unwrap_or_else(|| json!({})),
            _ => json!({}),
        };
        parts.push(json!({
            "type": "tool_use",
            "id": id,
            "name": function.get("name").and_then(|v| v.as_str()).unwrap_or(""),
            "input": input,
        }));
    }
}

fn openai_content_part_to_claude(part: &Value) -> Option<Value> {
    match part.get("type").and_then(|v| v.as_str()).unwrap_or("") {
        "text" => Some(json!({
            "type": "text",
            "text": part.get("text").and_then(|v| v.as_str()).unwrap_or(""),
        })),
        "image_url" => {
            let url = part
                .pointer("/image_url/url")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            convert_openai_image_url_to_claude_part(url)
        }
        "file" => {
            let file_data = part
                .pointer("/file/file_data")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            convert_openai_file_data_to_claude_part(file_data)
        }
        _ => None,
    }
}

fn convert_openai_image_url_to_claude_part(url: &str) -> Option<Value> {
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

fn convert_openai_file_data_to_claude_part(file_data: &str) -> Option<Value> {
    let (media_type, data) = parse_data_url(file_data)?;
    Some(json!({
        "type": "document",
        "source": {"type": "base64", "media_type": media_type, "data": data},
    }))
}

fn parse_data_url(value: &str) -> Option<(&str, &str)> {
    let rest = value.strip_prefix("data:")?;
    let (media_type, payload) = rest.split_once(";base64,")?;
    Some((media_type, payload))
}

fn convert_openai_tool_result_content(content: &Value) -> Value {
    match content {
        Value::String(text) => json!(text),
        Value::Array(items) => {
            let mut parts = Vec::new();
            for item in items {
                match item {
                    Value::String(text) => parts.push(json!({"type": "text", "text": text})),
                    Value::Object(_) => {
                        if let Some(converted) = openai_content_part_to_claude(item) {
                            parts.push(converted);
                        }
                    }
                    _ => {}
                }
            }
            if !parts.is_empty() || items.is_empty() {
                json!(parts)
            } else {
                content.clone()
            }
        }
        Value::Object(_) => {
            if let Some(converted) = openai_content_part_to_claude(content) {
                json!([converted])
            } else {
                content.clone()
            }
        }
        other => other.clone(),
    }
}

fn value_to_string(value: &Value) -> String {
    match value {
        Value::String(s) => s.clone(),
        other => other.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_openai_to_claude_basic() {
        let body = json!({
            "model": "gpt-4",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello!"}
            ]
        });
        let result = transform("gpt-4", body, false);
        assert_eq!(result["system"][0]["text"], "You are a helpful assistant.");
        assert_eq!(result["messages"][0]["role"], "user");
        assert_eq!(result["messages"][0]["content"][0]["text"], "Hello!");
    }

    #[test]
    fn test_openai_to_claude_reasoning_effort() {
        let body = json!({
            "model": "gpt-4",
            "reasoning_effort": "high",
            "messages": [{"role": "user", "content": "Think hard"}]
        });
        let result = transform("gpt-4", body, false);
        assert_eq!(result["thinking"]["budget_tokens"], 24576);
    }

    #[test]
    fn test_openai_to_claude_tool_calls() {
        let body = json!({
            "model": "gpt-4",
            "messages": [
                {"role": "assistant", "content": "", "tool_calls": [{
                    "id": "call_abc",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": r#"{"city":"Tokyo"}"#}
                }]},
                {"role": "tool", "tool_call_id": "call_abc", "content": "Sunny"}
            ]
        });
        let result = transform("gpt-4", body, false);
        assert_eq!(result["messages"][0]["content"][0]["type"], "tool_use");
        assert_eq!(result["messages"][0]["content"][0]["name"], "get_weather");
        assert_eq!(result["messages"][1]["content"][0]["type"], "tool_result");
        assert_eq!(result["messages"][1]["content"][0]["content"], "Sunny");
    }

    #[test]
    fn test_openai_to_claude_stream() {
        let body = json!({
            "model": "gpt-4",
            "stream": true,
            "messages": [{"role": "user", "content": "Hi"}]
        });
        let result = transform("gpt-4", body, true);
        assert_eq!(result["stream"], true);
    }
}
