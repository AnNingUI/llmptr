//! OpenAI Responses -> OpenAI Chat request translation.
//!
//! Mirrors CLIProxyAPI's `ConvertOpenAIResponsesRequestToOpenAIChatCompletions`.

use serde_json::{Value, json};
use std::collections::HashSet;

pub fn transform(model: &str, body: Value, stream: bool) -> Value {
    let mut out = json!({
        "model": model,
        "messages": [],
        "stream": stream,
    });

    if let Some(max_tokens) = body.get("max_output_tokens").and_then(Value::as_i64) {
        out["max_tokens"] = json!(max_tokens);
    }
    if let Some(parallel_tool_calls) = body.get("parallel_tool_calls").and_then(Value::as_bool) {
        out["parallel_tool_calls"] = json!(parallel_tool_calls);
    }

    if let Some(instructions) = body.get("instructions") {
        push_message(
            &mut out,
            json!({"role": "system", "content": value_to_string(instructions)}),
        );
    }

    match body.get("input") {
        Some(Value::Array(items)) => append_input_items(&mut out, items),
        Some(Value::String(input)) => {
            push_message(&mut out, json!({"role": "user", "content": input}));
        }
        _ => {}
    }

    append_tools(&mut out, &body);

    if let Some(effort) = body.pointer("/reasoning/effort").and_then(Value::as_str) {
        let effort = effort.trim().to_lowercase();
        if !effort.is_empty() {
            out["reasoning_effort"] = json!(effort);
        }
    }

    if let Some(tool_choice) = body.get("tool_choice") {
        out["tool_choice"] = tool_choice.clone();
    }

    out
}

fn append_input_items(out: &mut Value, items: &[Value]) {
    let output_call_ids: HashSet<String> = items
        .iter()
        .filter(|item| item.get("type").and_then(Value::as_str) == Some("function_call_output"))
        .filter_map(|item| item.get("call_id").and_then(Value::as_str))
        .map(str::trim)
        .filter(|call_id| !call_id.is_empty())
        .map(str::to_string)
        .collect();

    let mut pending_tool_calls: Vec<Value> = Vec::new();
    let mut pending_tool_call_ids: Vec<String> = Vec::new();
    let mut pending_reasoning = String::new();
    let mut awaiting_tool_outputs: HashSet<String> = HashSet::new();
    let mut deferred_messages: Vec<Value> = Vec::new();

    for item in items {
        let item_type = responses_item_type(item);
        if item_type != "function_call" {
            flush_pending_tool_calls(
                out,
                &mut pending_tool_calls,
                &mut pending_tool_call_ids,
                &mut pending_reasoning,
                &mut awaiting_tool_outputs,
            );
        }

        match item_type.as_str() {
            "message" | "" => append_message_item(
                out,
                item,
                &output_call_ids,
                &awaiting_tool_outputs,
                &mut deferred_messages,
                &mut pending_reasoning,
            ),
            "reasoning" => {
                pending_reasoning.push_str(&collect_reasoning_content(item));
            }
            "function_call" => {
                let mut tool_call = json!({
                    "id": "",
                    "type": "function",
                    "function": {"name": "", "arguments": ""},
                });
                if let Some(call_id) = item.get("call_id") {
                    tool_call["id"] = json!(value_to_string(call_id));
                }
                if let Some(name) = item.get("name") {
                    tool_call["function"]["name"] = json!(value_to_string(name));
                }
                if let Some(arguments) = item.get("arguments") {
                    tool_call["function"]["arguments"] = json!(value_to_string(arguments));
                }
                pending_tool_calls.push(tool_call);
                if let Some(call_id) = item.get("call_id").and_then(Value::as_str) {
                    let call_id = call_id.trim();
                    if !call_id.is_empty() {
                        pending_tool_call_ids.push(call_id.to_string());
                    }
                }
            }
            "function_call_output" => {
                let call_id = item
                    .get("call_id")
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .trim()
                    .to_string();
                let output = item.get("output").map(value_to_string).unwrap_or_default();
                push_message(
                    out,
                    json!({"role": "tool", "tool_call_id": call_id, "content": output}),
                );
                if !call_id.is_empty() {
                    awaiting_tool_outputs.remove(&call_id);
                }
                if awaiting_tool_outputs.is_empty() && !deferred_messages.is_empty() {
                    flush_deferred_messages(out, &mut deferred_messages);
                }
            }
            _ => {}
        }
    }

    flush_pending_tool_calls(
        out,
        &mut pending_tool_calls,
        &mut pending_tool_call_ids,
        &mut pending_reasoning,
        &mut awaiting_tool_outputs,
    );
    append_pending_reasoning_message(
        out,
        &output_call_ids,
        &awaiting_tool_outputs,
        &mut deferred_messages,
        &mut pending_reasoning,
    );
    flush_deferred_messages(out, &mut deferred_messages);
}

fn append_message_item(
    out: &mut Value,
    item: &Value,
    output_call_ids: &HashSet<String>,
    awaiting_tool_outputs: &HashSet<String>,
    deferred_messages: &mut Vec<Value>,
    pending_reasoning: &mut String,
) {
    let mut role = item
        .get("role")
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_string();
    if role == "developer" {
        role = "user".to_string();
    }

    if role != "assistant" {
        append_pending_reasoning_message(
            out,
            output_call_ids,
            awaiting_tool_outputs,
            deferred_messages,
            pending_reasoning,
        );
    }

    let mut message = json!({"role": role, "content": []});
    match item.get("content") {
        Some(Value::Array(content)) => {
            for content_item in content {
                match content_item
                    .get("type")
                    .and_then(Value::as_str)
                    .unwrap_or("input_text")
                {
                    "input_text" | "output_text" => {
                        let text = content_item
                            .get("text")
                            .map(value_to_string)
                            .unwrap_or_default();
                        message["content"]
                            .as_array_mut()
                            .expect("message.content is array")
                            .push(json!({"type": "text", "text": text}));
                    }
                    "input_image" => {
                        let image_url = content_item
                            .get("image_url")
                            .and_then(Value::as_str)
                            .unwrap_or("");
                        let mut part =
                            json!({"type": "image_url", "image_url": {"url": image_url}});
                        if let Some(detail) = content_item.get("detail") {
                            part["image_url"]["detail"] = json!(value_to_string(detail));
                        }
                        message["content"]
                            .as_array_mut()
                            .expect("message.content is array")
                            .push(part);
                    }
                    _ => {}
                }
            }
        }
        Some(Value::String(content)) => {
            message["content"] = json!(content);
        }
        _ => {}
    }

    if message.get("role").and_then(Value::as_str) == Some("assistant") {
        let reasoning = item
            .get("reasoning_content")
            .and_then(Value::as_str)
            .filter(|value| !value.is_empty())
            .map(str::to_string)
            .unwrap_or_else(|| std::mem::take(pending_reasoning));
        if !reasoning.is_empty() {
            message["reasoning_content"] = json!(reasoning);
        }
    }

    append_regular_message(
        out,
        message,
        output_call_ids,
        awaiting_tool_outputs,
        deferred_messages,
    );
}

fn flush_pending_tool_calls(
    out: &mut Value,
    pending_tool_calls: &mut Vec<Value>,
    pending_tool_call_ids: &mut Vec<String>,
    pending_reasoning: &mut String,
    awaiting_tool_outputs: &mut HashSet<String>,
) {
    if pending_tool_calls.is_empty() {
        return;
    }
    let mut message = json!({
        "role": "assistant",
        "tool_calls": std::mem::take(pending_tool_calls),
    });
    let reasoning = std::mem::take(pending_reasoning);
    if !reasoning.is_empty() {
        message["reasoning_content"] = json!(reasoning);
    }
    push_message(out, message);
    for call_id in pending_tool_call_ids.drain(..) {
        if !call_id.trim().is_empty() {
            awaiting_tool_outputs.insert(call_id);
        }
    }
}

fn append_pending_reasoning_message(
    out: &mut Value,
    output_call_ids: &HashSet<String>,
    awaiting_tool_outputs: &HashSet<String>,
    deferred_messages: &mut Vec<Value>,
    pending_reasoning: &mut String,
) {
    let reasoning = std::mem::take(pending_reasoning);
    if reasoning.is_empty() {
        return;
    }
    append_regular_message(
        out,
        json!({"role": "assistant", "content": "", "reasoning_content": reasoning}),
        output_call_ids,
        awaiting_tool_outputs,
        deferred_messages,
    );
}

fn append_regular_message(
    out: &mut Value,
    message: Value,
    output_call_ids: &HashSet<String>,
    awaiting_tool_outputs: &HashSet<String>,
    deferred_messages: &mut Vec<Value>,
) {
    if awaiting_tool_outputs
        .iter()
        .any(|call_id| output_call_ids.contains(call_id))
    {
        deferred_messages.push(message);
    } else {
        push_message(out, message);
    }
}

fn flush_deferred_messages(out: &mut Value, deferred_messages: &mut Vec<Value>) {
    for message in std::mem::take(deferred_messages) {
        push_message(out, message);
    }
}

fn append_tools(out: &mut Value, body: &Value) {
    let Some(tools) = body.get("tools").and_then(Value::as_array) else {
        return;
    };
    let mut converted = Vec::new();
    for tool in tools {
        converted.extend(convert_responses_tool_to_chat_tools(tool));
    }
    if !converted.is_empty() {
        out["tools"] = Value::Array(converted);
    }
}

fn convert_responses_tool_to_chat_tools(tool: &Value) -> Vec<Value> {
    match tool
        .get("type")
        .and_then(Value::as_str)
        .unwrap_or("")
        .trim()
    {
        "" | "function" => convert_responses_function_tool_to_chat(tool, "")
            .into_iter()
            .collect(),
        "namespace" => convert_responses_namespace_tool_to_chat(tool),
        _ => Vec::new(),
    }
}

fn convert_responses_namespace_tool_to_chat(tool: &Value) -> Vec<Value> {
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
        let qualified_name =
            qualify_responses_namespace_tool_name(namespace_name, responses_tool_name(child));
        if let Some(converted) = convert_responses_function_tool_to_chat(child, &qualified_name) {
            out.push(converted);
        }
    }
    out
}

fn convert_responses_function_tool_to_chat(tool: &Value, override_name: &str) -> Option<Value> {
    let name = if override_name.trim().is_empty() {
        responses_tool_name(tool).trim().to_string()
    } else {
        override_name.trim().to_string()
    };
    if name.is_empty() {
        return None;
    }

    let mut chat_tool = json!({
        "type": "function",
        "function": {
            "name": name,
            "description": "",
            "parameters": {},
        },
    });
    if let Some(description) = responses_tool_description(tool).filter(|value| !value.is_empty()) {
        chat_tool["function"]["description"] = json!(description);
    }
    if let Some(parameters) = responses_tool_parameters(tool) {
        chat_tool["function"]["parameters"] = parameters.clone();
    }
    Some(chat_tool)
}

fn collect_reasoning_content(item: &Value) -> String {
    let mut out = String::new();
    if let Some(summary) = item.get("summary").and_then(Value::as_array) {
        for summary_item in summary {
            if summary_item.get("type").and_then(Value::as_str) != Some("summary_text") {
                continue;
            }
            out.push_str(
                summary_item
                    .get("text")
                    .and_then(Value::as_str)
                    .unwrap_or(""),
            );
        }
    }
    if out.is_empty() {
        "[reasoning unavailable]".to_string()
    } else {
        out
    }
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

fn responses_tool_parameters(tool: &Value) -> Option<&Value> {
    [
        "/parameters",
        "/parametersJsonSchema",
        "/input_schema",
        "/function/parameters",
        "/function/parametersJsonSchema",
    ]
    .into_iter()
    .find_map(|path| tool.pointer(path))
}

fn qualify_responses_namespace_tool_name(namespace_name: &str, child_name: &str) -> String {
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

fn push_message(out: &mut Value, message: Value) {
    out["messages"]
        .as_array_mut()
        .expect("messages is array")
        .push(message);
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
    fn string_input_maps_to_user_message() {
        let result = transform("gpt-4", json!({"input": "Hello"}), false);
        assert_eq!(result["messages"][0]["role"], "user");
        assert_eq!(result["messages"][0]["content"], "Hello");
    }

    #[test]
    fn developer_message_maps_to_user() {
        let result = transform(
            "gpt-4",
            json!({"input": [{"role": "developer", "content": "You are helpful."}]}),
            false,
        );
        assert_eq!(result["messages"][0]["role"], "user");
        assert_eq!(result["messages"][0]["content"], "You are helpful.");
    }
}
