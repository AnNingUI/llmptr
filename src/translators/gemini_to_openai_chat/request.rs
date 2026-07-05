//! Google Gemini -> OpenAI Chat Completions request translation.

use serde_json::{Value, json};
use std::sync::atomic::{AtomicU64, Ordering};
use llmptr_infra::thinking;

static TOOL_CALL_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Convert a Gemini generateContent request to OpenAI Chat format.
pub fn transform(model: &str, body: Value, stream: bool) -> Value {
    let mut out = json!({
        "model": model,
        "messages": [],
        "stream": stream,
    });

    apply_generation_config(&mut out, &body);
    append_system_instruction(&mut out, &body);
    append_contents(&mut out, &body);
    append_tools(&mut out, &body);
    apply_tool_choice(&mut out, &body);

    out
}

fn apply_generation_config(out: &mut Value, body: &Value) {
    let Some(gen_config) = body.get("generationConfig") else {
        return;
    };

    if let Some(temp) = gen_config.get("temperature") {
        out["temperature"] = numeric_value(temp, true);
    }
    if let Some(max_tokens) = gen_config.get("maxOutputTokens") {
        out["max_tokens"] = numeric_value(max_tokens, false);
    }
    if let Some(top_p) = gen_config.get("topP") {
        out["top_p"] = numeric_value(top_p, true);
    }
    if let Some(top_k) = gen_config.get("topK") {
        out["top_k"] = numeric_value(top_k, false);
    }
    if let Some(stop_sequences) = gen_config.get("stopSequences").and_then(Value::as_array) {
        let stops: Vec<Value> = stop_sequences
            .iter()
            .map(|value| json!(value_to_string(value)))
            .collect();
        if !stops.is_empty() {
            out["stop"] = Value::Array(stops);
        }
    }
    if let Some(candidate_count) = gen_config.get("candidateCount") {
        out["n"] = numeric_value(candidate_count, false);
    }

    if let Some(thinking_config) = gen_config.get("thinkingConfig").and_then(Value::as_object) {
        let thinking_level = thinking_config
            .get("thinkingLevel")
            .or_else(|| thinking_config.get("thinking_level"))
            .and_then(Value::as_str)
            .map(|value| value.trim().to_lowercase())
            .filter(|value| !value.is_empty());
        if let Some(level) = thinking_level {
            out["reasoning_effort"] = json!(level);
        } else if let Some(budget) = thinking_config
            .get("thinkingBudget")
            .or_else(|| thinking_config.get("thinking_budget"))
            .and_then(Value::as_i64)
        {
            let (level, ok) = thinking::convert_budget_to_level(budget);
            if ok {
                out["reasoning_effort"] = json!(level);
            }
        }
    }
}

fn append_system_instruction(out: &mut Value, body: &Value) {
    let system_instruction = body
        .get("systemInstruction")
        .or_else(|| body.get("system_instruction"));
    let Some(system_instruction) = system_instruction else {
        return;
    };

    let mut content = Vec::new();
    if let Some(parts) = system_instruction.get("parts").and_then(Value::as_array) {
        for part in parts {
            if let Some(text) = part.get("text") {
                content.push(json!({"type": "text", "text": value_to_string(text)}));
            }
            if let Some(inline_data) = part.get("inlineData") {
                content.push(inline_data_to_openai_image(
                    inline_data,
                    "application/octet-stream",
                ));
            }
        }
    }

    if !content.is_empty() {
        push_message(out, json!({"role": "system", "content": content}));
    }
}

fn append_contents(out: &mut Value, body: &Value) {
    let Some(contents) = body.get("contents").and_then(Value::as_array) else {
        return;
    };

    let mut tool_call_ids = Vec::new();
    let mut tool_call_consume_idx = 0usize;

    for content in contents {
        let mut role = content
            .get("role")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string();
        if role == "model" {
            role = "assistant".to_string();
        }

        let mut msg = json!({"role": role, "content": ""});
        let mut text_builder = String::new();
        let mut content_parts = Vec::new();
        let mut only_text_content = true;
        let mut tool_calls = Vec::new();

        if let Some(parts) = content.get("parts").and_then(Value::as_array) {
            for part in parts {
                if let Some(text_value) = part.get("text") {
                    let text = value_to_string(text_value);
                    text_builder.push_str(&text);
                    content_parts.push(json!({"type": "text", "text": text}));
                }

                if let Some(inline_data) = part.get("inlineData") {
                    only_text_content = false;
                    content_parts.push(inline_data_to_openai_image(
                        inline_data,
                        "application/octet-stream",
                    ));
                }

                if let Some(function_call) = part.get("functionCall") {
                    let tool_call_id = gen_tool_call_id();
                    tool_call_ids.push(tool_call_id.clone());
                    let arguments = function_call
                        .get("args")
                        .map(json_raw_string)
                        .unwrap_or_else(|| "{}".to_string());
                    tool_calls.push(json!({
                        "id": tool_call_id,
                        "type": "function",
                        "function": {
                            "name": function_call.get("name").and_then(Value::as_str).unwrap_or(""),
                            "arguments": arguments,
                        },
                    }));
                }

                if let Some(function_response) = part.get("functionResponse") {
                    let mut tool_msg = json!({
                        "role": "tool",
                        "tool_call_id": "",
                        "content": "",
                    });
                    if let Some(response) = function_response.get("response") {
                        let content = response
                            .get("content")
                            .map(json_raw_string)
                            .unwrap_or_else(|| json_raw_string(response));
                        tool_msg["content"] = json!(content);
                    }
                    if tool_call_consume_idx < tool_call_ids.len() {
                        tool_msg["tool_call_id"] = json!(tool_call_ids[tool_call_consume_idx]);
                        tool_call_consume_idx += 1;
                    } else {
                        tool_msg["tool_call_id"] = json!(gen_tool_call_id());
                    }
                    push_message(out, tool_msg);
                }
            }
        }

        if !content_parts.is_empty() {
            msg["content"] = if only_text_content {
                json!(text_builder)
            } else {
                Value::Array(content_parts)
            };
        }
        if !tool_calls.is_empty() {
            msg["tool_calls"] = Value::Array(tool_calls);
        }
        push_message(out, msg);
    }
}

fn append_tools(out: &mut Value, body: &Value) {
    let Some(tools) = body.get("tools").and_then(Value::as_array) else {
        return;
    };

    let mut openai_tools = Vec::new();
    for tool in tools {
        let Some(function_declarations) =
            tool.get("functionDeclarations").and_then(Value::as_array)
        else {
            continue;
        };
        for declaration in function_declarations {
            let mut openai_tool = json!({
                "type": "function",
                "function": {
                    "name": declaration.get("name").and_then(Value::as_str).unwrap_or(""),
                    "description": declaration.get("description").and_then(Value::as_str).unwrap_or(""),
                },
            });
            if let Some(parameters) = declaration
                .get("parameters")
                .or_else(|| declaration.get("parametersJsonSchema"))
            {
                openai_tool["function"]["parameters"] = parameters.clone();
            }
            openai_tools.push(openai_tool);
        }
    }

    if !openai_tools.is_empty() {
        out["tools"] = Value::Array(openai_tools);
    }
}

fn apply_tool_choice(out: &mut Value, body: &Value) {
    let Some(function_calling_config) = body
        .pointer("/toolConfig/functionCallingConfig")
        .and_then(Value::as_object)
    else {
        return;
    };
    match function_calling_config
        .get("mode")
        .and_then(Value::as_str)
        .unwrap_or("")
    {
        "NONE" => out["tool_choice"] = json!("none"),
        "AUTO" => out["tool_choice"] = json!("auto"),
        "ANY" => out["tool_choice"] = json!("required"),
        _ => {}
    }
}

fn inline_data_to_openai_image(inline_data: &Value, default_mime: &str) -> Value {
    let mime_type = inline_data
        .get("mimeType")
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty())
        .unwrap_or(default_mime);
    let data = inline_data
        .get("data")
        .and_then(Value::as_str)
        .unwrap_or("");
    json!({
        "type": "image_url",
        "image_url": {"url": format!("data:{mime_type};base64,{data}")},
    })
}

fn push_message(out: &mut Value, message: Value) {
    out["messages"]
        .as_array_mut()
        .expect("messages is array")
        .push(message);
}

fn gen_tool_call_id() -> String {
    let next = TOOL_CALL_COUNTER.fetch_add(1, Ordering::Relaxed);
    format!("call_{next:024}")
}

fn numeric_value(value: &Value, as_float: bool) -> Value {
    if as_float {
        json!(value.as_f64().unwrap_or(0.0))
    } else {
        json!(value.as_i64().unwrap_or(0))
    }
}

fn json_raw_string(value: &Value) -> String {
    serde_json::to_string(value).unwrap_or_default()
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
    fn basic_request_maps_to_openai() {
        let body = json!({
            "contents": [{"role": "user", "parts": [{"text": "Hello"}]}]
        });
        let result = transform("gpt-4", body, false);
        assert_eq!(result["model"], "gpt-4");
        assert_eq!(result["stream"], false);
        assert_eq!(result["messages"][0]["role"], "user");
        assert_eq!(result["messages"][0]["content"], "Hello");
    }

    #[test]
    fn function_responses_consume_ids_fifo() {
        let body = json!({
            "contents": [
                {"role": "model", "parts": [
                    {"functionCall": {"name": "read_file", "args": {"path": "a.txt"}}},
                    {"functionCall": {"name": "grep", "args": {"pattern": "needle"}}}
                ]},
                {"role": "function", "parts": [
                    {"functionResponse": {"name": "read_file", "response": {"result": "a"}}},
                    {"functionResponse": {"name": "grep", "response": {"result": "b"}}}
                ]}
            ]
        });
        let result = transform("gpt-4", body, false);
        let first = result["messages"][0]["tool_calls"][0]["id"]
            .as_str()
            .unwrap();
        let second = result["messages"][0]["tool_calls"][1]["id"]
            .as_str()
            .unwrap();
        assert_eq!(result["messages"][1]["tool_call_id"], first);
        assert_eq!(result["messages"][2]["tool_call_id"], second);
    }
}
