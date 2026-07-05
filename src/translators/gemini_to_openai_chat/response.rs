//! Gemini response -> OpenAI Chat Completions response translation.

use serde_json::{Value, json};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use translator_infra::util;

#[derive(Debug, Default)]
struct StreamState {
    unix_timestamp: i64,
    function_index: HashMap<i64, i64>,
    saw_tool_call: HashMap<i64, bool>,
    upstream_finish_reason: HashMap<i64, String>,
    sanitized_name_map: Option<HashMap<String, String>>,
}

static FUNCTION_CALL_ID_COUNTER: AtomicU64 = AtomicU64::new(0);

pub fn transform_stream(
    _model: &str,
    original_request: &Value,
    _translated_request: &Value,
    chunk: Value,
    param: Option<&mut Box<dyn std::any::Any>>,
) -> Vec<Value> {
    let root = normalize_chunk_value(chunk);
    if root.as_str() == Some("[DONE]") {
        return vec![];
    }

    let mut local_state = StreamState::default();
    let state = if let Some(param) = param {
        if !param.is::<StreamState>() {
            *param = Box::<StreamState>::default();
        }
        param.downcast_mut::<StreamState>().unwrap()
    } else {
        &mut local_state
    };
    if state.sanitized_name_map.is_none() {
        state.sanitized_name_map = sanitized_tool_name_map(original_request);
    }

    let mut base = openai_stream_base();
    apply_common_response_fields(&mut base, &root, Some(state));
    apply_usage(&mut base, &root, true);

    let mut out = Vec::new();
    if let Some(candidates) = root.get("candidates") {
        if let Some(candidates) = candidates.as_array() {
            for candidate in candidates {
                let mut template = base.clone();
                let candidate_index = candidate.get("index").and_then(Value::as_i64).unwrap_or(0);
                template["choices"][0]["index"] = json!(candidate_index);

                if let Some(finish_reason) = candidate.get("finishReason").and_then(Value::as_str) {
                    state
                        .upstream_finish_reason
                        .insert(candidate_index, finish_reason.to_uppercase());
                }

                if let Some(parts) = candidate
                    .pointer("/content/parts")
                    .and_then(Value::as_array)
                {
                    for part in parts {
                        let thought_signature = part
                            .get("thoughtSignature")
                            .or_else(|| part.get("thought_signature"))
                            .and_then(Value::as_str)
                            .unwrap_or("");
                        let has_content_payload = part.get("text").is_some()
                            || part.get("functionCall").is_some()
                            || part.get("inlineData").is_some()
                            || part.get("inline_data").is_some();
                        if !thought_signature.is_empty() && !has_content_payload {
                            continue;
                        }

                        if let Some(text_value) = part.get("text") {
                            let text = value_to_string(text_value);
                            if part
                                .get("thought")
                                .and_then(Value::as_bool)
                                .unwrap_or(false)
                            {
                                template["choices"][0]["delta"]["reasoning_content"] = json!(text);
                            } else {
                                template["choices"][0]["delta"]["content"] = json!(text);
                            }
                            template["choices"][0]["delta"]["role"] = json!("assistant");
                        } else if let Some(function_call) = part.get("functionCall") {
                            state.saw_tool_call.insert(candidate_index, true);
                            if !template["choices"][0]["delta"]["tool_calls"].is_array() {
                                template["choices"][0]["delta"]["tool_calls"] = json!([]);
                            }
                            let tool_calls_len = template["choices"][0]["delta"]["tool_calls"]
                                .as_array()
                                .map(|items| items.len() as i64)
                                .unwrap_or(0);
                            let function_call_index =
                                if template["choices"][0]["delta"]["tool_calls"].is_array()
                                    && tool_calls_len > 0
                                {
                                    tool_calls_len
                                } else {
                                    let current =
                                        *state.function_index.get(&candidate_index).unwrap_or(&0);
                                    state.function_index.insert(candidate_index, current + 1);
                                    current
                                };
                            let raw_name = function_call
                                .get("name")
                                .and_then(Value::as_str)
                                .unwrap_or("");
                            let name = restore_sanitized_tool_name(
                                state.sanitized_name_map.as_ref(),
                                raw_name,
                            );
                            let mut tool_call = json!({
                                "id": generated_function_call_id(&name),
                                "index": function_call_index,
                                "type": "function",
                                "function": {
                                    "name": name,
                                    "arguments": "",
                                },
                            });
                            if let Some(args) = function_call.get("args") {
                                tool_call["function"]["arguments"] = json!(json_raw_string(args));
                            }
                            template["choices"][0]["delta"]["role"] = json!("assistant");
                            template["choices"][0]["delta"]["tool_calls"]
                                .as_array_mut()
                                .unwrap()
                                .push(tool_call);
                        } else if let Some(inline_data) =
                            part.get("inlineData").or_else(|| part.get("inline_data"))
                            && let Some(image_payload) = inline_data_to_openai_image(inline_data)
                        {
                            if !template["choices"][0]["delta"]["images"].is_array() {
                                template["choices"][0]["delta"]["images"] = json!([]);
                            }
                            let image_index = template["choices"][0]["delta"]["images"]
                                .as_array()
                                .map(|items| items.len())
                                .unwrap_or(0);
                            let mut image_payload = image_payload;
                            image_payload["index"] = json!(image_index);
                            template["choices"][0]["delta"]["role"] = json!("assistant");
                            template["choices"][0]["delta"]["images"]
                                .as_array_mut()
                                .unwrap()
                                .push(image_payload);
                        }
                    }
                }

                let upstream_finish_reason = state
                    .upstream_finish_reason
                    .get(&candidate_index)
                    .cloned()
                    .unwrap_or_default();
                let saw_tool_call = state
                    .saw_tool_call
                    .get(&candidate_index)
                    .copied()
                    .unwrap_or(false);
                let is_final_chunk =
                    !upstream_finish_reason.is_empty() && root.get("usageMetadata").is_some();
                if is_final_chunk {
                    let finish_reason = if saw_tool_call {
                        "tool_calls"
                    } else if upstream_finish_reason == "MAX_TOKENS" {
                        "max_tokens"
                    } else {
                        "stop"
                    };
                    template["choices"][0]["finish_reason"] = json!(finish_reason);
                    template["choices"][0]["native_finish_reason"] =
                        json!(upstream_finish_reason.to_lowercase());
                }

                out.push(template);
            }
        }
    } else if root.get("usageMetadata").is_some() {
        out.push(base);
    }

    out
}

pub fn transform_non_stream(
    _model: &str,
    original_request: &Value,
    _translated_request: &Value,
    response: Value,
    _param: Option<&mut Box<dyn std::any::Any>>,
) -> Value {
    let root = normalize_chunk_value(response);
    let sanitized_name_map = sanitized_tool_name_map(original_request);
    let mut out = json!({
        "id": "",
        "object": "chat.completion",
        "created": 0,
        "model": "model",
        "choices": [],
    });

    apply_common_response_fields(&mut out, &root, None);
    apply_usage(&mut out, &root, false);

    if let Some(candidates) = root.get("candidates").and_then(Value::as_array) {
        for candidate in candidates {
            let mut choice = json!({
                "index": candidate.get("index").and_then(Value::as_i64).unwrap_or(0),
                "message": {
                    "role": "assistant",
                    "content": null,
                    "reasoning_content": null,
                    "tool_calls": null,
                },
                "finish_reason": null,
                "native_finish_reason": null,
            });

            if let Some(finish_reason) = candidate.get("finishReason").and_then(Value::as_str) {
                choice["finish_reason"] = json!(finish_reason.to_lowercase());
                choice["native_finish_reason"] = json!(finish_reason.to_lowercase());
            }

            let mut has_function_call = false;
            if let Some(parts) = candidate
                .pointer("/content/parts")
                .and_then(Value::as_array)
            {
                for part in parts {
                    if let Some(text_value) = part.get("text") {
                        let text = value_to_string(text_value);
                        if part
                            .get("thought")
                            .and_then(Value::as_bool)
                            .unwrap_or(false)
                        {
                            let old = choice["message"]["reasoning_content"]
                                .as_str()
                                .unwrap_or("");
                            choice["message"]["reasoning_content"] = json!(format!("{old}{text}"));
                        } else {
                            let old = choice["message"]["content"].as_str().unwrap_or("");
                            choice["message"]["content"] = json!(format!("{old}{text}"));
                        }
                        choice["message"]["role"] = json!("assistant");
                    } else if let Some(function_call) = part.get("functionCall") {
                        has_function_call = true;
                        if !choice["message"]["tool_calls"].is_array() {
                            choice["message"]["tool_calls"] = json!([]);
                        }
                        let raw_name = function_call
                            .get("name")
                            .and_then(Value::as_str)
                            .unwrap_or("");
                        let name =
                            restore_sanitized_tool_name(sanitized_name_map.as_ref(), raw_name);
                        let mut tool_call = json!({
                            "id": generated_function_call_id(&name),
                            "type": "function",
                            "function": {
                                "name": name,
                                "arguments": "",
                            },
                        });
                        if let Some(args) = function_call.get("args") {
                            tool_call["function"]["arguments"] = json!(json_raw_string(args));
                        }
                        choice["message"]["role"] = json!("assistant");
                        choice["message"]["tool_calls"]
                            .as_array_mut()
                            .unwrap()
                            .push(tool_call);
                    } else if let Some(inline_data) =
                        part.get("inlineData").or_else(|| part.get("inline_data"))
                        && let Some(mut image_payload) = inline_data_to_openai_image(inline_data)
                    {
                        if !choice["message"]["images"].is_array() {
                            choice["message"]["images"] = json!([]);
                        }
                        let image_index = choice["message"]["images"]
                            .as_array()
                            .map(|items| items.len())
                            .unwrap_or(0);
                        image_payload["index"] = json!(image_index);
                        choice["message"]["role"] = json!("assistant");
                        choice["message"]["images"]
                            .as_array_mut()
                            .unwrap()
                            .push(image_payload);
                    }
                }
            }

            if has_function_call {
                choice["finish_reason"] = json!("tool_calls");
                choice["native_finish_reason"] = json!("tool_calls");
            }
            out["choices"].as_array_mut().unwrap().push(choice);
        }
    }

    out
}

fn openai_stream_base() -> Value {
    json!({
        "id": "",
        "object": "chat.completion.chunk",
        "created": 0,
        "model": "model",
        "choices": [{
            "index": 0,
            "delta": {
                "role": null,
                "content": null,
                "reasoning_content": null,
                "tool_calls": null,
            },
            "finish_reason": null,
            "native_finish_reason": null,
        }],
    })
}

fn apply_common_response_fields(out: &mut Value, root: &Value, state: Option<&mut StreamState>) {
    if let Some(model_version) = root.get("modelVersion").and_then(Value::as_str) {
        out["model"] = json!(model_version);
    }

    let mut created = 0_i64;
    if root.get("createTime").is_some() {
        created = parse_rfc3339_unix(root.get("createTime").and_then(Value::as_str).unwrap_or(""));
    }
    if let Some(state) = state {
        if root.get("createTime").is_some() {
            state.unix_timestamp = created;
        } else {
            created = state.unix_timestamp;
        }
    }
    out["created"] = json!(created);

    if let Some(response_id) = root.get("responseId").and_then(Value::as_str) {
        out["id"] = json!(response_id);
    }
}

fn apply_usage(out: &mut Value, root: &Value, _stream: bool) {
    let Some(usage) = root.get("usageMetadata") else {
        return;
    };
    if let Some(candidates) = usage.get("candidatesTokenCount") {
        out["usage"]["completion_tokens"] = numeric_i64(candidates);
    }
    if let Some(total) = usage.get("totalTokenCount") {
        out["usage"]["total_tokens"] = numeric_i64(total);
    }
    out["usage"]["prompt_tokens"] = json!(
        usage
            .get("promptTokenCount")
            .and_then(Value::as_i64)
            .unwrap_or(0)
    );
    if let Some(thoughts) = usage.get("thoughtsTokenCount").and_then(Value::as_i64)
        && thoughts > 0
    {
        out["usage"]["completion_tokens_details"]["reasoning_tokens"] = json!(thoughts);
    }
    if let Some(cached) = usage.get("cachedContentTokenCount").and_then(Value::as_i64)
        && cached > 0
    {
        out["usage"]["prompt_tokens_details"]["cached_tokens"] = json!(cached);
    }
}

fn inline_data_to_openai_image(inline_data: &Value) -> Option<Value> {
    let data = inline_data
        .get("data")
        .and_then(Value::as_str)
        .unwrap_or("");
    if data.is_empty() {
        return None;
    }
    let mime_type = inline_data
        .get("mimeType")
        .or_else(|| inline_data.get("mime_type"))
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty())
        .unwrap_or("image/png");
    Some(json!({
        "type": "image_url",
        "image_url": {"url": format!("data:{mime_type};base64,{data}")},
    }))
}

fn sanitized_tool_name_map(original_request: &Value) -> Option<HashMap<String, String>> {
    let tools = original_request.get("tools").and_then(Value::as_array)?;
    let mut out = HashMap::new();
    for tool in tools {
        let Some(name) = tool.get("name").and_then(Value::as_str).map(str::trim) else {
            continue;
        };
        if name.is_empty() {
            continue;
        }
        let sanitized = util::sanitize_function_name(name);
        if sanitized != name && !out.contains_key(&sanitized) {
            out.insert(sanitized, name.to_string());
        }
    }
    if out.is_empty() { None } else { Some(out) }
}

fn restore_sanitized_tool_name(map: Option<&HashMap<String, String>>, sanitized: &str) -> String {
    if sanitized.is_empty() {
        return String::new();
    }
    map.and_then(|map| map.get(sanitized).cloned())
        .unwrap_or_else(|| sanitized.to_string())
}

fn generated_function_call_id(name: &str) -> String {
    let counter = FUNCTION_CALL_ID_COUNTER.fetch_add(1, Ordering::Relaxed) + 1;
    format!("{name}-1000000000000000000-{counter}")
}

fn normalize_chunk_value(value: Value) -> Value {
    let Some(raw) = value.as_str() else {
        return value;
    };
    if raw.trim() == "[DONE]" {
        return Value::String("[DONE]".to_string());
    }
    let payload = raw
        .trim()
        .strip_prefix("data:")
        .map(str::trim)
        .unwrap_or_else(|| raw.trim());
    serde_json::from_str(payload).unwrap_or(value)
}

fn numeric_i64(value: &Value) -> Value {
    json!(value.as_i64().unwrap_or(0))
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

fn parse_rfc3339_unix(value: &str) -> i64 {
    if value.len() < 19 {
        return 0;
    }
    let Ok(year) = value[0..4].parse::<i32>() else {
        return 0;
    };
    let Ok(month) = value[5..7].parse::<u32>() else {
        return 0;
    };
    let Ok(day) = value[8..10].parse::<u32>() else {
        return 0;
    };
    let Ok(hour) = value[11..13].parse::<u32>() else {
        return 0;
    };
    let Ok(minute) = value[14..16].parse::<u32>() else {
        return 0;
    };
    let Ok(second) = value[17..19].parse::<u32>() else {
        return 0;
    };
    days_from_civil(year, month, day) * 86_400
        + hour as i64 * 3_600
        + minute as i64 * 60
        + second as i64
}

fn days_from_civil(year: i32, month: u32, day: u32) -> i64 {
    let year = year - (month <= 2) as i32;
    let era = if year >= 0 { year } else { year - 399 } / 400;
    let yoe = year - era * 400;
    let month = month as i32;
    let doy = (153 * (month + if month > 2 { -3 } else { 9 }) + 2) / 5 + day as i32 - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    (era * 146097 + doe - 719468) as i64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn non_stream_basic() {
        let resp = json!({
            "responseId": "resp-1",
            "modelVersion": "gemini-test",
            "candidates": [{
                "index": 0,
                "content": {"role": "model", "parts": [{"text": "Hello"}]},
                "finishReason": "STOP"
            }],
            "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 5, "totalTokenCount": 15}
        });
        let result = transform_non_stream("gpt-4", &json!({}), &json!({}), resp, None);
        assert_eq!(result["id"], "resp-1");
        assert_eq!(result["model"], "gemini-test");
        assert_eq!(result["choices"][0]["message"]["content"], "Hello");
        assert_eq!(result["choices"][0]["finish_reason"], "stop");
    }

    #[test]
    fn stream_finish_reason_waits_for_usage() {
        let mut param: Box<dyn std::any::Any> = Box::new(());
        let first = transform_stream(
            "gpt-4",
            &json!({}),
            &json!({}),
            json!({"candidates":[{"index":0,"content":{"parts":[{"functionCall":{"name":"list_dir","args":{"path":"C:/"}}}]}}]}),
            Some(&mut param),
        );
        assert!(first[0]["choices"][0]["finish_reason"].is_null());
        let final_chunk = transform_stream(
            "gpt-4",
            &json!({}),
            &json!({}),
            json!({"candidates":[{"index":0,"content":{"parts":[{"text":""}]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":1}}),
            Some(&mut param),
        );
        assert_eq!(final_chunk[0]["choices"][0]["finish_reason"], "tool_calls");
        assert_eq!(final_chunk[0]["choices"][0]["native_finish_reason"], "stop");
    }
}
