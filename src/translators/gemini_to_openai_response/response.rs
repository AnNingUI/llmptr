//! Gemini response -> OpenAI Responses response translation.

use serde_json::{Value, json};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use translator_infra::util;

const GEMINI_RESPONSES_THOUGHT_SIGNATURE: &str = "skip_thought_signature_validator";

static RESPONSE_ID_COUNTER: AtomicU64 = AtomicU64::new(0);
static FUNC_CALL_ID_COUNTER: AtomicU64 = AtomicU64::new(0);

#[derive(Debug, Default)]
struct StreamState {
    seq: i64,
    response_id: String,
    created_at: i64,
    started: bool,

    msg_opened: bool,
    msg_closed: bool,
    msg_index: i64,
    current_msg_id: String,
    text_buf: String,
    item_text_buf: String,

    reasoning_opened: bool,
    reasoning_index: i64,
    reasoning_item_id: String,
    reasoning_enc: String,
    reasoning_buf: String,
    reasoning_closed: bool,

    next_index: i64,
    func_args_buf: HashMap<i64, String>,
    func_names: HashMap<i64, String>,
    func_call_ids: HashMap<i64, String>,
    func_done: HashMap<i64, bool>,
    sanitized_name_map: Option<HashMap<String, String>>,
}

impl StreamState {
    fn next_seq(&mut self) -> i64 {
        self.seq += 1;
        self.seq
    }
}

pub fn transform_stream(
    _model: &str,
    original_request: &Value,
    _translated_request: &Value,
    chunk: Value,
    param: Option<&mut Box<dyn std::any::Any>>,
) -> Vec<Value> {
    let root = normalize_chunk_value(chunk);
    if root.as_str() == Some("[DONE]") || root.is_null() {
        return Vec::new();
    }
    let root = unwrap_gemini_response_root(root);
    if !root.is_object() {
        return Vec::new();
    }

    let mut local_state = StreamState::default();
    let st = if let Some(param) = param {
        if !param.is::<StreamState>() {
            *param = Box::<StreamState>::default();
        }
        param.downcast_mut::<StreamState>().unwrap()
    } else {
        &mut local_state
    };

    if st.sanitized_name_map.is_none() {
        st.sanitized_name_map = sanitized_tool_name_map(original_request);
    }

    let mut out = Vec::new();

    if !st.started {
        st.response_id = response_id_from_root(&root);
        st.created_at = root
            .get("createTime")
            .and_then(Value::as_str)
            .map(parse_rfc3339_unix)
            .filter(|value| *value != 0)
            .unwrap_or_else(now_unix);

        out.push(sse(
            "response.created",
            json!({
                "type": "response.created",
                "sequence_number": st.next_seq(),
                "response": {
                    "id": st.response_id,
                    "object": "response",
                    "created_at": st.created_at,
                    "status": "in_progress",
                    "background": false,
                    "error": null,
                    "output": [],
                },
            }),
        ));
        out.push(sse(
            "response.in_progress",
            json!({
                "type": "response.in_progress",
                "sequence_number": st.next_seq(),
                "response": {
                    "id": st.response_id,
                    "object": "response",
                    "created_at": st.created_at,
                    "status": "in_progress",
                },
            }),
        ));
        st.started = true;
        st.next_index = 0;
    }

    if let Some(parts) = root
        .pointer("/candidates/0/content/parts")
        .and_then(Value::as_array)
    {
        for part in parts {
            if part
                .get("thought")
                .and_then(Value::as_bool)
                .unwrap_or(false)
            {
                if st.reasoning_closed {
                    continue;
                }
                if let Some(sig) = part
                    .get("thoughtSignature")
                    .or_else(|| part.get("thought_signature"))
                    .and_then(Value::as_str)
                    .filter(|sig| !sig.is_empty() && *sig != GEMINI_RESPONSES_THOUGHT_SIGNATURE)
                {
                    st.reasoning_enc = sig.to_string();
                }
                if !st.reasoning_opened {
                    st.reasoning_opened = true;
                    st.reasoning_index = st.next_index;
                    st.next_index += 1;
                    st.reasoning_item_id = format!("rs_{}_{}", st.response_id, st.reasoning_index);
                    out.push(sse(
                        "response.output_item.added",
                        json!({
                            "type": "response.output_item.added",
                            "sequence_number": st.next_seq(),
                            "output_index": st.reasoning_index,
                            "item": {
                                "id": st.reasoning_item_id,
                                "type": "reasoning",
                                "status": "in_progress",
                                "encrypted_content": st.reasoning_enc,
                                "summary": [],
                            },
                        }),
                    ));
                    out.push(sse(
                        "response.reasoning_summary_part.added",
                        json!({
                            "type": "response.reasoning_summary_part.added",
                            "sequence_number": st.next_seq(),
                            "item_id": st.reasoning_item_id,
                            "output_index": st.reasoning_index,
                            "summary_index": 0,
                            "part": {"type": "summary_text", "text": ""},
                        }),
                    ));
                }
                if let Some(text) = part.get("text").and_then(Value::as_str)
                    && !text.is_empty()
                {
                    st.reasoning_buf.push_str(text);
                    out.push(sse(
                        "response.reasoning_summary_text.delta",
                        json!({
                            "type": "response.reasoning_summary_text.delta",
                            "sequence_number": st.next_seq(),
                            "item_id": st.reasoning_item_id,
                            "output_index": st.reasoning_index,
                            "summary_index": 0,
                            "delta": text,
                        }),
                    ));
                }
                continue;
            }

            if let Some(text) = part.get("text").and_then(Value::as_str) {
                if !text.is_empty() {
                    finalize_reasoning(st, &mut out);
                    if !st.msg_opened {
                        st.msg_opened = true;
                        st.msg_index = st.next_index;
                        st.next_index += 1;
                        st.current_msg_id = format!("msg_{}_0", st.response_id);
                        st.item_text_buf.clear();
                        out.push(sse(
                            "response.output_item.added",
                            json!({
                                "type": "response.output_item.added",
                                "sequence_number": st.next_seq(),
                                "output_index": st.msg_index,
                                "item": {
                                    "id": st.current_msg_id,
                                    "type": "message",
                                    "status": "in_progress",
                                    "content": [],
                                    "role": "assistant",
                                },
                            }),
                        ));
                        out.push(sse(
                            "response.content_part.added",
                            json!({
                                "type": "response.content_part.added",
                                "sequence_number": st.next_seq(),
                                "item_id": st.current_msg_id,
                                "output_index": st.msg_index,
                                "content_index": 0,
                                "part": {
                                    "type": "output_text",
                                    "annotations": [],
                                    "logprobs": [],
                                    "text": "",
                                },
                            }),
                        ));
                    }
                    st.text_buf.push_str(text);
                    st.item_text_buf.push_str(text);
                    out.push(sse(
                        "response.output_text.delta",
                        json!({
                            "type": "response.output_text.delta",
                            "sequence_number": st.next_seq(),
                            "item_id": st.current_msg_id,
                            "output_index": st.msg_index,
                            "content_index": 0,
                            "delta": text,
                            "logprobs": [],
                        }),
                    ));
                }
                continue;
            }

            if let Some(function_call) = part.get("functionCall") {
                finalize_reasoning(st, &mut out);
                finalize_message(st, &mut out);

                let raw_name = function_call
                    .get("name")
                    .and_then(Value::as_str)
                    .unwrap_or("");
                let name = restore_sanitized_tool_name(st.sanitized_name_map.as_ref(), raw_name);
                let idx = st.next_index;
                st.next_index += 1;
                st.func_args_buf.entry(idx).or_default();
                if st
                    .func_call_ids
                    .get(&idx)
                    .map(String::is_empty)
                    .unwrap_or(true)
                {
                    st.func_call_ids.insert(idx, generated_stream_call_id());
                }
                st.func_names.insert(idx, name.clone());

                let args_json = function_call
                    .get("args")
                    .map(json_raw_string)
                    .unwrap_or_else(|| "{}".to_string());
                if st
                    .func_args_buf
                    .get(&idx)
                    .map(String::is_empty)
                    .unwrap_or(true)
                    && !args_json.is_empty()
                {
                    st.func_args_buf.insert(idx, args_json.clone());
                }

                let call_id = st.func_call_ids.get(&idx).cloned().unwrap_or_default();
                out.push(sse(
                    "response.output_item.added",
                    json!({
                        "type": "response.output_item.added",
                        "sequence_number": st.next_seq(),
                        "output_index": idx,
                        "item": {
                            "id": format!("fc_{call_id}"),
                            "type": "function_call",
                            "status": "in_progress",
                            "arguments": "",
                            "call_id": call_id,
                            "name": name,
                        },
                    }),
                ));

                if !args_json.is_empty() {
                    out.push(sse(
                        "response.function_call_arguments.delta",
                        json!({
                            "type": "response.function_call_arguments.delta",
                            "sequence_number": st.next_seq(),
                            "item_id": format!("fc_{call_id}"),
                            "output_index": idx,
                            "delta": args_json,
                        }),
                    ));
                }

                if !st.func_done.get(&idx).copied().unwrap_or(false) {
                    out.push(sse(
                        "response.function_call_arguments.done",
                        json!({
                            "type": "response.function_call_arguments.done",
                            "sequence_number": st.next_seq(),
                            "item_id": format!("fc_{call_id}"),
                            "output_index": idx,
                            "arguments": args_json,
                        }),
                    ));
                    out.push(sse(
                        "response.output_item.done",
                        json!({
                            "type": "response.output_item.done",
                            "sequence_number": st.next_seq(),
                            "output_index": idx,
                            "item": {
                                "id": format!("fc_{call_id}"),
                                "type": "function_call",
                                "status": "completed",
                                "arguments": args_json,
                                "call_id": call_id,
                                "name": st.func_names.get(&idx).cloned().unwrap_or_default(),
                            },
                        }),
                    ));
                    st.func_done.insert(idx, true);
                }
            }
        }
    }

    if root
        .pointer("/candidates/0/finishReason")
        .and_then(Value::as_str)
        .filter(|reason| !reason.is_empty())
        .is_some()
    {
        finalize_reasoning(st, &mut out);
        finalize_message(st, &mut out);
        finalize_unfinished_functions(st, &mut out);

        let request_json = pick_request_value(original_request, _translated_request);
        let mut completed = json!({
            "type": "response.completed",
            "sequence_number": st.next_seq(),
            "response": {
                "id": st.response_id,
                "object": "response",
                "created_at": st.created_at,
                "status": "completed",
                "background": false,
                "error": null,
            },
        });
        echo_request_fields(&mut completed["response"], &request_json);

        let outputs = build_stream_outputs(st);
        if !outputs.is_empty() {
            completed["response"]["output"] = Value::Array(outputs);
        }
        if let Some(usage) = root.get("usageMetadata") {
            apply_stream_usage(&mut completed["response"], usage);
        }
        out.push(sse("response.completed", completed));
    }

    out
}

pub fn transform_non_stream(
    _model: &str,
    original_request: &Value,
    translated_request: &Value,
    response: Value,
    _param: Option<&mut Box<dyn std::any::Any>>,
) -> Value {
    let root = unwrap_gemini_response_root(normalize_chunk_value(response));
    let request_json = pick_request_value(original_request, translated_request);
    let sanitized_name_map = sanitized_tool_name_map(original_request);

    let id = response_id_from_root(&root);
    let created_at = root
        .get("createTime")
        .and_then(Value::as_str)
        .map(parse_rfc3339_unix)
        .filter(|value| *value != 0)
        .unwrap_or_else(now_unix);

    let mut out = json!({
        "id": id,
        "object": "response",
        "created_at": created_at,
        "status": "completed",
        "background": false,
        "error": null,
        "incomplete_details": null,
    });

    echo_request_fields(&mut out, &request_json);
    if out.get("model").is_none()
        && let Some(model) = root.get("modelVersion").and_then(Value::as_str)
    {
        out["model"] = json!(model);
    }

    let mut reasoning_text = String::new();
    let mut reasoning_encrypted = String::new();
    let mut message_text = String::new();
    let mut have_message = false;
    let mut output = Vec::new();

    if let Some(parts) = root
        .pointer("/candidates/0/content/parts")
        .and_then(Value::as_array)
    {
        for part in parts {
            if part
                .get("thought")
                .and_then(Value::as_bool)
                .unwrap_or(false)
            {
                if let Some(text) = part.get("text").and_then(Value::as_str) {
                    reasoning_text.push_str(text);
                }
                if let Some(sig) = part
                    .get("thoughtSignature")
                    .and_then(Value::as_str)
                    .filter(|sig| !sig.is_empty())
                {
                    reasoning_encrypted = sig.to_string();
                }
                continue;
            }

            if let Some(text) = part.get("text").and_then(Value::as_str)
                && !text.is_empty()
            {
                message_text.push_str(text);
                have_message = true;
                continue;
            }

            if let Some(function_call) = part.get("functionCall") {
                let raw_name = function_call
                    .get("name")
                    .and_then(Value::as_str)
                    .unwrap_or("");
                let name = restore_sanitized_tool_name(sanitized_name_map.as_ref(), raw_name);
                let call_id = generated_non_stream_call_id();
                let args = function_call
                    .get("args")
                    .map(json_raw_string)
                    .unwrap_or_default();
                output.push(json!({
                    "id": format!("fc_{call_id}"),
                    "type": "function_call",
                    "status": "completed",
                    "arguments": args,
                    "call_id": call_id,
                    "name": name,
                }));
            }
        }
    }

    if !reasoning_text.is_empty() || !reasoning_encrypted.is_empty() {
        let rid = id.strip_prefix("resp_").unwrap_or(&id);
        let mut item = json!({
            "id": format!("rs_{rid}"),
            "type": "reasoning",
            "encrypted_content": reasoning_encrypted,
        });
        if !reasoning_text.is_empty() {
            item["summary"] = json!([{"type": "summary_text", "text": reasoning_text}]);
        }
        output.push(item);
    }

    if have_message {
        let rid = id.strip_prefix("resp_").unwrap_or(&id);
        output.push(json!({
            "id": format!("msg_{rid}_0"),
            "type": "message",
            "status": "completed",
            "content": [{
                "type": "output_text",
                "annotations": [],
                "logprobs": [],
                "text": message_text,
            }],
            "role": "assistant",
        }));
    }

    if !output.is_empty() {
        out["output"] = Value::Array(output);
    }

    if let Some(usage) = root.get("usageMetadata") {
        apply_non_stream_usage(&mut out, usage);
    }

    out
}

fn finalize_reasoning(st: &mut StreamState, out: &mut Vec<Value>) {
    if !st.reasoning_opened || st.reasoning_closed {
        return;
    }
    let full = st.reasoning_buf.clone();
    out.push(sse(
        "response.reasoning_summary_text.done",
        json!({
            "type": "response.reasoning_summary_text.done",
            "sequence_number": st.next_seq(),
            "item_id": st.reasoning_item_id,
            "output_index": st.reasoning_index,
            "summary_index": 0,
            "text": full,
        }),
    ));
    out.push(sse(
        "response.reasoning_summary_part.done",
        json!({
            "type": "response.reasoning_summary_part.done",
            "sequence_number": st.next_seq(),
            "item_id": st.reasoning_item_id,
            "output_index": st.reasoning_index,
            "summary_index": 0,
            "part": {"type": "summary_text", "text": full},
        }),
    ));
    out.push(sse(
        "response.output_item.done",
        json!({
            "type": "response.output_item.done",
            "sequence_number": st.next_seq(),
            "output_index": st.reasoning_index,
            "item": {
                "id": st.reasoning_item_id,
                "type": "reasoning",
                "encrypted_content": st.reasoning_enc,
                "summary": [{"type": "summary_text", "text": full}],
            },
        }),
    ));
    st.reasoning_closed = true;
}

fn finalize_message(st: &mut StreamState, out: &mut Vec<Value>) {
    if !st.msg_opened || st.msg_closed {
        return;
    }
    let full_text = st.item_text_buf.clone();
    out.push(sse(
        "response.output_text.done",
        json!({
            "type": "response.output_text.done",
            "sequence_number": st.next_seq(),
            "item_id": st.current_msg_id,
            "output_index": st.msg_index,
            "content_index": 0,
            "text": full_text,
            "logprobs": [],
        }),
    ));
    out.push(sse(
        "response.content_part.done",
        json!({
            "type": "response.content_part.done",
            "sequence_number": st.next_seq(),
            "item_id": st.current_msg_id,
            "output_index": st.msg_index,
            "content_index": 0,
            "part": {
                "type": "output_text",
                "annotations": [],
                "logprobs": [],
                "text": full_text,
            },
        }),
    ));
    out.push(sse(
        "response.output_item.done",
        json!({
            "type": "response.output_item.done",
            "sequence_number": st.next_seq(),
            "output_index": st.msg_index,
            "item": {
                "id": st.current_msg_id,
                "type": "message",
                "status": "completed",
                "content": [{"type": "output_text", "text": full_text}],
                "role": "assistant",
            },
        }),
    ));
    st.msg_closed = true;
}

fn finalize_unfinished_functions(st: &mut StreamState, out: &mut Vec<Value>) {
    let mut indices: Vec<_> = st.func_args_buf.keys().copied().collect();
    indices.sort_unstable();
    for idx in indices {
        if st.func_done.get(&idx).copied().unwrap_or(false) {
            continue;
        }
        let args = st
            .func_args_buf
            .get(&idx)
            .cloned()
            .filter(|value| !value.is_empty())
            .unwrap_or_else(|| "{}".to_string());
        let call_id = st.func_call_ids.get(&idx).cloned().unwrap_or_default();
        out.push(sse(
            "response.function_call_arguments.done",
            json!({
                "type": "response.function_call_arguments.done",
                "sequence_number": st.next_seq(),
                "item_id": format!("fc_{call_id}"),
                "output_index": idx,
                "arguments": args,
            }),
        ));
        out.push(sse(
            "response.output_item.done",
            json!({
                "type": "response.output_item.done",
                "sequence_number": st.next_seq(),
                "output_index": idx,
                "item": {
                    "id": format!("fc_{call_id}"),
                    "type": "function_call",
                    "status": "completed",
                    "arguments": args,
                    "call_id": call_id,
                    "name": st.func_names.get(&idx).cloned().unwrap_or_default(),
                },
            }),
        ));
        st.func_done.insert(idx, true);
    }
}

fn build_stream_outputs(st: &StreamState) -> Vec<Value> {
    let mut output = Vec::new();
    for idx in 0..st.next_index {
        if st.reasoning_opened && idx == st.reasoning_index {
            output.push(json!({
                "id": st.reasoning_item_id,
                "type": "reasoning",
                "encrypted_content": st.reasoning_enc,
                "summary": [{"type": "summary_text", "text": st.reasoning_buf}],
            }));
            continue;
        }
        if st.msg_opened && idx == st.msg_index {
            output.push(json!({
                "id": st.current_msg_id,
                "type": "message",
                "status": "completed",
                "content": [{
                    "type": "output_text",
                    "annotations": [],
                    "logprobs": [],
                    "text": st.text_buf,
                }],
                "role": "assistant",
            }));
            continue;
        }
        if let Some(call_id) = st.func_call_ids.get(&idx) {
            if call_id.is_empty() {
                continue;
            }
            let args = st
                .func_args_buf
                .get(&idx)
                .cloned()
                .filter(|value| !value.is_empty())
                .unwrap_or_else(|| "{}".to_string());
            output.push(json!({
                "id": format!("fc_{call_id}"),
                "type": "function_call",
                "status": "completed",
                "arguments": args,
                "call_id": call_id,
                "name": st.func_names.get(&idx).cloned().unwrap_or_default(),
            }));
        }
    }
    output
}

fn apply_stream_usage(target: &mut Value, usage: &Value) {
    let input = usage
        .get("promptTokenCount")
        .and_then(Value::as_i64)
        .unwrap_or(0);
    let cached = usage
        .get("cachedContentTokenCount")
        .and_then(Value::as_i64)
        .unwrap_or(0);
    let output = usage
        .get("candidatesTokenCount")
        .and_then(Value::as_i64)
        .unwrap_or(0);
    let reasoning = usage
        .get("thoughtsTokenCount")
        .and_then(Value::as_i64)
        .unwrap_or(0);
    let total = usage
        .get("totalTokenCount")
        .and_then(Value::as_i64)
        .unwrap_or(0);
    target["usage"]["input_tokens"] = json!(input);
    target["usage"]["input_tokens_details"]["cached_tokens"] = json!(cached);
    target["usage"]["output_tokens"] = json!(output);
    target["usage"]["output_tokens_details"]["reasoning_tokens"] = json!(reasoning);
    target["usage"]["total_tokens"] = json!(total);
}

fn apply_non_stream_usage(target: &mut Value, usage: &Value) {
    let input = usage
        .get("promptTokenCount")
        .and_then(Value::as_i64)
        .unwrap_or(0);
    let cached = usage
        .get("cachedContentTokenCount")
        .and_then(Value::as_i64)
        .unwrap_or(0);
    target["usage"]["input_tokens"] = json!(input);
    target["usage"]["input_tokens_details"]["cached_tokens"] = json!(cached);
    if let Some(value) = usage.get("candidatesTokenCount").and_then(Value::as_i64) {
        target["usage"]["output_tokens"] = json!(value);
    }
    if let Some(value) = usage.get("thoughtsTokenCount").and_then(Value::as_i64) {
        target["usage"]["output_tokens_details"]["reasoning_tokens"] = json!(value);
    }
    if let Some(value) = usage.get("totalTokenCount").and_then(Value::as_i64) {
        target["usage"]["total_tokens"] = json!(value);
    }
}

fn echo_request_fields(target: &mut Value, request: &Value) {
    if !request.is_object() {
        return;
    }
    for field in [
        "instructions",
        "max_output_tokens",
        "max_tool_calls",
        "model",
        "parallel_tool_calls",
        "previous_response_id",
        "prompt_cache_key",
        "reasoning",
        "safety_identifier",
        "service_tier",
        "store",
        "temperature",
        "text",
        "tool_choice",
        "tools",
        "top_logprobs",
        "top_p",
        "truncation",
        "user",
        "metadata",
    ] {
        if let Some(value) = request.get(field) {
            target[field] = value.clone();
        }
    }
}

fn pick_request_value(original_request: &Value, translated_request: &Value) -> Value {
    if original_request.is_object() {
        unwrap_request_root(original_request.clone())
    } else if translated_request.is_object() {
        unwrap_request_root(translated_request.clone())
    } else {
        Value::Null
    }
}

fn unwrap_request_root(root: Value) -> Value {
    let Some(req) = root.get("request") else {
        return root;
    };
    if req.get("model").is_some() || req.get("input").is_some() || req.get("instructions").is_some()
    {
        req.clone()
    } else {
        root
    }
}

fn unwrap_gemini_response_root(root: Value) -> Value {
    let Some(resp) = root.get("response") else {
        return root;
    };
    if resp.get("candidates").is_some()
        || resp.get("responseId").is_some()
        || resp.get("usageMetadata").is_some()
    {
        resp.clone()
    } else {
        root
    }
}

fn normalize_chunk_value(value: Value) -> Value {
    let Some(raw) = value.as_str() else {
        return value;
    };
    let trimmed = raw.trim();
    if trimmed.is_empty() || trimmed == "[DONE]" {
        return Value::String(trimmed.to_string());
    }
    let payload = trimmed
        .strip_prefix("data:")
        .map(str::trim)
        .unwrap_or(trimmed);
    if payload == "[DONE]" {
        return Value::String("[DONE]".to_string());
    }
    serde_json::from_str(payload).unwrap_or_else(|_| Value::String(payload.to_string()))
}

fn response_id_from_root(root: &Value) -> String {
    let raw = root
        .get("responseId")
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty())
        .map(str::to_string)
        .unwrap_or_else(generated_response_id);
    if raw.starts_with("resp_") {
        raw
    } else {
        format!("resp_{raw}")
    }
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

fn sse(event: &str, data: Value) -> Value {
    Value::String(format!(
        "event: {event}\ndata: {}",
        serde_json::to_string(&data).unwrap_or_else(|_| "{}".to_string())
    ))
}

fn json_raw_string(value: &Value) -> String {
    serde_json::to_string(value).unwrap_or_default()
}

fn generated_response_id() -> String {
    format!(
        "resp_{:x}_{}",
        now_nanos(),
        RESPONSE_ID_COUNTER.fetch_add(1, Ordering::Relaxed) + 1
    )
}

fn generated_stream_call_id() -> String {
    format!(
        "call_{}_{}",
        now_nanos(),
        FUNC_CALL_ID_COUNTER.fetch_add(1, Ordering::Relaxed) + 1
    )
}

fn generated_non_stream_call_id() -> String {
    format!(
        "call_{:x}_{}",
        now_nanos(),
        FUNC_CALL_ID_COUNTER.fetch_add(1, Ordering::Relaxed) + 1
    )
}

fn now_unix() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64
}

fn now_nanos() -> u128 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos()
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
    fn non_stream_text_maps_to_response_output() {
        let resp = json!({
            "responseId": "req_1",
            "createTime": "2026-01-01T00:00:00Z",
            "modelVersion": "gemini-test",
            "candidates": [{"content": {"parts": [{"text": "Hello"}]}}],
        });
        let out = transform_non_stream("gpt", &json!({}), &json!({}), resp, None);
        assert_eq!(out["id"], "resp_req_1");
        assert_eq!(out["model"], "gemini-test");
        assert_eq!(out["output"][0]["content"][0]["text"], "Hello");
    }

    #[test]
    fn stream_emits_created_once() {
        let mut param: Box<dyn std::any::Any> = Box::new(());
        let out = transform_stream(
            "gpt",
            &json!({}),
            &json!({}),
            json!({"responseId":"req_stream","candidates":[{"content":{"parts":[{"text":"Hi"}]}}]}),
            Some(&mut param),
        );
        let events: Vec<_> = out
            .iter()
            .filter_map(Value::as_str)
            .filter_map(|s| s.lines().next())
            .collect();
        assert_eq!(events[0], "event: response.created");
        assert_eq!(events[1], "event: response.in_progress");
        assert!(events.contains(&"event: response.output_text.delta"));
    }
}
