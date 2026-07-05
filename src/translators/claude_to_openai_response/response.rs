//! Claude Messages response -> OpenAI Responses response translation.

use serde_json::{Value, json};
use std::collections::HashMap;

#[derive(Debug, Clone, Default)]
pub struct ResponseUsage {
    pub input_tokens: i64,
    pub output_tokens: i64,
    pub cache_creation_input_tokens: i64,
    pub cache_read_input_tokens: i64,
    pub has_usage: bool,
}

impl ResponseUsage {
    fn merge(&mut self, usage: &Value) {
        if !usage.is_object() {
            return;
        }
        self.has_usage = true;
        if let Some(v) = usage.get("input_tokens").and_then(Value::as_i64) {
            self.input_tokens = v;
        }
        if let Some(v) = usage.get("output_tokens").and_then(Value::as_i64) {
            self.output_tokens = v;
        }
        if let Some(v) = usage
            .get("cache_creation_input_tokens")
            .and_then(Value::as_i64)
        {
            self.cache_creation_input_tokens = v;
        }
        if let Some(v) = usage.get("cache_read_input_tokens").and_then(Value::as_i64) {
            self.cache_read_input_tokens = v;
        }
    }

    fn openai_responses_usage(&self) -> (i64, i64, i64, i64) {
        let cached_tokens = self.cache_read_input_tokens;
        let input_tokens = self.input_tokens + self.cache_creation_input_tokens + cached_tokens;
        let output_tokens = self.output_tokens;
        let total_tokens = input_tokens + output_tokens;
        (input_tokens, output_tokens, total_tokens, cached_tokens)
    }
}

#[derive(Debug, Clone, Default)]
pub struct ClaudeToResponseState {
    pub seq: i64,
    pub response_id: String,
    pub created_at: i64,
    pub current_msg_id: String,
    pub current_fc_id: String,
    pub in_text_block: bool,
    pub in_func_block: bool,
    pub message_open: bool,
    pub content_part_open: bool,
    pub func_args: HashMap<usize, String>,
    pub func_names: HashMap<usize, String>,
    pub func_call_ids: HashMap<usize, String>,
    pub text_buf: String,
    pub current_text_buf: String,
    pub message_annotations: Vec<Value>,
    pub reasoning_active: bool,
    pub reasoning_item_id: String,
    pub reasoning_buf: String,
    pub reasoning_signature: String,
    pub reasoning_part_added: bool,
    pub reasoning_index: usize,
    pub usage: ResponseUsage,
    pub request_json: Value,
}

impl ClaudeToResponseState {
    fn next_seq(&mut self) -> i64 {
        self.seq += 1;
        self.seq
    }

    fn reset_message_state(&mut self) {
        self.text_buf.clear();
        self.current_text_buf.clear();
        self.message_annotations.clear();
        self.reasoning_buf.clear();
        self.reasoning_active = false;
        self.in_text_block = false;
        self.in_func_block = false;
        self.message_open = false;
        self.content_part_open = false;
        self.current_msg_id.clear();
        self.current_fc_id.clear();
        self.reasoning_item_id.clear();
        self.reasoning_signature.clear();
        self.reasoning_index = 0;
        self.reasoning_part_added = false;
        self.func_args.clear();
        self.func_names.clear();
        self.func_call_ids.clear();
        self.usage = ResponseUsage::default();
    }

    fn finalize_assistant_message(&mut self) -> Vec<Value> {
        if !self.message_open {
            return Vec::new();
        }

        let full_text = self.text_buf.clone();
        let item_id = self.current_msg_id.clone();
        let mut out = Vec::new();

        out.push(sse(
            "response.output_text.done",
            json!({
                "type": "response.output_text.done",
                "sequence_number": self.next_seq(),
                "item_id": item_id,
                "output_index": 0,
                "content_index": 0,
                "text": full_text,
                "logprobs": [],
            }),
        ));

        let mut part_done = json!({
            "type": "response.content_part.done",
            "sequence_number": self.next_seq(),
            "item_id": self.current_msg_id,
            "output_index": 0,
            "content_index": 0,
            "part": {
                "type": "output_text",
                "annotations": [],
                "logprobs": [],
                "text": self.text_buf,
            },
        });
        if !self.message_annotations.is_empty() {
            part_done["part"]["annotations"] = json!(self.message_annotations);
        }
        out.push(sse("response.content_part.done", part_done));

        let mut item_done = json!({
            "type": "response.output_item.done",
            "sequence_number": self.next_seq(),
            "output_index": 0,
            "item": {
                "id": self.current_msg_id,
                "type": "message",
                "status": "completed",
                "content": [{
                    "type": "output_text",
                    "annotations": [],
                    "logprobs": [],
                    "text": self.text_buf,
                }],
                "role": "assistant",
            },
        });
        if !self.message_annotations.is_empty() {
            item_done["item"]["content"][0]["annotations"] = json!(self.message_annotations);
        }
        out.push(sse("response.output_item.done", item_done));

        self.in_text_block = false;
        self.message_open = false;
        self.content_part_open = false;
        self.current_text_buf.clear();
        out
    }
}

#[derive(Debug, Clone, Default)]
struct NonStreamToolState {
    id: String,
    name: String,
    args: String,
}

pub fn transform_stream(
    _model: &str,
    original_request: &Value,
    translated_request: &Value,
    chunk: Value,
    param: Option<&mut Box<dyn std::any::Any>>,
) -> Vec<Value> {
    let Some(param) = param else {
        return vec![chunk];
    };
    if !param.is::<ClaudeToResponseState>() {
        *param = Box::<ClaudeToResponseState>::default();
    }
    let st = param.downcast_mut::<ClaudeToResponseState>().unwrap();

    if st.request_json.is_null() {
        st.request_json = pick_request_value(original_request, translated_request);
    }

    let event_type = chunk.get("type").and_then(Value::as_str).unwrap_or("");
    let mut out = Vec::new();

    match event_type {
        "message_start" => {
            if let Some(message) = chunk.get("message") {
                st.response_id = message
                    .get("id")
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .to_string();
                st.created_at = now_unix();
                st.reset_message_state();
                st.usage.merge(message.get("usage").unwrap_or(&Value::Null));

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
            }
        }
        "content_block_start" => {
            let Some(content_block) = chunk.get("content_block") else {
                return out;
            };
            let idx = chunk.get("index").and_then(Value::as_u64).unwrap_or(0) as usize;
            match content_block
                .get("type")
                .and_then(Value::as_str)
                .unwrap_or("")
            {
                "text" => {
                    st.in_text_block = true;
                    if st.current_msg_id.is_empty() {
                        st.current_msg_id = format!("msg_{}_0", st.response_id);
                    }
                    if !st.message_open {
                        out.push(sse(
                            "response.output_item.added",
                            json!({
                                "type": "response.output_item.added",
                                "sequence_number": st.next_seq(),
                                "output_index": 0,
                                "item": {
                                    "id": st.current_msg_id,
                                    "type": "message",
                                    "status": "in_progress",
                                    "content": [],
                                    "role": "assistant",
                                },
                            }),
                        ));
                        st.message_open = true;
                    }
                    if !st.content_part_open {
                        out.push(sse(
                            "response.content_part.added",
                            json!({
                                "type": "response.content_part.added",
                                "sequence_number": st.next_seq(),
                                "item_id": st.current_msg_id,
                                "output_index": 0,
                                "content_index": 0,
                                "part": {
                                    "type": "output_text",
                                    "annotations": [],
                                    "logprobs": [],
                                    "text": "",
                                },
                            }),
                        ));
                        st.content_part_open = true;
                    }
                }
                "tool_use" => {
                    st.in_func_block = true;
                    st.current_fc_id = content_block
                        .get("id")
                        .and_then(Value::as_str)
                        .unwrap_or("")
                        .to_string();
                    let name = content_block
                        .get("name")
                        .and_then(Value::as_str)
                        .unwrap_or("")
                        .to_string();

                    let mut item = json!({
                        "id": format!("fc_{}", st.current_fc_id),
                        "type": "function_call",
                        "status": "in_progress",
                        "arguments": "",
                        "call_id": st.current_fc_id,
                        "name": name,
                    });
                    apply_fn_call_namespace(&mut item, &st.request_json, &name);
                    out.push(sse(
                        "response.output_item.added",
                        json!({
                            "type": "response.output_item.added",
                            "sequence_number": st.next_seq(),
                            "output_index": idx,
                            "item": item,
                        }),
                    ));

                    st.func_args.entry(idx).or_default();
                    st.func_call_ids.insert(idx, st.current_fc_id.clone());
                    st.func_names.insert(idx, name);
                }
                "thinking" => {
                    st.reasoning_active = true;
                    st.reasoning_index = idx;
                    st.reasoning_buf.clear();
                    st.reasoning_signature = content_block
                        .get("signature")
                        .and_then(Value::as_str)
                        .unwrap_or("")
                        .to_string();
                    st.reasoning_item_id = format!("rs_{}_{}", st.response_id, idx);

                    out.push(sse(
                        "response.output_item.added",
                        json!({
                            "type": "response.output_item.added",
                            "sequence_number": st.next_seq(),
                            "output_index": idx,
                            "item": {
                                "id": st.reasoning_item_id,
                                "type": "reasoning",
                                "status": "in_progress",
                                "encrypted_content": st.reasoning_signature,
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
                            "output_index": idx,
                            "summary_index": 0,
                            "part": {"type": "summary_text", "text": ""},
                        }),
                    ));
                    st.reasoning_part_added = true;
                }
                _ => {}
            }
        }
        "content_block_delta" => {
            let Some(delta) = chunk.get("delta") else {
                return out;
            };
            match delta.get("type").and_then(Value::as_str).unwrap_or("") {
                "text_delta" => {
                    if let Some(text) = delta.get("text").and_then(Value::as_str) {
                        out.push(sse(
                            "response.output_text.delta",
                            json!({
                                "type": "response.output_text.delta",
                                "sequence_number": st.next_seq(),
                                "item_id": st.current_msg_id,
                                "output_index": 0,
                                "content_index": 0,
                                "delta": text,
                                "logprobs": [],
                            }),
                        ));
                        st.text_buf.push_str(text);
                        st.current_text_buf.push_str(text);
                    }
                }
                "input_json_delta" => {
                    if !st.in_func_block || st.current_fc_id.is_empty() {
                        return Vec::new();
                    }
                    let idx = chunk.get("index").and_then(Value::as_u64).unwrap_or(0) as usize;
                    if let Some(partial_json) = delta.get("partial_json").and_then(Value::as_str) {
                        st.func_args.entry(idx).or_default().push_str(partial_json);
                        out.push(sse(
                            "response.function_call_arguments.delta",
                            json!({
                                "type": "response.function_call_arguments.delta",
                                "sequence_number": st.next_seq(),
                                "item_id": format!("fc_{}", st.current_fc_id),
                                "output_index": idx,
                                "delta": partial_json,
                            }),
                        ));
                    }
                }
                "thinking_delta" => {
                    if st.reasoning_active
                        && let Some(thinking) = delta.get("thinking").and_then(Value::as_str)
                    {
                        st.reasoning_buf.push_str(thinking);
                        out.push(sse(
                            "response.reasoning_summary_text.delta",
                            json!({
                                "type": "response.reasoning_summary_text.delta",
                                "sequence_number": st.next_seq(),
                                "item_id": st.reasoning_item_id,
                                "output_index": st.reasoning_index,
                                "summary_index": 0,
                                "delta": thinking,
                            }),
                        ));
                    }
                }
                "signature_delta" => {
                    if st.reasoning_active
                        && let Some(signature) = delta.get("signature").and_then(Value::as_str)
                    {
                        st.reasoning_signature = signature.to_string();
                    }
                    return Vec::new();
                }
                "citations_delta" => {
                    if let Some(citation) = delta.get("citation") {
                        st.message_annotations.push(citation.clone());
                    }
                    return Vec::new();
                }
                _ => {}
            }
        }
        "content_block_stop" => {
            let idx = chunk.get("index").and_then(Value::as_u64).unwrap_or(0) as usize;
            if st.in_text_block {
                st.in_text_block = false;
            } else if st.in_func_block {
                let raw_args = st.func_args.get(&idx).cloned().unwrap_or_default();
                let event_args = if raw_args.is_empty() {
                    "{}".to_string()
                } else {
                    raw_args.clone()
                };
                let call_id = if st.current_fc_id.is_empty() {
                    st.func_call_ids.get(&idx).cloned().unwrap_or_default()
                } else {
                    st.current_fc_id.clone()
                };
                let name = st.func_names.get(&idx).cloned().unwrap_or_default();

                out.push(sse(
                    "response.function_call_arguments.done",
                    json!({
                        "type": "response.function_call_arguments.done",
                        "sequence_number": st.next_seq(),
                        "item_id": format!("fc_{}", call_id),
                        "output_index": idx,
                        "arguments": event_args,
                    }),
                ));

                let mut item = json!({
                    "id": format!("fc_{}", call_id),
                    "type": "function_call",
                    "status": "completed",
                    "arguments": event_args,
                    "call_id": call_id,
                    "name": name,
                });
                apply_fn_call_namespace(&mut item, &st.request_json, &name);
                out.push(sse(
                    "response.output_item.done",
                    json!({
                        "type": "response.output_item.done",
                        "sequence_number": st.next_seq(),
                        "output_index": idx,
                        "item": item,
                    }),
                ));
                st.in_func_block = false;
            } else if st.reasoning_active {
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

                let mut item = json!({
                    "id": st.reasoning_item_id,
                    "type": "reasoning",
                    "encrypted_content": st.reasoning_signature,
                    "summary": [],
                });
                if !st.reasoning_buf.is_empty() {
                    item["summary"] = json!([{"type": "summary_text", "text": st.reasoning_buf}]);
                }
                out.push(sse(
                    "response.output_item.done",
                    json!({
                        "type": "response.output_item.done",
                        "sequence_number": st.next_seq(),
                        "output_index": st.reasoning_index,
                        "item": item,
                    }),
                ));
                st.reasoning_active = false;
                st.reasoning_part_added = false;
            }
        }
        "message_delta" => {
            st.usage.merge(chunk.get("usage").unwrap_or(&Value::Null));
        }
        "message_stop" => {
            out.extend(st.finalize_assistant_message());
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
            echo_request_fields(&mut completed["response"], &st.request_json);

            let outputs = build_stream_outputs(st);
            if !outputs.is_empty() {
                completed["response"]["output"] = Value::Array(outputs);
            }

            let reasoning_tokens = reasoning_token_estimate(&st.reasoning_buf);
            if st.usage.has_usage || reasoning_tokens > 0 {
                let (input_tokens, output_tokens, total_tokens, cached_tokens) =
                    st.usage.openai_responses_usage();
                completed["response"]["usage"]["input_tokens"] = json!(input_tokens);
                completed["response"]["usage"]["input_tokens_details"]["cached_tokens"] =
                    json!(cached_tokens);
                completed["response"]["usage"]["output_tokens"] = json!(output_tokens);
                if reasoning_tokens > 0 {
                    completed["response"]["usage"]["output_tokens_details"]["reasoning_tokens"] =
                        json!(reasoning_tokens);
                }
                if total_tokens > 0 || st.usage.has_usage {
                    completed["response"]["usage"]["total_tokens"] = json!(total_tokens);
                }
            }

            out.push(sse("response.completed", completed));
        }
        _ => {}
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
    let raw = response.as_str().unwrap_or("");
    let request_json = pick_request_value(original_request, translated_request);

    let mut response_id = String::new();
    let mut created_at = 0_i64;
    let mut current_msg_id = String::new();
    let mut text_buf = String::new();
    let mut reasoning_buf = String::new();
    let mut reasoning_active = false;
    let mut reasoning_item_id = String::new();
    let mut reasoning_signature = String::new();
    let mut annotations = Vec::new();
    let mut usage = ResponseUsage::default();
    let mut tool_calls: HashMap<usize, NonStreamToolState> = HashMap::new();

    for chunk in claude_sse_data_chunks(raw) {
        let Ok(root) = serde_json::from_str::<Value>(chunk) else {
            continue;
        };
        match root.get("type").and_then(Value::as_str).unwrap_or("") {
            "message_start" => {
                if let Some(message) = root.get("message") {
                    response_id = message
                        .get("id")
                        .and_then(Value::as_str)
                        .unwrap_or("")
                        .to_string();
                    created_at = now_unix();
                    usage.merge(message.get("usage").unwrap_or(&Value::Null));
                }
            }
            "content_block_start" => {
                let Some(content_block) = root.get("content_block") else {
                    continue;
                };
                let idx = root.get("index").and_then(Value::as_u64).unwrap_or(0) as usize;
                match content_block
                    .get("type")
                    .and_then(Value::as_str)
                    .unwrap_or("")
                {
                    "text" => {
                        current_msg_id = format!("msg_{}_0", response_id);
                    }
                    "tool_use" => {
                        let current_fc_id = content_block
                            .get("id")
                            .and_then(Value::as_str)
                            .unwrap_or("")
                            .to_string();
                        let name = content_block
                            .get("name")
                            .and_then(Value::as_str)
                            .unwrap_or("")
                            .to_string();
                        let entry = tool_calls.entry(idx).or_default();
                        entry.id = current_fc_id.clone();
                        entry.name = name;
                    }
                    "thinking" => {
                        reasoning_active = true;
                        reasoning_item_id = format!("rs_{}_{}", response_id, idx);
                        reasoning_signature = content_block
                            .get("signature")
                            .and_then(Value::as_str)
                            .unwrap_or("")
                            .to_string();
                    }
                    _ => {}
                }
            }
            "content_block_delta" => {
                let Some(delta) = root.get("delta") else {
                    continue;
                };
                match delta.get("type").and_then(Value::as_str).unwrap_or("") {
                    "text_delta" => {
                        if let Some(text) = delta.get("text").and_then(Value::as_str) {
                            text_buf.push_str(text);
                        }
                    }
                    "input_json_delta" => {
                        if let Some(partial_json) =
                            delta.get("partial_json").and_then(Value::as_str)
                        {
                            let idx =
                                root.get("index").and_then(Value::as_u64).unwrap_or(0) as usize;
                            tool_calls
                                .entry(idx)
                                .or_default()
                                .args
                                .push_str(partial_json);
                        }
                    }
                    "thinking_delta" => {
                        if reasoning_active
                            && let Some(thinking) = delta.get("thinking").and_then(Value::as_str)
                        {
                            reasoning_buf.push_str(thinking);
                        }
                    }
                    "signature_delta" => {
                        if reasoning_active
                            && let Some(signature) = delta.get("signature").and_then(Value::as_str)
                        {
                            reasoning_signature = signature.to_string();
                        }
                    }
                    "citations_delta" => {
                        if let Some(citation) = delta.get("citation") {
                            annotations.push(citation.clone());
                        }
                    }
                    _ => {}
                }
            }
            "message_delta" => {
                usage.merge(root.get("usage").unwrap_or(&Value::Null));
            }
            _ => {}
        }
    }

    let mut out = json!({
        "id": response_id,
        "object": "response",
        "created_at": created_at,
        "status": "completed",
        "background": false,
        "error": null,
        "incomplete_details": null,
        "output": [],
        "usage": {
            "input_tokens": 0,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens": 0,
            "output_tokens_details": {},
            "total_tokens": 0,
        },
    });
    echo_request_fields(&mut out, &request_json);

    let mut outputs = Vec::new();
    if !reasoning_buf.is_empty() || !reasoning_signature.is_empty() {
        let mut item = json!({
            "id": reasoning_item_id,
            "type": "reasoning",
            "encrypted_content": reasoning_signature,
            "summary": [],
        });
        if !reasoning_buf.is_empty() {
            item["summary"] = json!([{"type": "summary_text", "text": reasoning_buf}]);
        }
        outputs.push(item);
    }

    if !current_msg_id.is_empty() || !text_buf.is_empty() {
        let mut item = json!({
            "id": current_msg_id,
            "type": "message",
            "status": "completed",
            "content": [{
                "type": "output_text",
                "annotations": [],
                "logprobs": [],
                "text": text_buf,
            }],
            "role": "assistant",
        });
        if !annotations.is_empty() {
            item["content"][0]["annotations"] = Value::Array(annotations);
        }
        outputs.push(item);
    }

    let mut indices: Vec<_> = tool_calls.keys().copied().collect();
    indices.sort_unstable();
    for idx in indices {
        let Some(tool) = tool_calls.get(&idx) else {
            continue;
        };
        let args = if tool.args.is_empty() {
            "{}".to_string()
        } else {
            tool.args.clone()
        };
        let mut item = json!({
            "id": format!("fc_{}", tool.id),
            "type": "function_call",
            "status": "completed",
            "arguments": args,
            "call_id": tool.id,
            "name": tool.name,
        });
        apply_fn_call_namespace(&mut item, &request_json, &tool.name);
        outputs.push(item);
    }
    if !outputs.is_empty() {
        out["output"] = Value::Array(outputs);
    }

    let (input_tokens, output_tokens, total_tokens, cached_tokens) = usage.openai_responses_usage();
    out["usage"]["input_tokens"] = json!(input_tokens);
    out["usage"]["input_tokens_details"]["cached_tokens"] = json!(cached_tokens);
    out["usage"]["output_tokens"] = json!(output_tokens);
    out["usage"]["total_tokens"] = json!(total_tokens);
    let reasoning_tokens = out
        .get("output")
        .and_then(Value::as_array)
        .and_then(|items| {
            items
                .iter()
                .find(|item| item.get("type").and_then(Value::as_str) == Some("reasoning"))
        })
        .and_then(|item| item.pointer("/summary/0/text"))
        .and_then(Value::as_str)
        .map(reasoning_token_estimate)
        .unwrap_or(0);
    if reasoning_tokens > 0 {
        out["usage"]["output_tokens_details"]["reasoning_tokens"] = json!(reasoning_tokens);
    }

    out
}

fn build_stream_outputs(st: &ClaudeToResponseState) -> Vec<Value> {
    let mut output = Vec::new();
    if !st.reasoning_buf.is_empty() || st.reasoning_part_added || !st.reasoning_signature.is_empty()
    {
        let mut item = json!({
            "id": st.reasoning_item_id,
            "type": "reasoning",
            "encrypted_content": st.reasoning_signature,
            "summary": [],
        });
        if !st.reasoning_buf.is_empty() {
            item["summary"] = json!([{"type": "summary_text", "text": st.reasoning_buf}]);
        }
        output.push(item);
    }

    if !st.text_buf.is_empty() || st.in_text_block || !st.current_msg_id.is_empty() {
        let mut item = json!({
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
        });
        if !st.message_annotations.is_empty() {
            item["content"][0]["annotations"] = json!(st.message_annotations);
        }
        output.push(item);
    }

    let mut indices: Vec<_> = st.func_args.keys().copied().collect();
    indices.sort_unstable();
    for idx in indices {
        let args = st.func_args.get(&idx).cloned().unwrap_or_default();
        let mut call_id = st.func_call_ids.get(&idx).cloned().unwrap_or_default();
        if call_id.is_empty() && !st.current_fc_id.is_empty() {
            call_id = st.current_fc_id.clone();
        }
        let name = st.func_names.get(&idx).cloned().unwrap_or_default();
        let mut item = json!({
            "id": format!("fc_{}", call_id),
            "type": "function_call",
            "status": "completed",
            "arguments": args,
            "call_id": call_id,
            "name": name,
        });
        apply_fn_call_namespace(&mut item, &st.request_json, &name);
        output.push(item);
    }
    output
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

fn sse(event: &str, payload: Value) -> Value {
    let payload = serde_json::to_string(&payload).unwrap_or_else(|_| "{}".to_string());
    Value::String(format!("event: {event}\ndata: {payload}"))
}

fn now_unix() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64
}

fn pick_request_value(original_request: &Value, translated_request: &Value) -> Value {
    if original_request.is_object() {
        original_request.clone()
    } else if translated_request.is_object() {
        translated_request.clone()
    } else {
        Value::Null
    }
}

fn claude_sse_data_chunks(raw: &str) -> impl Iterator<Item = &str> {
    raw.lines()
        .filter_map(|line| line.trim_start().strip_prefix("data:").map(str::trim_start))
}

fn reasoning_token_estimate(text: &str) -> i64 {
    if text.is_empty() {
        0
    } else {
        (text.len() / 4) as i64
    }
}

fn split_qualified_fn_call<'a>(
    request_json: &'a Value,
    qualified_name: &'a str,
) -> (&'a str, &'a str) {
    if qualified_name.trim().is_empty() {
        return (qualified_name, "");
    }

    let Some(tools) = request_json.get("tools").and_then(Value::as_array) else {
        return (qualified_name, "");
    };

    for tool in tools {
        if tool.get("type").and_then(Value::as_str) != Some("namespace") {
            continue;
        }
        let Some(namespace) = tool.get("name").and_then(Value::as_str) else {
            continue;
        };
        if namespace.trim().is_empty() {
            continue;
        }
        let Some(children) = tool.get("tools").and_then(Value::as_array) else {
            continue;
        };
        for child in children {
            let child_name = responses_tool_name(child);
            if child_name.is_empty() {
                continue;
            }
            if qualify_namespace_tool_name(namespace, child_name) == qualified_name {
                return (child_name, namespace);
            }
        }
    }

    (qualified_name, "")
}

fn responses_tool_name(tool: &Value) -> &str {
    tool.get("name")
        .and_then(Value::as_str)
        .filter(|name| !name.trim().is_empty())
        .or_else(|| tool.pointer("/function/name").and_then(Value::as_str))
        .unwrap_or("")
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

fn apply_fn_call_namespace(item: &mut Value, request_json: &Value, qualified_name: &str) {
    if qualified_name.trim().is_empty() {
        return;
    }
    let (name, namespace) = split_qualified_fn_call(request_json, qualified_name);
    item["name"] = json!(name);
    if namespace.is_empty() {
        if let Some(obj) = item.as_object_mut() {
            obj.remove("namespace");
        }
    } else {
        item["namespace"] = json!(namespace);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sse_name(value: &Value) -> String {
        value
            .as_str()
            .and_then(|s| {
                s.lines()
                    .find_map(|line| line.strip_prefix("event: ").map(str::to_string))
            })
            .unwrap_or_default()
    }

    #[test]
    fn message_start_creates_response_events() {
        let mut param: Box<dyn std::any::Any> = Box::new(());
        let out = transform_stream(
            "claude",
            &json!({}),
            &json!({}),
            json!({"type":"message_start","message":{"id":"msg_1","model":"claude"}}),
            Some(&mut param),
        );
        assert!(out.iter().any(|v| sse_name(v) == "response.created"));
        assert!(out.iter().any(|v| sse_name(v) == "response.in_progress"));
    }

    #[test]
    fn text_delta_emits_output_text_delta() {
        let mut param: Box<dyn std::any::Any> = Box::new(());
        let _ = transform_stream(
            "claude",
            &json!({}),
            &json!({}),
            json!({"type":"message_start","message":{"id":"msg_1","model":"claude"}}),
            Some(&mut param),
        );
        let _ = transform_stream(
            "claude",
            &json!({}),
            &json!({}),
            json!({"type":"content_block_start","content_block":{"type":"text"},"index":0}),
            Some(&mut param),
        );
        let out = transform_stream(
            "claude",
            &json!({}),
            &json!({}),
            json!({"type":"content_block_delta","delta":{"type":"text_delta","text":"Hello"},"index":0}),
            Some(&mut param),
        );
        assert!(
            out.iter()
                .any(|v| sse_name(v) == "response.output_text.delta")
        );
    }

    #[test]
    fn non_stream_aggregates_sse_text() {
        let raw = [
            r#"data: {"type":"message_start","message":{"id":"msg_1","usage":{"input_tokens":5,"output_tokens":0}}}"#,
            r#"data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}"#,
            r#"data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello"}}"#,
            r#"data: {"type":"message_delta","usage":{"output_tokens":2}}"#,
            r#"data: {"type":"message_stop"}"#,
        ]
        .join("\n");
        let out = transform_non_stream("claude", &json!({}), &json!({}), json!(raw), None);
        assert_eq!(out["id"], "msg_1");
        assert_eq!(out["output"][0]["content"][0]["text"], "Hello");
        assert_eq!(out["usage"]["total_tokens"], 7);
    }
}
