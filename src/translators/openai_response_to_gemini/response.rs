//! Gemini SSE -> OpenAI Responses SSE -- streaming state machine.
//!
//! Port of Go's `gemini_openai-responses_response.go`
//! `ConvertGeminiResponseToOpenAIResponses` and
//! `ConvertGeminiResponseToOpenAIResponsesNonStream`.

use serde_json::{Value, json};

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Default)]
struct State {
    seq: i64,
    response_id: String,
    started: bool,
    // message aggregation
    msg_opened: bool,
    msg_closed: bool,
    msg_index: i64,
    current_msg_id: String,
    text_buf: String,
    // reasoning aggregation
    reasoning_opened: bool,
    reasoning_index: i64,
    reasoning_item_id: String,
    reasoning_enc: String,
    reasoning_buf: String,
    reasoning_closed: bool,
    // function call aggregation
    next_index: i64,
}

// ---------------------------------------------------------------------------
// Stream
// ---------------------------------------------------------------------------

pub fn transform_stream(
    model: &str,
    orig_req: &Value,
    trans_req: &Value,
    chunk: Value,
    param: Option<&mut Box<dyn std::any::Any>>,
) -> Vec<Value> {
    let st = get_or_init_state(param, model, orig_req);

    let chunk = match chunk {
        Value::String(ref s) if s.starts_with("data:") => {
            let payload = s.trim_start_matches("data:").trim();
            if payload == "[DONE]" {
                return vec![];
            }
            match serde_json::from_str::<Value>(payload) {
                Ok(v) => unwrap_gemini_response_root(v),
                Err(_) => return vec![],
            }
        }
        Value::String(_) => return vec![],
        other => unwrap_gemini_response_root(other),
    };

    let mut out: Vec<Value> = vec![];

    // Initialize response on first chunk.
    if !st.started {
        st.response_id = format!(
            "resp_{:x}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos()
        );
        st.started = true;

        let req = resolve_request(orig_req, trans_req);
        let mut created = json!({
            "type": "response.created",
            "sequence_number": 0,
            "response": build_base_response(st, model, req)
        });
        created["sequence_number"] = json!(next_seq(st));
        out.push(emit_event("response.created", &created));
    }

    next_seq(st);

    // Handle candidates.
    if let Some(candidates) = chunk.get("candidates").and_then(|v| v.as_array()) {
        for candidate in candidates {
            let parts = candidate
                .get("content")
                .and_then(|v| v.get("parts"))
                .and_then(|v| v.as_array());

            if let Some(parts) = parts {
                for part in parts {
                    if part
                        .get("thought")
                        .and_then(|v| v.as_bool())
                        .unwrap_or(false)
                    {
                        handle_reasoning_delta(st, part, &mut out);
                    } else if let Some(text) = part.get("text").and_then(|v| v.as_str()) {
                        handle_text_delta(st, text, &mut out);
                    } else if let Some(fc) = part.get("functionCall") {
                        handle_function_call(st, fc, &mut out, model);
                    }
                }
            }

            let finish_reason = candidate
                .get("finishReason")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            if !finish_reason.is_empty() {
                finalize_outputs(st, &mut out, model);
            }
        }
    }

    // Usage metadata.
    if let Some(um) = chunk.get("usageMetadata") {
        let req = resolve_request(orig_req, trans_req);
        let mut completed = json!({
            "type": "response.completed",
            "sequence_number": 0,
            "response": build_base_response(st, model, req)
        });
        completed["sequence_number"] = json!(next_seq(st));
        completed["response"]["usage"] = json!({
            "input_tokens": um.get("promptTokenCount").and_then(|v| v.as_i64()).unwrap_or(0),
            "output_tokens": um.get("candidatesTokenCount").and_then(|v| v.as_i64()).unwrap_or(0),
            "total_tokens": um.get("totalTokenCount").and_then(|v| v.as_i64()).unwrap_or(0),
        });
        out.push(emit_event("response.completed", &completed));
    }

    out
}

// ---------------------------------------------------------------------------
// Non-stream
// ---------------------------------------------------------------------------

pub fn transform_non_stream(
    model: &str,
    orig_req: &Value,
    trans_req: &Value,
    response: Value,
    _param: Option<&mut Box<dyn std::any::Any>>,
) -> Value {
    let root = unwrap_gemini_response_root(response);
    let req = resolve_request(orig_req, trans_req);
    let mut resp = build_base_response_raw(model, &root, req);

    let id = resp
        .get("id")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();
    let have_output = false;

    let mut reasoning_text = String::new();
    let mut reasoning_enc = String::new();
    let mut message_text = String::new();
    let mut have_message = false;

    // Process candidates[0].content.parts.
    if let Some(parts) = root
        .pointer("/candidates/0/content/parts")
        .and_then(|v| v.as_array())
    {
        for p in parts {
            if p.get("thought").and_then(|v| v.as_bool()).unwrap_or(false) {
                if let Some(t) = p.get("text").and_then(|v| v.as_str()) {
                    reasoning_text.push_str(t);
                }
                if let Some(sig) = p.get("thoughtSignature").and_then(|v| v.as_str())
                    && !sig.is_empty()
                {
                    reasoning_enc = sig.to_string();
                }
            } else if let Some(t) = p.get("text").and_then(|v| v.as_str()) {
                if !t.is_empty() {
                    message_text.push_str(t);
                    have_message = true;
                }
            } else if let Some(fc) = p.get("functionCall") {
                if !have_output {
                    resp["output"] = json!([]);
                }
                // fc is already a &Value
                let name = fc.get("name").and_then(|v| v.as_str()).unwrap_or("");
                let args = fc.get("args").cloned().unwrap_or(json!({}));
                let item = json!({
                    "id": format!("fc_{}", id),
                    "type": "function_call",
                    "status": "completed",
                    "arguments": serde_json::to_string(&args).unwrap_or_default(),
                    "call_id": format!("call_{}", id),
                    "name": name,
                });
                if let Some(arr) = resp["output"].as_array_mut() {
                    arr.push(item);
                }
            }
        }
    }

    // Reasoning output item.
    if !reasoning_text.is_empty() || !reasoning_enc.is_empty() {
        if !have_output {
            resp["output"] = json!([]);
        }
        let rid = id.trim_start_matches("resp_");
        let mut item = json!({
            "id": format!("rs_{}", rid),
            "type": "reasoning",
            "encrypted_content": reasoning_enc,
        });
        if !reasoning_text.is_empty() {
            item["summary"] = json!([{
                "type": "summary_text",
                "text": reasoning_text,
            }]);
        }
        if let Some(arr) = resp["output"].as_array_mut() {
            arr.push(item);
        }
    }

    // Message output item.
    if have_message {
        if !have_output {
            resp["output"] = json!([]);
        }
        let item = json!({
            "id": format!("msg_{}_0", id.trim_start_matches("resp_")),
            "type": "message",
            "status": "completed",
            "content": [{"type": "output_text", "annotations": [], "logprobs": [], "text": message_text}],
            "role": "assistant",
        });
        if let Some(arr) = resp["output"].as_array_mut() {
            arr.push(item);
        }
    }

    // Usage mapping.
    if let Some(um) = root.get("usageMetadata") {
        resp["usage"] = json!({
            "input_tokens": um.get("promptTokenCount").and_then(|v| v.as_i64()).unwrap_or(0),
            "output_tokens": um.get("candidatesTokenCount").and_then(|v| v.as_i64()).unwrap_or(0),
            "total_tokens": um.get("totalTokenCount").and_then(|v| v.as_i64()).unwrap_or(0),
        });
    }

    resp
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

#[allow(unused_variables)]
fn get_or_init_state<'a>(
    param: Option<&'a mut Box<dyn std::any::Any>>,
    model: &str,
    _orig_req: &Value,
) -> &'a mut State {
    let p = param.unwrap();
    if !p.is::<State>() {
        *p = Box::new(State::default());
    }
    p.downcast_mut::<State>().unwrap()
}

fn next_seq(st: &mut State) -> i64 {
    st.seq += 1;
    st.seq
}

fn resolve_request<'a>(orig: &'a Value, trans: &'a Value) -> &'a Value {
    if orig.get("model").is_some() || orig.get("input").is_some() {
        orig
    } else {
        trans
    }
}

fn unwrap_gemini_response_root(value: Value) -> Value {
    if let Some(resp) = value.get("response")
        && (resp.get("candidates").is_some() || resp.get("responseId").is_some())
    {
        return resp.clone();
    }
    value
}

fn emit_event(event: &str, payload: &Value) -> Value {
    // In the Rust system, pseudo-SSE events are stored as JSON values.
    // The "event:" prefix and "data:" wrapping are applied by the last-mile serializer.
    let mut out = payload.clone();
    if let Some(obj) = out.as_object_mut() {
        // Store event type for the serializer.
        obj.insert("_sse_event".to_string(), json!(event));
    }
    out
}

fn build_base_response(st: &State, model: &str, req: &Value) -> Value {
    let mut resp = json!({
        "id": st.response_id,
        "object": "response",
        "model": model,
        "status": "in_progress",
        "output": [],
    });

    // Copy request-level config.
    for key in &[
        "instructions",
        "input",
        "tools",
        "tool_choice",
        "temperature",
        "top_p",
        "max_output_tokens",
        "parallel_tool_calls",
        "reasoning",
        "metadata",
        "store",
    ] {
        if let Some(v) = req.get(*key) {
            resp[*key] = v.clone();
        }
    }

    resp
}

fn build_base_response_raw(model: &str, root: &Value, req: &Value) -> Value {
    let id = root
        .get("responseId")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string())
        .unwrap_or_else(|| {
            format!(
                "resp_{:x}",
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_nanos()
            )
        });

    let mut resp = json!({
        "id": id,
        "object": "response",
        "model": model,
        "status": "completed",
        "output": [],
    });

    for key in &[
        "instructions",
        "input",
        "tools",
        "tool_choice",
        "temperature",
        "top_p",
        "max_output_tokens",
        "parallel_tool_calls",
        "reasoning",
        "metadata",
        "store",
    ] {
        if let Some(v) = req.get(*key) {
            resp[*key] = v.clone();
        }
    }

    resp
}

fn handle_text_delta(st: &mut State, text: &str, out: &mut Vec<Value>) {
    if !st.msg_opened {
        st.msg_opened = true;
        st.msg_index += 1;
        st.current_msg_id = format!(
            "msg_{}_{}",
            st.response_id.trim_start_matches("resp_"),
            st.msg_index
        );

        let mut added = json!({
            "type": "response.content_part.added",
            "sequence_number": 0,
            "item_id": st.current_msg_id,
            "output_index": st.msg_index,
            "content_index": 0,
            "part": {"type": "output_text", "text": "", "annotations": [], "logprobs": []},
        });
        added["sequence_number"] = json!(next_seq(st));
        out.push(emit_event("response.content_part.added", &added));

        let mut msg_added = json!({
            "type": "response.output_item.added",
            "sequence_number": 0,
            "output_index": st.msg_index,
            "item": {"id": st.current_msg_id, "type": "message", "role": "assistant", "content": []},
        });
        msg_added["sequence_number"] = json!(next_seq(st));
        out.push(emit_event("response.output_item.added", &msg_added));
    }

    let mut delta = json!({
        "type": "response.output_text.delta",
        "sequence_number": 0,
        "item_id": st.current_msg_id,
        "output_index": st.msg_index,
        "content_index": 0,
        "delta": text,
    });
    delta["sequence_number"] = json!(next_seq(st));
    out.push(emit_event("response.output_text.delta", &delta));
}

fn handle_reasoning_delta(st: &mut State, part: &Value, out: &mut Vec<Value>) {
    if !st.reasoning_opened {
        st.reasoning_opened = true;
        st.reasoning_index += 1;
        st.reasoning_item_id = format!("rs_{}", st.response_id.trim_start_matches("resp_"));

        let mut added = json!({
            "type": "response.reasoning_summary_part.added",
            "sequence_number": 0,
            "item_id": st.reasoning_item_id,
            "output_index": st.reasoning_index,
            "summary_index": 0,
            "part": {"type": "summary_text", "text": ""},
        });
        added["sequence_number"] = json!(next_seq(st));
        out.push(emit_event("response.reasoning_summary_part.added", &added));

        let mut item_added = json!({
            "type": "response.output_item.added",
            "sequence_number": 0,
            "output_index": st.reasoning_index,
            "item": {"id": st.reasoning_item_id, "type": "reasoning", "summary": []},
        });
        item_added["sequence_number"] = json!(next_seq(st));
        out.push(emit_event("response.output_item.added", &item_added));
    }

    if let Some(text) = part.get("text").and_then(|v| v.as_str()) {
        st.reasoning_buf.push_str(text);
        let mut delta = json!({
            "type": "response.reasoning_summary_text.delta",
            "sequence_number": 0,
            "item_id": st.reasoning_item_id,
            "output_index": st.reasoning_index,
            "summary_index": 0,
            "delta": text,
        });
        delta["sequence_number"] = json!(next_seq(st));
        out.push(emit_event("response.reasoning_summary_text.delta", &delta));
    }

    if let Some(sig) = part.get("thoughtSignature").and_then(|v| v.as_str())
        && !sig.is_empty()
    {
        st.reasoning_enc = sig.to_string();
    }
}

fn finalize_reasoning(st: &mut State, out: &mut Vec<Value>) {
    if !st.reasoning_opened || st.reasoning_closed {
        return;
    }
    st.reasoning_closed = true;

    let mut text_done = json!({
        "type": "response.reasoning_summary_text.done",
        "sequence_number": 0,
        "item_id": st.reasoning_item_id,
        "output_index": st.reasoning_index,
        "summary_index": 0,
        "text": st.reasoning_buf,
    });
    text_done["sequence_number"] = json!(next_seq(st));
    out.push(emit_event(
        "response.reasoning_summary_text.done",
        &text_done,
    ));

    let mut part_done = json!({
        "type": "response.reasoning_summary_part.done",
        "sequence_number": 0,
        "item_id": st.reasoning_item_id,
        "output_index": st.reasoning_index,
        "summary_index": 0,
        "part": {"type": "summary_text", "text": st.reasoning_buf},
    });
    part_done["sequence_number"] = json!(next_seq(st));
    out.push(emit_event(
        "response.reasoning_summary_part.done",
        &part_done,
    ));

    let mut item_done = json!({
        "type": "response.output_item.done",
        "sequence_number": 0,
        "output_index": st.reasoning_index,
        "item": {
            "id": st.reasoning_item_id,
            "type": "reasoning",
            "encrypted_content": st.reasoning_enc,
            "status": "completed",
        },
    });
    item_done["sequence_number"] = json!(next_seq(st));
    out.push(emit_event("response.output_item.done", &item_done));
}

fn finalize_message(st: &mut State, out: &mut Vec<Value>) {
    if !st.msg_opened || st.msg_closed {
        return;
    }
    st.msg_closed = true;

    let mut part_done = json!({
        "type": "response.content_part.done",
        "sequence_number": 0,
        "item_id": st.current_msg_id,
        "output_index": st.msg_index,
        "content_index": 0,
        "part": {"type": "output_text", "text": st.text_buf, "annotations": [], "logprobs": []},
    });
    part_done["sequence_number"] = json!(next_seq(st));
    out.push(emit_event("response.content_part.done", &part_done));

    let mut item_done = json!({
        "type": "response.output_item.done",
        "sequence_number": 0,
        "output_index": st.msg_index,
        "item": {"id": st.current_msg_id, "type": "message", "status": "completed", "role": "assistant"},
    });
    item_done["sequence_number"] = json!(next_seq(st));
    out.push(emit_event("response.output_item.done", &item_done));
}

fn handle_function_call(st: &mut State, fc: &Value, out: &mut Vec<Value>, _model: &str) {
    let name = fc.get("name").and_then(|v| v.as_str()).unwrap_or("");
    let args = fc.get("args").cloned().unwrap_or(json!({}));
    let call_id = format!(
        "call_{:x}_{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos(),
        st.seq
    );

    st.next_index += 1;
    let idx = st.next_index;

    let mut item_added = json!({
        "type": "response.output_item.added",
        "sequence_number": 0,
        "output_index": idx,
        "item": {
            "id": format!("fc_{}", call_id),
            "type": "function_call",
            "name": name,
            "call_id": call_id,
        },
    });
    item_added["sequence_number"] = json!(next_seq(st));
    out.push(emit_event("response.output_item.added", &item_added));

    let mut args_delta = json!({
        "type": "response.function_call_arguments.delta",
        "sequence_number": 0,
        "output_index": idx,
        "delta": serde_json::to_string(&args).unwrap_or_default(),
    });
    args_delta["sequence_number"] = json!(next_seq(st));
    out.push(emit_event(
        "response.function_call_arguments.delta",
        &args_delta,
    ));

    let mut fc_done = json!({
        "type": "response.function_call_arguments.done",
        "sequence_number": 0,
        "output_index": idx,
        "arguments": serde_json::to_string(&args).unwrap_or_default(),
    });
    fc_done["sequence_number"] = json!(next_seq(st));
    out.push(emit_event(
        "response.function_call_arguments.done",
        &fc_done,
    ));

    let mut item_done = json!({
        "type": "response.output_item.done",
        "sequence_number": 0,
        "output_index": idx,
        "item": {
            "id": format!("fc_{}", call_id),
            "type": "function_call",
            "name": name,
            "call_id": call_id,
            "arguments": serde_json::to_string(&args).unwrap_or_default(),
            "status": "completed",
        },
    });
    item_done["sequence_number"] = json!(next_seq(st));
    out.push(emit_event("response.output_item.done", &item_done));
}

fn finalize_outputs(st: &mut State, out: &mut Vec<Value>, _model: &str) {
    finalize_reasoning(st, out);
    finalize_message(st, out);
}
