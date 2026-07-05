//! OpenAI Chat Completions → OpenAI Responses API response translation.
//!
//! Full state machine ported from Go's `openai/openai/responses/openai_openai-responses_response.go` (620 lines).
//! Handles SSE event lifecycle: response.created, output_item.added, output_text.delta,
//! function_call_arguments.delta, content_part.done, output_item.done, response.completed.

use serde_json::{Value, json};
use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicU64, Ordering};

static RESPONSE_ID_COUNTER: AtomicU64 = AtomicU64::new(0);

#[derive(Debug, Clone)]
pub(crate) struct ReasoningEntry {
    id: String,
    text: String,
    output_index: i32,
}

#[derive(Debug, Clone, Default)]
pub(crate) struct ChatToResponsesState {
    pub seq: i32,
    pub response_id: String,
    pub created_at: i64,
    pub started: bool,
    pub completion_pending: bool,
    pub completed_emitted: bool,
    pub reasoning_id: String,
    pub reasoning_index: i32,
    pub reasoning_buf: String,
    pub reasoning_entries: Vec<ReasoningEntry>,
    pub msg_text: HashMap<i32, String>,
    pub msg_output_ix: HashMap<i32, i32>,
    pub msg_item_added: HashSet<i32>,
    pub msg_content_added: HashSet<i32>,
    pub msg_item_done: HashSet<i32>,
    pub func_args: HashMap<String, String>,
    pub func_names: HashMap<String, String>,
    pub func_call_ids: HashMap<String, String>,
    pub func_output_ix: HashMap<String, i32>,
    pub func_item_done: HashSet<String>,
    pub next_output_ix: i32,
    pub prompt_tokens: i64,
    pub cached_tokens: i64,
    pub completion_tokens: i64,
    pub total_tokens: i64,
    pub reasoning_tokens: i64,
    pub usage_seen: bool,
}

impl ChatToResponsesState {
    fn next_seq(&mut self) -> i32 {
        self.seq += 1;
        self.seq
    }
    fn alloc_output_ix(&mut self) -> i32 {
        let i = self.next_output_ix;
        self.next_output_ix += 1;
        i
    }
}

fn sse(event: &str, data: Value) -> Value {
    Value::String(format!(
        "event: {event}\ndata: {}",
        serde_json::to_string(&data).unwrap_or_else(|_| "{}".to_string())
    ))
}

pub fn transform_non_stream(
    _model: &str,
    orig: &Value,
    trans: &Value,
    response: Value,
    _param: Option<&mut Box<dyn std::any::Any>>,
) -> Value {
    let generated_id;
    let rid = response.get("id").and_then(|v| v.as_str()).unwrap_or("");
    let rid = if rid.is_empty() {
        generated_id = generated_response_id();
        generated_id.as_str()
    } else {
        rid
    };
    let mut out = json!({
        "id": rid,
        "object": "response",
        "created_at": response.get("created").and_then(Value::as_i64).filter(|v| *v != 0).unwrap_or_else(now_unix),
        "status": "completed",
        "background": false,
        "error": null,
        "incomplete_details": null,
    });

    if trans.is_object() {
        echo_request_fields(&mut out, trans, true);
    }
    if out.get("model").is_none()
        && let Some(model) = response.get("model").and_then(Value::as_str)
    {
        out["model"] = json!(model);
    }

    let mut output_items: Vec<(i32, Value)> = Vec::new();
    let reasoning_text = response
        .pointer("/choices/0/message/reasoning_content")
        .and_then(Value::as_str)
        .unwrap_or("");
    let include_reasoning = !reasoning_text.is_empty() || trans.get("reasoning").is_some();
    if include_reasoning {
        let mut item = json!({
            "id": format!("rs_{}", rid.strip_prefix("resp_").unwrap_or(rid)),
            "type": "reasoning",
            "encrypted_content": "",
            "summary": [],
        });
        if !reasoning_text.is_empty() {
            item["summary"] = json!([{"type": "summary_text", "text": reasoning_text}]);
        }
        output_items.push((0, item));
    }

    let mut ci = 0i32;
    if let Some(choices) = response.get("choices").and_then(|v| v.as_array()) {
        for choice in choices {
            let idx = choice
                .get("index")
                .and_then(|v| v.as_i64())
                .unwrap_or(ci as i64) as i32;
            ci = idx + 1;
            if let Some(msg) = choice.get("message") {
                if let Some(t) = msg.get("content").and_then(|v| v.as_str())
                    && !t.is_empty()
                {
                    output_items.push((1, json!({"type":"message","id":format!("msg_{}_{}",rid,idx),"status":"completed","content":[{"type":"output_text","annotations":[],"logprobs":[],"text":t}],"role":"assistant"})));
                }
                if let Some(tcs) = msg.get("tool_calls").and_then(|v| v.as_array()) {
                    for (ti, tc) in tcs.iter().enumerate() {
                        let call_id = tc.get("id").and_then(|v| v.as_str()).unwrap_or("");
                        let name = tc
                            .pointer("/function/name")
                            .and_then(|v| v.as_str())
                            .unwrap_or("");
                        let args = tc
                            .pointer("/function/arguments")
                            .and_then(|v| v.as_str())
                            .unwrap_or("");
                        let item = function_call_item(
                            orig,
                            name,
                            json!({
                                "type": "function_call",
                                "id": format!("fc_{call_id}"),
                                "status": "completed",
                                "arguments": args,
                                "call_id": call_id,
                                "name": "",
                            }),
                        );
                        output_items.push((2 + ti as i32, item));
                    }
                }
            }
        }
    }
    output_items.sort_by_key(|k| k.0);
    if !output_items.is_empty() {
        out["output"] = json!(output_items.into_iter().map(|(_, v)| v).collect::<Vec<_>>());
    }
    if let Some(usage) = response.get("usage") {
        let has_token_usage = usage.get("prompt_tokens").is_some()
            || usage.get("completion_tokens").is_some()
            || usage.get("total_tokens").is_some();
        if has_token_usage {
            let pt = usage
                .get("prompt_tokens")
                .and_then(|v| v.as_i64())
                .unwrap_or(0);
            let ct = usage
                .get("completion_tokens")
                .and_then(|v| v.as_i64())
                .unwrap_or(0);
            let total = usage
                .get("total_tokens")
                .and_then(|v| v.as_i64())
                .unwrap_or(0);
            out["usage"]["input_tokens"] = json!(pt);
            if let Some(cached) = usage
                .pointer("/prompt_tokens_details/cached_tokens")
                .and_then(|v| v.as_i64())
            {
                out["usage"]["input_tokens_details"] = json!({"cached_tokens": cached});
            }
            out["usage"]["output_tokens"] = json!(ct);
            if let Some(rt) = usage
                .pointer("/output_tokens_details/reasoning_tokens")
                .and_then(|v| v.as_i64())
            {
                out["usage"]["output_tokens_details"] = json!({"reasoning_tokens": rt});
            }
            out["usage"]["total_tokens"] = json!(total);
        } else {
            out["usage"] = usage.clone();
        }
    }
    out
}

pub fn transform_stream(
    _model: &str,
    original_request: &Value,
    _translated_request: &Value,
    chunk: Value,
    param: Option<&mut Box<dyn std::any::Any>>,
) -> Vec<Value> {
    let normalized = normalize_stream_chunk(chunk);
    let st = if let Some(p) = param {
        if !p.is::<ChatToResponsesState>() {
            *p = Box::<ChatToResponsesState>::default();
        }
        p.downcast_mut::<ChatToResponsesState>().unwrap()
    } else {
        return vec![normalized];
    };
    let request_for_ns = if original_request.is_object() {
        original_request
    } else {
        _translated_request
    };

    if normalized.as_str() == Some("[DONE]") {
        if st.completion_pending && !st.completed_emitted {
            st.completed_emitted = true;
            return vec![sse(
                "response.completed",
                build_completed(st, request_for_ns),
            )];
        }
        return vec![];
    }

    let chunk = normalized;
    if !is_chat_completion_chunk(&chunk) {
        return Vec::new();
    }

    let mut out: Vec<Value> = Vec::new();

    if let Some(uv) = chunk.get("usage")
        && uv.is_object()
    {
        if let Some(v) = uv.get("prompt_tokens").and_then(|v| v.as_i64()) {
            st.prompt_tokens = v;
            st.usage_seen = true;
        }
        if let Some(v) = uv
            .pointer("/prompt_tokens_details/cached_tokens")
            .and_then(|v| v.as_i64())
        {
            st.cached_tokens = v;
            st.usage_seen = true;
        }
        if let Some(v) = uv.get("completion_tokens").and_then(|v| v.as_i64()) {
            st.completion_tokens = v;
            st.usage_seen = true;
        }
        if let Some(v) = uv.get("output_tokens").and_then(|v| v.as_i64()) {
            st.completion_tokens = v;
            st.usage_seen = true;
        }
        if let Some(v) = uv
            .pointer("/output_tokens_details/reasoning_tokens")
            .or_else(|| uv.pointer("/completion_tokens_details/reasoning_tokens"))
            .and_then(|v| v.as_i64())
        {
            st.reasoning_tokens = v;
            st.usage_seen = true;
        }
        if let Some(v) = uv.get("total_tokens").and_then(|v| v.as_i64()) {
            st.total_tokens = v;
            st.usage_seen = true;
        }
    }

    if !st.started {
        st.response_id = chunk
            .get("id")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        st.created_at = chunk.get("created").and_then(|v| v.as_i64()).unwrap_or(0);
        out.push(sse("response.created", json!({"type":"response.created","sequence_number":st.next_seq(),"response":{"id":&st.response_id,"object":"response","created_at":st.created_at,"status":"in_progress","background":false,"error":null,"output":[]}})));
        out.push(sse("response.in_progress", json!({"type":"response.in_progress","sequence_number":st.next_seq(),"response":{"id":&st.response_id,"object":"response","created_at":st.created_at,"status":"in_progress"}})));
        st.started = true;
        st.msg_text.clear();
        st.msg_item_added.clear();
        st.msg_content_added.clear();
        st.msg_item_done.clear();
        st.func_args.clear();
        st.func_call_ids.clear();
        st.func_names.clear();
        st.func_output_ix.clear();
        st.func_item_done.clear();
        st.next_output_ix = 0;
        st.completion_pending = false;
        st.completed_emitted = false;
        st.prompt_tokens = 0;
        st.cached_tokens = 0;
        st.completion_tokens = 0;
        st.total_tokens = 0;
        st.reasoning_tokens = 0;
        st.usage_seen = false;
        st.reasoning_entries.clear();
    }

    if let Some(choices) = chunk.get("choices").and_then(|v| v.as_array()) {
        for choice in choices {
            let idx = choice.get("index").and_then(|v| v.as_i64()).unwrap_or(0) as i32;
            if let Some(d) = choice.get("delta") {
                if let Some(text) = d.get("content").and_then(|v| v.as_str())
                    && !text.is_empty()
                {
                    if !st.reasoning_id.is_empty() {
                        stop_reasoning(st, &mut out);
                    }
                    if !st.msg_output_ix.contains_key(&idx) {
                        let ox = st.alloc_output_ix();
                        st.msg_output_ix.insert(idx, ox);
                    }
                    let ox = st.msg_output_ix[&idx];
                    if !st.msg_item_added.contains(&idx) {
                        out.push(sse("response.output_item.added", json!({"type":"response.output_item.added","sequence_number":st.next_seq(),"output_index":ox,"item":{"id":format!("msg_{}_{}",st.response_id,idx),"type":"message","status":"in_progress","content":[],"role":"assistant"}})));
                        st.msg_item_added.insert(idx);
                    }
                    if !st.msg_content_added.contains(&idx) {
                        out.push(sse("response.content_part.added", json!({"type":"response.content_part.added","sequence_number":st.next_seq(),"item_id":format!("msg_{}_{}",st.response_id,idx),"output_index":ox,"content_index":0,"part":{"type":"output_text","annotations":[],"logprobs":[],"text":""}})));
                        st.msg_content_added.insert(idx);
                    }
                    out.push(sse("response.output_text.delta", json!({"type":"response.output_text.delta","sequence_number":st.next_seq(),"item_id":format!("msg_{}_{}",st.response_id,idx),"output_index":ox,"content_index":0,"delta":text,"logprobs":[]})));
                    st.msg_text.entry(idx).or_default().push_str(text);
                }
                if let Some(rc) = d.get("reasoning_content").and_then(|v| v.as_str())
                    && !rc.is_empty()
                {
                    if st.reasoning_id.is_empty() {
                        st.reasoning_id = format!("rs_{}_{}", st.response_id, idx);
                        st.reasoning_index = st.alloc_output_ix();
                        out.push(sse("response.output_item.added", json!({"type":"response.output_item.added","sequence_number":st.next_seq(),"output_index":st.reasoning_index,"item":{"id":&st.reasoning_id,"type":"reasoning","status":"in_progress","summary":[]}})));
                        out.push(sse("response.reasoning_summary_part.added", json!({"type":"response.reasoning_summary_part.added","sequence_number":st.next_seq(),"item_id":&st.reasoning_id,"output_index":st.reasoning_index,"summary_index":0,"part":{"type":"summary_text","text":""}})));
                    }
                    st.reasoning_buf.push_str(rc);
                    out.push(sse("response.reasoning_summary_text.delta", json!({"type":"response.reasoning_summary_text.delta","sequence_number":st.next_seq(),"item_id":&st.reasoning_id,"output_index":st.reasoning_index,"summary_index":0,"delta":rc})));
                }
                if let Some(tcs) = d.get("tool_calls").and_then(|v| v.as_array()) {
                    if !st.reasoning_id.is_empty() {
                        stop_reasoning(st, &mut out);
                    }
                    if st.msg_item_added.contains(&idx) && !st.msg_item_done.contains(&idx) {
                        finalize_msg(st, idx, &mut out);
                    }
                    for tc in tcs {
                        let ti = tc.get("index").and_then(|v| v.as_i64()).unwrap_or(0) as i32;
                        let key = format!("{}:{}", idx, ti);
                        let nid = tc
                            .get("id")
                            .and_then(|v| v.as_str())
                            .unwrap_or("")
                            .to_string();
                        if let Some(n) = tc.pointer("/function/name").and_then(|v| v.as_str())
                            && !n.is_empty()
                        {
                            st.func_names.insert(key.clone(), n.to_string());
                        }
                        let should_emit_item = st
                            .func_call_ids
                            .get(&key)
                            .map(|s| s.is_empty())
                            .unwrap_or(true)
                            && !nid.is_empty();
                        if should_emit_item {
                            st.func_call_ids.insert(key.clone(), nid.clone());
                            let ox = st.alloc_output_ix();
                            st.func_output_ix.insert(key.clone(), ox);
                        }
                        let eid = st.func_call_ids.get(&key).cloned().unwrap_or_default();
                        if should_emit_item {
                            let ox = st.func_output_ix[&key];
                            let item = function_call_item(
                                request_for_ns,
                                &st.func_names.get(&key).cloned().unwrap_or_default(),
                                json!({
                                    "id": format!("fc_{}", eid),
                                    "type": "function_call",
                                    "status": "in_progress",
                                    "arguments": "",
                                    "call_id": &eid,
                                    "name": "",
                                }),
                            );
                            out.push(sse("response.output_item.added", json!({"type":"response.output_item.added","sequence_number":st.next_seq(),"output_index":ox,"item":item})));
                        }
                        st.func_args.entry(key.clone()).or_default();
                        if let Some(args) =
                            tc.pointer("/function/arguments").and_then(|v| v.as_str())
                            && !args.is_empty()
                        {
                            let ref_id = st
                                .func_call_ids
                                .get(&key)
                                .cloned()
                                .unwrap_or_else(|| nid.clone());
                            if !ref_id.is_empty() {
                                out.push(sse("response.function_call_arguments.delta", json!({"type":"response.function_call_arguments.delta","sequence_number":st.next_seq(),"item_id":format!("fc_{}",ref_id),"output_index":st.func_output_ix[&key],"delta":args})));
                            }
                            st.func_args.get_mut(&key).unwrap().push_str(args);
                        }
                    }
                }
            }
            if let Some(fr) = choice.get("finish_reason").and_then(|v| v.as_str())
                && !fr.is_empty()
            {
                let ids: Vec<i32> = st.msg_item_added.iter().copied().collect();
                let mut ids = ids;
                ids.sort_by_key(|i| st.msg_output_ix.get(i).copied().unwrap_or(i32::MAX));
                for i in ids {
                    if st.msg_item_added.contains(&i) && !st.msg_item_done.contains(&i) {
                        finalize_msg(st, i, &mut out);
                    }
                }
                if !st.reasoning_id.is_empty() {
                    stop_reasoning(st, &mut out);
                }
                let mut ks: Vec<String> = st.func_call_ids.keys().cloned().collect();
                ks.sort_by(|a, b| {
                    let left = st.func_output_ix.get(a).copied().unwrap_or(i32::MAX);
                    let right = st.func_output_ix.get(b).copied().unwrap_or(i32::MAX);
                    left.cmp(&right).then_with(|| a.cmp(b))
                });
                for k in &ks {
                    let cid = st.func_call_ids.get(k).cloned().unwrap_or_default();
                    if cid.is_empty() || st.func_item_done.contains(k) {
                        continue;
                    }
                    let ox = st.func_output_ix[k];
                    let args = st
                        .func_args
                        .get(k)
                        .cloned()
                        .unwrap_or_else(|| "{}".to_string());
                    out.push(sse("response.function_call_arguments.done", json!({"type":"response.function_call_arguments.done","sequence_number":st.next_seq(),"item_id":format!("fc_{}",cid),"output_index":ox,"arguments":&args})));
                    let item = function_call_item(
                        request_for_ns,
                        st.func_names.get(k).map(String::as_str).unwrap_or(""),
                        json!({
                            "id": format!("fc_{}", cid),
                            "type": "function_call",
                            "status": "completed",
                            "arguments": &args,
                            "call_id": &cid,
                            "name": "",
                        }),
                    );
                    out.push(sse("response.output_item.done", json!({"type":"response.output_item.done","sequence_number":st.next_seq(),"output_index":ox,"item":item})));
                    st.func_item_done.insert(k.clone());
                }
                st.completion_pending = true;
            }
        }
    }
    out
}

fn stop_reasoning(st: &mut ChatToResponsesState, out: &mut Vec<Value>) {
    let text = std::mem::take(&mut st.reasoning_buf);
    out.push(sse("response.reasoning_summary_text.done", json!({"type":"response.reasoning_summary_text.done","sequence_number":st.next_seq(),"item_id":&st.reasoning_id,"output_index":st.reasoning_index,"summary_index":0,"text":&text})));
    out.push(sse("response.reasoning_summary_part.done", json!({"type":"response.reasoning_summary_part.done","sequence_number":st.next_seq(),"item_id":&st.reasoning_id,"output_index":st.reasoning_index,"summary_index":0,"part":{"type":"summary_text","text":&text}})));
    out.push(sse("response.output_item.done", json!({"type":"response.output_item.done","sequence_number":st.next_seq(),"output_index":st.reasoning_index,"item":{"id":&st.reasoning_id,"type":"reasoning","encrypted_content":"","summary":[{"type":"summary_text","text":&text}]}})));
    st.reasoning_entries.push(ReasoningEntry {
        id: std::mem::take(&mut st.reasoning_id),
        text,
        output_index: st.reasoning_index,
    });
}

fn finalize_msg(st: &mut ChatToResponsesState, idx: i32, out: &mut Vec<Value>) {
    let ox = st.msg_output_ix.get(&idx).copied().unwrap_or(0);
    let ft = st.msg_text.get(&idx).cloned().unwrap_or_default();
    out.push(sse("response.output_text.done", json!({"type":"response.output_text.done","sequence_number":st.next_seq(),"item_id":format!("msg_{}_{}",st.response_id,idx),"output_index":ox,"content_index":0,"text":&ft,"logprobs":[]})));
    out.push(sse("response.content_part.done", json!({"type":"response.content_part.done","sequence_number":st.next_seq(),"item_id":format!("msg_{}_{}",st.response_id,idx),"output_index":ox,"content_index":0,"part":{"type":"output_text","annotations":[],"logprobs":[],"text":&ft}})));
    out.push(sse("response.output_item.done", json!({"type":"response.output_item.done","sequence_number":st.next_seq(),"output_index":ox,"item":{"type":"message","status":"completed","content":[{"type":"output_text","annotations":[],"logprobs":[],"text":&ft}],"role":"assistant","id":format!("msg_{}_{}",st.response_id,idx)}})));
    st.msg_item_done.insert(idx);
}

fn build_completed(st: &ChatToResponsesState, rj: &Value) -> Value {
    let mut c = json!({"type":"response.completed","sequence_number":st.seq+1,"response":{"id":&st.response_id,"object":"response","created_at":st.created_at,"status":"completed","background":false,"error":null}});
    echo_request_fields(&mut c["response"], rj, false);
    let mut items: Vec<(i32, Value)> = Vec::new();
    for r in &st.reasoning_entries {
        items.push((r.output_index, json!({"id":&r.id,"type":"reasoning","summary":[{"type":"summary_text","text":&r.text}]})));
    }
    let mut mids: Vec<i32> = st.msg_item_added.iter().copied().collect();
    mids.sort();
    for i in mids {
        let ox = st.msg_output_ix.get(&i).copied().unwrap_or(0);
        let t = st.msg_text.get(&i).cloned().unwrap_or_default();
        items.push((ox, json!({"id":format!("msg_{}_{}",st.response_id,i),"type":"message","status":"completed","content":[{"type":"output_text","annotations":[],"logprobs":[],"text":&t}],"role":"assistant"})));
    }
    for (k, cid) in &st.func_call_ids {
        if cid.is_empty() {
            continue;
        }
        let ox = st.func_output_ix.get(k).copied().unwrap_or(0);
        let a = st
            .func_args
            .get(k)
            .cloned()
            .unwrap_or_else(|| "{}".to_string());
        let item = function_call_item(
            rj,
            st.func_names.get(k).map(String::as_str).unwrap_or(""),
            json!({
                "id": format!("fc_{}", cid),
                "type": "function_call",
                "status": "completed",
                "arguments": &a,
                "call_id": cid,
                "name": "",
            }),
        );
        items.push((ox, item));
    }
    items.sort_by_key(|k| k.0);
    if !items.is_empty() {
        c["response"]["output"] = json!(items.into_iter().map(|(_, v)| v).collect::<Vec<_>>());
    }
    if st.usage_seen {
        let total = if st.total_tokens > 0 {
            st.total_tokens
        } else {
            st.prompt_tokens + st.completion_tokens
        };
        c["response"]["usage"] = json!({"input_tokens":st.prompt_tokens,"input_tokens_details":{"cached_tokens":st.cached_tokens},"output_tokens":st.completion_tokens,"total_tokens":total});
        if st.reasoning_tokens > 0 {
            c["response"]["usage"]["output_tokens_details"] =
                json!({"reasoning_tokens": st.reasoning_tokens});
        }
    }
    c
}

fn normalize_stream_chunk(value: Value) -> Value {
    let Some(raw) = value.as_str() else {
        return value;
    };
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return Value::String(String::new());
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

fn is_chat_completion_chunk(chunk: &Value) -> bool {
    if let Some(object) = chunk.get("object").and_then(Value::as_str)
        && !object.is_empty()
        && object != "chat.completion.chunk"
    {
        return false;
    }
    chunk.get("choices").and_then(Value::as_array).is_some()
}

fn echo_request_fields(target: &mut Value, request: &Value, non_stream: bool) {
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
    if non_stream
        && target.get("max_output_tokens").is_none()
        && request.get("max_output_tokens").is_none()
        && let Some(value) = request.get("max_tokens")
    {
        target["max_output_tokens"] = value.clone();
    }
}

fn function_call_item(request: &Value, qualified_name: &str, mut item: Value) -> Value {
    let (name, namespace) = split_qualified_function_call_from_request(request, qualified_name);
    item["name"] = json!(name);
    if namespace.is_empty() {
        if let Some(obj) = item.as_object_mut() {
            obj.remove("namespace");
        }
    } else {
        item["namespace"] = json!(namespace);
    }
    item
}

fn split_qualified_function_call_from_request<'a>(
    request: &'a Value,
    qualified_name: &'a str,
) -> (&'a str, &'a str) {
    let qualified_name = qualified_name.trim();
    if qualified_name.is_empty() {
        return ("", "");
    }
    let Some(tools) = request.get("tools").and_then(Value::as_array) else {
        return (qualified_name, "");
    };

    // First pass: try qualified-name match (exact)
    // Second pass fallback: bare-name match (when upstream returns unqualified name)
    let mut best_namespace: &str = "";
    let mut best_child: &str = "";

    for tool in tools {
        if tool.get("type").and_then(Value::as_str) != Some("namespace") {
            continue;
        }
        let namespace = tool
            .get("name")
            .and_then(Value::as_str)
            .unwrap_or("")
            .trim();
        if namespace.is_empty() {
            continue;
        }
        let Some(children) = tool.get("tools").and_then(Value::as_array) else {
            continue;
        };

        // Per-namespace bare-name match tracking
        let mut bare_match_namespace: &str = "";
        let mut bare_match_child: &str = "";

        for child in children {
            let child_name = responses_tool_name(child);
            if child_name.is_empty() {
                continue;
            }
            // Track bare-name match if upstream returned unqualified name
            if child_name == qualified_name && bare_match_child.is_empty() {
                bare_match_namespace = namespace;
                bare_match_child = child_name;
            }
            // Qualified-name match takes priority
            if qualify_responses_namespace_tool_name(namespace, child_name) == qualified_name {
                best_namespace = namespace;
                best_child = child_name;
                return (best_child, best_namespace);
            }
        }

        // Fallback: use bare-name match if no qualified match found in this namespace
        if best_child.is_empty() && !bare_match_child.is_empty() {
            best_namespace = bare_match_namespace;
            best_child = bare_match_child;
        }
    }

    if best_namespace.is_empty() || best_child.is_empty() {
        (qualified_name, "")
    } else {
        (best_child, best_namespace)
    }
}

fn responses_tool_name(tool: &Value) -> &str {
    tool.get("name")
        .and_then(Value::as_str)
        .filter(|name| !name.trim().is_empty())
        .or_else(|| tool.pointer("/function/name").and_then(Value::as_str))
        .unwrap_or("")
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

fn generated_response_id() -> String {
    format!(
        "resp_{:x}_{}",
        now_nanos(),
        RESPONSE_ID_COUNTER.fetch_add(1, Ordering::Relaxed) + 1
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

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_chat_to_responses_non_stream() {
        let resp = json!({"id":"chatcmpl-abc","model":"gpt-4","choices":[{"message":{"role":"assistant","content":"Hello!"},"finish_reason":"stop"}],"usage":{"prompt_tokens":5,"completion_tokens":3}});
        let r = transform_non_stream("gpt-4", &json!({}), &json!({}), resp, None);
        assert_eq!(r["output"][0]["content"][0]["text"], "Hello!");
        assert_eq!(r["usage"]["input_tokens"], 5);
    }
}
