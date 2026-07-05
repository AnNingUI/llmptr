//! Codex (Responses API SSE) → Claude SSE — full state machine translation.
//!
//! This is a direct port of the Go `codex_claude_response.go` state machine,
//! handling all edge cases:
//! - Pending function calls (name can arrive after call_id)
//! - Thinking / reasoning block lifecycle with signature
//! - Web search embedded tool use/results
//! - Multi-tool parallel calls with BlockIndex tracking
//! - Text content accumulation (delta + output_item.done fallback)
//! - SSE error event remapping
//! - Non-stream aggregation

use serde_json::{Value, json};
use std::collections::HashMap;

// ── SSE helpers ──────────────────────────────────────────────

// sse_event and sse_bytes intentionally omitted — the Rust version
// stores pseudo-SSE events as JSON values internally; the last-mile
// serialisation step emits the "data: " prefix from the middlebox.

// ── Pending function call tracking ────────────────────────────

#[derive(Debug, Clone)]
pub struct PendingFnCall {
    pub call_id: String,
    pub arguments: String,
    pub block_index: usize,
    pub has_received_arguments_delta: bool,
    pub start_emitted: bool,
}

fn fn_call_key(output_index: Option<u64>, call_id: &str) -> String {
    if let Some(idx) = output_index {
        format!("output:{}", idx)
    } else if !call_id.is_empty() {
        format!("call:{}", call_id)
    } else {
        "last".to_string()
    }
}

// ── Web search helpers ────────────────────────────────────────

fn web_search_tool_use_id(item: &Value, last_id: &str, block_index: usize) -> String {
    for key in &["id", "output_item_id", "call_id"] {
        if let Some(v) = item.get(*key).and_then(|x| x.as_str())
            && !v.is_empty()
        {
            return v.to_string();
        }
    }
    if !last_id.is_empty() {
        return last_id.to_string();
    }
    for key in &["item_id"] {
        if let Some(v) = item.get(*key).and_then(|x| x.as_str())
            && !v.is_empty()
        {
            return v.to_string();
        }
    }
    format!("web_search_{}", block_index)
}

fn web_search_query(item: &Value) -> String {
    for path in &["action.query", "query", "input.query"] {
        if let Some(v) = item
            .pointer(&format!("/{}", path.replace('.', "/")))
            .and_then(|x| x.as_str())
            && !v.is_empty()
        {
            return v.to_string();
        }
    }
    String::new()
}

fn web_search_result_content(item: &Value) -> Vec<Value> {
    let results = item.get("results").and_then(|v| v.as_array());
    let results = results.or_else(|| item.get("results").and_then(|v| v.as_array()));

    let mut out = Vec::new();
    if let Some(arr) = results {
        for r in arr {
            let url = r.get("url").and_then(|v| v.as_str()).unwrap_or("");
            if url.is_empty() {
                continue;
            }
            let title = r.get("title").and_then(|v| v.as_str()).unwrap_or(url);
            out.push(json!({
                "type": "web_search_result",
                "title": title,
                "url": url,
            }));
        }
    }
    out
}

// ── Core state machine ────────────────────────────────────────

/// Full streaming state for Codex → Claude conversion.
///
/// Mirrors Go `ConvertCodexResponseToClaudeParams` exactly.
#[derive(Debug, Clone, Default)]
pub struct CodexToClaudeState {
    pub has_tool_call: bool,
    pub block_index: usize,
    pub has_received_arguments_delta: bool,
    pub has_text_delta: bool,
    pub text_block_open: bool,
    pub thinking_block_open: bool,
    pub thinking_stop_pending: bool,
    pub thinking_signature: String,
    pub thinking_summary_seen: bool,
    pub web_search_tool_use_ids: Vec<String>,
    pub web_search_tool_result_ids: Vec<String>,
    pub last_web_search_tool_use_id: String,
    pub pending_function_calls: HashMap<String, PendingFnCall>,
    pub last_pending_function_call_key: String,
    pub tool_name_rev: HashMap<String, String>, // short → original name
}

impl CodexToClaudeState {
    pub fn new(tool_name_rev: HashMap<String, String>) -> Self {
        Self {
            tool_name_rev,
            ..Default::default()
        }
    }

    // ── thinking block helpers ──────────────────────────────

    fn start_thinking_block(&mut self) -> Vec<Value> {
        if self.thinking_block_open {
            return vec![];
        }
        self.thinking_block_open = true;
        self.thinking_stop_pending = false;
        vec![json!({
            "type": "content_block_start",
            "index": self.block_index,
            "content_block": {"type": "thinking", "thinking": ""}
        })]
    }

    fn finalize_thinking_block(&mut self) -> Vec<Value> {
        if !self.thinking_block_open {
            return vec![];
        }
        let mut out = Vec::new();

        // Signature delta
        if !self.thinking_signature.is_empty() {
            out.push(json!({
                "type": "content_block_delta",
                "index": self.block_index,
                "delta": {"type": "signature_delta", "signature": &self.thinking_signature}
            }));
        }

        // Content block stop
        out.push(json!({
            "type": "content_block_stop",
            "index": self.block_index
        }));

        self.block_index += 1;
        self.thinking_block_open = false;
        self.thinking_stop_pending = false;
        out
    }

    fn finalize_signature_only_thinking(&mut self) -> Vec<Value> {
        if self.thinking_signature.is_empty() {
            return vec![];
        }
        let mut out = self.start_thinking_block();
        out.extend(self.finalize_thinking_block());
        out
    }

    // ── text block helpers ──────────────────────────────────

    fn start_text_block(&mut self) -> Vec<Value> {
        if self.text_block_open {
            return vec![];
        }
        self.text_block_open = true;
        vec![json!({
            "type": "content_block_start",
            "index": self.block_index,
            "content_block": {"type": "text", "text": ""}
        })]
    }

    fn stop_text_block(&mut self) -> Vec<Value> {
        if !self.text_block_open {
            return vec![];
        }
        let out = vec![json!({
            "type": "content_block_stop",
            "index": self.block_index
        })];
        self.text_block_open = false;
        self.block_index += 1;
        out
    }

    // ── function call helpers ───────────────────────────────

    fn append_fn_call_start(&mut self, call_id: &str, name: &str) -> Vec<Value> {
        let resolved_name = self
            .tool_name_rev
            .get(name)
            .cloned()
            .unwrap_or_else(|| name.to_string());
        vec![json!({
            "type": "content_block_start",
            "index": self.block_index,
            "content_block": {
                "type": "tool_use",
                "id": call_id,
                "name": resolved_name,
                "input": {}
            }
        })]
    }

    fn append_fn_call_arg_delta(&self, args: &str) -> Vec<Value> {
        if args.is_empty() {
            return vec![];
        }
        vec![json!({
            "type": "content_block_delta",
            "index": self.block_index,
            "delta": {"type": "input_json_delta", "partial_json": args}
        })]
    }

    fn append_fn_call_stop(&self) -> Vec<Value> {
        vec![json!({
            "type": "content_block_stop",
            "index": self.block_index
        })]
    }

    // ── web search helpers ──────────────────────────────────

    fn append_web_search_server_tool_use(&mut self, item: &Value) -> Vec<Value> {
        let tool_use_id =
            web_search_tool_use_id(item, &self.last_web_search_tool_use_id, self.block_index);
        if tool_use_id.is_empty() {
            return vec![];
        }

        let already_started = self.web_search_tool_use_ids.contains(&tool_use_id);
        let query = web_search_query(item);
        if already_started && query.is_empty() {
            return vec![];
        }

        let mut out = Vec::new();
        let _tool_use_id_clone = tool_use_id.clone();

        if !already_started {
            out.extend(self.finalize_thinking_block());
            out.push(json!({
                "type": "content_block_start",
                "index": self.block_index,
                "content_block": {
                    "type": "server_tool_use",
                    "id": &tool_use_id,
                    "name": "web_search",
                    "input": {}
                }
            }));
        }

        if !query.is_empty() {
            let partial = json!({"query": query}).to_string();
            out.push(json!({
                "type": "content_block_delta",
                "index": self.block_index,
                "delta": {"type": "input_json_delta", "partial_json": partial}
            }));
        }

        if !already_started {
            out.push(json!({
                "type": "content_block_stop",
                "index": self.block_index
            }));
            self.web_search_tool_use_ids.push(tool_use_id.clone());
            self.block_index += 1;
        }

        if let Some(last) = self.web_search_tool_use_ids.last() {
            self.last_web_search_tool_use_id.clone_from(last);
        }
        out
    }

    fn append_web_search_tool_result(&mut self, item: &Value) -> Vec<Value> {
        let tool_use_id =
            web_search_tool_use_id(item, &self.last_web_search_tool_use_id, self.block_index);
        if tool_use_id.is_empty() {
            return vec![];
        }

        let mut out = self.append_web_search_server_tool_use(item);

        let id_seen = self.web_search_tool_result_ids.contains(&tool_use_id);
        if id_seen {
            return out;
        }

        let query = web_search_query(item);
        let results = web_search_result_content(item);
        if query.is_empty() && results.is_empty() && item.get("action").is_none() {
            return out;
        }

        let tid_clone = tool_use_id.clone();
        out.push(json!({
            "type": "content_block_start",
            "index": self.block_index,
            "content_block": {
                "type": "web_search_tool_result",
                "tool_use_id": &tid_clone,
                "content": results,
            }
        }));
        out.push(json!({
            "type": "content_block_stop",
            "index": self.block_index
        }));
        self.web_search_tool_result_ids.push(tid_clone.clone());
        self.block_index += 1;

        if self.last_web_search_tool_use_id == tid_clone {
            self.last_web_search_tool_use_id.clear();
        }
        out
    }
}

// ── Main streaming transform ──────────────────────────────────

/// Transform a Codex SSE streaming event to Claude SSE events.
///
/// This is a direct port of Go's `ConvertCodexResponseToClaude`, supporting:
/// - `response.created` → `message_start`
/// - `response.reasoning_summary_part.added` / `.delta` / `.done` → thinking blocks
/// - `response.content_part.added` / `response.output_text.delta` / `response.content_part.done` → text blocks
/// - `response.output_item.added` / `.done` (function_call, reasoning, web_search_call) → tool_use / thinking
/// - `response.function_call_arguments.delta/.done` → input_json_delta
/// - `response.completed` / `.incomplete` → `message_delta` + `message_stop`
/// - `error` → Claude error SSE
pub fn transform_stream(
    _model: &str,
    _original_request: &Value,
    _translated_request: &Value,
    chunk: Value,
    mut param: Option<&mut Box<dyn std::any::Any>>,
) -> Vec<Value> {
    let st: &mut CodexToClaudeState = if let Some(p) = &mut param {
        p.downcast_mut().unwrap()
    } else {
        return vec![chunk];
    };

    let event_type = chunk.get("type").and_then(|v| v.as_str()).unwrap_or("");
    let mut out = Vec::new();

    // ── ThinkingStopPending gate ────────────────────────────
    if st.thinking_block_open && st.thinking_stop_pending {
        match event_type {
            "response.content_part.added" | "response.completed" | "response.incomplete" => {
                out.extend(st.finalize_thinking_block());
            }
            _ => {}
        }
    }

    match event_type {
        "error" => {
            out.push(codex_error_to_claude_error(&chunk));
        }

        // ── response.created: begin message ───────────────────
        "response.created" => {
            let resp = chunk.get("response");
            let msg_id = resp
                .and_then(|r| r.get("id"))
                .and_then(|v| v.as_str())
                .unwrap_or("");
            let model = resp
                .and_then(|r| r.get("model"))
                .and_then(|v| v.as_str())
                .unwrap_or("");

            let msg = json!({
                "type": "message_start",
                "message": {
                    "id": msg_id,
                    "type": "message",
                    "role": "assistant",
                    "model": model,
                    "stop_sequence": null,
                    "usage": {"input_tokens": 0, "output_tokens": 0},
                    "content": [],
                    "stop_reason": null,
                }
            });
            out.push(msg);
        }

        // ── reasoning blocks ─────────────────────────────────
        "response.reasoning_summary_part.added" => {
            if st.thinking_block_open && st.thinking_stop_pending {
                out.extend(st.finalize_thinking_block());
            }
            st.thinking_summary_seen = true;
            out.extend(st.start_thinking_block());
        }

        "response.reasoning_summary_text.delta" => {
            if let Some(delta) = chunk.get("delta").and_then(|v| v.as_str()) {
                out.push(json!({
                    "type": "content_block_delta",
                    "index": st.block_index,
                    "delta": {"type": "thinking_delta", "thinking": delta}
                }));
            }
        }

        "response.reasoning_summary_part.done" => {
            st.thinking_stop_pending = true;
        }

        // ── text content blocks ──────────────────────────────
        "response.content_part.added" => {
            if chunk.pointer("/part/type").and_then(|v| v.as_str()) == Some("output_text") {
                out.extend(st.start_text_block());
            }
        }

        "response.output_text.delta" => {
            st.has_text_delta = true;
            out.extend(st.finalize_thinking_block());
            out.extend(st.start_text_block());

            if let Some(delta) = chunk.get("delta").and_then(|v| v.as_str()) {
                out.push(json!({
                    "type": "content_block_delta",
                    "index": st.block_index,
                    "delta": {"type": "text_delta", "text": delta}
                }));
            }
        }

        "response.content_part.done" => {
            if chunk.pointer("/part/type").and_then(|v| v.as_str()) == Some("output_text") {
                out.extend(st.stop_text_block());
            }
        }

        // ── web_search_call lifecycle — defer, populated items ─
        "response.web_search_call.searching"
        | "response.web_search_call.completed"
        | "response.web_search_call.in_progress" => {}

        // ── response.completed / incomplete ─────────────────
        "response.completed" | "response.incomplete" => {
            let resp = chunk.get("response").unwrap_or(&chunk);
            let stop_reason = codex_stop_reason(resp);
            let mapped = map_codex_stop_reason(&stop_reason, st.has_tool_call);

            // Usage from response.usage
            let usage = resp.get("usage");
            let (input_tokens, output_tokens, cached_tokens) = extract_responses_usage(usage);

            let mut delta = json!({
                "type": "message_delta",
                "delta": {
                    "stop_reason": mapped,
                    "stop_sequence": null,
                },
                "usage": {
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                }
            });
            if cached_tokens > 0 {
                delta["usage"]["cache_read_input_tokens"] = json!(cached_tokens);
            }

            // stop_sequence passthrough
            if let Some(seq) = resp.get("stop_sequence").and_then(|v| v.as_str())
                && !seq.is_empty()
            {
                delta["delta"]["stop_sequence"] = json!(seq);
            }

            out.push(delta);
            out.push(json!({"type": "message_stop"}));
        }

        // ── output_item.added ────────────────────────────────
        "response.output_item.added" => {
            let item = chunk.get("item").unwrap_or(&chunk);
            let item_type = item.get("type").and_then(|v| v.as_str()).unwrap_or("");

            if item_type == "function_call" {
                out.extend(st.finalize_thinking_block());
                out.extend(st.stop_text_block());
                st.has_tool_call = true;
                st.has_received_arguments_delta = false;

                let call_id = item
                    .get("call_id")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                let name = item
                    .get("name")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                let output_index = chunk.get("output_index").and_then(|v| v.as_u64());
                let key = fn_call_key(output_index, &call_id);

                if name.is_empty() {
                    // Defer: name arrives later in output_item.done
                    st.pending_function_calls.insert(
                        key.clone(),
                        PendingFnCall {
                            call_id: call_id.clone(),
                            arguments: String::new(),
                            block_index: st.block_index,
                            has_received_arguments_delta: false,
                            start_emitted: false,
                        },
                    );
                    st.last_pending_function_call_key = key;
                    st.block_index += 1;
                } else {
                    // Name present now: emit immediately
                    st.pending_function_calls.remove(&key);
                    out.extend(st.append_fn_call_start(&call_id, &name));
                    out.extend(st.append_fn_call_arg_delta(""));
                }
            } else if item_type == "reasoning" {
                st.thinking_summary_seen = false;
                if let Some(sig) = item.get("encrypted_content").and_then(|v| v.as_str()) {
                    st.thinking_signature = sig.to_string();
                }
            }
            // web_search_call: defer until output_item.done
        }

        // ── output_item.done ────────────────────────────────
        "response.output_item.done" => {
            let item = chunk.get("item").unwrap_or(&chunk);
            let item_type = item.get("type").and_then(|v| v.as_str()).unwrap_or("");

            if item_type == "message" {
                if st.has_text_delta {
                    // Text already streamed via deltas, nothing to do
                } else {
                    // Fallback: emit full text from output_item.done
                    let content = item.get("content").and_then(|v| v.as_array());
                    let text = content
                        .map(|arr| {
                            arr.iter()
                                .filter(|p| {
                                    p.get("type").and_then(|v| v.as_str()) == Some("output_text")
                                })
                                .filter_map(|p| p.get("text").and_then(|v| v.as_str()))
                                .collect::<Vec<_>>()
                                .join("")
                        })
                        .unwrap_or_default();

                    if !text.is_empty() {
                        out.extend(st.finalize_thinking_block());
                        out.extend(st.start_text_block());

                        out.push(json!({
                            "type": "content_block_delta",
                            "index": st.block_index,
                            "delta": {"type": "text_delta", "text": text}
                        }));
                        out.extend(st.stop_text_block());
                        st.has_text_delta = true;
                    }
                }
            } else if item_type == "function_call" {
                let output_index = chunk.get("output_index").and_then(|v| v.as_u64());
                let call_id = item.get("call_id").and_then(|v| v.as_str()).unwrap_or("");
                let key = fn_call_key(output_index, call_id);

                // Clone pending data before any mutable borrow of st
                let pending_data = st.pending_function_calls.get(&key).map(|p| {
                    (
                        p.call_id.clone(),
                        p.arguments.clone(),
                        p.block_index,
                        p.start_emitted,
                    )
                });

                if let Some((ref p_call_id, ref p_args, block_idx, start_emitted)) = pending_data {
                    if !start_emitted {
                        let name = item
                            .get("name")
                            .and_then(|v| v.as_str())
                            .unwrap_or("")
                            .to_string();
                        if name.is_empty() {
                            // Still no name, skip
                        } else {
                            let cid = if p_call_id.is_empty() {
                                item.get("call_id").and_then(|v| v.as_str()).unwrap_or("")
                            } else {
                                p_call_id.as_str()
                            };

                            out.extend(st.append_fn_call_start(cid, &name));
                            let args = if !p_args.is_empty() {
                                p_args.clone()
                            } else {
                                item.get("arguments")
                                    .and_then(|v| v.as_str())
                                    .unwrap_or("")
                                    .to_string()
                            };
                            if !args.is_empty() {
                                out.push(json!({
                                    "type": "content_block_delta",
                                    "index": block_idx,
                                    "delta": {"type": "input_json_delta", "partial_json": args}
                                }));
                            }
                            out.push(json!({
                                "type": "content_block_stop",
                                "index": block_idx
                            }));

                            st.pending_function_calls.remove(&key);
                            if st.last_pending_function_call_key == key {
                                st.last_pending_function_call_key.clear();
                            }
                        }
                    }
                } else {
                    // Normal path: stop the function call block
                    out.extend(st.append_fn_call_stop());
                    st.block_index += 1;
                }
            } else if item_type == "reasoning" {
                if let Some(sig) = item.get("encrypted_content").and_then(|v| v.as_str())
                    && !sig.is_empty()
                {
                    st.thinking_signature = sig.to_string();
                }
                if st.thinking_summary_seen {
                    out.extend(st.finalize_thinking_block());
                } else {
                    out.extend(st.finalize_signature_only_thinking());
                }
                st.thinking_signature.clear();
                st.thinking_summary_seen = false;
            } else if item_type == "web_search_call" {
                out.extend(st.append_web_search_tool_result(item));
            }
        }

        // ── function_call_arguments.delta ────────────────────
        "response.function_call_arguments.delta" => {
            let delta = chunk.get("delta").and_then(|v| v.as_str()).unwrap_or("");
            let output_index = chunk.get("output_index").and_then(|v| v.as_u64());
            let key = if let Some(oi) = output_index {
                format!("output:{}", oi)
            } else if !st.last_pending_function_call_key.is_empty() {
                st.last_pending_function_call_key.clone()
            } else {
                "last".to_string()
            };

            let consumed = st
                .pending_function_calls
                .get_mut(&key)
                .map(|pending| {
                    if !pending.start_emitted {
                        pending.has_received_arguments_delta = true;
                        pending.arguments.push_str(delta);
                        true
                    } else {
                        false
                    }
                })
                .unwrap_or(false);

            if !consumed {
                st.has_received_arguments_delta = true;
                out.push(json!({
                    "type": "content_block_delta",
                    "index": st.block_index,
                    "delta": {"type": "input_json_delta", "partial_json": delta}
                }));
            }
        }

        // ── function_call_arguments.done ─────────────────────
        "response.function_call_arguments.done" => {
            let output_index = chunk.get("output_index").and_then(|v| v.as_u64());
            let key = fn_call_key(output_index, "");

            let had_pending = st
                .pending_function_calls
                .get_mut(&key)
                .map(|pending| {
                    if !pending.start_emitted {
                        if !pending.has_received_arguments_delta {
                            pending.arguments = chunk
                                .get("arguments")
                                .and_then(|v| v.as_str())
                                .unwrap_or("")
                                .to_string();
                        }
                        true // consumed by pending
                    } else {
                        false
                    }
                })
                .unwrap_or(false);

            if !had_pending
                && !st.has_received_arguments_delta
                && let Some(args) = chunk.get("arguments").and_then(|v| v.as_str())
                && !args.is_empty()
            {
                out.push(json!({
                    "type": "content_block_delta",
                    "index": st.block_index,
                    "delta": {"type": "input_json_delta", "partial_json": args}
                }));
            }
        }

        // ── ping — noop ─────────────────────────────────────
        "ping" => {}

        _ => {}
    }

    out
}

// ── Non-stream support ────────────────────────────────────────

/// Aggregate all SSE chunks into a single Claude response (non-stream).
pub fn transform_non_stream(
    _model: &str,
    original_request: &Value,
    _translated_request: &Value,
    response: Value,
    _param: Option<&mut Box<dyn std::any::Any>>,
) -> Value {
    // Build reverse tool name map
    let tool_name_rev = build_reverse_name_map(original_request);

    // Must be response.completed or response.incomplete
    let ty = response.get("type").and_then(|v| v.as_str()).unwrap_or("");
    if ty != "response.completed" && ty != "response.incomplete" {
        return json!({});
    }

    let resp = response.get("response").unwrap_or(&response);
    if resp.is_null() {
        return json!({});
    }

    let resp_id = resp.get("id").and_then(|v| v.as_str()).unwrap_or("");
    let model = resp.get("model").and_then(|v| v.as_str()).unwrap_or(_model);
    let (input_tokens, output_tokens, cached_tokens) = extract_responses_usage(resp.get("usage"));

    let mut out = json!({
        "id": resp_id,
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": [],
        "stop_reason": null,
        "stop_sequence": null,
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        }
    });
    if cached_tokens > 0 {
        out["usage"]["cache_read_input_tokens"] = json!(cached_tokens);
    }

    let mut has_tool_call = false;
    let mut web_search_seen: Vec<String> = Vec::new();

    if let Some(output) = resp.get("output").and_then(|v| v.as_array()) {
        for item in output {
            match item.get("type").and_then(|v| v.as_str()).unwrap_or("") {
                "reasoning" => {
                    let thinking_text = extract_reasoning_text(item);
                    let signature = item
                        .get("encrypted_content")
                        .and_then(|v| v.as_str())
                        .unwrap_or("");

                    if !thinking_text.is_empty() || !signature.is_empty() {
                        let mut block = json!({"type": "thinking", "thinking": thinking_text});
                        if !signature.is_empty() {
                            block["signature"] = json!(signature);
                        }
                        out["content"].as_array_mut().unwrap().push(block);
                    }
                }

                "message" => {
                    if let Some(content) = item.get("content").and_then(|v| v.as_array()) {
                        for part in content {
                            if part.get("type").and_then(|v| v.as_str()) == Some("output_text")
                                && let Some(text) = part.get("text").and_then(|v| v.as_str())
                                && !text.is_empty()
                            {
                                out["content"].as_array_mut().unwrap().push(json!({
                                    "type": "text",
                                    "text": text
                                }));
                            }
                        }
                    }
                }

                "web_search_call" => {
                    let id = item.get("id").and_then(|v| v.as_str()).unwrap_or("");
                    if id.is_empty() || web_search_seen.contains(&id.to_string()) {
                        // Skip duplicate/empty
                    } else {
                        let query = web_search_query(item);
                        let results = web_search_result_content(item);
                        if !query.is_empty() || !results.is_empty() {
                            out["content"].as_array_mut().unwrap().push(json!({
                                "type": "server_tool_use",
                                "id": id,
                                "name": "web_search",
                                "input": {"query": query},
                            }));

                            out["content"].as_array_mut().unwrap().push(json!({
                                "type": "web_search_tool_result",
                                "tool_use_id": id,
                                "content": results,
                            }));

                            web_search_seen.push(id.to_string());
                        }
                    }
                }

                "function_call" => {
                    has_tool_call = true;
                    let name = item.get("name").and_then(|v| v.as_str()).unwrap_or("");
                    let resolved = tool_name_rev
                        .get(name)
                        .cloned()
                        .unwrap_or_else(|| name.to_string());
                    let call_id = item.get("call_id").and_then(|v| v.as_str()).unwrap_or("");
                    let mut input = json!({});
                    if let Some(args_str) = item.get("arguments").and_then(|v| v.as_str())
                        && let Ok(parsed) = serde_json::from_str::<Value>(args_str)
                        && parsed.is_object()
                    {
                        input = parsed;
                    }

                    out["content"].as_array_mut().unwrap().push(json!({
                        "type": "tool_use",
                        "id": call_id,
                        "name": resolved,
                        "input": input,
                    }));
                }

                _ => {}
            }
        }
    }

    let stop_reason = codex_stop_reason(resp);
    out["stop_reason"] = json!(map_codex_stop_reason(&stop_reason, has_tool_call));

    // stop_sequence
    if let Some(seq) = resp.get("stop_sequence").and_then(|v| v.as_str())
        && !seq.is_empty()
    {
        out["stop_sequence"] = json!(seq);
    }

    out
}

// ── Helper functions ──────────────────────────────────────────

fn codex_stop_reason(resp: &Value) -> String {
    if let Some(reason) = resp.get("stop_reason").and_then(|v| v.as_str()) {
        if reason == "stop"
            && let Some(seq) = resp.get("stop_sequence").and_then(|v| v.as_str())
            && !seq.is_empty()
        {
            return "stop_sequence".to_string();
        }
        return reason.to_string();
    }
    if let Some(reason) = resp
        .pointer("/incomplete_details/reason")
        .and_then(|v| v.as_str())
        && !reason.is_empty()
    {
        return reason.to_string();
    }
    if let Some(seq) = resp.get("stop_sequence").and_then(|v| v.as_str())
        && !seq.is_empty()
    {
        return "stop_sequence".to_string();
    }
    String::new()
}

fn map_codex_stop_reason(reason: &str, has_tool_call: bool) -> &'static str {
    if has_tool_call {
        return "tool_use";
    }
    match reason {
        "" | "stop" | "completed" => "end_turn",
        "max_tokens" | "max_output_tokens" => "max_tokens",
        "tool_use" | "tool_calls" | "function_call" => "tool_use",
        "end_turn" => "end_turn",
        "stop_sequence" => "stop_sequence",
        "pause_turn" => "pause_turn",
        "refusal" => "refusal",
        "model_context_window_exceeded" => "model_context_window_exceeded",
        "content_filter" => "refusal",
        _ => "end_turn",
    }
}

fn extract_responses_usage(usage: Option<&Value>) -> (u64, u64, u64) {
    let usage = match usage {
        Some(u) => u,
        None => return (0, 0, 0),
    };
    if usage.is_null() {
        return (0, 0, 0);
    }

    let input = usage
        .get("input_tokens")
        .and_then(|v| v.as_u64())
        .unwrap_or(0);
    let output = usage
        .get("output_tokens")
        .and_then(|v| v.as_u64())
        .unwrap_or(0);
    let cached = usage
        .pointer("/input_tokens_details/cached_tokens")
        .and_then(|v| v.as_u64())
        .unwrap_or(0);

    let adjusted_input = if cached > 0 && input >= cached {
        input - cached
    } else {
        input
    };
    (adjusted_input, output, cached)
}

fn codex_error_to_claude_error(chunk: &Value) -> Value {
    let err = chunk.get("error").unwrap_or(chunk);
    let err_type = err
        .get("type")
        .and_then(|v| v.as_str())
        .unwrap_or("api_error");
    let code = err.get("code").and_then(|v| v.as_str()).unwrap_or("");
    let message = err.get("message").and_then(|v| v.as_str()).unwrap_or("");

    let final_msg = if !message.is_empty() { message } else { code };
    let final_type = if code == "cyber_policy" || err_type == "invalid_request" {
        "invalid_request_error"
    } else {
        err_type
    };

    json!({
        "type": "error",
        "error": {"type": final_type, "message": final_msg}
    })
}

fn extract_reasoning_text(item: &Value) -> String {
    let mut buf = String::new();

    if let Some(summary) = item.get("summary") {
        if let Some(arr) = summary.as_array() {
            for part in arr {
                if let Some(t) = part.get("text").and_then(|v| v.as_str()) {
                    buf.push_str(t);
                } else if let Some(s) = part.as_str() {
                    buf.push_str(s);
                }
            }
        } else if let Some(s) = summary.as_str() {
            buf.push_str(s);
        }
    }

    if buf.is_empty()
        && let Some(content) = item.get("content")
    {
        if let Some(arr) = content.as_array() {
            for part in arr {
                if let Some(t) = part.get("text").and_then(|v| v.as_str()) {
                    buf.push_str(t);
                } else if let Some(s) = part.as_str() {
                    buf.push_str(s);
                }
            }
        } else if let Some(s) = content.as_str() {
            buf.push_str(s);
        }
    }

    buf
}

fn build_reverse_name_map(original_request: &Value) -> HashMap<String, String> {
    let mut rev = HashMap::new();
    if let Some(tools) = original_request.get("tools").and_then(|v| v.as_array()) {
        let mut names: Vec<String> = Vec::new();
        for t in tools {
            if let Some(n) = t.get("name").and_then(|v| v.as_str())
                && !n.is_empty()
            {
                names.push(n.to_string());
            }
        }
        // Build the short name map (same algorithm as Go)
        let short_map = build_short_name_map(&names);
        for (orig, short) in &short_map {
            rev.insert(short.clone(), orig.clone());
        }
    }
    rev
}

fn build_short_name_map(names: &[String]) -> HashMap<String, String> {
    const LIMIT: usize = 64;
    let mut used: std::collections::HashSet<String> = std::collections::HashSet::new();
    let mut m = HashMap::new();

    let base_candidate = |n: &str| -> String {
        if n.len() <= LIMIT {
            return n.to_string();
        }
        if let Some(rest) = n.strip_prefix("mcp__")
            && let Some(last) = rest.rfind("__")
        {
            let cand = format!("mcp__{}", &rest[last + 2..]);
            if cand.len() > LIMIT {
                return cand[..LIMIT].to_string();
            }
            return cand;
        }
        n[..LIMIT].to_string()
    };

    let make_unique = |cand: &str, used: &std::collections::HashSet<String>| -> String {
        if !used.contains(cand) {
            return cand.to_string();
        }
        let mut i = 1usize;
        loop {
            let suffix = format!("_{}", i);
            let allowed = LIMIT.saturating_sub(suffix.len());
            let mut tmp = cand[..allowed.min(cand.len())].to_string();
            tmp.push_str(&suffix);
            if !used.contains(&tmp) {
                return tmp;
            }
            i += 1;
        }
    };

    for name in names {
        let cand = base_candidate(name);
        let uniq = make_unique(&cand, &used);
        used.insert(uniq.clone());
        m.insert(name.clone(), uniq);
    }
    m
}

// ── Tests ─────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_response_created_to_message_start() {
        let chunk = json!({
            "type": "response.created",
            "response": {"id": "resp_123", "model": "gpt-4"}
        });
        let state = CodexToClaudeState::default();
        let mut param: Box<dyn std::any::Any> = Box::new(state.clone());
        let results = transform_stream("gpt-4", &json!({}), &json!({}), chunk, Some(&mut param));
        let _new_state = param.downcast_ref::<CodexToClaudeState>().unwrap();

        assert!(!results.is_empty());
        assert_eq!(results[0]["type"], "message_start");
        assert_eq!(results[0]["message"]["id"], "resp_123");
    }

    #[test]
    fn test_output_text_delta() {
        let state = CodexToClaudeState::default();
        let mut param: Box<dyn std::any::Any> = Box::new(state);

        let chunk1 = json!({"type": "response.output_text.delta", "delta": "Hello"});
        let r1 = transform_stream("gpt-4", &json!({}), &json!({}), chunk1, Some(&mut param));
        assert!(!r1.is_empty());

        let chunk2 = json!({"type": "response.output_text.delta", "delta": " World"});
        let r2 = transform_stream("gpt-4", &json!({}), &json!({}), chunk2, Some(&mut param));
        assert!(!r2.is_empty());
    }

    #[test]
    fn test_reasoning_delta() {
        let state = CodexToClaudeState::default();
        let mut param: Box<dyn std::any::Any> = Box::new(state);

        let chunk1 = json!({"type": "response.reasoning_summary_part.added"});
        let r1 = transform_stream("gpt-4", &json!({}), &json!({}), chunk1, Some(&mut param));
        assert!(r1.iter().any(|v| v["type"] == "content_block_start"));

        let chunk2 =
            json!({"type": "response.reasoning_summary_text.delta", "delta": "thinking..."});
        let r2 = transform_stream("gpt-4", &json!({}), &json!({}), chunk2, Some(&mut param));
        assert!(r2.iter().any(|v| v["type"] == "content_block_delta"));
    }

    #[test]
    fn test_function_call_name_deferred() {
        let state = CodexToClaudeState::default();
        let mut param: Box<dyn std::any::Any> = Box::new(state);

        // output_item.added with no name (deferred)
        let chunk1 = json!({
            "type": "response.output_item.added",
            "item": {"type": "function_call", "call_id": "call_1", "name": ""}
        });
        let r1 = transform_stream("gpt-4", &json!({}), &json!({}), chunk1, Some(&mut param));
        assert!(r1.is_empty()); // should defer

        // arguments delta while pending
        let chunk2 = json!({
            "type": "response.function_call_arguments.delta",
            "delta": r#"{"city":"Paris"}"#,
        });
        let r2 = transform_stream("gpt-4", &json!({}), &json!({}), chunk2, Some(&mut param));
        assert!(r2.is_empty()); // still deferred

        // output_item.done with name
        let chunk3 = json!({
            "type": "response.output_item.done",
            "item": {"type": "function_call", "call_id": "call_1", "name": "get_weather", "arguments": r#"{"city":"Paris"}"#}
        });
        let r3 = transform_stream("gpt-4", &json!({}), &json!({}), chunk3, Some(&mut param));
        assert!(r3.iter().any(|v| v["type"] == "content_block_start"));
    }

    #[test]
    fn test_response_completed() {
        let state = CodexToClaudeState::default();
        let mut param: Box<dyn std::any::Any> = Box::new(state);

        let chunk = json!({
            "type": "response.completed",
            "response": {
                "status": "completed",
                "usage": {"input_tokens": 10, "output_tokens": 5}
            }
        });
        let r = transform_stream("gpt-4", &json!({}), &json!({}), chunk, Some(&mut param));
        assert!(r.iter().any(|v| v["type"] == "message_delta"));
        assert!(r.iter().any(|v| v["type"] == "message_stop"));
    }

    #[test]
    fn test_web_search() {
        let state = CodexToClaudeState::default();
        let mut param: Box<dyn std::any::Any> = Box::new(state);

        let chunk = json!({
            "type": "response.output_item.done",
            "item": {
                "type": "web_search_call",
                "id": "ws_1",
                "query": "latest news",
                "results": [{"url": "https://example.com", "title": "Example"}]
            }
        });
        let r = transform_stream("gpt-4", &json!({}), &json!({}), chunk, Some(&mut param));
        let types: Vec<&str> = r.iter().filter_map(|v| v["type"].as_str()).collect();
        assert!(types.contains(&"content_block_start"));
    }

    #[test]
    fn test_error_event() {
        let chunk = json!({
            "type": "error",
            "error": {"type": "api_error", "message": "Rate limit exceeded"}
        });
        let state = CodexToClaudeState::default();
        let mut param: Box<dyn std::any::Any> = Box::new(state);
        let r = transform_stream("gpt-4", &json!({}), &json!({}), chunk, Some(&mut param));
        assert_eq!(r[0]["type"], "error");
        assert!(
            r[0]["error"]["message"]
                .as_str()
                .unwrap_or("")
                .contains("Rate limit")
        );
    }
}
