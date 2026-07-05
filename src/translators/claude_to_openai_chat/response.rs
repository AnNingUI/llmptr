//! OpenAI ChatCompletions response → Claude Messages response translation.
//!
//! Translates streaming SSE chunks and non-streaming responses from OpenAI
//! format back to Claude Messages format (SSE events for streaming).

use serde_json::{Value, json};
use std::collections::{BTreeMap, HashMap};
use translator_infra::util;

/// State accumulated across streaming chunks.
#[derive(Debug, Clone)]
pub struct StreamState {
    pub message_id: String,
    pub model: String,
    pub created_at: i64,
    pub saw_tool_call: bool,
    pub finish_reason: String,
    pub text_content_block_index: i64,
    pub thinking_content_block_index: i64,
    pub next_content_block_index: i64,
    pub text_content_block_started: bool,
    pub thinking_content_block_started: bool,
    pub content_blocks_stopped: bool,
    pub message_started: bool,
    pub message_delta_sent: bool,
    pub message_stop_sent: bool,
    pub tool_call_accumulators: BTreeMap<usize, ToolCallAcc>,
    /// Input tokens accumulated across message_delta events.
    pub input_tokens: u64,
    /// Output tokens accumulated across message_delta events.
    pub output_tokens: u64,
    /// Cached read tokens.
    pub cache_read_input_tokens: u64,
    /// Tool name mapping from original request (simplified → original).
    pub tool_name_map: HashMap<String, String>,
}

#[derive(Debug, Clone)]
pub struct ToolCallAcc {
    pub id: String,
    pub name: String,
    pub arguments: String,
    pub start_emitted: bool,
    pub block_index: i64,
}

impl Default for StreamState {
    fn default() -> Self {
        Self {
            message_id: String::new(),
            model: String::new(),
            created_at: 0,
            saw_tool_call: false,
            finish_reason: String::new(),
            text_content_block_index: -1,
            thinking_content_block_index: -1,
            next_content_block_index: 0,
            text_content_block_started: false,
            thinking_content_block_started: false,
            content_blocks_stopped: false,
            message_started: false,
            message_delta_sent: false,
            message_stop_sent: false,
            tool_call_accumulators: BTreeMap::new(),
            input_tokens: 0,
            output_tokens: 0,
            cache_read_input_tokens: 0,
            tool_name_map: HashMap::new(),
        }
    }
}

impl StreamState {
    /// Merge usage from a chunk's `usage` field (typically from message_delta events).
    /// Uses last-write-wins for input/output tokens (they represent total not delta),
    /// matching Go's `claudeUsageTokens.Merge()` behavior.
    fn merge_usage(&mut self, usage: &Value) {
        if usage.is_null() || !usage.is_object() {
            return;
        }
        if let Some(v) = usage.get("input_tokens").and_then(|v| v.as_u64()) {
            self.input_tokens = v;
        }
        if let Some(v) = usage.get("output_tokens").and_then(|v| v.as_u64()) {
            self.output_tokens = v;
        }
        if let Some(v) = usage
            .pointer("/input_tokens_details/cached_tokens")
            .or_else(|| usage.pointer("/prompt_tokens_details/cached_tokens"))
            .and_then(|v| v.as_u64())
        {
            self.cache_read_input_tokens = v;
        }
    }
}

// ── streaming response transform ─────────────────────────────

/// Transform an OpenAI streaming SSE chunk to Claude SSE events.
pub fn transform_stream(
    _model: &str,
    original_request: &Value,
    _translated_request: &Value,
    chunk: Value,
    param: Option<&mut Box<dyn std::any::Any>>,
) -> Vec<Value> {
    // Extract/recover param state
    let state = if let Some(p) = param {
        if !p.is::<StreamState>() {
            *p = Box::<StreamState>::default();
        }
        p.downcast_mut::<StreamState>().unwrap()
    } else {
        return vec![chunk];
    };

    // Initialize tool name map from original request
    if state.tool_name_map.is_empty() && original_request.is_object() {
        state.tool_name_map = util::tool_name_map_from_request(original_request);
    }

    // Check for [DONE] marker
    if chunk.as_str() == Some("[DONE]") {
        return into_sse_events(handle_done(state));
    }

    // Body payload
    let body = if let Some(obj) = chunk.as_object() {
        obj
    } else {
        return vec![chunk];
    };

    let mut results: Vec<Value> = Vec::new();

    // Initialize state from first chunk
    if state.message_id.is_empty() {
        state.message_id = body
            .get("id")
            .and_then(|v| v.as_str())
            .unwrap_or("msg_unknown")
            .to_string();
    }
    if state.model.is_empty() {
        state.model = body
            .get("model")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
    }
    if state.created_at == 0 {
        state.created_at = body.get("created").and_then(|v| v.as_i64()).unwrap_or(0);
    }

    let choices = match body.get("choices").and_then(|v| v.as_array()) {
        Some(c) if !c.is_empty() => &c[0],
        _ => {
            if !state.finish_reason.is_empty()
                && !state.message_delta_sent
                && let Some(usage) = body.get("usage")
                && !usage.is_null()
            {
                state.merge_usage(usage);
                results.push(make_message_delta(state, Some(usage)));
                state.message_delta_sent = true;
                emit_message_stop(state, &mut results);
            }
            return into_sse_events(results);
        }
    };

    let delta = match choices.get("delta") {
        Some(d) => d,
        None => return results,
    };

    // ── message_start (first chunk only) ──────────────
    if !state.message_started {
        results.push(make_message_start(state));
        state.message_started = true;
    }

    // ── thinking / reasoning_content ───────────────────
    if let Some(reasoning) = delta.get("reasoning_content").and_then(|v| v.as_str())
        && !reasoning.is_empty()
    {
        stop_text(state, &mut results);
        if !state.thinking_content_block_started {
            if state.thinking_content_block_index < 0 {
                state.thinking_content_block_index = state.next_content_block_index;
                state.next_content_block_index += 1;
            }
            results.push(make_thinking_block_start(state));
            state.thinking_content_block_started = true;
        }

        results.push(make_thinking_delta(state, reasoning));
    }

    // ── text content ──────────────────────────────────
    if let Some(text) = delta.get("content").and_then(|v| v.as_str())
        && !text.is_empty()
    {
        if !state.text_content_block_started {
            stop_thinking(state, &mut results);
            if state.text_content_block_index < 0 {
                state.text_content_block_index = state.next_content_block_index;
                state.next_content_block_index += 1;
            }
            results.push(make_text_block_start(state));
            state.text_content_block_started = true;
        }

        results.push(make_text_delta(state, text));
    }

    // ── tool_calls ────────────────────────────────────
    if let Some(tcs) = delta.get("tool_calls").and_then(|v| v.as_array()) {
        for tc in tcs {
            let idx = tc.get("index").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
            let acc = state
                .tool_call_accumulators
                .entry(idx)
                .or_insert_with(|| ToolCallAcc {
                    id: String::new(),
                    name: String::new(),
                    arguments: String::new(),
                    start_emitted: false,
                    block_index: -1,
                });

            if let Some(id) = tc.get("id").and_then(|v| v.as_str())
                && !id.is_empty()
            {
                acc.id = id.to_string();
            }

            if let Some(func) = tc.get("function") {
                if !acc.start_emitted
                    && let Some(name) = func.get("name").and_then(|v| v.as_str())
                    && !name.is_empty()
                {
                    acc.name = util::map_tool_name(&state.tool_name_map, name);
                }
                if let Some(args) = func.get("arguments").and_then(|v| v.as_str()) {
                    acc.arguments.push_str(args);
                }
            }

            if !acc.start_emitted
                && !acc.name.is_empty()
                && !acc.id.is_empty()
                && !state.content_blocks_stopped
            {
                emit_tool_use_start(state, idx, &mut results);
            }
        }
    }

    // ── finish_reason (end of stream signals) ──────────
    if let Some(reason) = choices.get("finish_reason").and_then(|v| v.as_str())
        && !reason.is_empty()
    {
        state.finish_reason = reason.to_string();
        if state.saw_tool_call {
            state.finish_reason = "tool_calls".to_string();
        }

        // Stop all content blocks
        stop_thinking(state, &mut results);
        stop_text(state, &mut results);

        if !state.content_blocks_stopped {
            stop_all_tool_calls(state, &mut results);
        }

        // Merge usage from any chunk that carries usage, even mid-stream
        if let Some(usage) = body.get("usage")
            && !usage.is_null()
        {
            state.merge_usage(usage);
        }

        // message_delta if usage present in same chunk
        if let Some(usage) = body.get("usage")
            && !usage.is_null()
        {
            results.push(make_message_delta(state, Some(usage)));
            state.message_delta_sent = true;
            emit_message_stop(state, &mut results);
        }
    }

    into_sse_events(results)
}

// ── non-streaming response transform ─────────────────────────

/// Transform a non-streaming OpenAI Chat response to Claude format.
pub fn transform_non_stream(
    _model: &str,
    original_request: &Value,
    _translated_request: &Value,
    response: Value,
    _param: Option<&mut Box<dyn std::any::Any>>,
) -> Value {
    let tool_name_map = util::tool_name_map_from_request(original_request);
    let mut out = json!({
        "id": response.get("id").and_then(|v| v.as_str()).unwrap_or("msg_unknown"),
        "type": "message",
        "role": "assistant",
        "model": response.get("model").and_then(|v| v.as_str()).unwrap_or(""),
        "content": [],
        "stop_reason": null,
        "stop_sequence": null,
        "usage": {
            "input_tokens": 0,
            "output_tokens": 0,
        }
    });

    let choices = match response.get("choices").and_then(|v| v.as_array()) {
        Some(c) if !c.is_empty() => &c[0],
        _ => return out,
    };

    let msg = match choices.get("message") {
        Some(m) => m,
        None => return out,
    };

    // reasoning_content → thinking blocks
    // content
    if let Some(content) = msg.get("content") {
        match content {
            Value::String(s) if !s.is_empty() => {
                out["content"]
                    .as_array_mut()
                    .unwrap()
                    .push(json!({"type": "text", "text": s}));
            }
            Value::Array(parts) => {
                for part in parts {
                    let ptype = part.get("type").and_then(|v| v.as_str()).unwrap_or("");
                    match ptype {
                        "text" => {
                            if let Some(t) = part.get("text").and_then(|v| v.as_str()) {
                                out["content"]
                                    .as_array_mut()
                                    .unwrap()
                                    .push(json!({"type": "text", "text": t}));
                            }
                        }
                        "image_url" => {
                            // Claude doesn't have image_url in response, convert differently
                            // This is unusual but possible with some providers
                        }
                        _ => {}
                    }
                }
            }
            _ => {}
        }
    }

    // tool_calls → tool_use blocks
    if let Some(reasoning) = msg.get("reasoning_content").and_then(|v| v.as_str()) {
        if !reasoning.is_empty() {
            out["content"]
                .as_array_mut()
                .unwrap()
                .push(json!({"type": "thinking", "thinking": reasoning}));
        }
    } else if let Some(reasoning_arr) = msg.get("reasoning_content").and_then(|v| v.as_array()) {
        for item in reasoning_arr {
            if let Some(text) = item.get("text").and_then(|v| v.as_str())
                && !text.is_empty()
            {
                out["content"]
                    .as_array_mut()
                    .unwrap()
                    .push(json!({"type": "thinking", "thinking": text}));
                break;
            }
        }
    }

    let has_tool_calls = if let Some(tcs) = msg.get("tool_calls").and_then(|v| v.as_array()) {
        for tc in tcs {
            let name = util::map_tool_name(
                &tool_name_map,
                tc.get("function")
                    .and_then(|f| f.get("name"))
                    .and_then(|v| v.as_str())
                    .unwrap_or(""),
            );
            let id =
                util::sanitize_claude_tool_id(tc.get("id").and_then(|v| v.as_str()).unwrap_or(""));
            let args = tc.get("function").and_then(|f| f.get("arguments"));

            let input = match args {
                Some(Value::String(s)) => {
                    let fixed = util::fix_json(s);
                    serde_json::from_str(&fixed).unwrap_or(json!({}))
                }
                Some(other) => other.clone(),
                None => json!({}),
            };

            out["content"].as_array_mut().unwrap().push(json!({
                "type": "tool_use",
                "id": id,
                "name": name,
                "input": input,
            }));
        }
        true
    } else {
        false
    };

    // stop_reason
    let finish_reason = choices
        .get("finish_reason")
        .and_then(|v| v.as_str())
        .unwrap_or("");
    out["stop_reason"] = json!(map_finish_reason(finish_reason, has_tool_calls));

    // usage
    if let Some(usage) = response.get("usage") {
        let input_tokens = usage
            .get("prompt_tokens")
            .and_then(|v| v.as_u64())
            .unwrap_or(0);
        let output_tokens = usage
            .get("completion_tokens")
            .and_then(|v| v.as_u64())
            .unwrap_or(0);
        let cached = usage
            .pointer("/prompt_tokens_details/cached_tokens")
            .and_then(|v| v.as_u64())
            .unwrap_or(0);

        let adjusted_input = if cached > 0 && input_tokens >= cached {
            input_tokens - cached
        } else {
            input_tokens
        };

        out["usage"] = json!({
            "input_tokens": adjusted_input,
            "output_tokens": output_tokens,
        });
        if cached > 0 {
            out["usage"]["cache_read_input_tokens"] = json!(cached);
        }
    }

    out
}

// ── helpers ──────────────────────────────────────────────────

fn make_message_start(state: &StreamState) -> Value {
    json!({
        "type": "message_start",
        "message": {
            "id": state.message_id,
            "type": "message",
            "role": "assistant",
            "model": state.model,
            "content": [],
            "stop_reason": null,
            "stop_sequence": null,
            "usage": {
                "input_tokens": state.input_tokens,
                "output_tokens": state.output_tokens,
            },
        }
    })
}

fn make_thinking_block_start(state: &StreamState) -> Value {
    json!({
        "type": "content_block_start",
        "index": state.thinking_content_block_index,
        "content_block": {"type": "thinking", "thinking": ""},
    })
}

fn make_thinking_delta(state: &StreamState, text: &str) -> Value {
    json!({
        "type": "content_block_delta",
        "index": state.thinking_content_block_index,
        "delta": {"type": "thinking_delta", "thinking": text},
    })
}

fn make_text_block_start(state: &StreamState) -> Value {
    json!({
        "type": "content_block_start",
        "index": state.text_content_block_index,
        "content_block": {"type": "text", "text": ""},
    })
}

fn make_text_delta(state: &StreamState, text: &str) -> Value {
    json!({
        "type": "content_block_delta",
        "index": state.text_content_block_index,
        "delta": {"type": "text_delta", "text": text},
    })
}

fn make_message_delta(state: &StreamState, usage: Option<&Value>) -> Value {
    let reason = if state.saw_tool_call {
        "tool_use"
    } else {
        map_finish_reason(&state.finish_reason, false)
    };

    let (input_tokens, output_tokens, cached) = if let Some(u) = usage {
        let it = u
            .get("prompt_tokens")
            .and_then(|v| v.as_u64())
            .unwrap_or(state.input_tokens);
        let ot = u
            .get("completion_tokens")
            .and_then(|v| v.as_u64())
            .unwrap_or(state.output_tokens);
        let ca = u
            .pointer("/prompt_tokens_details/cached_tokens")
            .and_then(|v| v.as_u64())
            .unwrap_or(state.cache_read_input_tokens);
        (it, ot, ca)
    } else {
        (
            state.input_tokens,
            state.output_tokens,
            state.cache_read_input_tokens,
        )
    };

    let adjusted = if cached > 0 && input_tokens >= cached {
        input_tokens - cached
    } else {
        input_tokens
    };

    let mut delta = json!({
        "type": "message_delta",
        "delta": {
            "stop_reason": reason,
            "stop_sequence": null,
        },
        "usage": {
            "input_tokens": adjusted,
            "output_tokens": output_tokens,
        }
    });

    if cached > 0 {
        delta["usage"]["cache_read_input_tokens"] = json!(cached);
    }

    delta
}

fn emit_message_stop(state: &mut StreamState, results: &mut Vec<Value>) {
    if state.message_stop_sent {
        return;
    }
    results.push(json!({"type": "message_stop"}));
    state.message_stop_sent = true;
}

fn into_sse_events(events: Vec<Value>) -> Vec<Value> {
    events.into_iter().map(to_sse_event).collect()
}

fn to_sse_event(event: Value) -> Value {
    let event_type = event
        .get("type")
        .and_then(|v| v.as_str())
        .unwrap_or("message");
    let payload = sse_payload(&event);
    Value::String(format!("event: {event_type}\ndata: {payload}\n\n"))
}

fn sse_payload(event: &Value) -> String {
    match event.get("type").and_then(|v| v.as_str()).unwrap_or("") {
        "message_start" => {
            let message = event.get("message").unwrap_or(&Value::Null);
            let usage = message.get("usage").unwrap_or(&Value::Null);
            format!(
                "{{\"type\":\"message_start\",\"message\":{{\"id\":{},\"type\":\"message\",\"role\":\"assistant\",\"model\":{},\"content\":[],\"stop_reason\":null,\"stop_sequence\":null,\"usage\":{{\"input_tokens\":{},\"output_tokens\":{}}}}}}}",
                json_string(message.get("id").and_then(|v| v.as_str()).unwrap_or("")),
                json_string(message.get("model").and_then(|v| v.as_str()).unwrap_or("")),
                usage
                    .get("input_tokens")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(0),
                usage
                    .get("output_tokens")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(0),
            )
        }
        "content_block_start" => {
            let index = event.get("index").and_then(|v| v.as_i64()).unwrap_or(0);
            let block = event.get("content_block").unwrap_or(&Value::Null);
            match block.get("type").and_then(|v| v.as_str()).unwrap_or("") {
                "thinking" => format!(
                    "{{\"type\":\"content_block_start\",\"index\":{},\"content_block\":{{\"type\":\"thinking\",\"thinking\":\"\"}}}}",
                    index
                ),
                "text" => format!(
                    "{{\"type\":\"content_block_start\",\"index\":{},\"content_block\":{{\"type\":\"text\",\"text\":\"\"}}}}",
                    index
                ),
                "tool_use" => format!(
                    "{{\"type\":\"content_block_start\",\"index\":{},\"content_block\":{{\"type\":\"tool_use\",\"id\":{},\"name\":{},\"input\":{{}}}}}}",
                    index,
                    json_string(block.get("id").and_then(|v| v.as_str()).unwrap_or("")),
                    json_string(block.get("name").and_then(|v| v.as_str()).unwrap_or("")),
                ),
                _ => serde_json::to_string(event).unwrap_or_else(|_| "{}".to_string()),
            }
        }
        "content_block_delta" => {
            let index = event.get("index").and_then(|v| v.as_i64()).unwrap_or(0);
            let delta = event.get("delta").unwrap_or(&Value::Null);
            match delta.get("type").and_then(|v| v.as_str()).unwrap_or("") {
                "thinking_delta" => format!(
                    "{{\"type\":\"content_block_delta\",\"index\":{},\"delta\":{{\"type\":\"thinking_delta\",\"thinking\":{}}}}}",
                    index,
                    json_string(delta.get("thinking").and_then(|v| v.as_str()).unwrap_or("")),
                ),
                "text_delta" => format!(
                    "{{\"type\":\"content_block_delta\",\"index\":{},\"delta\":{{\"type\":\"text_delta\",\"text\":{}}}}}",
                    index,
                    json_string(delta.get("text").and_then(|v| v.as_str()).unwrap_or("")),
                ),
                "input_json_delta" => format!(
                    "{{\"type\":\"content_block_delta\",\"index\":{},\"delta\":{{\"type\":\"input_json_delta\",\"partial_json\":{}}}}}",
                    index,
                    json_string(
                        delta
                            .get("partial_json")
                            .and_then(|v| v.as_str())
                            .unwrap_or("")
                    ),
                ),
                _ => serde_json::to_string(event).unwrap_or_else(|_| "{}".to_string()),
            }
        }
        "content_block_stop" => format!(
            "{{\"type\":\"content_block_stop\",\"index\":{}}}",
            event.get("index").and_then(|v| v.as_i64()).unwrap_or(0)
        ),
        "message_delta" => {
            let delta = event.get("delta").unwrap_or(&Value::Null);
            let usage = event.get("usage").unwrap_or(&Value::Null);
            let mut payload = format!(
                "{{\"type\":\"message_delta\",\"delta\":{{\"stop_reason\":{},\"stop_sequence\":null}},\"usage\":{{\"input_tokens\":{},\"output_tokens\":{}",
                json_string(
                    delta
                        .get("stop_reason")
                        .and_then(|v| v.as_str())
                        .unwrap_or("end_turn")
                ),
                usage
                    .get("input_tokens")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(0),
                usage
                    .get("output_tokens")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(0),
            );
            if let Some(cached) = usage
                .get("cache_read_input_tokens")
                .and_then(|v| v.as_u64())
            {
                payload.push_str(&format!(",\"cache_read_input_tokens\":{}", cached));
            }
            payload.push_str("}}");
            payload
        }
        "message_stop" => "{\"type\":\"message_stop\"}".to_string(),
        _ => serde_json::to_string(event).unwrap_or_else(|_| "{}".to_string()),
    }
}

fn json_string(value: &str) -> String {
    serde_json::to_string(value).unwrap_or_else(|_| "\"\"".to_string())
}

fn emit_tool_use_start(state: &mut StreamState, idx: usize, results: &mut Vec<Value>) {
    // Stop other content blocks BEFORE accessing tool_call_accumulators
    stop_thinking(state, results);
    stop_text(state, results);

    let name;
    let id;
    let block_index;
    {
        let acc = state.tool_call_accumulators.get_mut(&idx).unwrap();

        if acc.block_index < 0 {
            acc.block_index = state.next_content_block_index;
            state.next_content_block_index += 1;
        }

        name = acc.name.clone();
        id = acc.id.clone();
        block_index = acc.block_index;
        acc.start_emitted = true;
    }

    // Sanitize tool call ID
    let safe_id = util::sanitize_claude_tool_id(&id);

    results.push(json!({
        "type": "content_block_start",
        "index": block_index,
        "content_block": {
            "type": "tool_use",
            "id": safe_id,
            "name": name,
            "input": {},
        }
    }));
    state.saw_tool_call = true;
}

fn stop_thinking(state: &mut StreamState, results: &mut Vec<Value>) {
    if !state.thinking_content_block_started {
        return;
    }
    results.push(json!({
        "type": "content_block_stop",
        "index": state.thinking_content_block_index,
    }));
    state.thinking_content_block_started = false;
    state.thinking_content_block_index = -1;
}

fn stop_text(state: &mut StreamState, results: &mut Vec<Value>) {
    if !state.text_content_block_started {
        return;
    }
    results.push(json!({
        "type": "content_block_stop",
        "index": state.text_content_block_index,
    }));
    state.text_content_block_started = false;
    state.text_content_block_index = -1;
}

fn stop_all_tool_calls(state: &mut StreamState, results: &mut Vec<Value>) {
    for acc in state.tool_call_accumulators.values() {
        if !acc.start_emitted {
            continue;
        }
        if !acc.arguments.is_empty() {
            let fixed = util::fix_json(&acc.arguments);
            results.push(json!({
                "type": "content_block_delta",
                "index": acc.block_index,
                "delta": {
                    "type": "input_json_delta",
                    "partial_json": fixed,
                }
            }));
        }
        results.push(json!({
            "type": "content_block_stop",
            "index": acc.block_index,
        }));
    }
    state.content_blocks_stopped = true;
}

fn handle_done(state: &mut StreamState) -> Vec<Value> {
    let mut results = Vec::new();

    // Stop open content blocks before message_stop (matches Go behavior)
    if state.thinking_content_block_started {
        results.push(json!({
            "type": "content_block_stop",
            "index": state.thinking_content_block_index,
        }));
        state.thinking_content_block_started = false;
        state.thinking_content_block_index = -1;
    }
    if state.text_content_block_started {
        results.push(json!({
            "type": "content_block_stop",
            "index": state.text_content_block_index,
        }));
        state.text_content_block_started = false;
        state.text_content_block_index = -1;
    }
    if !state.content_blocks_stopped {
        for acc in state.tool_call_accumulators.values() {
            if !acc.start_emitted {
                continue;
            }
            if !acc.arguments.is_empty() {
                results.push(json!({
                    "type": "content_block_delta",
                    "index": acc.block_index,
                    "delta": {
                        "type": "input_json_delta",
                        "partial_json": util::fix_json(&acc.arguments),
                    }
                }));
            }
            results.push(json!({
                "type": "content_block_stop",
                "index": acc.block_index,
            }));
        }
        state.content_blocks_stopped = true;
    }

    // message_delta if not yet sent
    if !state.finish_reason.is_empty() && !state.message_delta_sent {
        results.push(make_message_delta(state, None));
        state.message_delta_sent = true;
    }

    // message_stop
    if !state.message_stop_sent {
        state.message_stop_sent = true;
        results.push(json!({"type": "message_stop"}));
    }

    results
}

fn map_finish_reason(openai_reason: &str, saw_tool_call: bool) -> &'static str {
    if saw_tool_call {
        return "tool_use";
    }
    match openai_reason {
        "stop" => "end_turn",
        "length" => "max_tokens",
        "tool_calls" => "tool_use",
        "content_filter" => "end_turn",
        "function_call" => "tool_use",
        _ => "end_turn",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_non_stream_basic() {
        let response = json!({
            "id": "chatcmpl-123",
            "model": "gpt-4",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Hello! How can I help?"
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 8
            }
        });

        let result = transform_non_stream("gpt-4", &json!({}), &json!({}), response, None);
        assert_eq!(result["type"], "message");
        assert_eq!(result["content"][0]["type"], "text");
        assert_eq!(result["content"][0]["text"], "Hello! How can I help?");
        assert_eq!(result["stop_reason"], "end_turn");
        assert_eq!(result["usage"]["input_tokens"], 10);
        assert_eq!(result["usage"]["output_tokens"], 8);
    }

    #[test]
    fn test_non_stream_with_tool_calls() {
        let response = json!({
            "id": "chatcmpl-456",
            "model": "gpt-4",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{
                        "id": "call_123",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": r#"{"city":"Paris"}"#
                        }
                    }]
                },
                "finish_reason": "tool_calls"
            }],
            "usage": {"prompt_tokens": 20, "completion_tokens": 10}
        });

        let result = transform_non_stream("gpt-4", &json!({}), &json!({}), response, None);
        assert_eq!(result["content"][0]["type"], "tool_use");
        assert_eq!(result["content"][0]["name"], "get_weather");
        assert_eq!(result["content"][0]["input"]["city"], "Paris");
        assert_eq!(result["stop_reason"], "tool_use");
    }

    #[test]
    fn test_non_stream_with_reasoning() {
        let response = json!({
            "id": "chatcmpl-789",
            "model": "gpt-4",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "The answer is 42.",
                    "reasoning_content": "Let me calculate..."
                },
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 15, "completion_tokens": 12}
        });

        let result = transform_non_stream("gpt-4", &json!({}), &json!({}), response, None);
        assert_eq!(result["content"][0]["type"], "text");
        assert_eq!(result["content"][0]["text"], "The answer is 42.");
        assert_eq!(result["content"][1]["type"], "thinking");
        assert_eq!(result["content"][1]["thinking"], "Let me calculate...");
    }
}
