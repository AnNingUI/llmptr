//! Antigravity (Gemini AI Studio) → Claude Messages response translation.
//!
//! Full state machine ported from Go's `antigravity/claude/antigravity_claude_response.go` (628 lines).
//! Manages response type states: 0=none, 1=content, 2=thinking, 3=function

use crate::translators::antigravity_web_search;
use serde_json::{Value, json};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use llmptr_infra::signature;

/// Response type state: 0=none, 1=content, 2=thinking, 3=function
const RT_NONE: i32 = 0;
const RT_CONTENT: i32 = 1;
const RT_THINKING: i32 = 2;
const RT_FUNCTION: i32 = 3;

/// Global tool use ID counter.
static TOOL_USE_ID_CTR: AtomicU64 = AtomicU64::new(0);

/// State maintained across streaming chunks.
#[derive(Debug, Clone, Default)]
pub struct AntigravityClaudeParams {
    pub has_first_response: bool,
    pub response_type: i32,
    pub response_index: i32,
    pub has_finish_reason: bool,
    pub finish_reason: String,
    pub has_usage_metadata: bool,
    pub prompt_token_count: i64,
    pub candidates_token_count: i64,
    pub thoughts_token_count: i64,
    pub total_token_count: i64,
    pub cached_token_count: i64,
    pub has_sent_final_events: bool,
    pub has_tool_use: bool,
    pub has_content: bool,
    pub current_thinking_text: String,
    pub tool_name_map: HashMap<String, String>,
    pub has_web_search_tool: bool,
    pub web_search_requests: i64,
    pub web_search_text_buf: String,
}

/// Convert a single streaming Antigravity (Gemini) chunk to Claude SSE events.
pub fn transform_stream(
    _model: &str,
    original_request: &Value,
    _translated_request: &Value,
    chunk: Value,
    param: Option<&mut Box<dyn std::any::Any>>,
) -> Vec<Value> {
    let params = if let Some(p) = param {
        p.downcast_mut::<AntigravityClaudeParams>().unwrap()
    } else {
        return vec![chunk];
    };

    // Initialize tool name map from original request on first call
    if params.tool_name_map.is_empty() && original_request.is_object() {
        params.tool_name_map = build_tool_name_map(original_request);
    }

    // Handle [DONE] marker
    if chunk.as_str() == Some("[DONE]") {
        let mut results = Vec::new();
        if params.has_content {
            append_final_events(params, &mut results, true);
            results.push(json!({"type": "message_stop"}));
        }
        return results;
    }

    let mut out = Vec::new();

    let sse_event = |event: &str, payload: Value| -> Value {
        let mut ev = json!({"__sse": true, "event": event});
        ev["data"] = payload;
        ev
    };

    let append_event = |out: &mut Vec<Value>, event: &str, payload: Value| {
        out.push(sse_event(event, payload));
    };

    // message_start on first chunk
    if !params.has_first_response {
        let mut msg_start = json!({
            "type": "message_start",
            "message": {
                "id": "msg_antigravity",
                "type": "message",
                "role": "assistant",
                "content": [],
                "model": "claude",
                "stop_reason": null,
                "stop_sequence": null,
                "usage": {"input_tokens": 0, "output_tokens": 0},
            }
        });

        // Try to fill usage from cpaUsageMetadata
        if let Some(prompt) = chunk
            .pointer("/response/cpaUsageMetadata/promptTokenCount")
            .and_then(|v| v.as_i64())
        {
            msg_start["message"]["usage"]["input_tokens"] = json!(prompt);
        }
        if let Some(cand) = chunk
            .pointer("/response/cpaUsageMetadata/candidatesTokenCount")
            .and_then(|v| v.as_i64())
        {
            msg_start["message"]["usage"]["output_tokens"] = json!(cand);
        }
        // Model/id from response
        if let Some(model) = chunk
            .pointer("/response/modelVersion")
            .and_then(|v| v.as_str())
            && !model.is_empty()
        {
            msg_start["message"]["model"] = json!(model);
        }
        if let Some(rid) = chunk
            .pointer("/response/responseId")
            .and_then(|v| v.as_str())
            && !rid.is_empty()
        {
            msg_start["message"]["id"] = json!(rid);
        }
        append_event(&mut out, "message_start", msg_start);
        params.has_first_response = true;
    }

    // Check for web search grounding BEFORE processing normal parts
    if antigravity_web_search::should_translate_web_search_grounding(
        original_request,
        _translated_request,
    ) && !params.has_web_search_tool
    {
        if let Some(grounding_metadata) =
            antigravity_web_search::antigravity_grounding_metadata(&chunk)
        {
            let tool_use_id = antigravity_web_search::new_claude_web_search_tool_use_id();
            let text_content = {
                let mut buf = std::mem::take(&mut params.web_search_text_buf);
                buf.push_str(&antigravity_web_search::antigravity_text_content(&chunk));
                buf
            };
            params.response_index = antigravity_web_search::append_claude_web_search_stream_blocks(
                &mut out,
                params.response_index,
                &tool_use_id,
                &text_content,
                &grounding_metadata,
            );
            params.has_web_search_tool = true;
            params.web_search_requests = 1;
            params.has_content = true;
            params.response_type = RT_NONE;
            // Skip normal parts processing
        } else {
            // Buffer text for later web search grounding
            if let Some(part_results) = chunk
                .pointer("/response/candidates/0/content/parts")
                .and_then(|v| v.as_array())
            {
                for part in part_results {
                    if part.get("thought") == Some(&json!(true))
                        || part.get("functionCall").is_some()
                    {
                        continue;
                    }
                    if let Some(t) = part.get("text").and_then(|v| v.as_str()) {
                        params.web_search_text_buf.push_str(t);
                    }
                }
            }
        }
    } else {
        // Normal parts processing
        process_normal_parts(&mut out, params, &chunk);
    }

    // Finish reason
    if let Some(reason) = chunk
        .pointer("/response/candidates/0/finishReason")
        .and_then(|v| v.as_str())
        && !reason.is_empty()
    {
        params.has_finish_reason = true;
        params.finish_reason = reason.to_string();
    }

    // If web search streaming mode with buffered text and finish reason, flush buffer
    if antigravity_web_search::should_translate_web_search_grounding(
        original_request,
        _translated_request,
    ) && !params.has_web_search_tool
        && params.has_finish_reason
    {
        let text = std::mem::take(&mut params.web_search_text_buf);
        if !text.is_empty() {
            out.push(json!({
                "__sse": true, "event": "content_block_start",
                "data": {
                    "type": "content_block_start",
                    "index": params.response_index,
                    "content_block": {"type": "text", "text": ""},
                }
            }));
            out.push(json!({
                "__sse": true, "event": "content_block_delta",
                "data": {
                    "type": "content_block_delta",
                    "index": params.response_index,
                    "delta": {"type": "text_delta", "text": text},
                }
            }));
            params.response_type = RT_CONTENT;
            params.has_content = true;
        }
    }

    // Usage metadata
    if let Some(usage) = chunk
        .pointer("/response/usageMetadata")
        .and_then(|v| v.as_object())
    {
        params.has_usage_metadata = true;
        params.cached_token_count = usage
            .get("cachedContentTokenCount")
            .and_then(|v| v.as_i64())
            .unwrap_or(0);
        let prompt = usage
            .get("promptTokenCount")
            .and_then(|v| v.as_i64())
            .unwrap_or(0);
        params.prompt_token_count = prompt - params.cached_token_count;
        params.candidates_token_count = usage
            .get("candidatesTokenCount")
            .and_then(|v| v.as_i64())
            .unwrap_or(0);
        params.thoughts_token_count = usage
            .get("thoughtsTokenCount")
            .and_then(|v| v.as_i64())
            .unwrap_or(0);
        params.total_token_count = usage
            .get("totalTokenCount")
            .and_then(|v| v.as_i64())
            .unwrap_or(0);
        if params.candidates_token_count == 0 && params.total_token_count > 0 {
            params.candidates_token_count =
                params.total_token_count - params.prompt_token_count - params.thoughts_token_count;
            if params.candidates_token_count < 0 {
                params.candidates_token_count = 0;
            }
        }
    }

    // Final events when both usage and finish reason available
    if params.has_usage_metadata && params.has_finish_reason {
        append_final_events(params, &mut out, false);
    }

    out
}

/// Append a thinking signature delta to the output.
fn append_thinking_signature(
    out: &mut Vec<Value>,
    params: &mut AntigravityClaudeParams,
    signature: &str,
) {
    if signature.is_empty() || params.response_type != RT_THINKING {
        return;
    }
    let sig_val = format_claude_signature_value(signature);
    out.push(json!({
        "__sse": true, "event": "content_block_delta",
        "data": {
            "type": "content_block_delta",
            "index": params.response_index,
            "delta": {"type": "signature_delta", "signature": sig_val},
        }
    }));
    params.has_content = true;
}

/// Format a Claude signature value via signature module.
fn format_claude_signature_value(signature: &str) -> String {
    signature::normalize_claude_native_sig(signature, false).unwrap_or_else(|| {
        if signature.starts_with('R') || signature.starts_with('E') {
            signature.to_string()
        } else {
            String::new()
        }
    })
}

/// Append final events (content_block_stop + message_delta).
fn append_final_events(params: &mut AntigravityClaudeParams, out: &mut Vec<Value>, _force: bool) {
    if params.has_sent_final_events {
        return;
    }
    if !params.has_usage_metadata && !_force {
        return;
    }
    if !params.has_content {
        return;
    }

    if params.response_type != RT_NONE {
        out.push(json!({"type": "content_block_stop", "index": params.response_index}));
        params.response_type = RT_NONE;
    }

    let stop_reason = resolve_stop_reason(params);
    let usage_output = params.candidates_token_count + params.thoughts_token_count;
    let mut delta = json!({
        "type": "message_delta",
        "delta": {"stop_reason": &stop_reason, "stop_sequence": null},
        "usage": {"input_tokens": params.prompt_token_count, "output_tokens": usage_output},
    });

    if params.cached_token_count > 0 {
        delta["usage"]["cache_read_input_tokens"] = json!(params.cached_token_count);
    }

    out.push(delta);
    params.has_sent_final_events = true;
}

/// Resolve the stop reason string.
fn resolve_stop_reason(params: &AntigravityClaudeParams) -> &'static str {
    if params.has_tool_use {
        return "tool_use";
    }
    match params.finish_reason.as_str() {
        "MAX_TOKENS" => "max_tokens",
        "STOP" | "FINISH_REASON_UNSPECIFIED" | "UNKNOWN" => "end_turn",
        _ => "end_turn",
    }
}

/// Convert a non-streaming Antigravity response to Claude format.
pub fn transform_non_stream(
    _model: &str,
    original_request: &Value,
    _translated_request: &Value,
    response: Value,
    _param: Option<&mut Box<dyn std::any::Any>>,
) -> Value {
    let tool_name_map = build_tool_name_map(original_request);

    let prompt_tokens = response
        .pointer("/response/usageMetadata/promptTokenCount")
        .and_then(|v| v.as_i64())
        .unwrap_or(0);
    let candidate_tokens = response
        .pointer("/response/usageMetadata/candidatesTokenCount")
        .and_then(|v| v.as_i64())
        .unwrap_or(0);
    let thought_tokens = response
        .pointer("/response/usageMetadata/thoughtsTokenCount")
        .and_then(|v| v.as_i64())
        .unwrap_or(0);
    let total_tokens = response
        .pointer("/response/usageMetadata/totalTokenCount")
        .and_then(|v| v.as_i64())
        .unwrap_or(0);
    let cached_tokens = response
        .pointer("/response/usageMetadata/cachedContentTokenCount")
        .and_then(|v| v.as_i64())
        .unwrap_or(0);
    let mut output_tokens = candidate_tokens + thought_tokens;
    if output_tokens == 0 && total_tokens > 0 {
        output_tokens = total_tokens - prompt_tokens;
        if output_tokens < 0 {
            output_tokens = 0;
        }
    }

    let rid = response
        .pointer("/response/responseId")
        .and_then(|v| v.as_str())
        .unwrap_or("msg_unknown");
    let model_ver = response
        .pointer("/response/modelVersion")
        .and_then(|v| v.as_str())
        .unwrap_or("");

    let mut out = json!({
        "id": rid,
        "type": "message",
        "role": "assistant",
        "model": model_ver,
        "content": null,
        "stop_reason": null,
        "stop_sequence": null,
        "usage": {"input_tokens": prompt_tokens, "output_tokens": output_tokens},
    });

    if cached_tokens > 0 {
        out["usage"]["cache_read_input_tokens"] = json!(cached_tokens);
    }

    // Build content array from parts
    let parts = response
        .pointer("/response/candidates/0/content/parts")
        .and_then(|v| v.as_array());
    let mut text_buf = String::new();
    let mut thinking_buf = String::new();
    let mut thinking_sig = String::new();
    let mut has_tool_call = false;

    let flush_text = |out: &mut Value, buf: &mut String| {
        if buf.is_empty() {
            return;
        }
        if !out.get("content").map(|v| v.is_array()).unwrap_or(false) {
            out["content"] = json!([]);
        }
        out["content"]
            .as_array_mut()
            .unwrap()
            .push(json!({"type": "text", "text": buf.clone()}));
        buf.clear();
    };

    let flush_thinking = |out: &mut Value, buf: &mut String, sig: &mut String| {
        if buf.is_empty() && sig.is_empty() {
            return;
        }
        if !out.get("content").map(|v| v.is_array()).unwrap_or(false) {
            out["content"] = json!([]);
        }
        let mut block = json!({"type": "thinking", "thinking": buf.clone()});
        if !sig.is_empty() {
            block["signature"] = json!(format_claude_signature_value(sig));
        }
        out["content"].as_array_mut().unwrap().push(block);
        buf.clear();
        sig.clear();
    };

    if let Some(part_results) = parts {
        for part in part_results {
            let sig = part
                .get("thoughtSignature")
                .and_then(|v| v.as_str())
                .or_else(|| part.get("thought_signature").and_then(|v| v.as_str()))
                .unwrap_or("")
                .to_string();
            let has_sig = !sig.is_empty() && part.get("functionCall").is_none();
            let is_thought = part.get("thought") == Some(&json!(true)) || has_sig;

            if has_sig {
                thinking_sig = sig;
            }

            if let Some(t) = part.get("text").and_then(|v| v.as_str())
                && !t.is_empty()
            {
                if is_thought {
                    flush_text(&mut out, &mut text_buf);
                    thinking_buf.push_str(t);
                    continue;
                }
                flush_thinking(&mut out, &mut thinking_buf, &mut thinking_sig);
                text_buf.push_str(t);
                continue;
            }

            if let Some(fc) = part.get("functionCall").and_then(|v| v.as_object()) {
                flush_thinking(&mut out, &mut thinking_buf, &mut thinking_sig);
                flush_text(&mut out, &mut text_buf);
                has_tool_call = true;

                let name = restore_sanitized_tool_name(
                    &tool_name_map,
                    fc.get("name").and_then(|v| v.as_str()).unwrap_or(""),
                );

                if !out.get("content").map(|v| v.is_array()).unwrap_or(false) {
                    out["content"] = json!([]);
                }
                let mut tool_block = json!({
                    "type": "tool_use", "id": "", "name": name, "input": {},
                });
                if let Some(args) = fc.get("args") {
                    tool_block["input"] = args.clone();
                }
                out["content"].as_array_mut().unwrap().push(tool_block);
            }
        }
    }

    flush_thinking(&mut out, &mut thinking_buf, &mut thinking_sig);
    flush_text(&mut out, &mut text_buf);

    // Web search grounding for non-stream responses
    if antigravity_web_search::should_translate_web_search_grounding(
        original_request,
        _translated_request,
    ) && let Some(grounding_metadata) =
        antigravity_web_search::antigravity_grounding_metadata(&response)
    {
        let tool_use_id = antigravity_web_search::new_claude_web_search_tool_use_id();
        let text_content = antigravity_web_search::antigravity_text_content(&response);
        out["content"] = json!(antigravity_web_search::build_claude_web_search_content(
            &tool_use_id,
            &text_content,
            &grounding_metadata,
        ));
        out["stop_reason"] = json!("end_turn");
        out["usage"]["server_tool_use"]["web_search_requests"] = json!(1);
        return out;
    }

    // Stop reason
    let stop = if has_tool_call {
        "tool_use"
    } else {
        match response
            .pointer("/response/candidates/0/finishReason")
            .and_then(|v| v.as_str())
        {
            Some("MAX_TOKENS") => "max_tokens",
            _ => "end_turn",
        }
    };
    out["stop_reason"] = json!(stop);

    // Remove usage if zero
    if prompt_tokens == 0
        && output_tokens == 0
        && response.pointer("/response/usageMetadata").is_none()
    {
        out["usage"] = json!({"input_tokens": 0, "output_tokens": 0});
    }

    out
}

/// Build a reverse tool name map from the original request.
fn build_tool_name_map(original_request: &Value) -> HashMap<String, String> {
    let mut m = HashMap::new();
    if let Some(tools) = original_request.get("tools").and_then(|v| v.as_array()) {
        for tool in tools {
            if let Some(name) = tool.get("name").and_then(|v| v.as_str()) {
                let short = name.split("__").last().unwrap_or(name);
                if short != name {
                    m.insert(short.to_string(), name.to_string());
                }
                m.insert(name.to_string(), name.to_string());
            }
        }
    }
    m
}

/// Restore a sanitized tool name back to its original form.
fn restore_sanitized_tool_name(map: &HashMap<String, String>, name: &str) -> String {
    map.get(name).cloned().unwrap_or_else(|| name.to_string())
}

/// Process normal parts array (text, thinking, function calls) without web search grounding.
fn process_normal_parts(out: &mut Vec<Value>, params: &mut AntigravityClaudeParams, chunk: &Value) {
    let parts = chunk
        .pointer("/response/candidates/0/content/parts")
        .and_then(|v| v.as_array());
    if parts.is_none() {
        return;
    }
    let part_results = parts.unwrap();

    for part in part_results {
        let text = part.get("text").and_then(|v| v.as_str()).unwrap_or("");
        let is_thought = part.get("thought") == Some(&json!(true));
        let fc = part.get("functionCall");
        let sig = part
            .get("thoughtSignature")
            .and_then(|v| v.as_str())
            .or_else(|| part.get("thought_signature").and_then(|v| v.as_str()))
            .unwrap_or("");

        let has_thought_sig = !sig.is_empty() && fc.is_none();

        // Signature-only parts
        if has_thought_sig && text.is_empty() {
            append_thinking_signature(out, params, sig);
            continue;
        }

        // Text content
        if !text.is_empty() {
            if is_thought || has_thought_sig {
                if has_thought_sig {
                    if params.response_type != RT_THINKING {
                        if params.response_type != RT_NONE {
                            out.push(json!({"__sse": true, "event": "content_block_stop",
                                "data": {"type": "content_block_stop", "index": params.response_index}}));
                            params.response_index += 1;
                        }
                        out.push(json!({"__sse": true, "event": "content_block_start",
                            "data": {"type": "content_block_start", "index": params.response_index,
                                "content_block": {"type": "thinking", "thinking": ""}}}));
                        params.response_type = RT_THINKING;
                        params.current_thinking_text.clear();
                    }
                    params.current_thinking_text.push_str(text);
                    out.push(json!({"__sse": true, "event": "content_block_delta",
                        "data": {"type": "content_block_delta", "index": params.response_index,
                            "delta": {"type": "thinking_delta", "thinking": text}}}));
                    append_thinking_signature(out, params, sig);
                } else if params.response_type == RT_THINKING {
                    params.current_thinking_text.push_str(text);
                    out.push(json!({"__sse": true, "event": "content_block_delta",
                        "data": {"type": "content_block_delta", "index": params.response_index,
                            "delta": {"type": "thinking_delta", "thinking": text}}}));
                    params.has_content = true;
                } else {
                    if params.response_type != RT_NONE {
                        out.push(json!({"__sse": true, "event": "content_block_stop",
                            "data": {"type": "content_block_stop", "index": params.response_index}}));
                        params.response_index += 1;
                    }
                    out.push(json!({"__sse": true, "event": "content_block_start",
                        "data": {"type": "content_block_start", "index": params.response_index,
                            "content_block": {"type": "thinking", "thinking": ""}}}));
                    out.push(json!({"__sse": true, "event": "content_block_delta",
                        "data": {"type": "content_block_delta", "index": params.response_index,
                            "delta": {"type": "thinking_delta", "thinking": text}}}));
                    params.response_type = RT_THINKING;
                    params.has_content = true;
                    params.current_thinking_text = text.to_string();
                }
            } else {
                let finish = chunk
                    .pointer("/response/candidates/0/finishReason")
                    .and_then(|v| v.as_str());
                if !text.is_empty() || finish.is_none() {
                    if params.response_type == RT_CONTENT {
                        out.push(json!({"__sse": true, "event": "content_block_delta",
                            "data": {"type": "content_block_delta", "index": params.response_index,
                                "delta": {"type": "text_delta", "text": text}}}));
                        params.has_content = true;
                    } else {
                        if params.response_type != RT_NONE {
                            out.push(json!({"__sse": true, "event": "content_block_stop",
                                "data": {"type": "content_block_stop", "index": params.response_index}}));
                            params.response_index += 1;
                        }
                        if !text.is_empty() {
                            out.push(json!({"__sse": true, "event": "content_block_start",
                                "data": {"type": "content_block_start", "index": params.response_index,
                                    "content_block": {"type": "text", "text": ""}}}));
                            out.push(json!({"__sse": true, "event": "content_block_delta",
                                "data": {"type": "content_block_delta", "index": params.response_index,
                                    "delta": {"type": "text_delta", "text": text}}}));
                            params.response_type = RT_CONTENT;
                            params.has_content = true;
                        }
                    }
                }
            }
        } else if let Some(fc_obj) = fc.and_then(|v| v.as_object()) {
            params.has_tool_use = true;
            let fc_name = fc_obj.get("name").and_then(|v| v.as_str()).unwrap_or("");
            let restored = restore_sanitized_tool_name(&params.tool_name_map, fc_name);

            if params.response_type != RT_NONE {
                out.push(json!({"__sse": true, "event": "content_block_stop",
                    "data": {"type": "content_block_stop", "index": params.response_index}}));
                params.response_index += 1;
            }

            let call_id = format!(
                "{}-{}-{}",
                restored,
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_nanos(),
                TOOL_USE_ID_CTR.fetch_add(1, Ordering::Relaxed)
            );

            out.push(json!({"__sse": true, "event": "content_block_start",
                "data": {"type": "content_block_start", "index": params.response_index,
                    "content_block": {"type": "tool_use", "id": call_id, "name": restored, "input": {}}}}));

            if let Some(args) = fc_obj.get("args") {
                out.push(json!({"__sse": true, "event": "content_block_delta",
                    "data": {"type": "content_block_delta", "index": params.response_index,
                        "delta": {"type": "input_json_delta", "partial_json": args.clone()}}}));
            }

            params.response_type = RT_FUNCTION;
            params.has_content = true;
        }
    }
}
