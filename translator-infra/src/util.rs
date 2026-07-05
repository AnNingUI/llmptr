//! Utility functions for AI API protocol translation.
//!
//! Ported from Go's `CLIProxyAPI/internal/util/`.

// ── Claude Tool ID sanitization ────────────────────────────

/// Sanitize a tool call ID for Claude's strict `^[a-zA-Z0-9_-]+$` regex.
/// Replaces invalid characters with '_'; generates a fallback if the result is empty.
pub fn sanitize_claude_tool_id(id: &str) -> String {
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);

    let s: String = id.chars()
        .map(|c| if c.is_ascii_alphanumeric() || c == '_' || c == '-' { c } else { '_' })
        .collect();

    if s.is_empty() || s.trim_matches('_').is_empty() {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        format!("toolu_{}_{}", nanos, COUNTER.fetch_add(1, Ordering::Relaxed))
    } else {
        s
    }
}

// ── Claude Code attribution detection ──────────────────────

/// Check if a text string is a Claude Code attribution that should be stripped.
pub fn is_claude_attribution(text: &str) -> bool {
    text.contains("The assistant is Claude")
        || text.contains("由 Claude 开发")
        || text.contains("Claude Code是由")
        || text.contains("Claude Code is")
}

// ── Sanitize function name for Gemini compatibility ────────

/// Replace non-alphanumeric characters (except underscore, dash, dot) with underscores.
pub fn sanitize_function_name(name: &str) -> String {
    if name.is_empty() {
        return String::new();
    }

    let mut sanitized: String = name
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '_' || c == '.' || c == ':' || c == '-' {
                c
            } else {
                '_'
            }
        })
        .collect();

    if let Some(first) = sanitized.as_bytes().first().copied() {
        let valid_start = first.is_ascii_alphabetic() || first == b'_';
        if !valid_start {
            if sanitized.len() >= 64 {
                sanitized.truncate(63);
            }
            sanitized.insert(0, '_');
        }
    } else {
        sanitized.push('_');
    }

    if sanitized.len() > 64 {
        sanitized.truncate(64);
    }
    sanitized
}

// ── Tool name mapping ──────────────────────────────────────

use std::collections::HashMap;

/// Build a tool name mapping from a Claude request's tools array.
/// Maps simplified names back to original Claude-qualified names.
pub fn tool_name_map_from_request(tools_json: &serde_json::Value) -> HashMap<String, String> {
    let mut m = HashMap::new();
    if let Some(tools) = tools_json.get("tools").and_then(|v| v.as_array()) {
        for tool in tools {
            if let Some(name) = tool.get("name").and_then(|v| v.as_str()) {
                let simplified = name.split("__").last().unwrap_or(name);
                if simplified != name {
                    m.insert(simplified.to_string(), name.to_string());
                }
                m.insert(name.to_string(), name.to_string());
            }
        }
    }
    m
}

/// Look up a tool name in the reverse mapping (simplified → original).
pub fn map_tool_name(tool_name_map: &HashMap<String, String>, name: &str) -> String {
    tool_name_map.get(name).cloned().unwrap_or_else(|| name.to_string())
}

// ── Fix malformed JSON ─────────────────────────────────────

/// Convert single-quoted strings to double-quoted in JSON-like strings.
/// This handles a common issue where AI providers send malformed JSON with single quotes.
pub fn fix_json(input: &str) -> String {
    let mut out = String::with_capacity(input.len());
    let mut in_double = false;
    let mut in_single = false;
    let mut escaped = false;
    let chars: Vec<char> = input.chars().collect();
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        if in_double {
            out.push(c);
            if escaped { escaped = false; }
            else if c == '\\' { escaped = true; }
            else if c == '"' { in_double = false; }
        } else if in_single {
            if escaped {
                escaped = false;
                match c {
                    '\\' => { out.push('\\'); out.push('\\'); }
                    '\'' => out.push('\''),
                    '"' => { out.push('\\'); out.push('"'); }
                    'n' | 'r' | 't' | 'b' | 'f' | '/' => { out.push('\\'); out.push(c); }
                    'u' => {
                        out.push_str("\\u");
                        for _ in 0..4 {
                            if i + 1 < chars.len() {
                                let n = chars[i + 1];
                                if n.is_ascii_hexdigit() { out.push(n); i += 1; }
                                else { break; }
                            } else { break; }
                        }
                    }
                    _ => out.push(c),
                }
            } else if c == '\\' { escaped = true; }
            else if c == '\'' { in_single = false; }
            else if c == '"' { out.push('\\'); out.push('"'); }
            else { out.push(c); }
        } else {
            if c == '"' { in_double = true; out.push(c); }
            else if c == '\'' { in_single = true; out.push('"'); }
            else { out.push(c); }
        }
        i += 1;
    }
    out
}

// ── Claude tool result content extraction ──────────────────

/// Result of converting a Claude tool result content block.
pub struct ToolResultContent {
    /// The aggregated text content.
    pub result: String,
    /// Whether the result was raw JSON (vs plain string).
    pub is_raw: bool,
    /// Images extracted from multi-part content.
    pub images: Vec<ToolResultImage>,
}

/// An image embedded in a tool result.
pub struct ToolResultImage {
    pub mime_type: String,
    pub data: String,
}

/// Extract text and images from a Claude tool result content block.
pub fn convert_tool_result_content(content: &serde_json::Value) -> ToolResultContent {
    match content {
        serde_json::Value::String(s) => ToolResultContent {
            result: s.clone(),
            is_raw: false,
            images: vec![],
        },
        serde_json::Value::Array(parts) => {
            let mut texts: Vec<String> = Vec::new();
            let mut images: Vec<ToolResultImage> = Vec::new();
            let mut is_raw = false;

            for part in parts {
                match part.get("type").and_then(|v| v.as_str()) {
                    Some("text") => {
                        if let Some(t) = part.get("text").and_then(|v| v.as_str()) {
                            texts.push(t.to_string());
                        }
                    }
                    Some("image") => {
                        if let Some(src) = part.get("source") {
                            let mt = src.get("media_type").and_then(|v| v.as_str()).unwrap_or("image/png");
                            let data = src.get("data").and_then(|v| v.as_str()).unwrap_or("");
                            if !data.is_empty() {
                                images.push(ToolResultImage {
                                    mime_type: mt.to_string(),
                                    data: data.to_string(),
                                });
                            }
                        }
                    }
                    _ => {
                        is_raw = true;
                    }
                }
            }

            ToolResultContent {
                result: texts.join("\n\n"),
                is_raw,
                images,
            }
        }
        _ => ToolResultContent {
            result: serde_json::to_string(content).unwrap_or_default(),
            is_raw: true,
            images: vec![],
        },
    }
}

// ── Clean JSON Schema for Gemini ───────────────────────────

/// Clean a JSON Schema for Gemini compatibility.
/// Removes fields Gemini doesn't understand and preserves unsupported semantics as
/// short description hints where CLIProxyAPI does the same.
pub fn clean_json_schema_for_gemini(schema_raw: &str) -> String {
    use serde_json::Value;
    if schema_raw.is_empty() || schema_raw == "null" {
        return r#"{"type":"object","properties":{}}"#.to_string();
    }
    let mut schema: Value = match serde_json::from_str(schema_raw) {
        Ok(s) => s,
        Err(_) => return r#"{"type":"object","properties":{}}"#.to_string(),
    };

    if !schema.is_object() {
        return r#"{"type":"object","properties":{}}"#.to_string();
    }

    clean_gemini_schema_value(&mut schema, false);
    cleanup_required_fields(&mut schema);

    serde_json::to_string(&schema).unwrap_or_else(|_| r#"{"type":"object","properties":{}}"#.to_string())
}

fn clean_gemini_schema_value(value: &mut serde_json::Value, is_properties_map: bool) {
    use serde_json::Value;

    match value {
        Value::Object(obj) => {
            if !is_properties_map {
                if matches!(obj.get("additionalProperties"), Some(Value::Bool(false))) {
                    append_description_hint(obj, "No extra properties allowed");
                }
                normalize_enum(obj);

                for key in gemini_unsupported_schema_keys() {
                    obj.remove(*key);
                }
                let extension_keys: Vec<String> = obj
                    .keys()
                    .filter(|key| key.starts_with("x-"))
                    .cloned()
                    .collect();
                for key in extension_keys {
                    obj.remove(&key);
                }
            }

            let keys: Vec<String> = obj.keys().cloned().collect();
            for key in keys {
                if let Some(child) = obj.get_mut(&key) {
                    clean_gemini_schema_value(child, key == "properties");
                }
            }
        }
        Value::Array(items) => {
            for item in items {
                clean_gemini_schema_value(item, false);
            }
        }
        _ => {}
    }
}

fn normalize_enum(obj: &mut serde_json::Map<String, serde_json::Value>) {
    use serde_json::Value;

    let Some(enum_values) = obj.get_mut("enum").and_then(Value::as_array_mut) else {
        return;
    };

    let string_values: Vec<String> = enum_values.iter().map(gjson_string_value).collect();
    *enum_values = string_values
        .iter()
        .map(|value| Value::String(value.clone()))
        .collect();
    obj.insert("type".to_string(), Value::String("string".to_string()));

    if (2..=10).contains(&string_values.len()) {
        append_description_hint(obj, &format!("Allowed: {}", string_values.join(", ")));
    }
}

fn gjson_string_value(value: &serde_json::Value) -> String {
    match value {
        serde_json::Value::String(value) => value.clone(),
        serde_json::Value::Number(value) => value.to_string(),
        serde_json::Value::Bool(value) => value.to_string(),
        serde_json::Value::Null => String::new(),
        other => serde_json::to_string(other).unwrap_or_default(),
    }
}

fn append_description_hint(obj: &mut serde_json::Map<String, serde_json::Value>, hint: &str) {
    let description = obj
        .get("description")
        .and_then(serde_json::Value::as_str)
        .map(|existing| {
            if existing.is_empty() {
                hint.to_string()
            } else {
                format!("{existing} ({hint})")
            }
        })
        .unwrap_or_else(|| hint.to_string());
    obj.insert(
        "description".to_string(),
        serde_json::Value::String(description),
    );
}

fn cleanup_required_fields(value: &mut serde_json::Value) {
    use serde_json::Value;

    match value {
        Value::Object(obj) => {
            let keys: Vec<String> = obj.keys().cloned().collect();
            for key in keys {
                if let Some(child) = obj.get_mut(&key) {
                    cleanup_required_fields(child);
                }
            }

            let Some(properties) = obj.get("properties").and_then(Value::as_object) else {
                return;
            };
            let Some(required) = obj.get("required").and_then(Value::as_array) else {
                return;
            };
            let filtered: Vec<Value> = required
                .iter()
                .filter_map(Value::as_str)
                .filter(|name| properties.contains_key(*name))
                .map(|name| Value::String(name.to_string()))
                .collect();
            if filtered.is_empty() {
                obj.remove("required");
            } else if filtered.len() != required.len() {
                obj.insert("required".to_string(), Value::Array(filtered));
            }
        }
        Value::Array(items) => {
            for item in items {
                cleanup_required_fields(item);
            }
        }
        _ => {}
    }
}

fn gemini_unsupported_schema_keys() -> &'static [&'static str] {
    &[
        "minLength",
        "maxLength",
        "exclusiveMinimum",
        "exclusiveMaximum",
        "pattern",
        "minItems",
        "maxItems",
        "uniqueItems",
        "format",
        "default",
        "examples",
        "$schema",
        "$defs",
        "definitions",
        "const",
        "$ref",
        "$id",
        "additionalProperties",
        "propertyNames",
        "patternProperties",
        "$comment",
        "enumDescriptions",
        "enumTitles",
        "prefill",
        "deprecated",
        "nullable",
        "title",
        "strict",
        "input_examples",
        "cache_control",
        "defer_loading",
        "eager_input_streaming",
    ]
}

// ── Codex name shortening (64-char limit) ─────────────────

const CODEX_NAME_LIMIT: usize = 64;

/// Shorten a function name to fit Codex's 64-character limit.
/// Preserves the `mcp__` prefix and last segment when possible.
pub fn shorten_name_if_needed(name: &str) -> String {
    if name.len() <= CODEX_NAME_LIMIT {
        return name.to_string();
    }
    if let Some(stripped) = name.strip_prefix("mcp__") {
        if let Some(last_idx) = stripped.rfind("__") {
            let mut cand = String::from("mcp__") + &stripped[last_idx + 2..];
            cand.truncate(CODEX_NAME_LIMIT);
            return cand;
        }
    }
    name.chars().take(CODEX_NAME_LIMIT).collect()
}

/// Build a short-name map ensuring uniqueness for Codex's 64-char limit.
/// Returns `original_name -> short_name`.
pub fn build_short_name_map(names: &[String]) -> HashMap<String, String> {
    use std::collections::HashSet;
    let mut used: HashSet<String> = HashSet::new();
    let mut m = HashMap::new();

    let base_candidate = |n: &str| -> String {
        if n.len() <= CODEX_NAME_LIMIT {
            return n.to_string();
        }
        if let Some(stripped) = n.strip_prefix("mcp__") {
            if let Some(last_idx) = stripped.rfind("__") {
                let mut cand = String::from("mcp__") + &stripped[last_idx + 2..];
                cand.truncate(CODEX_NAME_LIMIT);
                return cand;
            }
        }
        n.chars().take(CODEX_NAME_LIMIT).collect()
    };

    for n in names {
        let base = base_candidate(n);
        if used.insert(base.clone()) {
            m.insert(n.clone(), base);
        } else {
            let mut i = 1usize;
            loop {
                let suffix = format!("_{}", i);
                let allowed = CODEX_NAME_LIMIT.saturating_sub(suffix.len());
                let mut tmp: String = base.chars().take(allowed).collect();
                tmp.push_str(&suffix);
                if used.insert(tmp.clone()) {
                    m.insert(n.clone(), tmp);
                    break;
                }
                i += 1;
            }
        }
    }
    m
}

/// Build a reverse map (short -> original) from a Claude original request JSON.
pub fn build_reverse_name_map(tools_json: &serde_json::Value) -> HashMap<String, String> {
    let mut orig_names: Vec<String> = Vec::new();
    if let Some(tools) = tools_json.get("tools").and_then(|v| v.as_array()) {
        for tool in tools {
            if let Some(name) = tool.get("name").and_then(|v| v.as_str()) {
                orig_names.push(name.to_string());
            }
        }
    }
    let short_map = build_short_name_map(&orig_names);
    short_map.into_iter().map(|(k, v)| (v, k)).collect()
}
