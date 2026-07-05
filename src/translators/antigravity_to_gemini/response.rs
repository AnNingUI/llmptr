//! Antigravity → Gemini response passthrough with cpaUsageMetadata restoration.

use serde_json::Value;

/// Restore cpaUsageMetadata back to usageMetadata.
/// The executor renames usageMetadata→cpaUsageMetadata in non-terminal chunks
/// to preserve usage data while hiding it from clients. When returning standard
/// Gemini format, we must restore the original name.
fn restore_usage_metadata(mut chunk: Value) -> Value {
    if let Some(cpa) = chunk.get("cpaUsageMetadata") {
        chunk["usageMetadata"] = cpa.clone();
        let obj = chunk.as_object_mut().unwrap();
        obj.remove("cpaUsageMetadata");
    }
    // Also check nested under response.cpaUsageMetadata
    if let Some(cpa) = chunk.pointer("/response/cpaUsageMetadata") {
        chunk["response"]["usageMetadata"] = cpa.clone();
        if let Some(resp) = chunk
            .pointer_mut("/response")
            .and_then(|v| v.as_object_mut())
        {
            resp.remove("cpaUsageMetadata");
        }
    }
    chunk
}

pub fn transform_stream(
    m: &str,
    a: &Value,
    b: &Value,
    c: Value,
    d: Option<&mut Box<dyn std::any::Any>>,
) -> Vec<Value> {
    let mut results =
        crate::translators::gemini_to_gemini::response::passthrough_stream(m, a, b, c, d);
    for chunk in &mut results {
        *chunk = restore_usage_metadata(std::mem::take(chunk));
    }
    results
}

pub fn transform_non_stream(
    m: &str,
    a: &Value,
    b: &Value,
    c: Value,
    d: Option<&mut Box<dyn std::any::Any>>,
) -> Value {
    let result =
        crate::translators::gemini_to_gemini::response::passthrough_non_stream(m, a, b, c, d);
    restore_usage_metadata(result)
}
