//! Antigravity -> OpenAI Responses SSE -- wraps the Gemini->Responses converter.

use serde_json::Value;

/// Stream: extract `response` field from Antigravity wrapper, then delegate to Gemini->Responses.
pub fn transform_stream(
    model: &str,
    orig_req: &Value,
    trans_req: &Value,
    chunk: Value,
    param: Option<&mut Box<dyn std::any::Any>>,
) -> Vec<Value> {
    let inner = chunk.get("response").cloned().unwrap_or(chunk);

    crate::translators::openai_response_to_gemini::response::transform_stream(
        model, orig_req, trans_req, inner, param,
    )
}

/// Non-stream: extract `response` field, unwrap request wrappers, delegate to Gemini->Responses.
pub fn transform_non_stream(
    model: &str,
    orig_req: &Value,
    trans_req: &Value,
    response: Value,
    param: Option<&mut Box<dyn std::any::Any>>,
) -> Value {
    let inner = response.get("response").cloned().unwrap_or(response);

    let orig = orig_req
        .get("request")
        .cloned()
        .unwrap_or_else(|| orig_req.clone());
    let trans = trans_req
        .get("request")
        .cloned()
        .unwrap_or_else(|| trans_req.clone());

    crate::translators::openai_response_to_gemini::response::transform_non_stream(
        model, &orig, &trans, inner, param,
    )
}
