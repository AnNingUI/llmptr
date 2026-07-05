// WHY: collapsible-if false positives on Go-ported state machines.
// Nested ifs in sequential state checks are intentional for readability.
// #![allow(clippy::collapsible_if)]
//! # llmptr
//!
//! AI API protocol translation matrix — convert requests and responses between
//! OpenAI ChatCompletions, OpenAI Responses, Claude Messages, Gemini, Codex,
//! Antigravity, Kimi, and xAI formats.
//!
//! ## Architecture
//!
//! A central [`Registry`] maps `(from, to)` format pairs to transform functions.
//! Each translator module registers itself via [`Registry::register`], typically
//! in an init-style pattern (or explicitly during setup).
//!
//! Transforms operate on **raw JSON** (`serde_json::Value`), keeping the
//! translation layer schema-agnostic at the registry level.
//!
//! ## Supported formats
//!
//! See [`Format`] for the complete list.

pub mod error;
pub mod format;
pub mod models;
pub mod pipeline;
pub mod registry;
pub mod translators;

// Re-exports
pub use error::Error;
pub use format::Format;
pub use pipeline::Pipeline;
pub use registry::Registry;

use once_cell::sync::Lazy;

/// The global default translator registry.
pub static DEFAULT_REGISTRY: Lazy<Registry> = Lazy::new(|| {
    let registry = Registry::new();
    translators::register_all(&registry);
    registry
});

/// Register a translator pair on the global default registry.
pub fn register(
    from: Format,
    to: Format,
    request: Option<crate::registry::RequestTransform>,
    response: Option<crate::registry::ResponseTransform>,
) {
    DEFAULT_REGISTRY.register(from, to, request, response);
}

/// Translate a request on the global default registry.
pub fn translate_request(
    from: Format,
    to: Format,
    model: &str,
    body: serde_json::Value,
    stream: bool,
) -> serde_json::Value {
    DEFAULT_REGISTRY.translate_request(from, to, model, body, stream)
}

/// Translate a streaming response chunk on the global default registry.
pub fn translate_stream(
    from: Format,
    to: Format,
    model: &str,
    original_request: &serde_json::Value,
    translated_request: &serde_json::Value,
    chunk: serde_json::Value,
    param: Option<&mut Box<dyn std::any::Any>>,
) -> Vec<serde_json::Value> {
    DEFAULT_REGISTRY.translate_stream(
        from,
        to,
        model,
        original_request,
        translated_request,
        chunk,
        param,
    )
}

/// Translate a non-streaming response on the global default registry.
pub fn translate_non_stream(
    from: Format,
    to: Format,
    model: &str,
    original_request: &serde_json::Value,
    translated_request: &serde_json::Value,
    response: serde_json::Value,
    param: Option<&mut Box<dyn std::any::Any>>,
) -> serde_json::Value {
    DEFAULT_REGISTRY.translate_non_stream(
        from,
        to,
        model,
        original_request,
        translated_request,
        response,
        param,
    )
}

/// Translate a token count response on the global default registry.
pub fn translate_token_count(
    from: Format,
    to: Format,
    count: i64,
    raw: serde_json::Value,
) -> serde_json::Value {
    DEFAULT_REGISTRY.translate_token_count(from, to, count, raw)
}
