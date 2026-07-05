//! # Translator modules
//!
//! Each submodule implements one **source->target** direction of the N×N translation matrix.

// Claude <-> others
use crate::format::Format;
use crate::registry::{
    NonStreamResponseTransform, Registry, ResponseTokenCountTransform, ResponseTransform,
    StreamResponseTransform,
};
use serde_json::json;

pub mod claude_to_antigravity;
pub mod claude_to_codex;
pub mod claude_to_gemini;
pub mod claude_to_openai_chat;
pub mod claude_to_openai_response;
pub mod gemini_to_claude;
pub mod openai_chat_to_claude;
pub mod openai_response_to_claude;

// Gemini <-> others
pub mod codex_to_claude;
pub mod codex_to_gemini;
pub mod gemini_to_codex;
pub mod gemini_to_openai_chat;
pub mod gemini_to_openai_response;
pub mod openai_chat_to_gemini;
pub mod openai_response_to_gemini;

// OpenAI Chat <-> OpenAI Response
pub mod openai_chat_to_openai_response;
pub mod openai_response_to_openai_chat;

// OpenAI Chat <-> Codex
pub mod codex_to_openai_chat;
pub mod openai_chat_to_codex;

// Codex <-> others
pub mod codex_to_openai_response;
pub mod openai_response_to_codex;

// Antigravity <-> others
pub mod antigravity_to_claude;
pub mod antigravity_to_gemini;
pub mod antigravity_to_openai_chat;
pub mod antigravity_to_openai_response;
pub mod openai_chat_to_antigravity;
pub mod openai_response_to_antigravity;

// Self↔Self normalizers
pub mod claude_to_claude;
pub mod gemini_to_gemini;
pub mod openai_chat_to_openai_chat;
pub mod openai_response_to_openai_response;

// Shared Antigravity web search helpers
pub(crate) mod antigravity_web_search;

fn response(
    stream: StreamResponseTransform,
    non_stream: NonStreamResponseTransform,
) -> Option<ResponseTransform> {
    Some(ResponseTransform::new(Some(stream), Some(non_stream)))
}

fn response_with_token_count(
    stream: StreamResponseTransform,
    non_stream: NonStreamResponseTransform,
    token_count: ResponseTokenCountTransform,
) -> Option<ResponseTransform> {
    Some(ResponseTransform::new(Some(stream), Some(non_stream)).with_token_count(Some(token_count)))
}

fn gemini_token_count(count: i64) -> serde_json::Value {
    json!({
        "totalTokens": count,
        "promptTokensDetails": [
            {
                "modality": "TEXT",
                "tokenCount": count
            }
        ]
    })
}

fn claude_token_count(count: i64) -> serde_json::Value {
    json!({ "input_tokens": count })
}

/// Register the built-in translator pairs that have Rust implementations.
///
/// The `from -> to` pair follows the request direction. The response transform
/// attached to that pair converts the upstream `to` response back to `from`.
pub fn register_all(registry: &Registry) {
    use Format::*;

    // Claude target.
    registry.register(
        Gemini,
        Claude,
        Some(gemini_to_claude::request::transform),
        response_with_token_count(
            claude_to_gemini::response::transform_stream,
            claude_to_gemini::response::transform_non_stream,
            gemini_token_count,
        ),
    );
    registry.register(
        OpenAIChat,
        Claude,
        Some(openai_chat_to_claude::request::transform),
        response(
            openai_chat_to_claude::response::transform_stream,
            openai_chat_to_claude::response::transform_non_stream,
        ),
    );
    registry.register(
        OpenAIResponse,
        Claude,
        Some(openai_response_to_claude::request::transform),
        response(
            claude_to_openai_response::response::transform_stream,
            claude_to_openai_response::response::transform_non_stream,
        ),
    );

    // Codex target.
    registry.register(
        Claude,
        Codex,
        Some(claude_to_codex::request::transform),
        response_with_token_count(
            claude_to_codex::response::transform_stream,
            claude_to_codex::response::transform_non_stream,
            claude_token_count,
        ),
    );
    registry.register(
        Gemini,
        Codex,
        Some(gemini_to_codex::request::transform),
        response_with_token_count(
            gemini_to_codex::response::transform_stream,
            gemini_to_codex::response::transform_non_stream,
            gemini_token_count,
        ),
    );
    registry.register(
        OpenAIChat,
        Codex,
        Some(openai_chat_to_codex::request::transform),
        response(
            codex_to_openai_chat::response::transform_stream,
            codex_to_openai_chat::response::transform_non_stream,
        ),
    );
    registry.register(
        OpenAIResponse,
        Codex,
        Some(openai_response_to_codex::request::transform),
        response(
            openai_response_to_codex::response::transform_stream,
            openai_response_to_codex::response::transform_non_stream,
        ),
    );

    // Codex source (reverse pairs for response direction).
    registry.register(
        Codex,
        Claude,
        Some(codex_to_claude::request::transform),
        response(
            claude_to_codex::response::transform_stream,
            claude_to_codex::response::transform_non_stream,
        ),
    );
    registry.register(
        Codex,
        Gemini,
        Some(codex_to_gemini::request::transform),
        response(
            gemini_to_codex::response::transform_stream,
            gemini_to_codex::response::transform_non_stream,
        ),
    );
    registry.register(
        Codex,
        OpenAIChat,
        Some(codex_to_openai_chat::request::transform),
        response(
            openai_chat_to_codex::response::transform_stream,
            openai_chat_to_codex::response::transform_non_stream,
        ),
    );

    // Gemini target.
    registry.register(
        Claude,
        Gemini,
        Some(claude_to_gemini::request::transform),
        response_with_token_count(
            gemini_to_claude::response::transform_stream,
            gemini_to_claude::response::transform_non_stream,
            claude_token_count,
        ),
    );
    registry.register(
        Gemini,
        Gemini,
        Some(gemini_to_gemini::request::normalize),
        response_with_token_count(
            gemini_to_gemini::response::passthrough_stream,
            gemini_to_gemini::response::passthrough_non_stream,
            gemini_token_count,
        ),
    );
    registry.register(
        OpenAIChat,
        Gemini,
        Some(openai_chat_to_gemini::request::transform),
        response(
            gemini_to_openai_chat::response::transform_stream,
            gemini_to_openai_chat::response::transform_non_stream,
        ),
    );
    registry.register(
        OpenAIResponse,
        Gemini,
        Some(openai_response_to_gemini::request::transform),
        response(
            gemini_to_openai_response::response::transform_stream,
            gemini_to_openai_response::response::transform_non_stream,
        ),
    );

    // OpenAI Chat target.
    registry.register(
        Claude,
        OpenAIChat,
        Some(claude_to_openai_chat::request::transform),
        response_with_token_count(
            claude_to_openai_chat::response::transform_stream,
            claude_to_openai_chat::response::transform_non_stream,
            claude_token_count,
        ),
    );
    registry.register(
        Gemini,
        OpenAIChat,
        Some(gemini_to_openai_chat::request::transform),
        response_with_token_count(
            openai_chat_to_gemini::response::transform_stream,
            openai_chat_to_gemini::response::transform_non_stream,
            gemini_token_count,
        ),
    );
    registry.register(
        OpenAIChat,
        OpenAIChat,
        Some(openai_chat_to_openai_chat::request::normalize),
        response(
            openai_chat_to_openai_chat::response::passthrough_stream,
            openai_chat_to_openai_chat::response::passthrough_non_stream,
        ),
    );
    registry.register(
        OpenAIResponse,
        OpenAIChat,
        Some(openai_response_to_openai_chat::request::transform),
        response(
            openai_chat_to_openai_response::response::transform_stream,
            openai_chat_to_openai_response::response::transform_non_stream,
        ),
    );

    // Antigravity target.
    registry.register(
        Claude,
        Antigravity,
        Some(claude_to_antigravity::transform),
        response_with_token_count(
            antigravity_to_claude::response::transform_stream,
            antigravity_to_claude::response::transform_non_stream,
            claude_token_count,
        ),
    );
    registry.register(
        Gemini,
        Antigravity,
        Some(antigravity_to_gemini::request::wrap_gemini_request),
        response_with_token_count(
            antigravity_to_gemini::response::transform_stream,
            antigravity_to_gemini::response::transform_non_stream,
            gemini_token_count,
        ),
    );
    registry.register(
        OpenAIChat,
        Antigravity,
        Some(openai_chat_to_antigravity::request::transform),
        response(
            antigravity_to_openai_chat::response::transform_stream,
            antigravity_to_openai_chat::response::transform_non_stream,
        ),
    );
    registry.register(
        OpenAIResponse,
        Antigravity,
        Some(openai_response_to_antigravity::request::transform),
        response(
            openai_response_to_antigravity::response::transform_stream,
            openai_response_to_antigravity::response::transform_non_stream,
        ),
    );

    // OpenAI Responses target.
    registry.register(
        OpenAIChat,
        OpenAIResponse,
        Some(openai_chat_to_openai_response::request::transform),
        response(
            openai_response_to_openai_chat::response::transform_stream,
            openai_response_to_openai_chat::response::transform_non_stream,
        ),
    );
    registry.register(
        Gemini,
        OpenAIResponse,
        Some(gemini_to_openai_response::request::transform),
        response(
            openai_response_to_gemini::response::transform_stream,
            openai_response_to_gemini::response::transform_non_stream,
        ),
    );
    registry.register(
        Claude,
        OpenAIResponse,
        Some(claude_to_openai_response::request::transform),
        response(
            openai_response_to_claude::response::transform_stream,
            openai_response_to_claude::response::transform_non_stream,
        ),
    );
    registry.register(
        Codex,
        OpenAIResponse,
        Some(codex_to_openai_response::request::transform),
        Some(ResponseTransform::new(
            Some(codex_to_openai_response::response::transform_stream),
            Some(codex_to_openai_response::response::transform_non_stream),
        )),
    );
    registry.register(
        Antigravity,
        OpenAIResponse,
        Some(antigravity_to_openai_response::request::transform),
        Some(ResponseTransform::new(
            Some(antigravity_to_openai_response::response::transform_stream),
            Some(antigravity_to_openai_response::response::transform_non_stream),
        )),
    );
    registry.register(
        OpenAIResponse,
        OpenAIResponse,
        Some(openai_response_to_openai_response::request::normalize),
        response(
            openai_response_to_openai_response::response::passthrough_stream,
            openai_response_to_openai_response::response::passthrough_non_stream,
        ),
    );
    // Claude self-normalizer.
    registry.register(
        Claude,
        Claude,
        Some(claude_to_claude::request::normalize),
        response(
            claude_to_claude::response::passthrough_stream,
            claude_to_claude::response::passthrough_non_stream,
        ),
    );
}

#[cfg(test)]
mod registry_tests {
    use super::*;
    use crate::DEFAULT_REGISTRY;

    #[test]
    fn registers_current_builtin_request_pairs() {
        use Format::*;

        let pairs = [
            (Claude, Claude),
            (Gemini, Claude),
            (OpenAIChat, Claude),
            (OpenAIResponse, Claude),
            (Claude, Codex),
            (Gemini, Codex),
            (OpenAIChat, Codex),
            (OpenAIResponse, Codex),
            (Codex, Claude),
            (Codex, Gemini),
            (Codex, OpenAIChat),
            (Claude, Gemini),
            (Gemini, Gemini),
            (OpenAIChat, Gemini),
            (OpenAIResponse, Gemini),
            (Claude, OpenAIChat),
            (Gemini, OpenAIChat),
            (OpenAIChat, OpenAIChat),
            (OpenAIResponse, OpenAIChat),
            (Claude, Antigravity),
            (Gemini, Antigravity),
            (OpenAIChat, Antigravity),
            (OpenAIResponse, Antigravity),
            (OpenAIChat, OpenAIResponse),
            (Gemini, OpenAIResponse),
            (Claude, OpenAIResponse),
            (Codex, OpenAIResponse),
            (Antigravity, OpenAIResponse),
            (OpenAIResponse, OpenAIResponse),
        ];

        for (from, to) in pairs {
            assert!(
                DEFAULT_REGISTRY.has_request_transformer(from, to),
                "missing request transformer for {from:?} -> {to:?}"
            );
        }
    }
}
