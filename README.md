# llmptr

[![Crates.io](https://img.shields.io/crates/v/llmptr)](https://crates.io/crates/llmptr)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

**AI API protocol translation matrix** — convert requests and responses between 8+ AI API formats.

## Formats

| Format           | Identifier          | Protocol                                                  |
| ---------------- | ------------------- | --------------------------------------------------------- |
| `OpenAIChat`     | `"openai_chat"`     | OpenAI Chat Completions (`/v1/chat/completions`)          |
| `OpenAIResponse` | `"openai_response"` | OpenAI Responses API (`/v1/responses`) — Codex CLI native |
| `Claude`         | `"claude"`          | Anthropic Claude Messages API (`/v1/messages`)            |
| `Gemini`         | `"gemini"`          | Google Gemini API (`/v1beta/models`)                      |
| `Codex`          | `"codex"`           | OpenAI Codex (subscription-based, Responses-aligned)      |
| `Antigravity`    | `"antigravity"`     | Gemini AI Studio via WebSocket (wraps Gemini)             |
| `Kimi`           | `"kimi"`            | Moonshot Kimi API                                         |
| `XAI`            | `"xai"`             | xAI / Grok API                                            |

## Translation Matrix (29 pairs)

Every `from → to` pair has **bidirectional** request + response + token-count translation.

| Source ↓ / Target → | Claude | Gemini | OpenAIChat | OpenAIResponse |
| ------------------- | ------ | ------ | ---------- | -------------- |
| **Claude**          | ✅ self | ✅      | ✅          | ✅              |
| **Gemini**          | ✅      | ✅ self | ✅          | ✅              |
| **OpenAIChat**      | ✅      | ✅      | ✅ self     | ✅              |
| **OpenAIResponse**  | ✅      | ✅      | ✅          | ✅ self         |

Plus full coverage for **Codex** and **Antigravity** shells as both source and target.

## Quick Start

```toml
[dependencies]
llmptr = "0.1"
```

```rust
use llmptr::{Format, translate_request, translate_non_stream};

// Translate a Claude request to Gemini format
let claude_body = serde_json::json!({
    "model": "claude-sonnet-4-20250514",
    "messages": [{"role": "user", "content": "Hello"}],
    "max_tokens": 1024
});

let gemini_body = translate_request(
    Format::Claude,
    Format::Gemini,
    "gemini-2.0-flash",
    claude_body,
    false, // non-streaming
);

// Later, translate the Gemini response back to Claude format
let gemini_response = serde_json::json!({
    "candidates": [{"content": {"parts": [{"text": "Hi there!"}]}}]
});

let claude_response = translate_non_stream(
    Format::Gemini,
    Format::Claude,
    "claude-sonnet-4-20250514",
    &original_request,
    &gemini_body, // translated request
    gemini_response,
    None, // optional state param
);
```

## Architecture

```
┌──────────┐    translate_request(from, to)     ┌──────────┐
│  Client  │ ─────────────────────────────────> │  Server  │
│ (Format) │ <───────────────────────────────── │ (Format) │
└──────────┘   translate_non_stream(from, to)   └──────────┘
                    ↑ responses[to][from]
```

- **`Registry`** — thread-safe `(from, to)` → transform function map
- **`Pipeline`** — middleware chain wrapping registry lookups
- **29 built-in translator modules** — one per pair, each with `request.rs` + `response.rs`

## Using a Custom Registry

```rust
use llmptr::{Format, Registry};

let registry = Registry::new();

// Register a custom pair
registry.register(
    Format::OpenAIChat,
    Format::Claude,
    Some(my_request_transform),
    Some(my_response_transform),
);

let body = registry.translate_request(
    Format::OpenAIChat,
    Format::Claude,
    "claude-sonnet-4-20250514",
    my_body,
    false,
);
```

## Features

- **Request translation**: model name normalization, role mapping, tool/function conversion, content restructuring
- **Streaming response**: full SSE state machines (Codex, Claude, Gemini, OpenAI events)
- **Non-streaming response**: aggregated batch conversion
- **Token count translation**: format-specific token response shaping
- **Passthrough fallback**: unregistered pairs pass through with model field normalization
- **Thread-safe**: `RwLock`-protected registry

## License

MIT

## Related Projects

- [CLIProxyAPI](https://github.com/router-for-me/CLIProxyAPI) — Go reference implementation
- Part of the **ruston** monorepo — AI proxy infrastructure in Rust
