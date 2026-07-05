/// Protocol format identifiers for the translation matrix.
///
/// Each variant identifies an AI API protocol dialect.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Format {
    /// OpenAI **Chat Completions** API (`/v1/chat/completions`)
    OpenAIChat,
    /// OpenAI **Responses** API (`/v1/responses`) — Codex CLI native format
    OpenAIResponse,
    /// Anthropic Claude **Messages** API (`/v1/messages`)
    Claude,
    /// Google Gemini API (`/v1beta/models`)
    Gemini,
    /// OpenAI Codex (subscription-based GPT access via reverse-engineered auth)
    Codex,
    /// Antigravity (Gemini AI Studio via WebSocket)
    Antigravity,
    /// Moonshot Kimi API
    Kimi,
    /// xAI / Grok API
    XAI,
}

impl std::str::FromStr for Format {
    type Err = ();

    /// Parse a format from its string identifier.
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().replace('-', "_").as_str() {
            "openai" | "openai_chat" | "chat_completions" | "openai_chat_completions" => {
                Ok(Self::OpenAIChat)
            }
            "openai_response" | "openai_responses" => Ok(Self::OpenAIResponse),
            "claude" | "anthropic" => Ok(Self::Claude),
            "gemini" | "google" => Ok(Self::Gemini),
            "codex" => Ok(Self::Codex),
            "antigravity" | "aistudio" => Ok(Self::Antigravity),
            "kimi" => Ok(Self::Kimi),
            "xai" | "grok" => Ok(Self::XAI),
            _ => Err(()),
        }
    }
}

impl Format {
    /// Return the canonical string representation.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::OpenAIChat => "openai_chat",
            Self::OpenAIResponse => "openai_response",
            Self::Claude => "claude",
            Self::Gemini => "gemini",
            Self::Codex => "codex",
            Self::Antigravity => "antigravity",
            Self::Kimi => "kimi",
            Self::XAI => "xai",
        }
    }
}

impl std::fmt::Display for Format {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::str::FromStr;

    #[test]
    fn parses_cliproxyapi_format_identifiers() {
        let cases = [
            ("openai", Format::OpenAIChat),
            ("openai_chat", Format::OpenAIChat),
            ("chat-completions", Format::OpenAIChat),
            ("openai-chat-completions", Format::OpenAIChat),
            ("openai-response", Format::OpenAIResponse),
            ("openai_response", Format::OpenAIResponse),
            ("openai-responses", Format::OpenAIResponse),
            ("claude", Format::Claude),
            ("anthropic", Format::Claude),
            ("gemini", Format::Gemini),
            ("google", Format::Gemini),
            ("codex", Format::Codex),
            ("antigravity", Format::Antigravity),
            ("aistudio", Format::Antigravity),
        ];

        for (raw, expected) in cases {
            assert_eq!(Format::from_str(raw), Ok(expected), "format alias {raw}");
        }
    }

    #[test]
    fn display_uses_stable_canonical_names() {
        assert_eq!(Format::OpenAIChat.to_string(), "openai_chat");
        assert_eq!(Format::OpenAIResponse.to_string(), "openai_response");
        assert_eq!(Format::Claude.to_string(), "claude");
        assert_eq!(Format::Gemini.to_string(), "gemini");
        assert_eq!(Format::Codex.to_string(), "codex");
        assert_eq!(Format::Antigravity.to_string(), "antigravity");
    }
}
