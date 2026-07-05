/// Error types for the translator matrix.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// Unsupported format pair: no translator registered.
    #[error("no translator registered for {from} -> {to}")]
    NoTranslator { from: String, to: String },

    /// JSON serialization/deserialization error.
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    /// Invalid model name format.
    #[error("invalid model name: {0}")]
    InvalidModel(String),

    /// Missing required field in payload.
    #[error("missing required field: {0}")]
    MissingField(String),

    /// Unsupported thinking mode for provider.
    #[error("unsupported thinking mode '{mode}' for provider {provider}")]
    UnsupportedThinking { provider: String, mode: String },

    /// Internal transform error.
    #[error("transform error: {0}")]
    Transform(String),
}
