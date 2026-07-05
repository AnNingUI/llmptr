//! Registry — model information and lookup utilities.
//!
//! Provides static model definitions and a global model lookup that combines
//! static definitions with a dynamic Lazy registry. Ported from Go's
//! `CLIProxyAPI/internal/registry/`.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::RwLock;

// ── Model types ────────────────────────────────────────────

/// Information about an available model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    pub id: String,
    #[serde(default)]
    pub object: String,
    #[serde(default)]
    pub created: i64,
    #[serde(default)]
    pub owned_by: String,
    #[serde(default)]
    pub r#type: String,
    #[serde(default)]
    pub display_name: String,
    #[serde(default)]
    pub name: String,
    #[serde(default)]
    pub version: String,
    #[serde(default)]
    pub description: String,
    #[serde(default)]
    pub input_token_limit: i64,
    #[serde(default)]
    pub output_token_limit: i64,
    #[serde(default)]
    pub supported_generation_methods: Vec<String>,
    #[serde(default)]
    pub context_length: i64,
    #[serde(default)]
    pub max_completion_tokens: i64,
    #[serde(default)]
    pub supported_parameters: Vec<String>,
    #[serde(default)]
    pub supported_input_modalities: Vec<String>,
    #[serde(default)]
    pub supported_output_modalities: Vec<String>,
    #[serde(default)]
    pub supports_web_search: bool,
    #[serde(default)]
    pub thinking: Option<ThinkingSupport>,
}

/// Reasoning/thinking budget capabilities for a model family.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThinkingSupport {
    #[serde(default)]
    pub min: i64,
    #[serde(default)]
    pub max: i64,
    #[serde(default)]
    pub zero_allowed: bool,
    #[serde(default)]
    pub dynamic_allowed: bool,
    #[serde(default)]
    pub levels: Vec<String>,
}

// ── Global registry ────────────────────────────────────────

type ModelMap = HashMap<String, ModelInfo>;

static GLOBAL_REGISTRY: once_cell::sync::Lazy<RwLock<ModelMap>> =
    once_cell::sync::Lazy::new(|| RwLock::new(HashMap::new()));

/// Register a model dynamically (e.g., from client discovery).
pub fn register_model(info: ModelInfo) {
    let id = info.id.clone();
    if let Ok(mut registry) = GLOBAL_REGISTRY.write() {
        registry.insert(id, info);
    }
}

/// Register a batch of models.
pub fn register_models(models: Vec<ModelInfo>) {
    if let Ok(mut registry) = GLOBAL_REGISTRY.write() {
        for info in models {
            registry.insert(info.id.clone(), info);
        }
    }
}

/// Look up a model by ID, searching dynamic registry first, then static definitions.
pub fn lookup_model_info(model_id: &str) -> Option<ModelInfo> {
    if model_id.is_empty() {
        return None;
    }

    // Dynamic registry first
    if let Ok(registry) = GLOBAL_REGISTRY.read() {
        if let Some(info) = registry.get(model_id) {
            return Some(info.clone());
        }
    }

    // Static definitions fallback
    lookup_static_model_info(model_id)
}

/// Look up a model from static definitions only.
pub fn lookup_static_model_info(model_id: &str) -> Option<ModelInfo> {
    if model_id.is_empty() {
        return None;
    }
    for model in STATIC_MODELS.iter() {
        if model.id == model_id {
            return Some(model.clone());
        }
    }
    None
}

// ── Static model definitions ───────────────────────────────

fn claude_model(id: &str, display: &str, owned_by: &str, ctx: i64, max_out: i64, thinking: Option<ThinkingSupport>) -> ModelInfo {
    ModelInfo {
        id: id.to_string(),
        object: "model".to_string(),
        created: 1704067200,
        owned_by: owned_by.to_string(),
        r#type: "claude".to_string(),
        display_name: display.to_string(),
        context_length: ctx,
        max_completion_tokens: max_out,
        thinking,
        ..Default::default()
    }
}

fn gemini_model(id: &str, display: &str, owned_by: &str, input_limit: i64, output_limit: i64, thinking: Option<ThinkingSupport>) -> ModelInfo {
    ModelInfo {
        id: id.to_string(),
        object: "model".to_string(),
        owned_by: owned_by.to_string(),
        r#type: "gemini".to_string(),
        display_name: display.to_string(),
        input_token_limit: input_limit,
        output_token_limit: output_limit,
        thinking,
        ..Default::default()
    }
}

static STATIC_MODELS: once_cell::sync::Lazy<Vec<ModelInfo>> = once_cell::sync::Lazy::new(|| {
    vec![
        claude_model("claude-sonnet-4-20250514", "Claude Sonnet 4", "anthropic", 200_000, 64_000,
            Some(ThinkingSupport { min: 1024, max: 64_000, zero_allowed: true, dynamic_allowed: true, levels: vec![] })),
        claude_model("claude-3-5-sonnet-20241022", "Claude 3.5 Sonnet", "anthropic", 200_000, 8_192,
            Some(ThinkingSupport { min: 1024, max: 8_192, zero_allowed: true, dynamic_allowed: true, levels: vec![] })),
        claude_model("claude-opus-4-20250514", "Claude Opus 4", "anthropic", 200_000, 64_000,
            Some(ThinkingSupport { min: 1024, max: 64_000, zero_allowed: true, dynamic_allowed: true, levels: vec!["none".to_string(), "low".to_string(), "medium".to_string(), "high".to_string(), "xhigh".to_string(), "max".to_string()] })),
        claude_model("claude-haiku-3-5-20241022", "Claude 3.5 Haiku", "anthropic", 200_000, 8_192, None),
        gemini_model("gemini-2.0-flash", "Gemini 2.0 Flash", "google", 1_048_576, 8_192,
            Some(ThinkingSupport { min: 0, max: 0, zero_allowed: true, dynamic_allowed: true, levels: vec![] })),
        gemini_model("gemini-2.0-pro", "Gemini 2.0 Pro", "google", 2_097_152, 8_192,
            Some(ThinkingSupport { min: 0, max: 0, zero_allowed: true, dynamic_allowed: true, levels: vec![] })),
        gemini_model("gemini-2.5-flash", "Gemini 2.5 Flash", "google", 1_048_576, 64_000,
            Some(ThinkingSupport { min: 0, max: 64_000, zero_allowed: true, dynamic_allowed: true, levels: vec!["none".to_string(), "low".to_string(), "medium".to_string(), "high".to_string()] })),
        gemini_model("gemini-2.5-pro", "Gemini 2.5 Pro", "google", 2_097_152, 64_000,
            Some(ThinkingSupport { min: 0, max: 64_000, zero_allowed: true, dynamic_allowed: true, levels: vec!["none".to_string(), "low".to_string(), "medium".to_string(), "high".to_string()] })),
    ]
});

/// Get all static models.
pub fn static_models() -> &'static [ModelInfo] {
    &STATIC_MODELS
}

impl Default for ModelInfo {
    fn default() -> Self {
        Self {
            id: String::new(),
            object: "model".to_string(),
            created: 0,
            owned_by: String::new(),
            r#type: String::new(),
            display_name: String::new(),
            name: String::new(),
            version: String::new(),
            description: String::new(),
            input_token_limit: 0,
            output_token_limit: 0,
            supported_generation_methods: vec![],
            context_length: 0,
            max_completion_tokens: 0,
            supported_parameters: vec![],
            supported_input_modalities: vec![],
            supported_output_modalities: vec![],
            supports_web_search: false,
            thinking: None,
        }
    }
}
