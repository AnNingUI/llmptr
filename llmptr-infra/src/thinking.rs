//! Thinking configuration — unified reasoning config for all AI providers.
//!
//! Ported from Go's `CLIProxyAPI/internal/thinking/`.

use crate::registry::lookup_model_info;

/// Thinking mode: how to interpret the config.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThinkingMode {
    Budget,  // numeric token budget
    Level,   // discrete effort level (low/medium/high)
    None,    // disabled
    Auto,    // automatic/dynamic
}

/// Discrete thinking/effort level names.
pub mod levels {
    pub const NONE: &str = "none";
    pub const AUTO: &str = "auto";
    pub const MINIMAL: &str = "minimal";
    pub const LOW: &str = "low";
    pub const MEDIUM: &str = "medium";
    pub const HIGH: &str = "high";
    pub const XHIGH: &str = "xhigh";
    pub const MAX: &str = "max";
}

/// Unified thinking configuration.
#[derive(Debug, Clone)]
pub struct ThinkingConfig {
    pub mode: ThinkingMode,
    /// Token budget (only when mode is Budget). 0 = off, -1 = auto.
    pub budget: i64,
    /// Effort level string (only when mode is Level).
    pub level: String,
}

/// Convert a discrete effort level to a token budget.
/// Returns `(budget, ok)` where budget=0=off, budget=-1=auto.
pub fn convert_level_to_budget(level: &str) -> (i64, bool) {
    match level.trim().to_lowercase().as_str() {
        levels::NONE => (0, true),
        levels::AUTO => (-1, true),
        levels::MINIMAL => (512, true),
        levels::LOW => (1024, true),
        levels::MEDIUM => (8192, true),
        levels::HIGH => (24576, true),
        levels::XHIGH => (32768, true),
        levels::MAX => (128000, true),
        _ => (0, false),
    }
}

/// Convert a token budget to a discrete effort level.
/// Returns `(level, ok)`.
pub fn convert_budget_to_level(budget: i64) -> (&'static str, bool) {
    match budget {
        0 => (levels::NONE, true),
        -1 => (levels::AUTO, true),
        1..=512 => (levels::MINIMAL, true),
        513..=1024 => (levels::LOW, true),
        1025..=8192 => (levels::MEDIUM, true),
        8193..=24576 => (levels::HIGH, true),
        _ if budget > 24576 => (levels::XHIGH, true),
        _ => (levels::HIGH, false),
    }
}

/// Map a Claude-compatible effort level from an upstream/OAI/other level string.
/// Normalizes common aliases.
pub fn map_to_claude_effort(level: &str, supports_max: bool) -> (&str, bool) {
    match level.trim().to_lowercase().as_str() {
        levels::NONE => (levels::NONE, true),
        levels::AUTO => (levels::AUTO, true),
        levels::MINIMAL => (levels::LOW, true),
        levels::LOW => (levels::LOW, true),
        levels::MEDIUM => (levels::MEDIUM, true),
        levels::HIGH => (levels::HIGH, true),
        levels::XHIGH => (levels::XHIGH, true),
        levels::MAX if supports_max => (levels::MAX, true),
        levels::MAX => (levels::XHIGH, true),
        _ => ("", false),
    }
}

/// Check if a thinking level is supported by a model's ThinkingSupport.
pub fn has_level(levels_list: &[String], level: &str) -> bool {
    if !level.is_empty() {
        if levels_list.iter().any(|l| l == level) {
            return true;
        }
        // Also check common aliases
        let (mapped, ok) = map_to_claude_effort(level, true);
        if ok && !mapped.is_empty() && mapped != level {
            return levels_list.iter().any(|l| l == mapped);
        }
    }
    false
}

/// Apply thinking config to a Gemini-style request body.
/// This is a simplified version of the Go `thinking/provider/gemini` applier.
pub fn apply_gemini_thinking(body: serde_json::Value, config: &ThinkingConfig) -> serde_json::Value {
    use serde_json::json;
    let mut out = body;

    match config.mode {
        ThinkingMode::None => {
            out["generationConfig"]["thinkingConfig"] = json!({"thinkingLevel": "none", "includeThoughts": false});
        }
        ThinkingMode::Auto => {
            out["generationConfig"]["thinkingConfig"] = json!({"includeThoughts": true});
        }
        ThinkingMode::Budget => {
            if config.budget > 0 {
                out["generationConfig"]["thinkingConfig"] = json!({
                    "thinkingBudget": config.budget,
                    "includeThoughts": true,
                });
            } else {
                out["generationConfig"]["thinkingConfig"] = json!({"thinkingLevel": "none", "includeThoughts": false});
            }
        }
        ThinkingMode::Level => {
            if config.level == levels::NONE {
                out["generationConfig"]["thinkingConfig"] = json!({"thinkingLevel": "none", "includeThoughts": false});
            } else {
                out["generationConfig"]["thinkingConfig"] = json!({
                    "thinkingLevel": config.level,
                    "includeThoughts": true,
                });
            }
        }
    }

    out
}

/// Apply thinking config to a Claude-style request body (simplified).
pub fn apply_claude_thinking(body: serde_json::Value, config: &ThinkingConfig, model_id: &str) -> serde_json::Value {
    use serde_json::json;
    let mut out = body;
    let model_info = lookup_model_info(model_id);

    let supports_adaptive = model_info.as_ref()
        .and_then(|m| m.thinking.as_ref())
        .map(|t| !t.levels.is_empty())
        .unwrap_or(false);
    let supports_max = supports_adaptive && has_level(
        model_info.as_ref().and_then(|m| m.thinking.as_ref()).map(|t| &t.levels).unwrap_or(&vec![]),
        levels::MAX,
    );

    match config.mode {
        ThinkingMode::None => {
            out["thinking"] = json!({"type": "disabled"});
            if supports_adaptive {
                out["output_config"]["effort"] = json!(levels::NONE);
            }
        }
        ThinkingMode::Auto => {
            if supports_adaptive {
                out["thinking"] = json!({"type": "adaptive"});
            } else {
                out["thinking"] = json!({"type": "enabled"});
            }
        }
        ThinkingMode::Budget => {
            let budget = config.budget;
            if budget <= 0 {
                out["thinking"] = json!({"type": "disabled"});
                return out;
            }
            if supports_adaptive {
                let (level, _) = convert_budget_to_level(budget);
                let mapped = map_to_claude_effort(level, supports_max).0.to_string();
                out["thinking"] = json!({"type": "adaptive"});
                out["output_config"]["effort"] = json!(mapped);
            } else {
                out["thinking"] = json!({"type": "enabled", "budget_tokens": budget});
            }
        }
        ThinkingMode::Level => {
            let level = &config.level;
            if level == levels::NONE {
                out["thinking"] = json!({"type": "disabled"});
                return out;
            }
            if supports_adaptive {
                let mapped = map_to_claude_effort(level, supports_max).0.to_string();
                out["thinking"] = json!({"type": "adaptive"});
                out["output_config"]["effort"] = json!(mapped);
            } else {
                let (budget, ok) = convert_level_to_budget(level);
                if ok && budget > 0 {
                    out["thinking"] = json!({"type": "enabled", "budget_tokens": budget});
                } else if ok && budget == -1 {
                    out["thinking"] = json!({"type": "enabled"});
                }
            }
        }
    }

    out
}
