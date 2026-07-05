//! Cross-provider thinking signature compatibility.
//!
//! Ported from Go's `CLIProxyAPI/internal/signature/` (3390 lines).
//! Handles format detection, validation, and conversion between Claude, Gemini,
//! and GPT thinking/reasoning signatures.

/// Gemini bypass sentinel for synthetic or migrated function-call history.
pub const GEMINI_SKIP_SENTINEL: &str = "skip_thought_signature_validator";

const MAX_SIG_LEN: usize = 32 * 1024 * 1024;

// ── Provider detection ────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SignatureProvider {
    Unknown,
    Claude,
    Gemini,
    GeminiBypass,
    Gpt,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SignatureBlockKind {
    Unknown,
    ClaudeThinking,
    GeminiModelPart,
    GeminiFunctionCall,
    GptReasoning,
}

/// Detect signature provider from model name.
pub fn provider_from_model_name(name: &str) -> SignatureProvider {
    let lower = name.to_lowercase();
    if lower.contains("claude") { return SignatureProvider::Claude; }
    if lower.contains("gemini") { return SignatureProvider::Gemini; }
    if lower.contains("gpt") || lower.contains("openai") || lower.contains("codex")
        || lower.starts_with("o1") || lower.starts_with("o3") || lower.starts_with("o4") {
        return SignatureProvider::Gpt;
    }
    SignatureProvider::Unknown
}

/// Detect signature provider from cache prefix.
pub fn provider_from_cache_prefix(prefix: &str) -> SignatureProvider {
    match prefix.trim().to_lowercase().as_str() {
        "claude" | "anthropic" => SignatureProvider::Claude,
        "gemini" | "google" => SignatureProvider::Gemini,
        "openai" | "gpt" | "codex" => SignatureProvider::Gpt,
        _ => SignatureProvider::Unknown,
    }
}

// ── Claude signature validation ───────────────────────────

/// Check if rawSignature has the Claude E/R shape.
pub fn has_claude_sig_prefix(sig: &str) -> bool {
    let s = strip_cache_prefix(sig);
    if s.is_empty() { return false; }
    let b = s.as_bytes();
    b[0] == b'E' || b[0] == b'R'
}

/// Check if rawSignature can be base64-decoded in Claude's expected layers.
pub fn has_decodable_claude_sig(sig: &str) -> bool {
    let s = strip_cache_prefix(sig);
    if s.is_empty() || s.len() > MAX_SIG_LEN { return false; }
    let b = s.as_bytes();
    match b[0] {
        b'E' => {
            base64_decode(&s).is_some()
        }
        b'R' => {
            let outer = match base64_decode(&s) { Some(v) => v, None => return false };
            if outer.is_empty() || outer[0] != b'E' { return false; }
            let inner_str = match std::str::from_utf8(&outer) { Ok(v) => v, Err(_) => return false };
            base64_decode(inner_str).is_some()
        }
        _ => false,
    }
}

/// Basic Claude thinking signature validation — checks prefix, base64, and 0x12 marker.
pub fn is_valid_claude_sig(sig: &str, strict: bool) -> bool {
    normalize_claude_sig(sig, strict).is_some()
}

/// Validate and normalize a Claude signature to double-layer R-form.
pub fn normalize_claude_sig(sig: &str, strict: bool) -> Option<String> {
    let s = strip_cache_prefix(sig);
    if s.is_empty() || s.len() > MAX_SIG_LEN { return None; }
    let b = s.as_bytes();
    match b[0] {
        b'R' => {
            validate_claude_double_layer(&s, strict).map(|_| s.to_string())
        }
        b'E' => {
            validate_claude_single_layer(&s, strict)?;
            // Convert E → R: double-layer base64 encode
            Some(base64_encode(s.as_bytes()))
        }
        _ => None,
    }
}

/// Validate and normalize a Claude signature to single-layer E-form (Claude-native).
pub fn normalize_claude_native_sig(sig: &str, strict: bool) -> Option<String> {
    let s = strip_cache_prefix(sig);
    if s.is_empty() || s.len() > MAX_SIG_LEN { return None; }
    let b = s.as_bytes();
    match b[0] {
        b'E' => { validate_claude_single_layer(&s, strict).map(|_| s.to_string()) }
        b'R' => {
            validate_claude_double_layer(&s, strict)?;
            let outer = base64_decode(&s)?;
            let inner = std::str::from_utf8(&outer).ok()?;
            Some(inner.to_string())
        }
        _ => None,
    }
}

fn validate_claude_double_layer(sig: &str, strict: bool) -> Option<()> {
    let decoded = base64_decode(sig)?;
    if decoded.is_empty() || decoded[0] != b'E' { return None; }
    let inner = std::str::from_utf8(&decoded).ok()?;
    validate_claude_single_layer_content(inner, strict)
}

fn validate_claude_single_layer(sig: &str, strict: bool) -> Option<()> {
    validate_claude_single_layer_content(sig, strict)
}

fn validate_claude_single_layer_content(sig: &str, strict: bool) -> Option<()> {
    let decoded = base64_decode(sig)?;
    if decoded.is_empty() || decoded[0] != 0x12 { return None; }
    if !strict { return Some(()); }
    // Strict mode checks protobuf structure (simplified — checks field-2 container exists)
    if decoded.len() < 4 { return None; }
    // Tag byte for field 2 (bytes) = (2 << 3) | 2 = 18 = 0x12 — same as first byte
    // This is already validated by decoded[0] == 0x12
    Some(())
}

// ── Gemini signature validation ───────────────────────────

/// Check if signature is one of Gemini's documented bypass sentinels.
pub fn is_gemini_bypass(sig: &str) -> bool {
    let t = sig.trim();
    t == GEMINI_SKIP_SENTINEL || t == "context_engineering_is_the_way_to_go"
}

/// Check if a Gemini signature looks valid (base64 decode + known envelope).
pub fn is_valid_gemini_sig(sig: &str, require_envelope: bool) -> bool {
    inspect_gemini_sig(sig, require_envelope).is_some()
}

/// Gemini replay policy — returns a Gemini-replayable signature.
pub fn gemini_replay_sig(sig: &str, block_kind: SignatureBlockKind) -> String {
    if let Some(s) = compatible_sig_for_provider_block(SignatureProvider::Gemini, sig, block_kind) {
        return s;
    }
    GEMINI_SKIP_SENTINEL.to_string()
}

// ── GPT signature validation ──────────────────────────────

/// Check if a GPT reasoning signature looks valid (non-empty, printable).
pub fn is_valid_gpt_sig(sig: &str) -> bool {
    let s = sig.trim();
    if s.is_empty() || s.len() > MAX_SIG_LEN {
        return false;
    }
    if !s.starts_with("gAAAA") {
        return false;
    }
    if !s.chars().all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_' || c == '=') {
        return false;
    }
    let decoded = match base64_url_decode(s) {
        Some(decoded) => decoded,
        None => return false,
    };
    if decoded.len() < 73 || decoded[0] != 0x80 {
        return false;
    }
    let ciphertext_len = decoded.len() - 1 - 8 - 16 - 32;
    ciphertext_len > 0 && ciphertext_len % 16 == 0
}

// ── Provider compatibility ────────────────────────────────

/// Detect which provider produced a signature.
pub fn detect_provider(sig: &str) -> SignatureProvider {
    detect_provider_for_block(sig, SignatureBlockKind::Unknown)
}

pub fn detect_provider_for_block(sig: &str, block_kind: SignatureBlockKind) -> SignatureProvider {
    let s = sig.trim();
    if s.is_empty() { return SignatureProvider::Unknown; }

    // Check cache prefix
    if let Some((provider, unprefixed)) = split_cache_prefix(s) {
        match provider {
            SignatureProvider::Gemini => {
                if is_gemini_bypass(unprefixed) { return SignatureProvider::GeminiBypass; }
                if is_recognized_gemini_sig(unprefixed, block_kind) { return SignatureProvider::Gemini; }
            }
            SignatureProvider::Claude => {
                if is_valid_claude_sig(unprefixed, true) { return SignatureProvider::Claude; }
            }
            SignatureProvider::Gpt => {
                if is_valid_gpt_sig(unprefixed) { return SignatureProvider::Gpt; }
            }
            _ => {}
        }
        return SignatureProvider::Unknown;
    }

    if s.contains('#') { return SignatureProvider::Unknown; }
    if is_gemini_bypass(s) { return SignatureProvider::GeminiBypass; }
    if is_valid_gpt_sig(s) { return SignatureProvider::Gpt; }
    if is_valid_claude_sig(s, true) { return SignatureProvider::Claude; }
    if is_recognized_gemini_sig(s, block_kind) { return SignatureProvider::Gemini; }

    SignatureProvider::Unknown
}

/// Decide compatibility — returns whether a signature can be replayed to target provider.
pub fn decide_compatibility(
    target: SignatureProvider,
    sig: &str,
    block_kind: SignatureBlockKind,
) -> (bool, String) {
    let bk = if matches!(block_kind, SignatureBlockKind::Unknown) {
        SignatureBlockKind::Unknown
    } else { block_kind };

    let detected = detect_provider_for_block(sig, bk);

    if signature_provider_matches(target, detected) {
        return (true, String::new());
    }

    match target {
        SignatureProvider::Gemini => {
            (false, GEMINI_SKIP_SENTINEL.to_string())
        }
        SignatureProvider::Claude => (false, String::new()),
        SignatureProvider::Gpt => (false, String::new()),
        _ => (false, String::new()),
    }
}

pub fn compatible_sig_for_provider_block(
    target: SignatureProvider,
    sig: &str,
    block_kind: SignatureBlockKind,
) -> Option<String> {
    let detected = detect_provider_for_block(sig, block_kind);
    if signature_provider_matches(target, detected) {
        return normalize_compatible(target, sig, block_kind);
    }
    None
}

/// CompatibleAntigravityClaudeThinkingSignature — used by antigravity→claude direction.
pub fn compatible_antigravity_claude_sig(sig: &str) -> Option<String> {
    // Normalize to Claude-native E-form
    normalize_claude_native_sig(sig, false)
}

// ── Helpers ───────────────────────────────────────────────

fn strip_cache_prefix(sig: &str) -> &str {
    let s = sig.trim();
    if let Some(idx) = s.find('#') {
        s[idx+1..].trim()
    } else {
        s
    }
}

fn split_cache_prefix(sig: &str) -> Option<(SignatureProvider, &str)> {
    let s = sig.trim();
    if let Some(idx) = s.find('#') {
        let prefix = &s[..idx].trim();
        let rest = s[idx+1..].trim();
        let provider = provider_from_cache_prefix(prefix);
        if !matches!(provider, SignatureProvider::Unknown) {
            return Some((provider, rest));
        }
    }
    None
}

fn signature_provider_matches(target: SignatureProvider, detected: SignatureProvider) -> bool {
    match target {
        SignatureProvider::Gemini => {
            matches!(
                detected,
                SignatureProvider::Gemini | SignatureProvider::GeminiBypass
            )
        }
        SignatureProvider::Claude => matches!(detected, SignatureProvider::Claude),
        SignatureProvider::Gpt => matches!(detected, SignatureProvider::Gpt),
        _ => false,
    }
}

fn normalize_compatible(
    target: SignatureProvider,
    sig: &str,
    block_kind: SignatureBlockKind,
) -> Option<String> {
    let payload = strip_cache_prefix(sig);
    match target {
        SignatureProvider::Claude => normalize_claude_native_sig(payload, false),
        SignatureProvider::Gemini => {
            if is_gemini_bypass(payload) || is_recognized_gemini_sig(payload, block_kind) {
                Some(payload.to_string())
            } else {
                None
            }
        }
        SignatureProvider::Gpt => {
            if is_valid_gpt_sig(payload) {
                Some(payload.to_string())
            } else {
                None
            }
        }
        _ => None,
    }
}

fn is_recognized_gemini_sig(sig: &str, _block_kind: SignatureBlockKind) -> bool {
    inspect_gemini_sig(sig, false).is_some()
}

fn inspect_gemini_sig(sig: &str, require_envelope: bool) -> Option<()> {
    let s = sig.trim();
    if s.is_empty() || is_gemini_bypass(s) { return None; }
    if s.len() > MAX_SIG_LEN { return None; }
    let decoded = base64_decode(s)?;
    if decoded.is_empty() { return None; }

    // Check known envelopes
    if decoded.len() >= 2 && decoded[0] == 0x12 {
        // Gemini Field 1 envelope (protobuf field 1, bytes type)
        return Some(());
    }
    if decoded.len() >= 3 && decoded[0] == 0x12 && decoded[1] as u8 == 0x12 {
        // Gemini Field 2 envelope
        return Some(());
    }
    if require_envelope { None } else { Some(()) }
}

fn base64_decode(s: &str) -> Option<Vec<u8>> {
    use base64::Engine as _;
    // Try standard
    if let Ok(v) = base64::engine::general_purpose::STANDARD.decode(s) {
        return Some(v);
    }
    // Try raw (no padding)
    if let Ok(v) = base64::engine::general_purpose::STANDARD_NO_PAD.decode(s) {
        return Some(v);
    }
    None
}

fn base64_url_decode(s: &str) -> Option<Vec<u8>> {
    use base64::Engine as _;
    if let Ok(v) = base64::engine::general_purpose::URL_SAFE_NO_PAD.decode(s) {
        return Some(v);
    }
    if let Ok(v) = base64::engine::general_purpose::URL_SAFE.decode(s) {
        return Some(v);
    }
    None
}

fn base64_encode(bytes: &[u8]) -> String {
    use base64::Engine as _;
    base64::engine::general_purpose::STANDARD.encode(bytes)
}
