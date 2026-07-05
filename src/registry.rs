use std::collections::HashMap;
use std::sync::RwLock;

use crate::format::Format;

/// Function that transforms a request payload from one format to another.
///
/// Arguments: (model_name, raw_json_body, is_stream). Returns transformed body.
pub type RequestTransform =
    fn(model: &str, body: serde_json::Value, stream: bool) -> serde_json::Value;

/// Function that transforms a streaming response chunk.
///
/// Returns zero or more transformed chunks (some providers emit multiple events per input).
pub type StreamResponseTransform = fn(
    model: &str,
    original_request: &serde_json::Value,
    translated_request: &serde_json::Value,
    chunk: serde_json::Value,
    param: Option<&mut Box<dyn std::any::Any>>,
) -> Vec<serde_json::Value>;

/// Function that transforms a non-streaming response.
pub type NonStreamResponseTransform = fn(
    model: &str,
    original_request: &serde_json::Value,
    translated_request: &serde_json::Value,
    response: serde_json::Value,
    param: Option<&mut Box<dyn std::any::Any>>,
) -> serde_json::Value;

/// Function that transforms a token count response.
pub type ResponseTokenCountTransform = fn(count: i64) -> serde_json::Value;

/// A pair of response transforms (streaming + non-streaming).
#[derive(Clone)]
pub struct ResponseTransform {
    pub stream: Option<StreamResponseTransform>,
    pub non_stream: Option<NonStreamResponseTransform>,
    pub token_count: Option<ResponseTokenCountTransform>,
}

impl ResponseTransform {
    pub fn new(
        stream: Option<StreamResponseTransform>,
        non_stream: Option<NonStreamResponseTransform>,
    ) -> Self {
        Self {
            stream,
            non_stream,
            token_count: None,
        }
    }

    pub fn with_token_count(mut self, token_count: Option<ResponseTokenCountTransform>) -> Self {
        self.token_count = token_count;
        self
    }

    pub fn has_any(&self) -> bool {
        self.stream.is_some() || self.non_stream.is_some() || self.token_count.is_some()
    }
}

/// Central translation registry mapping `(from, to)` pairs to transform functions.
///
/// Thread-safe: reads/writes are synchronized via `RwLock`.
pub struct Registry {
    requests: RwLock<HashMap<Format, HashMap<Format, RequestTransform>>>,
    responses: RwLock<HashMap<Format, HashMap<Format, ResponseTransform>>>,
}

impl Registry {
    /// Create an empty registry.
    pub fn new() -> Self {
        Self {
            requests: RwLock::new(HashMap::new()),
            responses: RwLock::new(HashMap::new()),
        }
    }

    /// Register request and/or response transforms for a `(from, to)` pair.
    ///
    /// Setting `request` to `None` registers only response transforms, and vice versa.
    pub fn register(
        &self,
        from: Format,
        to: Format,
        request: Option<RequestTransform>,
        response: Option<ResponseTransform>,
    ) {
        if let Some(f) = request {
            self.requests
                .write()
                .unwrap()
                .entry(from)
                .or_default()
                .insert(to, f);
        }
        if let Some(r) = response {
            // Response translators are indexed by (to, from) because
            // translate_non_stream / translate_stream look up via
            // responses[to][from] — matching Go's convention.
            self.responses
                .write()
                .unwrap()
                .entry(to)
                .or_default()
                .insert(from, r);
        }
    }

    /// Translate a request payload.
    ///
    /// If no translator is registered, the body is returned with the model field updated.
    pub fn translate_request(
        &self,
        from: Format,
        to: Format,
        model: &str,
        body: serde_json::Value,
        stream: bool,
    ) -> serde_json::Value {
        let fn_opt = self
            .requests
            .read()
            .unwrap()
            .get(&from)
            .and_then(|by_target| by_target.get(&to))
            .copied();

        match fn_opt {
            Some(f) => f(model, body, stream),
            None => normalize_model_field(body, model),
        }
    }

    /// Translate a non-streaming response.
    #[allow(clippy::too_many_arguments)]
    pub fn translate_non_stream(
        &self,
        from: Format,
        to: Format,
        model: &str,
        original_request: &serde_json::Value,
        translated_request: &serde_json::Value,
        response: serde_json::Value,
        param: Option<&mut Box<dyn std::any::Any>>,
    ) -> serde_json::Value {
        // Note: the Go code uses `to` as the _source_ for response lookup:
        //   responses[to][from].NonStream(body)
        let fn_opt = self
            .responses
            .read()
            .unwrap()
            .get(&to)
            .and_then(|sources| sources.get(&from))
            .and_then(|rt| rt.non_stream);

        match fn_opt {
            Some(f) => f(model, original_request, translated_request, response, param),
            None => response,
        }
    }

    /// Translate a streaming response chunk. Returns zero or more transformed chunks.
    #[allow(clippy::too_many_arguments)]
    pub fn translate_stream(
        &self,
        from: Format,
        to: Format,
        model: &str,
        original_request: &serde_json::Value,
        translated_request: &serde_json::Value,
        chunk: serde_json::Value,
        param: Option<&mut Box<dyn std::any::Any>>,
    ) -> Vec<serde_json::Value> {
        // responses[to][from].Stream(body)
        let fn_opt = self
            .responses
            .read()
            .unwrap()
            .get(&to)
            .and_then(|sources| sources.get(&from))
            .and_then(|rt| rt.stream);

        match fn_opt {
            Some(f) => f(model, original_request, translated_request, chunk, param),
            None => vec![chunk],
        }
    }

    /// Translate a token count response.
    pub fn translate_token_count(
        &self,
        from: Format,
        to: Format,
        count: i64,
        raw: serde_json::Value,
    ) -> serde_json::Value {
        let fn_opt = self
            .responses
            .read()
            .unwrap()
            .get(&to)
            .and_then(|sources| sources.get(&from))
            .and_then(|rt| rt.token_count);

        match fn_opt {
            Some(f) => f(count),
            None => raw,
        }
    }

    /// Check if a request translator exists for the given pair.
    pub fn has_request_transformer(&self, from: Format, to: Format) -> bool {
        self.requests
            .read()
            .unwrap()
            .get(&from)
            .and_then(|m| m.get(&to))
            .is_some()
    }

    /// Check if a response translator (streaming or non-streaming) exists.
    ///
    /// Uses `responses[to][from]` to match the lookup convention of
    /// `translate_non_stream`/`translate_stream`/`translate_token_count`.
    pub fn has_response_transformer(&self, from: Format, to: Format) -> bool {
        self.responses
            .read()
            .unwrap()
            .get(&to)
            .and_then(|m| m.get(&from))
            .map(ResponseTransform::has_any)
            .unwrap_or(false)
    }

    /// Check if a streaming response translator exists for the registered pair.
    pub fn has_stream_response_transformer(&self, from: Format, to: Format) -> bool {
        self.responses
            .read()
            .unwrap()
            .get(&to)
            .and_then(|m| m.get(&from))
            .map(|rt| rt.stream.is_some())
            .unwrap_or(false)
    }

    /// Check if a non-streaming response translator exists for the registered pair.
    pub fn has_non_stream_response_transformer(&self, from: Format, to: Format) -> bool {
        self.responses
            .read()
            .unwrap()
            .get(&to)
            .and_then(|m| m.get(&from))
            .map(|rt| rt.non_stream.is_some())
            .unwrap_or(false)
    }
}

impl Default for Registry {
    fn default() -> Self {
        Self::new()
    }
}

fn normalize_model_field(mut body: serde_json::Value, model: &str) -> serde_json::Value {
    if model.is_empty() {
        return body;
    }

    let current = body.get("model").and_then(|v| v.as_str());
    if current == Some(model) {
        return body;
    }

    if let Some(obj) = body.as_object_mut() {
        obj.insert(
            "model".to_string(),
            serde_json::Value::String(model.to_string()),
        );
    }
    body
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn passthrough_stream(
        _model: &str,
        _original_request: &serde_json::Value,
        _translated_request: &serde_json::Value,
        chunk: serde_json::Value,
        _param: Option<&mut Box<dyn std::any::Any>>,
    ) -> Vec<serde_json::Value> {
        vec![chunk]
    }

    fn passthrough_non_stream(
        _model: &str,
        _original_request: &serde_json::Value,
        _translated_request: &serde_json::Value,
        response: serde_json::Value,
        _param: Option<&mut Box<dyn std::any::Any>>,
    ) -> serde_json::Value {
        response
    }

    fn token_count(count: i64) -> serde_json::Value {
        json!({"count": count})
    }

    #[test]
    fn fallback_normalizes_model_field() {
        let registry = Registry::new();
        let body = json!({"model":"prefixed/gpt-5","input":"ping"});

        let result = registry.translate_request(
            Format::OpenAIResponse,
            Format::OpenAIResponse,
            "gpt-5",
            body,
            false,
        );

        assert_eq!(result["model"], "gpt-5");
        assert_eq!(result["input"], "ping");
    }

    #[test]
    fn response_detection_uses_registration_direction() {
        let registry = Registry::new();
        registry.register(
            Format::OpenAIChat,
            Format::Claude,
            None,
            Some(ResponseTransform::new(
                Some(passthrough_stream),
                Some(passthrough_non_stream),
            )),
        );

        assert!(registry.has_response_transformer(Format::OpenAIChat, Format::Claude));
        assert!(registry.has_stream_response_transformer(Format::OpenAIChat, Format::Claude));
        assert!(registry.has_non_stream_response_transformer(Format::OpenAIChat, Format::Claude));
        assert!(!registry.has_response_transformer(Format::Claude, Format::OpenAIChat));
    }

    #[test]
    fn token_count_lookup_uses_response_execution_direction() {
        let registry = Registry::new();
        registry.register(
            Format::OpenAIChat,
            Format::Claude,
            None,
            Some(ResponseTransform::new(None, None).with_token_count(Some(token_count))),
        );

        let result = registry.translate_token_count(
            Format::OpenAIChat,
            Format::Claude,
            42,
            json!({"raw": true}),
        );

        assert_eq!(result, json!({"count": 42}));
        assert!(registry.has_response_transformer(Format::OpenAIChat, Format::Claude));
    }
}
