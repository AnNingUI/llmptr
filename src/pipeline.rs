use std::sync::Arc;

use crate::format::Format;
use crate::registry::Registry;

/// Middleware type for request transformation.
pub type RequestMiddleware =
    Box<dyn Fn(&PipelineCtx, PipelineRequest, RequestHandler) -> PipelineRequest + Send + Sync>;

/// Middleware type for response transformation.
pub type ResponseMiddleware =
    Box<dyn Fn(&PipelineCtx, PipelineResponse, ResponseHandler) -> PipelineResponse + Send + Sync>;

/// Handler that performs the core request translation.
pub type RequestHandler =
    Box<dyn Fn(&PipelineCtx, PipelineRequest) -> PipelineRequest + Send + Sync>;

/// Handler that performs the core response translation.
pub type ResponseHandler =
    Box<dyn Fn(&PipelineCtx, PipelineResponse) -> PipelineResponse + Send + Sync>;

/// Context passed through the pipeline.
#[derive(Debug, Clone)]
pub struct PipelineCtx {
    pub from: Format,
    pub to: Format,
    pub model: String,
    pub stream: bool,
}

/// A request envelope in the pipeline.
#[derive(Debug, Clone)]
pub struct PipelineRequest {
    pub body: serde_json::Value,
    pub format: Format,
}

/// A response envelope in the pipeline.
#[derive(Debug, Clone)]
pub struct PipelineResponse {
    pub body: serde_json::Value,
    pub format: Format,
    /// True if this is a streaming chunk.
    pub is_stream: bool,
}

/// A pipeline chains request/response middleware around a registry lookup.
pub struct Pipeline {
    registry: Arc<Registry>,
    request_middleware: Vec<RequestMiddleware>,
    response_middleware: Vec<ResponseMiddleware>,
}

impl Pipeline {
    /// Create a new pipeline bound to the given registry.
    pub fn new(registry: Registry) -> Self {
        Self {
            registry: Arc::new(registry),
            request_middleware: Vec::new(),
            response_middleware: Vec::new(),
        }
    }

    /// Create a new pipeline from an already-shared registry.
    pub fn from_arc(registry: Arc<Registry>) -> Self {
        Self {
            registry,
            request_middleware: Vec::new(),
            response_middleware: Vec::new(),
        }
    }

    /// Add a request middleware (executed in registration order).
    pub fn use_request(&mut self, mw: RequestMiddleware) {
        self.request_middleware.push(mw);
    }

    /// Add a response middleware (executed in registration order).
    pub fn use_response(&mut self, mw: ResponseMiddleware) {
        self.response_middleware.push(mw);
    }

    /// Translate a request through the pipeline (without middleware chaining).
    ///
    /// Middleware chaining requires a more involved ownership pattern. For now this
    /// delegates directly to the registry, which matches the common-path usage.
    pub fn translate_request(
        &self,
        from: Format,
        to: Format,
        model: &str,
        body: serde_json::Value,
        stream: bool,
    ) -> serde_json::Value {
        self.registry
            .translate_request(from, to, model, body, stream)
    }

    /// Translate a response through the pipeline.
    #[allow(clippy::too_many_arguments)]
    pub fn translate_response(
        &self,
        from: Format,
        to: Format,
        model: &str,
        original_request: &serde_json::Value,
        translated_request: &serde_json::Value,
        response: serde_json::Value,
        stream: bool,
        param: Option<&mut Box<dyn std::any::Any>>,
    ) -> serde_json::Value {
        if stream {
            let chunks = self.registry.translate_stream(
                from,
                to,
                model,
                original_request,
                translated_request,
                response,
                param,
            );
            chunks.into_iter().next().unwrap_or(serde_json::Value::Null)
        } else {
            self.registry.translate_non_stream(
                from,
                to,
                model,
                original_request,
                translated_request,
                response,
                param,
            )
        }
    }
}
