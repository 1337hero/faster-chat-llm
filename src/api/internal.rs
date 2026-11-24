//! Internal request type that all API adapters translate to.

use serde_json::{Map, Value};
use crate::sampler::SampleParams;

/// Thinking/reasoning mode configuration.
#[derive(Clone, Copy, Debug)]
pub enum Thinking {
    /// Thinking enabled, extract <think> blocks.
    Enabled,
    /// Thinking disabled.
    Disabled,
}

impl Thinking {
    pub fn is_enabled(&self) -> bool {
        matches!(self, Thinking::Enabled)
    }
}

/// A generation request, normalized from any API format.
#[derive(Debug)]
pub struct GenRequest {
    /// Rendered prompt text (after chat template or raw for completions).
    pub prompt_text: String,
    /// OR: raw token IDs (for /v1/completions or /completion with token array).
    pub prompt_ids: Option<Vec<u32>>,
    /// Maximum tokens to generate.
    pub max_tokens: usize,
    /// Sampling parameters.
    pub sampling: SampleParams,
    /// Stop sequences (strings).
    pub stop: Vec<String>,
    /// Whether thinking/reasoning is enabled.
    pub thinking: Thinking,
    /// Stream the response?
    pub stream: bool,
    /// Include usage in streaming response (OpenAI stream_options).
    pub stream_include_usage: bool,
    /// Echo the prompt (for /v1/completions).
    pub echo: bool,
    /// Return raw token IDs (llama-native return_tokens).
    pub return_tokens: bool,
    /// Tools provided in the request (for detecting tool_calls finish reason).
    pub has_tools: bool,
}

impl Default for GenRequest {
    fn default() -> Self {
        Self {
            prompt_text: String::new(),
            prompt_ids: None,
            max_tokens: usize::MAX,
            sampling: SampleParams {
                temperature: 1.0,
                top_k: 40,
                top_p: 0.95,
                min_p: 0.0,
                seed: 0,
            },
            stop: Vec::new(),
            thinking: Thinking::Enabled,
            stream: false,
            stream_include_usage: false,
            echo: false,
            return_tokens: false,
            has_tools: false,
        }
    }
}

/// Helper to extract sampling params from a JSON request.
pub fn extract_sampling(req: &Value, defaults: &SampleParams) -> SampleParams {
    let f = |k: &str| req.get(k).and_then(Value::as_f64);
    let u = |k: &str| req.get(k).and_then(Value::as_u64);
    SampleParams {
        temperature: f("temperature").map_or(defaults.temperature, |v| v as f32),
        top_k: u("top_k").map_or(defaults.top_k, |v| v as usize),
        top_p: f("top_p").map_or(defaults.top_p, |v| v as f32),
        min_p: f("min_p").map_or(defaults.min_p, |v| v as f32),
        seed: u("seed").unwrap_or_else(|| {
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_or(0, |d| d.as_nanos() as u64)
        }),
    }
}

/// Extract stop sequences from request (handles string or array).
pub fn extract_stop(req: &Value) -> Vec<String> {
    match req.get("stop") {
        Some(Value::String(s)) => vec![s.clone()],
        Some(Value::Array(a)) => a.iter().filter_map(|v| v.as_str().map(str::to_string)).collect(),
        _ => Vec::new(),
    }
}

/// Merge extra kwargs into a base map.
pub fn merge_kwargs(base: &Map<String, Value>, extra: Option<&Value>) -> Map<String, Value> {
    let mut m = base.clone();
    if let Some(Value::Object(e)) = extra {
        m.extend(e.clone());
    }
    m
}

/// OpenAI-style error response.
pub fn openai_error(message: &str, error_type: &str, code: Option<&str>) -> Value {
    let mut err = serde_json::json!({
        "error": {
            "message": message,
            "type": error_type
        }
    });
    if let Some(c) = code {
        err["error"]["code"] = Value::String(c.to_string());
    }
    err
}

/// Anthropic-style error response.
pub fn anthropic_error(error_type: &str, message: &str) -> Value {
    serde_json::json!({
        "type": "error",
        "error": {
            "type": error_type,
            "message": message
        }
    })
}
