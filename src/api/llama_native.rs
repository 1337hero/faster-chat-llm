//! llama-server native API: /completion, /tokenize, /detokenize, /apply-template, /props, /slots
//!
//! These match the llama.cpp server's non-OpenAI endpoints.

use std::io::Write;
use std::net::TcpStream;
use std::sync::mpsc::Receiver;

use serde_json::{Map, Value, json};

use crate::chat::ChatTemplate;
use crate::sampler::SampleParams;
use crate::sched::{Done, Event};
use crate::tokenizer::Tokenizer;

use super::internal::{GenRequest, Thinking, openai_error};

/// Parse a /completion request.
pub fn parse_completion_request(
    req: &Value,
    tok: &Tokenizer,
    defaults: &SampleParams,
) -> Result<GenRequest, String> {
    // Parse prompt: string, token array, or mixed
    let (prompt_text, prompt_ids) = match req.get("prompt") {
        Some(Value::String(s)) => (s.clone(), None),
        Some(Value::Array(arr)) => {
            let all_numbers = arr.iter().all(Value::is_number);
            if all_numbers {
                let ids = arr
                    .iter()
                    .map(|value| super::internal::parse_token_id(value, tok))
                    .collect::<Result<Vec<_>, _>>()?;
                (String::new(), Some(ids))
            } else {
                // Mixed: concatenate strings and decode tokens
                let mut text = String::new();
                for item in arr {
                    match item {
                        Value::String(s) => text.push_str(s),
                        Value::Number(_) => {
                            let id = super::internal::parse_token_id(item, tok)?;
                            text.push_str(&tok.decode(&[id]));
                        }
                        _ => return Err("prompt entries must be strings or token IDs".into()),
                    }
                }
                (text, None)
            }
        }
        _ => return Err("missing prompt".into()),
    };

    // Sampling params (llama-native names)
    let f = |k: &str| req.get(k).and_then(Value::as_f64);
    let u = |k: &str| req.get(k).and_then(Value::as_u64);
    let i = |k: &str| req.get(k).and_then(Value::as_i64);

    let sampling = SampleParams {
        temperature: f("temperature").map_or(defaults.temperature, |v| v as f32),
        top_k: u("top_k").map_or(defaults.top_k, |v| v as usize),
        top_p: f("top_p").map_or(defaults.top_p, |v| v as f32),
        min_p: f("min_p").map_or(defaults.min_p, |v| v as f32),
        seed: match i("seed") {
            Some(s) if s >= 0 => s as u64,
            _ => std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_or(0, |d| d.as_nanos() as u64),
        },
    };

    // Stop sequences
    let stop = match req.get("stop") {
        Some(Value::String(s)) => vec![s.clone()],
        Some(Value::Array(a)) => a
            .iter()
            .filter_map(|v| v.as_str().map(str::to_string))
            .collect(),
        _ => Vec::new(),
    };

    // n_predict maps to max_tokens
    let max_tokens = match i("n_predict") {
        Some(n) if n >= 0 => n as usize,
        Some(-1) | None => usize::MAX,
        _ => usize::MAX,
    };

    let stream = req.get("stream").and_then(Value::as_bool).unwrap_or(false);
    let return_tokens = req
        .get("return_tokens")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    // cache_prompt is accepted but we always cache

    Ok(GenRequest {
        prompt_text,
        prompt_ids,
        max_tokens,
        sampling,
        stop,
        thinking: Thinking::Disabled, // Raw completion, no thinking extraction
        stream,
        stream_include_usage: false,
        echo: false,
        return_tokens,
        has_tools: false,
    })
}

/// Build timings object from Done.
fn timings(d: &Done) -> Value {
    json!({
        "prompt_n": d.prompt_tokens - d.cached_tokens,
        "prompt_ms": d.prefill_s * 1e3,
        "prompt_per_second": if d.prefill_s > 0.0 {
            (d.prompt_tokens - d.cached_tokens) as f64 / d.prefill_s
        } else { 0.0 },
        "predicted_n": d.completion_tokens,
        "predicted_ms": d.decode_s * 1e3,
        "predicted_per_second": if d.decode_s > 0.0 {
            d.completion_tokens.saturating_sub(1) as f64 / d.decode_s
        } else { 0.0 },
    })
}

/// Handle non-streaming completion (llama-native format).
pub fn handle_completion_sync(
    rx: Receiver<Event>,
    model: &str,
    return_tokens: bool,
    tok: &Tokenizer,
) -> Value {
    let mut content = String::new();
    let mut tokens: Vec<u32> = Vec::new();
    let mut done_info = None;

    for ev in rx {
        match ev {
            Event::Text(t) => {
                content.push_str(&t);
                // TODO: return_tokens requires per-token ids from the driver
                // For now, we re-encode the text (not perfectly accurate for streaming)
            }
            Event::Done(d) => done_info = Some(d),
        }
    }

    let Some(d) = done_info else {
        return openai_error("engine stopped", "server_error", None);
    };

    // Re-encode for tokens if requested (ponytail: imperfect; proper impl needs driver change)
    if return_tokens {
        tokens = tok.encode(&content, false);
    }

    let stop_type = match d.finish_reason {
        "stop" => "eos",
        "length" => "limit",
        _ => "none",
    };

    json!({
        "content": content,
        "tokens": if return_tokens { Value::Array(tokens.iter().map(|&t| json!(t)).collect()) } else { Value::Array(vec![]) },
        "model": model,
        "stop": true,
        "stop_type": stop_type,
        "tokens_predicted": d.completion_tokens,
        "tokens_evaluated": d.prompt_tokens,
        "tokens_cached": d.cached_tokens,
        "timings": timings(&d),
    })
}

/// Handle streaming completion (llama-native format).
pub fn handle_completion_stream(
    mut conn: TcpStream,
    rx: Receiver<Event>,
    model: &str,
    return_tokens: bool,
    _tok: &Tokenizer,
) -> std::io::Result<()> {
    write!(
        conn,
        "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: close\r\n\r\n"
    )?;

    let send = |conn: &mut TcpStream, v: &Value| -> std::io::Result<()> {
        write!(conn, "data: {v}\n\n")?;
        conn.flush()
    };

    for ev in rx {
        match ev {
            Event::Text(t) => {
                let mut chunk = json!({
                    "content": t,
                    "stop": false
                });
                // TODO: return_tokens needs per-token from driver
                if return_tokens {
                    chunk["tokens"] = Value::Array(vec![]);
                }
                send(&mut conn, &chunk)?;
            }
            Event::Done(d) => {
                let stop_type = match d.finish_reason {
                    "stop" => "eos",
                    "length" => "limit",
                    _ => "none",
                };
                send(
                    &mut conn,
                    &json!({
                        "content": "",
                        "stop": true,
                        "stop_type": stop_type,
                        "model": model,
                        "tokens_predicted": d.completion_tokens,
                        "tokens_evaluated": d.prompt_tokens,
                        "tokens_cached": d.cached_tokens,
                        "timings": timings(&d),
                    }),
                )?;
            }
        }
    }

    Ok(())
}

/// Handle /tokenize endpoint.
pub fn tokenize(req: &Value, tok: &Tokenizer) -> Value {
    let content = req.get("content").and_then(Value::as_str).unwrap_or("");
    let add_special = req
        .get("add_special")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let with_pieces = req
        .get("with_pieces")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    // parse_special is accepted but we always parse specials

    let tokens = tok.encode(content, add_special);

    if with_pieces {
        let pieces: Vec<Value> = tokens
            .iter()
            .map(|&id| {
                let bytes = tok.decode_id(id);
                let piece = match std::str::from_utf8(bytes) {
                    Ok(s) => Value::String(s.to_string()),
                    Err(_) => Value::Array(bytes.iter().map(|&b| json!(b)).collect()),
                };
                json!({"id": id, "piece": piece})
            })
            .collect();
        json!({"tokens": pieces})
    } else {
        json!({"tokens": tokens})
    }
}

/// Handle /detokenize endpoint.
pub fn detokenize(req: &Value, tok: &Tokenizer) -> Value {
    let tokens: Vec<u32> = req
        .get("tokens")
        .and_then(Value::as_array)
        .map(|a| {
            a.iter()
                .filter_map(|v| v.as_u64().map(|n| n as u32))
                .collect()
        })
        .unwrap_or_default();

    let content = tok.decode(&tokens);
    json!({"content": content})
}

/// Handle /apply-template endpoint.
pub fn apply_template(
    req: &Value,
    template: &ChatTemplate,
    default_kwargs: &Map<String, Value>,
) -> Result<Value, String> {
    let messages = req.get("messages").ok_or("missing messages")?;
    let tools = req.get("tools");
    let kwargs = super::internal::merge_kwargs(default_kwargs, req.get("chat_template_kwargs"));

    let prompt = template.render(messages, tools, &kwargs)?;
    Ok(json!({"prompt": prompt}))
}

/// Handle /props endpoint.
pub fn props(model: &str, defaults: &SampleParams, n_slots: usize, ctx_size: usize) -> Value {
    json!({
        "default_generation_settings": {
            "n_ctx": ctx_size,
            "temperature": defaults.temperature,
            "top_k": defaults.top_k,
            "top_p": defaults.top_p,
            "min_p": defaults.min_p,
        },
        "total_slots": n_slots,
        "model_path": model,
        "chat_template": "(embedded in model)",
    })
}

/// Handle /slots endpoint (minimal stub showing slot state).
pub fn slots(n_slots: usize) -> Value {
    // ponytail: real implementation would need driver cooperation to report slot state
    // This stub returns all slots as idle
    let slots: Vec<Value> = (0..n_slots)
        .map(|i| {
            json!({
                "id": i,
                "is_processing": false,
                "n_ctx": 0,
            })
        })
        .collect();
    Value::Array(slots)
}

/// Handle /health endpoint.
pub fn health() -> Value {
    json!({"status": "ok"})
}
