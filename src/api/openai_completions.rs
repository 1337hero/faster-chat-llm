//! OpenAI legacy Completions API: /v1/completions
//!
//! Accepts prompt as string or token array, returns text_completion objects.

use std::io::Write;
use std::net::TcpStream;
use std::sync::mpsc::Receiver;

use serde_json::{Value, json};

use crate::sched::{Done, Event};
use crate::sampler::SampleParams;
use crate::tokenizer::Tokenizer;

use super::internal::{GenRequest, Thinking, extract_sampling, extract_stop, openai_error};

/// Parse a completions request into a GenRequest.
pub fn parse_request(
    req: &Value,
    _tok: &Tokenizer,
    defaults: &SampleParams,
) -> Result<GenRequest, String> {
    // Validate n parameter
    if let Some(n) = req.get("n").and_then(Value::as_u64) {
        if n > 1 {
            return Err("n > 1 is not supported; this model serves one completion per request".into());
        }
    }

    let (prompt_text, prompt_ids) = match req.get("prompt") {
        Some(Value::String(s)) => (s.clone(), None),
        Some(Value::Array(arr)) => {
            // Check if it's an array of token IDs (all numbers)
            let all_numbers = arr.iter().all(|v| v.is_u64() || v.is_i64());
            if all_numbers {
                let ids: Vec<u32> = arr.iter().filter_map(|v| v.as_u64().map(|n| n as u32)).collect();
                (String::new(), Some(ids))
            } else {
                // Array of strings - concatenate
                let text: String = arr.iter().filter_map(|v| v.as_str()).collect::<Vec<_>>().join("");
                (text, None)
            }
        }
        _ => return Err("missing or invalid prompt".into()),
    };

    let sampling = extract_sampling(req, defaults);
    let stop = extract_stop(req);
    let max_tokens = req.get("max_tokens")
        .and_then(Value::as_u64)
        .map_or(16, |v| v as usize); // Default 16 for completions
    let stream = req.get("stream").and_then(Value::as_bool).unwrap_or(false);
    let echo = req.get("echo").and_then(Value::as_bool).unwrap_or(false);

    Ok(GenRequest {
        prompt_text,
        prompt_ids,
        max_tokens,
        sampling,
        stop,
        thinking: Thinking::Disabled, // No thinking for raw completions
        stream,
        stream_include_usage: false,
        echo,
        return_tokens: false,
        has_tools: false,
    })
}

/// Generate response ID and timestamp.
fn gen_ids() -> (String, u64) {
    let now = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_nanos());
    let id = format!("cmpl-{:x}", now);
    let created = (now / 1_000_000_000) as u64;
    (id, created)
}

/// Build usage object.
fn usage(d: &Done) -> Value {
    json!({
        "prompt_tokens": d.prompt_tokens,
        "completion_tokens": d.completion_tokens,
        "total_tokens": d.prompt_tokens + d.completion_tokens
    })
}

/// Handle non-streaming completion.
pub fn handle_sync(
    rx: Receiver<Event>,
    model: &str,
    echo_prefix: &str, // The prompt text to echo if echo=true
) -> Value {
    let (id, created) = gen_ids();
    let mut text = String::new();

    if !echo_prefix.is_empty() {
        text.push_str(echo_prefix);
    }

    let mut done_info = None;

    for ev in rx {
        match ev {
            Event::Text(t) => text.push_str(&t),
            Event::Done(d) => done_info = Some(d),
        }
    }

    let Some(d) = done_info else {
        return openai_error("engine stopped", "server_error", None);
    };

    json!({
        "id": id,
        "object": "text_completion",
        "created": created,
        "model": model,
        "choices": [{
            "index": 0,
            "text": text,
            "finish_reason": d.finish_reason,
            "logprobs": Value::Null
        }],
        "usage": usage(&d)
    })
}

/// Handle streaming completion.
pub fn handle_stream(
    mut conn: TcpStream,
    rx: Receiver<Event>,
    model: &str,
    echo_prefix: &str,
) -> std::io::Result<()> {
    let (id, created) = gen_ids();

    write!(conn, "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: close\r\n\r\n")?;

    let chunk = |text: &str, finish: Value| -> Value {
        json!({
            "id": id,
            "object": "text_completion",
            "created": created,
            "model": model,
            "choices": [{"index": 0, "text": text, "finish_reason": finish, "logprobs": Value::Null}]
        })
    };

    let send = |conn: &mut TcpStream, v: &Value| -> std::io::Result<()> {
        write!(conn, "data: {v}\n\n")?;
        conn.flush()
    };

    // Echo prefix if requested
    if !echo_prefix.is_empty() {
        send(&mut conn, &chunk(echo_prefix, Value::Null))?;
    }

    for ev in rx {
        match ev {
            Event::Text(t) => {
                send(&mut conn, &chunk(&t, Value::Null))?;
            }
            Event::Done(d) => {
                send(&mut conn, &chunk("", json!(d.finish_reason)))?;
            }
        }
    }

    write!(conn, "data: [DONE]\n\n")?;
    conn.flush()
}
