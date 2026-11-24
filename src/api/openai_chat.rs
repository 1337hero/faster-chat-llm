//! OpenAI Chat Completions API: /v1/chat/completions
//!
//! Supports tool calling: parses model's <tool_call> XML into structured tool_calls.

use std::io::Write;
use std::net::TcpStream;
use std::sync::mpsc::Receiver;

use serde_json::{Map, Value, json};

use crate::chat::ChatTemplate;
use crate::sampler::SampleParams;
use crate::sched::{Done, Event};
use crate::tokenizer::Tokenizer;

use super::internal::{
    GenRequest, Thinking, extract_sampling, extract_stop, merge_kwargs, openai_error,
};
use super::toolcall::{ToolCallParser, tool_calls_to_openai};

/// Splits streamed text into reasoning (inside <think>...</think>) and answer.
pub struct ThinkSplit {
    thinking: bool,
    buf: String,
    /// Set once the closing tag has passed and until the answer's first visible character:
    /// the blank lines the model puts after </think> belong to neither part.
    skip_blank: bool,
}

impl ThinkSplit {
    const CLOSE: &'static str = "</think>";

    pub fn new(thinking: bool) -> Self {
        Self {
            thinking,
            buf: String::new(),
            skip_blank: false,
        }
    }

    /// Returns (reasoning, content) deltas that are safe to send now.
    pub fn push(&mut self, text: &str, last: bool) -> (String, String) {
        if !self.thinking {
            let text = if self.skip_blank {
                text.trim_start()
            } else {
                text
            };
            self.skip_blank &= text.is_empty();
            return (String::new(), text.to_string());
        }
        self.buf.push_str(text);
        if let Some(pos) = self.buf.find(Self::CLOSE) {
            self.thinking = false;
            let reasoning = self.buf[..pos].trim_end().to_string();
            let content = self.buf[pos + Self::CLOSE.len()..].trim_start().to_string();
            self.skip_blank = content.is_empty();
            self.buf.clear();
            return (reasoning, content);
        }
        // Hold back a tail that could be the start of the closing tag.
        let mut keep = if last {
            0
        } else {
            (Self::CLOSE.len() - 1).min(self.buf.len())
        };
        while !self.buf.is_char_boundary(self.buf.len() - keep) {
            keep += 1;
        }
        let out: String = self.buf.drain(..self.buf.len() - keep).collect();
        (out, String::new())
    }
}

/// Parse a chat completions request into a GenRequest.
pub fn parse_request(
    req: &Value,
    template: &ChatTemplate,
    _tok: &Tokenizer,
    default_kwargs: &Map<String, Value>,
    defaults: &SampleParams,
) -> Result<GenRequest, String> {
    // Validate n parameter
    if let Some(n) = req.get("n").and_then(Value::as_u64)
        && n > 1
    {
        return Err("n > 1 is not supported; this model serves one completion per request".into());
    }

    let messages = req.get("messages").ok_or("missing messages")?;

    // Handle tool_choice
    let tools = match req.get("tool_choice") {
        Some(Value::String(s)) if s == "none" => None,
        _ => req.get("tools"),
    };

    let kwargs = merge_kwargs(default_kwargs, req.get("chat_template_kwargs"));

    // response_format is accepted but ignored (the model is text-only)

    let prompt_text = template.render(messages, tools, &kwargs)?;
    let thinking = kwargs
        .get("enable_thinking")
        .and_then(Value::as_bool)
        .unwrap_or(true);

    let sampling = extract_sampling(req, defaults);
    let stop = extract_stop(req);
    let max_tokens = req
        .get("max_completion_tokens")
        .or(req.get("max_tokens"))
        .and_then(Value::as_u64)
        .map_or(usize::MAX, |v| v as usize);
    let stream = req.get("stream").and_then(Value::as_bool).unwrap_or(false);
    let include_usage = req
        .pointer("/stream_options/include_usage")
        .and_then(Value::as_bool)
        .unwrap_or(false);

    Ok(GenRequest {
        prompt_text,
        prompt_ids: None,
        max_tokens,
        sampling,
        stop,
        thinking: if thinking {
            Thinking::Enabled
        } else {
            Thinking::Disabled
        },
        stream,
        stream_include_usage: include_usage,
        echo: false,
        return_tokens: false,
        has_tools: tools.is_some(),
    })
}

/// Generate response ID and timestamp for chat completion.
fn gen_ids() -> (String, u64) {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_nanos());
    let id = format!("chatcmpl-{:x}", now);
    let created = (now / 1_000_000_000) as u64;
    (id, created)
}

/// Build usage object from Done.
fn usage(d: &Done) -> Value {
    json!({
        "prompt_tokens": d.prompt_tokens,
        "completion_tokens": d.completion_tokens,
        "total_tokens": d.prompt_tokens + d.completion_tokens,
        "prompt_tokens_details": {"cached_tokens": d.cached_tokens},
    })
}

/// Build timings object from Done.
fn timings(d: &Done) -> Value {
    json!({
        "prompt_n": d.prompt_tokens - d.cached_tokens,
        "prompt_ms": d.prefill_s * 1e3,
        "predicted_n": d.completion_tokens,
        "predicted_ms": d.decode_s * 1e3,
        "predicted_per_second": if d.decode_s > 0.0 { d.completion_tokens.saturating_sub(1) as f64 / d.decode_s } else { 0.0 },
        "draft_n": d.drafted,
        "draft_n_accepted": d.accepted,
    })
}

/// Handle non-streaming chat completion.
pub fn handle_sync(
    rx: Receiver<Event>,
    model: &str,
    thinking_enabled: bool,
    has_tools: bool,
) -> Value {
    let (id, created) = gen_ids();
    let mut split = ThinkSplit::new(thinking_enabled);
    let mut tool_parser = ToolCallParser::new();

    let (mut reasoning, mut content) = (String::new(), String::new());
    let mut calls = Vec::new();
    let mut done_info = None;

    for ev in rx {
        match ev {
            Event::Text(t) => {
                let (r, c) = split.push(&t, false);
                reasoning.push_str(&r);
                let (safe_text, found) = tool_parser.push(&c, false);
                content.push_str(&safe_text);
                calls.extend(found);
            }
            Event::Done(d) => done_info = Some(d),
        }
    }

    // Flush
    let (r, c) = split.push("", true);
    reasoning.push_str(&r);
    let (safe_text, found) = tool_parser.push(&c, true);
    content.push_str(&safe_text);
    calls.extend(found);

    let Some(d) = done_info else {
        return openai_error("engine stopped", "server_error", None);
    };

    // Build message
    let mut msg = json!({"role": "assistant"});

    if !calls.is_empty() {
        msg["tool_calls"] = tool_calls_to_openai(&calls);
        // Content may be empty when only tool calls
        if !content.is_empty() {
            msg["content"] = Value::String(content);
        } else {
            msg["content"] = Value::Null;
        }
    } else {
        msg["content"] = Value::String(content);
    }

    if !reasoning.is_empty() {
        msg["reasoning_content"] = Value::String(reasoning);
    }

    // Determine finish reason
    let finish_reason = if !calls.is_empty() && has_tools {
        "tool_calls"
    } else {
        d.finish_reason
    };

    json!({
        "id": id,
        "object": "chat.completion",
        "created": created,
        "model": model,
        "choices": [{
            "index": 0,
            "message": msg,
            "finish_reason": finish_reason
        }],
        "usage": usage(&d),
        "timings": timings(&d),
    })
}

/// Handle streaming chat completion.
pub fn handle_stream(
    mut conn: TcpStream,
    rx: Receiver<Event>,
    model: &str,
    thinking_enabled: bool,
    has_tools: bool,
    include_usage: bool,
) -> std::io::Result<()> {
    let (id, created) = gen_ids();

    write!(
        conn,
        "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: close\r\n\r\n"
    )?;

    let chunk = |delta: Value, finish: Value| -> Value {
        json!({
            "id": id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]
        })
    };

    let send = |conn: &mut TcpStream, v: &Value| -> std::io::Result<()> {
        write!(conn, "data: {v}\n\n")?;
        conn.flush()
    };

    // Initial chunk with role
    send(
        &mut conn,
        &chunk(json!({"role": "assistant", "content": ""}), Value::Null),
    )?;

    let mut split = ThinkSplit::new(thinking_enabled);
    let mut tool_parser = ToolCallParser::new();
    let mut all_calls = Vec::new();

    for ev in rx {
        match ev {
            Event::Text(t) => {
                let (r, c) = split.push(&t, false);
                if !r.is_empty() {
                    send(
                        &mut conn,
                        &chunk(json!({"reasoning_content": r}), Value::Null),
                    )?;
                }
                // Parse tool calls; only emit safe text
                let (safe_text, calls) = tool_parser.push(&c, false);
                if !safe_text.is_empty() {
                    send(
                        &mut conn,
                        &chunk(json!({"content": safe_text}), Value::Null),
                    )?;
                }
                all_calls.extend(calls);
            }
            Event::Done(d) => {
                // Flush remaining
                let (r, c) = split.push("", true);
                if !r.is_empty() {
                    send(
                        &mut conn,
                        &chunk(json!({"reasoning_content": r}), Value::Null),
                    )?;
                }
                let (safe_text, calls) = tool_parser.push(&c, true);
                if !safe_text.is_empty() {
                    send(
                        &mut conn,
                        &chunk(json!({"content": safe_text}), Value::Null),
                    )?;
                }
                all_calls.extend(calls);

                // Emit tool calls if any
                if !all_calls.is_empty() {
                    let mut tc = tool_calls_to_openai(&all_calls);
                    if let Some(calls) = tc.as_array_mut() {
                        for (index, call) in calls.iter_mut().enumerate() {
                            call["index"] = json!(index);
                        }
                    }
                    send(&mut conn, &chunk(json!({"tool_calls": tc}), Value::Null))?;
                }

                // Finish reason
                let finish_reason = if !all_calls.is_empty() && has_tools {
                    "tool_calls"
                } else {
                    d.finish_reason
                };

                let mut fin = chunk(json!({}), json!(finish_reason));
                fin["timings"] = timings(&d);
                send(&mut conn, &fin)?;

                if include_usage {
                    send(
                        &mut conn,
                        &json!({
                            "id": id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": model,
                            "choices": [],
                            "usage": usage(&d)
                        }),
                    )?;
                }
            }
        }
    }

    write!(conn, "data: [DONE]\n\n")?;
    conn.flush()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn think_split_drops_blank_lines_after_close() {
        let mut s = ThinkSplit::new(true);
        assert_eq!(
            s.push("why</think>", false),
            ("why".to_string(), String::new())
        );
        assert_eq!(s.push("\n\n", false), (String::new(), String::new()));
        assert_eq!(
            s.push("51\n\nmore", false),
            (String::new(), "51\n\nmore".to_string())
        );
    }

    #[test]
    fn think_split_basic() {
        let mut s = ThinkSplit::new(true);
        let (r, c) = s.push("thinking here</think>answer", true);
        assert_eq!(r, "thinking here");
        assert_eq!(c, "answer");
    }

    #[test]
    fn think_split_disabled() {
        let mut s = ThinkSplit::new(false);
        let (r, c) = s.push("<think>foo</think>bar", true);
        assert_eq!(r, "");
        assert_eq!(c, "<think>foo</think>bar");
    }

    #[test]
    fn think_split_incremental() {
        let mut s = ThinkSplit::new(true);
        let (r1, c1) = s.push("part1</thi", false);
        assert_eq!(c1, "");
        // Holds back potential tag

        let (r2, c2) = s.push("nk>after", true);
        let full_r = format!("{}{}", r1, r2);
        assert!(full_r.contains("part1"));
        assert_eq!(c2, "after");
    }
}
