//! Anthropic Messages API: /v1/messages (and /v1/messages/count_tokens)
//!
//! Supports content blocks, tools, thinking, and streaming SSE events.

use std::io::Write;
use std::net::TcpStream;
use std::sync::mpsc::Receiver;

use serde_json::{Map, Value, json};

use crate::chat::ChatTemplate;
use crate::sampler::SampleParams;
use crate::sched::{Done, Event};
use crate::tokenizer::Tokenizer;

use super::internal::{GenRequest, Thinking, anthropic_error};
use super::openai_chat::ThinkSplit;
use super::toolcall::{ToolCall, ToolCallParser, tool_calls_to_anthropic};

/// Parse an Anthropic messages request into a GenRequest.
pub fn parse_request(
    req: &Value,
    template: &ChatTemplate,
    _tok: &Tokenizer,
    default_kwargs: &Map<String, Value>,
    defaults: &SampleParams,
) -> Result<GenRequest, String> {
    // Convert messages to OpenAI format for the template
    let messages = convert_messages_to_openai(req)?;

    // Convert tools to OpenAI format
    let tools = convert_tools(req.get("tools"));

    // Handle tool_choice
    let tools = match req.get("tool_choice") {
        Some(obj) if obj.get("type").and_then(Value::as_str) == Some("none") => None,
        _ => tools,
    };

    // The Messages API thinks only when asked: {"type": "enabled", ...} turns it on, and a
    // request without a `thinking` field gets none, whatever the server's own default is.
    let mut kwargs = default_kwargs.clone();
    let wants_thinking = req.pointer("/thinking/type").and_then(Value::as_str) == Some("enabled");
    kwargs.insert("enable_thinking".into(), Value::Bool(wants_thinking));

    let prompt_text = template.render(&messages, tools.as_ref(), &kwargs)?;
    let thinking_enabled = kwargs
        .get("enable_thinking")
        .and_then(Value::as_bool)
        .unwrap_or(true);

    // Extract sampling params (Anthropic names)
    let f = |k: &str| req.get(k).and_then(Value::as_f64);
    let u = |k: &str| req.get(k).and_then(Value::as_u64);
    let sampling = SampleParams {
        temperature: f("temperature").map_or(defaults.temperature, |v| v as f32),
        top_k: u("top_k").map_or(defaults.top_k, |v| v as usize),
        top_p: f("top_p").map_or(defaults.top_p, |v| v as f32),
        min_p: defaults.min_p,
        seed: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.as_nanos() as u64),
    };

    // Extract stop sequences
    let stop = match req.get("stop_sequences") {
        Some(Value::Array(a)) => a
            .iter()
            .filter_map(|v| v.as_str().map(str::to_string))
            .collect(),
        _ => Vec::new(),
    };

    let max_tokens = u("max_tokens").map_or(4096, |v| v as usize);
    let stream = req.get("stream").and_then(Value::as_bool).unwrap_or(false);

    Ok(GenRequest {
        prompt_text,
        prompt_ids: None,
        max_tokens,
        sampling,
        stop,
        thinking: if thinking_enabled {
            Thinking::Enabled
        } else {
            Thinking::Disabled
        },
        stream,
        stream_include_usage: false,
        echo: false,
        return_tokens: false,
        has_tools: tools.is_some(),
    })
}

/// Convert Anthropic messages format to OpenAI format.
fn convert_messages_to_openai(req: &Value) -> Result<Value, String> {
    let mut messages = Vec::new();

    // Add system message if present
    match req.get("system") {
        Some(Value::String(s)) => {
            messages.push(json!({"role": "system", "content": s}));
        }
        Some(Value::Array(blocks)) => {
            // System as array of blocks
            let text: String = blocks
                .iter()
                .filter_map(|b| {
                    if b.get("type").and_then(Value::as_str) == Some("text") {
                        b.get("text").and_then(Value::as_str).map(str::to_string)
                    } else {
                        None
                    }
                })
                .collect::<Vec<_>>()
                .join("\n");
            if !text.is_empty() {
                messages.push(json!({"role": "system", "content": text}));
            }
        }
        _ => {}
    }

    // Convert each message
    let msgs = req
        .get("messages")
        .and_then(Value::as_array)
        .ok_or("missing messages")?;
    for msg in msgs {
        let role = msg.get("role").and_then(Value::as_str).unwrap_or("user");

        match msg.get("content") {
            Some(Value::String(s)) => {
                messages.push(json!({"role": role, "content": s}));
            }
            Some(Value::Array(blocks)) => {
                // Convert content blocks
                let ContentBlocks {
                    text,
                    tool_calls,
                    tool_results,
                } = parse_content_blocks(blocks)?;

                if role == "assistant" {
                    if !tool_calls.is_empty() {
                        // Assistant with tool calls
                        let tc: Vec<Value> = tool_calls
                            .iter()
                            .map(|(id, name, input)| {
                                json!({
                                    "id": id,
                                    "type": "function",
                                    "function": {
                                        "name": name,
                                        "arguments": input.to_string()
                                    }
                                })
                            })
                            .collect();
                        let mut m = json!({"role": "assistant", "tool_calls": tc});
                        if !text.is_empty() {
                            m["content"] = Value::String(text);
                        }
                        messages.push(m);
                    } else {
                        messages.push(json!({"role": "assistant", "content": text}));
                    }
                } else if role == "user" {
                    // Add tool results as separate tool messages first
                    for (id, content) in &tool_results {
                        messages.push(json!({
                            "role": "tool",
                            "tool_call_id": id,
                            "content": content
                        }));
                    }
                    if !text.is_empty() {
                        messages.push(json!({"role": "user", "content": text}));
                    }
                } else {
                    if !text.is_empty() {
                        messages.push(json!({"role": role, "content": text}));
                    }
                }
            }
            _ => {}
        }
    }

    Ok(Value::Array(messages))
}

struct ContentBlocks {
    text: String,
    tool_calls: Vec<(String, String, Value)>,
    tool_results: Vec<(String, String)>,
}

fn parse_content_blocks(blocks: &[Value]) -> Result<ContentBlocks, String> {
    let mut text_parts = Vec::new();
    let mut tool_calls = Vec::new();
    let mut tool_results = Vec::new();

    for block in blocks {
        match block.get("type").and_then(Value::as_str) {
            Some("text") => {
                if let Some(t) = block.get("text").and_then(Value::as_str) {
                    text_parts.push(t.to_string());
                }
            }
            Some("thinking") => {
                // Thinking blocks are passed through as text with markers
                if let Some(_t) = block.get("thinking").and_then(Value::as_str) {
                    // The template handles thinking, so we don't add markers here
                }
            }
            Some("tool_use") => {
                let id = block
                    .get("id")
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .to_string();
                let name = block
                    .get("name")
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .to_string();
                let input = block.get("input").cloned().unwrap_or(json!({}));
                tool_calls.push((id, name, input));
            }
            Some("tool_result") => {
                let id = block
                    .get("tool_use_id")
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .to_string();
                let content = match block.get("content") {
                    Some(Value::String(s)) => s.clone(),
                    Some(Value::Array(arr)) => arr
                        .iter()
                        .filter_map(|b| {
                            if b.get("type").and_then(Value::as_str) == Some("text") {
                                b.get("text").and_then(Value::as_str).map(str::to_string)
                            } else {
                                None
                            }
                        })
                        .collect::<Vec<_>>()
                        .join("\n"),
                    _ => String::new(),
                };
                tool_results.push((id, content));
            }
            Some("image") => {
                return Err("image input is not supported; this model is text-only".into());
            }
            _ => {}
        }
    }

    Ok(ContentBlocks {
        text: text_parts.join("\n"),
        tool_calls,
        tool_results,
    })
}

/// Convert Anthropic tools to OpenAI format for the template.
fn convert_tools(tools: Option<&Value>) -> Option<Value> {
    let arr = tools?.as_array()?;
    let converted: Vec<Value> = arr
        .iter()
        .filter_map(|t| {
            let name = t.get("name")?.as_str()?;
            let description = t.get("description").and_then(Value::as_str).unwrap_or("");
            let schema = t
                .get("input_schema")
                .cloned()
                .unwrap_or(json!({"type": "object"}));
            Some(json!({
                "type": "function",
                "function": {
                    "name": name,
                    "description": description,
                    "parameters": schema
                }
            }))
        })
        .collect();
    if converted.is_empty() {
        None
    } else {
        Some(Value::Array(converted))
    }
}

/// Generate message ID.
fn gen_id() -> String {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_nanos());
    format!("msg_{:x}", now)
}

/// Build usage object.
fn usage(d: &Done) -> Value {
    json!({
        "input_tokens": d.prompt_tokens,
        "output_tokens": d.completion_tokens,
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": d.cached_tokens
    })
}

/// Determine stop reason from Done.
fn stop_reason(d: &Done, has_tool_calls: bool) -> &'static str {
    if has_tool_calls {
        return "tool_use";
    }
    match d.finish_reason {
        "stop" => "end_turn",
        "length" => "max_tokens",
        _ => "end_turn",
    }
}

/// Handle non-streaming message.
pub fn handle_sync(
    rx: Receiver<Event>,
    model: &str,
    thinking_enabled: bool,
    _has_tools: bool,
) -> Value {
    let id = gen_id();
    let mut split = ThinkSplit::new(thinking_enabled);
    let mut tool_parser = ToolCallParser::new();

    let (mut reasoning, mut content) = (String::new(), String::new());
    let mut calls: Vec<ToolCall> = Vec::new();
    let mut done_info = None;

    for ev in rx {
        match ev {
            Event::Text(t) => {
                let (r, c) = split.push(&t, false);
                reasoning.push_str(&r);
                let (safe, found) = tool_parser.push(&c, false);
                content.push_str(&safe);
                calls.extend(found);
            }
            Event::Done(d) => done_info = Some(d),
        }
    }

    let (r, c) = split.push("", true);
    reasoning.push_str(&r);
    let (safe, found) = tool_parser.push(&c, true);
    content.push_str(&safe);
    calls.extend(found);

    let Some(d) = done_info else {
        return anthropic_error("api_error", "engine stopped");
    };

    // Build content blocks
    let mut blocks = Vec::new();

    if !reasoning.is_empty() && thinking_enabled {
        blocks.push(json!({
            "type": "thinking",
            "thinking": reasoning,
            "signature": ""
        }));
    }

    if !content.is_empty() {
        blocks.push(json!({
            "type": "text",
            "text": content
        }));
    }

    // Add tool_use blocks
    for block in tool_calls_to_anthropic(&calls) {
        blocks.push(block);
    }

    json!({
        "id": id,
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": blocks,
        "stop_reason": stop_reason(&d, !calls.is_empty()),
        "stop_sequence": Value::Null,
        "usage": usage(&d)
    })
}

/// Handle streaming message with SSE events.
pub fn handle_stream(
    mut conn: TcpStream,
    rx: Receiver<Event>,
    model: &str,
    thinking_enabled: bool,
    _has_tools: bool,
) -> std::io::Result<()> {
    let id = gen_id();

    write!(
        conn,
        "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: close\r\n\r\n"
    )?;

    let send = |conn: &mut TcpStream, event: &str, data: &Value| -> std::io::Result<()> {
        write!(conn, "event: {event}\ndata: {data}\n\n")?;
        conn.flush()
    };

    // message_start
    send(
        &mut conn,
        "message_start",
        &json!({
            "type": "message_start",
            "message": {
                "id": id,
                "type": "message",
                "role": "assistant",
                "model": model,
                "content": [],
                "stop_reason": Value::Null,
                "stop_sequence": Value::Null,
                "usage": {"input_tokens": 0, "output_tokens": 0}
            }
        }),
    )?;

    let mut split = ThinkSplit::new(thinking_enabled);
    let mut tool_parser = ToolCallParser::new();

    let mut block_idx = 0;
    let mut thinking_started = false;
    let mut text_started = false;
    let mut all_calls: Vec<ToolCall> = Vec::new();

    for ev in rx {
        match ev {
            Event::Text(t) => {
                let (r, c) = split.push(&t, false);

                // Emit thinking deltas
                if !r.is_empty() && thinking_enabled {
                    if !thinking_started {
                        send(
                            &mut conn,
                            "content_block_start",
                            &json!({
                                "type": "content_block_start",
                                "index": block_idx,
                                "content_block": {"type": "thinking", "thinking": "", "signature": ""}
                            }),
                        )?;
                        thinking_started = true;
                    }
                    send(
                        &mut conn,
                        "content_block_delta",
                        &json!({
                            "type": "content_block_delta",
                            "index": block_idx,
                            "delta": {"type": "thinking_delta", "thinking": r}
                        }),
                    )?;
                }

                // Parse tool calls
                let (safe, calls) = tool_parser.push(&c, false);

                // Emit text deltas
                if !safe.is_empty() {
                    if !text_started {
                        if thinking_started {
                            send(
                                &mut conn,
                                "content_block_stop",
                                &json!({
                                    "type": "content_block_stop",
                                    "index": block_idx
                                }),
                            )?;
                            block_idx += 1;
                        }
                        send(
                            &mut conn,
                            "content_block_start",
                            &json!({
                                "type": "content_block_start",
                                "index": block_idx,
                                "content_block": {"type": "text", "text": ""}
                            }),
                        )?;
                        text_started = true;
                    }
                    send(
                        &mut conn,
                        "content_block_delta",
                        &json!({
                            "type": "content_block_delta",
                            "index": block_idx,
                            "delta": {"type": "text_delta", "text": safe}
                        }),
                    )?;
                }

                all_calls.extend(calls);
            }
            Event::Done(d) => {
                // Flush
                let (r, c) = split.push("", true);
                if !r.is_empty() && thinking_enabled && !thinking_started {
                    send(
                        &mut conn,
                        "content_block_start",
                        &json!({
                            "type": "content_block_start",
                            "index": block_idx,
                            "content_block": {"type": "thinking", "thinking": "", "signature": ""}
                        }),
                    )?;
                    thinking_started = true;
                }
                if !r.is_empty() && thinking_started {
                    send(
                        &mut conn,
                        "content_block_delta",
                        &json!({
                            "type": "content_block_delta",
                            "index": block_idx,
                            "delta": {"type": "thinking_delta", "thinking": r}
                        }),
                    )?;
                }

                let (safe, calls) = tool_parser.push(&c, true);
                all_calls.extend(calls);

                if !safe.is_empty() {
                    if !text_started {
                        if thinking_started {
                            send(
                                &mut conn,
                                "content_block_stop",
                                &json!({
                                    "type": "content_block_stop",
                                    "index": block_idx
                                }),
                            )?;
                            block_idx += 1;
                        }
                        send(
                            &mut conn,
                            "content_block_start",
                            &json!({
                                "type": "content_block_start",
                                "index": block_idx,
                                "content_block": {"type": "text", "text": ""}
                            }),
                        )?;
                        text_started = true;
                    }
                    send(
                        &mut conn,
                        "content_block_delta",
                        &json!({
                            "type": "content_block_delta",
                            "index": block_idx,
                            "delta": {"type": "text_delta", "text": safe}
                        }),
                    )?;
                }

                // Close text block if open
                if text_started || thinking_started {
                    send(
                        &mut conn,
                        "content_block_stop",
                        &json!({
                            "type": "content_block_stop",
                            "index": block_idx
                        }),
                    )?;
                    block_idx += 1;
                }

                // Emit tool_use blocks
                for call in &all_calls {
                    send(
                        &mut conn,
                        "content_block_start",
                        &json!({
                            "type": "content_block_start",
                            "index": block_idx,
                            "content_block": {
                                "type": "tool_use",
                                "id": call.id,
                                "name": call.name,
                                "input": {}
                            }
                        }),
                    )?;
                    send(
                        &mut conn,
                        "content_block_delta",
                        &json!({
                            "type": "content_block_delta",
                            "index": block_idx,
                            "delta": {
                                "type": "input_json_delta",
                                "partial_json": call.arguments.to_string()
                            }
                        }),
                    )?;
                    send(
                        &mut conn,
                        "content_block_stop",
                        &json!({
                            "type": "content_block_stop",
                            "index": block_idx
                        }),
                    )?;
                    block_idx += 1;
                }

                // message_delta with stop_reason and usage
                send(
                    &mut conn,
                    "message_delta",
                    &json!({
                        "type": "message_delta",
                        "delta": {
                            "stop_reason": stop_reason(&d, !all_calls.is_empty()),
                            "stop_sequence": Value::Null
                        },
                        "usage": {"output_tokens": d.completion_tokens}
                    }),
                )?;

                // message_stop
                send(
                    &mut conn,
                    "message_stop",
                    &json!({
                        "type": "message_stop"
                    }),
                )?;
            }
        }
    }

    Ok(())
}

/// Count tokens for /v1/messages/count_tokens endpoint.
pub fn count_tokens(
    req: &Value,
    template: &ChatTemplate,
    tok: &Tokenizer,
    default_kwargs: &Map<String, Value>,
) -> Result<Value, String> {
    let messages = convert_messages_to_openai(req)?;
    let tools = convert_tools(req.get("tools"));

    let mut kwargs = default_kwargs.clone();
    if let Some(thinking) = req.get("thinking")
        && let Some(enabled) = thinking.get("enabled").and_then(Value::as_bool)
    {
        kwargs.insert("enable_thinking".into(), Value::Bool(enabled));
    }

    let prompt_text = template.render(&messages, tools.as_ref(), &kwargs)?;
    let tokens = tok.encode(&prompt_text, true);

    Ok(json!({
        "input_tokens": tokens.len()
    }))
}
