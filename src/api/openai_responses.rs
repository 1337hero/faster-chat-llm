//! OpenAI Responses API: /v1/responses
//!
//! Stateless implementation with proper SSE semantic events.

use std::io::Write;
use std::net::TcpStream;
use std::sync::mpsc::Receiver;

use serde_json::{Map, Value, json};

use crate::chat::ChatTemplate;
use crate::sched::{Done, Event};
use crate::sampler::SampleParams;
use crate::tokenizer::Tokenizer;

use super::internal::{GenRequest, Thinking, extract_sampling, merge_kwargs, openai_error};
use super::openai_chat::ThinkSplit;
use super::toolcall::{ToolCallParser, ToolCall};

/// Parse a responses request into a GenRequest.
pub fn parse_request(
    req: &Value,
    template: &ChatTemplate,
    _tok: &Tokenizer,
    default_kwargs: &Map<String, Value>,
    defaults: &SampleParams,
) -> Result<GenRequest, String> {
    // Reject unsupported features
    if req.get("previous_response_id").is_some() {
        return Err("previous_response_id is not supported; this is a stateless endpoint".into());
    }
    if let Some(Value::Bool(true)) = req.get("store") {
        return Err("store: true is not supported; responses are not persisted".into());
    }

    // Convert input to messages format
    let messages = convert_input_to_messages(req)?;

    // Convert flat tools to OpenAI format
    let tools = convert_tools(req.get("tools"));

    let kwargs = merge_kwargs(default_kwargs, req.get("chat_template_kwargs"));

    let prompt_text = template.render(&messages, tools.as_ref(), &kwargs)?;
    let thinking = kwargs.get("enable_thinking").and_then(Value::as_bool).unwrap_or(true);

    let sampling = extract_sampling(req, defaults);

    // Extract stop from stop_sequences (Responses API name)
    let stop = match req.get("stop_sequences") {
        Some(Value::Array(a)) => a.iter().filter_map(|v| v.as_str().map(str::to_string)).collect(),
        _ => Vec::new(),
    };

    let max_tokens = req.get("max_output_tokens")
        .and_then(Value::as_u64)
        .map_or(usize::MAX, |v| v as usize);
    let stream = req.get("stream").and_then(Value::as_bool).unwrap_or(false);

    Ok(GenRequest {
        prompt_text,
        prompt_ids: None,
        max_tokens,
        sampling,
        stop,
        thinking: if thinking { Thinking::Enabled } else { Thinking::Disabled },
        stream,
        stream_include_usage: false,
        echo: false,
        return_tokens: false,
        has_tools: tools.is_some(),
    })
}

/// Convert Responses API input to messages array.
fn convert_input_to_messages(req: &Value) -> Result<Value, String> {
    let mut messages = Vec::new();

    // Add system/instructions if present
    if let Some(instr) = req.get("instructions").and_then(Value::as_str) {
        messages.push(json!({"role": "system", "content": instr}));
    }

    match req.get("input") {
        Some(Value::String(s)) => {
            messages.push(json!({"role": "user", "content": s}));
        }
        Some(Value::Array(items)) => {
            for item in items {
                match item.get("type").and_then(Value::as_str) {
                    Some("message") => {
                        let role = item.get("role").and_then(Value::as_str).unwrap_or("user");
                        let content = convert_message_content(item)?;
                        messages.push(json!({"role": role, "content": content}));
                    }
                    Some("function_call") => {
                        // Assistant made a function call
                        let call_id = item.get("call_id").and_then(Value::as_str).unwrap_or("");
                        let name = item.get("name").and_then(Value::as_str).unwrap_or("");
                        let args = item.get("arguments").and_then(Value::as_str).unwrap_or("{}");
                        messages.push(json!({
                            "role": "assistant",
                            "tool_calls": [{
                                "id": call_id,
                                "type": "function",
                                "function": {"name": name, "arguments": args}
                            }]
                        }));
                    }
                    Some("function_call_output") => {
                        // Tool result
                        let call_id = item.get("call_id").and_then(Value::as_str).unwrap_or("");
                        let output = item.get("output").and_then(Value::as_str).unwrap_or("");
                        messages.push(json!({
                            "role": "tool",
                            "tool_call_id": call_id,
                            "content": output
                        }));
                    }
                    Some("input_image") => {
                        return Err("image input is not supported; this model is text-only".into());
                    }
                    _ => {
                        // Skip unknown types
                    }
                }
            }
        }
        None => return Err("missing input".into()),
        _ => return Err("invalid input format".into()),
    }

    Ok(Value::Array(messages))
}

/// Convert message content (may have parts).
fn convert_message_content(item: &Value) -> Result<Value, String> {
    if let Some(content) = item.get("content") {
        match content {
            Value::String(s) => return Ok(Value::String(s.clone())),
            Value::Array(parts) => {
                let mut text_parts = Vec::new();
                for part in parts {
                    match part.get("type").and_then(Value::as_str) {
                        Some("input_text") | Some("output_text") => {
                            if let Some(t) = part.get("text").and_then(Value::as_str) {
                                text_parts.push(t.to_string());
                            }
                        }
                        Some("input_image") => {
                            return Err("image input is not supported; this model is text-only".into());
                        }
                        _ => {}
                    }
                }
                return Ok(Value::String(text_parts.join("")));
            }
            _ => {}
        }
    }
    Ok(Value::String(String::new()))
}

/// Convert flat function tools to OpenAI format.
fn convert_tools(tools: Option<&Value>) -> Option<Value> {
    let arr = tools?.as_array()?;
    let converted: Vec<Value> = arr.iter().filter_map(|t| {
        let name = t.get("name")?.as_str()?;
        let description = t.get("description").and_then(Value::as_str).unwrap_or("");
        let parameters = t.get("parameters").cloned().unwrap_or(json!({"type": "object"}));
        Some(json!({
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": parameters
            }
        }))
    }).collect();
    if converted.is_empty() { None } else { Some(Value::Array(converted)) }
}

/// Generate response ID.
fn gen_id() -> String {
    let now = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_nanos());
    format!("resp_{:x}", now)
}

/// Build usage object.
fn usage(d: &Done) -> Value {
    json!({
        "input_tokens": d.prompt_tokens,
        "output_tokens": d.completion_tokens,
        "total_tokens": d.prompt_tokens + d.completion_tokens
    })
}

/// Handle non-streaming response.
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
    let mut done_info = None;

    for ev in rx {
        match ev {
            Event::Text(t) => {
                let (r, c) = split.push(&t, false);
                reasoning.push_str(&r);
                let (safe, _) = tool_parser.push(&c, false);
                content.push_str(&safe);
            }
            Event::Done(d) => done_info = Some(d),
        }
    }

    let (r, c) = split.push("", true);
    reasoning.push_str(&r);
    let (safe, calls) = tool_parser.push(&c, true);
    content.push_str(&safe);

    let Some(d) = done_info else {
        return openai_error("engine stopped", "server_error", None);
    };

    // Build output items
    let mut output = Vec::new();
    let mut item_idx = 0;

    // Reasoning item (if present)
    if !reasoning.is_empty() && thinking_enabled {
        output.push(json!({
            "type": "reasoning",
            "id": format!("item_{}", item_idx),
            "summary": [{
                "type": "summary_text",
                "text": reasoning
            }]
        }));
        item_idx += 1;
    }

    // Message item with text
    if !content.is_empty() || calls.is_empty() {
        output.push(json!({
            "type": "message",
            "id": format!("item_{}", item_idx),
            "role": "assistant",
            "content": [{
                "type": "output_text",
                "text": content
            }]
        }));
        item_idx += 1;
    }

    // Function call items
    for call in &calls {
        output.push(json!({
            "type": "function_call",
            "id": format!("item_{}", item_idx),
            "call_id": call.id,
            "name": call.name,
            "arguments": call.arguments.to_string()
        }));
        item_idx += 1;
    }

    let status = if !calls.is_empty() { "incomplete" } else { "completed" };

    json!({
        "id": id,
        "object": "response",
        "status": status,
        "model": model,
        "output": output,
        "usage": usage(&d)
    })
}

/// Handle streaming response with semantic events.
pub fn handle_stream(
    mut conn: TcpStream,
    rx: Receiver<Event>,
    model: &str,
    thinking_enabled: bool,
    _has_tools: bool,
) -> std::io::Result<()> {
    let id = gen_id();

    write!(conn, "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: close\r\n\r\n")?;

    let send = |conn: &mut TcpStream, event: &str, data: &Value| -> std::io::Result<()> {
        write!(conn, "event: {event}\ndata: {data}\n\n")?;
        conn.flush()
    };

    // response.created
    send(&mut conn, "response.created", &json!({
        "type": "response.created",
        "response": {"id": id, "object": "response", "status": "in_progress", "model": model}
    }))?;

    // response.in_progress
    send(&mut conn, "response.in_progress", &json!({
        "type": "response.in_progress",
        "response": {"id": id, "status": "in_progress"}
    }))?;

    let mut split = ThinkSplit::new(thinking_enabled);
    let mut tool_parser = ToolCallParser::new();

    let mut item_idx = 0;
    let mut reasoning_started = false;
    let mut message_started = false;
    let mut all_calls: Vec<ToolCall> = Vec::new();

    for ev in rx {
        match ev {
            Event::Text(t) => {
                let (r, c) = split.push(&t, false);

                // Emit reasoning deltas
                if !r.is_empty() && thinking_enabled {
                    if !reasoning_started {
                        // Start reasoning item
                        send(&mut conn, "response.output_item.added", &json!({
                            "type": "response.output_item.added",
                            "output_index": item_idx,
                            "item": {"type": "reasoning", "id": format!("item_{}", item_idx)}
                        }))?;
                        reasoning_started = true;
                    }
                    send(&mut conn, "response.reasoning_summary_text.delta", &json!({
                        "type": "response.reasoning_summary_text.delta",
                        "item_id": format!("item_{}", item_idx),
                        "output_index": item_idx,
                        "delta": r
                    }))?;
                }

                // Parse tool calls from content
                let (safe, calls) = tool_parser.push(&c, false);

                // Emit text deltas
                if !safe.is_empty() {
                    if !message_started {
                        if reasoning_started {
                            // Close reasoning item first
                            send(&mut conn, "response.output_item.done", &json!({
                                "type": "response.output_item.done",
                                "output_index": item_idx,
                                "item": {"type": "reasoning", "id": format!("item_{}", item_idx)}
                            }))?;
                            item_idx += 1;
                        }
                        // Start message item
                        send(&mut conn, "response.output_item.added", &json!({
                            "type": "response.output_item.added",
                            "output_index": item_idx,
                            "item": {"type": "message", "id": format!("item_{}", item_idx), "role": "assistant"}
                        }))?;
                        send(&mut conn, "response.content_part.added", &json!({
                            "type": "response.content_part.added",
                            "item_id": format!("item_{}", item_idx),
                            "output_index": item_idx,
                            "content_index": 0,
                            "part": {"type": "output_text", "text": ""}
                        }))?;
                        message_started = true;
                    }
                    send(&mut conn, "response.output_text.delta", &json!({
                        "type": "response.output_text.delta",
                        "item_id": format!("item_{}", item_idx),
                        "output_index": item_idx,
                        "content_index": 0,
                        "delta": safe
                    }))?;
                }

                all_calls.extend(calls);
            }
            Event::Done(d) => {
                // Flush
                let (r, c) = split.push("", true);
                if !r.is_empty() && thinking_enabled && !reasoning_started {
                    send(&mut conn, "response.output_item.added", &json!({
                        "type": "response.output_item.added",
                        "output_index": item_idx,
                        "item": {"type": "reasoning", "id": format!("item_{}", item_idx)}
                    }))?;
                    reasoning_started = true;
                }
                if !r.is_empty() && reasoning_started {
                    send(&mut conn, "response.reasoning_summary_text.delta", &json!({
                        "type": "response.reasoning_summary_text.delta",
                        "item_id": format!("item_{}", item_idx),
                        "output_index": item_idx,
                        "delta": r
                    }))?;
                }

                let (safe, calls) = tool_parser.push(&c, true);
                all_calls.extend(calls);

                if !safe.is_empty() {
                    if !message_started {
                        if reasoning_started {
                            send(&mut conn, "response.output_item.done", &json!({
                                "type": "response.output_item.done",
                                "output_index": item_idx,
                                "item": {"type": "reasoning", "id": format!("item_{}", item_idx)}
                            }))?;
                            item_idx += 1;
                        }
                        send(&mut conn, "response.output_item.added", &json!({
                            "type": "response.output_item.added",
                            "output_index": item_idx,
                            "item": {"type": "message", "id": format!("item_{}", item_idx), "role": "assistant"}
                        }))?;
                        send(&mut conn, "response.content_part.added", &json!({
                            "type": "response.content_part.added",
                            "item_id": format!("item_{}", item_idx),
                            "output_index": item_idx,
                            "content_index": 0,
                            "part": {"type": "output_text", "text": ""}
                        }))?;
                        message_started = true;
                    }
                    send(&mut conn, "response.output_text.delta", &json!({
                        "type": "response.output_text.delta",
                        "item_id": format!("item_{}", item_idx),
                        "output_index": item_idx,
                        "content_index": 0,
                        "delta": safe
                    }))?;
                }

                // Close message item if started
                if message_started {
                    send(&mut conn, "response.output_text.done", &json!({
                        "type": "response.output_text.done",
                        "item_id": format!("item_{}", item_idx),
                        "output_index": item_idx,
                        "content_index": 0
                    }))?;
                    send(&mut conn, "response.content_part.done", &json!({
                        "type": "response.content_part.done",
                        "item_id": format!("item_{}", item_idx),
                        "output_index": item_idx,
                        "content_index": 0
                    }))?;
                    send(&mut conn, "response.output_item.done", &json!({
                        "type": "response.output_item.done",
                        "output_index": item_idx,
                        "item": {"type": "message", "id": format!("item_{}", item_idx)}
                    }))?;
                    item_idx += 1;
                } else if reasoning_started {
                    send(&mut conn, "response.output_item.done", &json!({
                        "type": "response.output_item.done",
                        "output_index": item_idx,
                        "item": {"type": "reasoning", "id": format!("item_{}", item_idx)}
                    }))?;
                    item_idx += 1;
                }

                // Emit function call items
                for call in &all_calls {
                    send(&mut conn, "response.output_item.added", &json!({
                        "type": "response.output_item.added",
                        "output_index": item_idx,
                        "item": {
                            "type": "function_call",
                            "id": format!("item_{}", item_idx),
                            "call_id": call.id,
                            "name": call.name,
                            "arguments": ""
                        }
                    }))?;
                    send(&mut conn, "response.function_call_arguments.delta", &json!({
                        "type": "response.function_call_arguments.delta",
                        "item_id": format!("item_{}", item_idx),
                        "output_index": item_idx,
                        "delta": call.arguments.to_string()
                    }))?;
                    send(&mut conn, "response.function_call_arguments.done", &json!({
                        "type": "response.function_call_arguments.done",
                        "item_id": format!("item_{}", item_idx),
                        "output_index": item_idx,
                        "arguments": call.arguments.to_string()
                    }))?;
                    send(&mut conn, "response.output_item.done", &json!({
                        "type": "response.output_item.done",
                        "output_index": item_idx,
                        "item": {
                            "type": "function_call",
                            "id": format!("item_{}", item_idx),
                            "call_id": call.id,
                            "name": call.name,
                            "arguments": call.arguments.to_string()
                        }
                    }))?;
                    item_idx += 1;
                }

                let status = if !all_calls.is_empty() { "incomplete" } else { "completed" };

                // response.completed
                send(&mut conn, "response.completed", &json!({
                    "type": "response.completed",
                    "response": {
                        "id": id,
                        "object": "response",
                        "status": status,
                        "model": model,
                        "usage": usage(&d)
                    }
                }))?;
            }
        }
    }

    Ok(())
}
