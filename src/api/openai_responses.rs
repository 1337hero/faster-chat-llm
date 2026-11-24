//! OpenAI Responses API: /v1/responses
//!
//! Stateless implementation with proper SSE semantic events.

use std::io::Write;
use std::net::TcpStream;
use std::sync::mpsc::Receiver;

use serde_json::{Map, Value, json};

use crate::chat::ChatTemplate;
use crate::sampler::SampleParams;
use crate::sched::{Done, Event};
use crate::tokenizer::Tokenizer;

use super::internal::{GenRequest, Thinking, extract_sampling, merge_kwargs, openai_error};
use super::openai_chat::ThinkSplit;
use super::toolcall::{ToolCall, ToolCallParser};

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
    let thinking = kwargs
        .get("enable_thinking")
        .and_then(Value::as_bool)
        .unwrap_or(true);

    let sampling = extract_sampling(req, defaults);

    // Extract stop from stop_sequences (Responses API name)
    let stop = match req.get("stop_sequences") {
        Some(Value::Array(a)) => a
            .iter()
            .filter_map(|v| v.as_str().map(str::to_string))
            .collect(),
        _ => Vec::new(),
    };

    let max_tokens = req
        .get("max_output_tokens")
        .and_then(Value::as_u64)
        .map_or(usize::MAX, |v| v as usize);
    let stream = req.get("stream").and_then(Value::as_bool).unwrap_or(false);

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
                        let args = item
                            .get("arguments")
                            .and_then(Value::as_str)
                            .unwrap_or("{}");
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
                            return Err(
                                "image input is not supported; this model is text-only".into()
                            );
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
    let converted: Vec<Value> = arr
        .iter()
        .filter_map(|t| {
            let name = t.get("name")?.as_str()?;
            let description = t.get("description").and_then(Value::as_str).unwrap_or("");
            let parameters = t
                .get("parameters")
                .cloned()
                .unwrap_or(json!({"type": "object"}));
            Some(json!({
                "type": "function",
                "function": {
                    "name": name,
                    "description": description,
                    "parameters": parameters
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

/// Generate response ID.
fn gen_id() -> String {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_nanos());
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

/// The finished output items of a response: optional reasoning, the message, then tool calls.
fn output_items(reasoning: &str, content: &str, calls: &[ToolCall]) -> Vec<Value> {
    let mut output = Vec::new();
    if !reasoning.is_empty() {
        output.push(json!({
            "type": "reasoning", "id": format!("rs_{}", output.len()),
            "summary": [{"type": "summary_text", "text": reasoning}]
        }));
    }
    if !content.is_empty() || calls.is_empty() {
        output.push(json!({
            "type": "message", "id": format!("msg_{}", output.len()), "role": "assistant", "status": "completed",
            "content": [{"type": "output_text", "text": content, "annotations": []}]
        }));
    }
    for call in calls {
        output.push(json!({
            "type": "function_call", "id": format!("fc_{}", output.len()), "status": "completed",
            "call_id": call.id, "name": call.name, "arguments": call.arguments.to_string()
        }));
    }
    output
}

fn response_object(id: &str, model: &str, status: &str, output: Vec<Value>, usage: Value) -> Value {
    json!({
        "id": id, "object": "response", "created_at": created_at(), "status": status, "model": model,
        "output": output, "usage": usage,
        "parallel_tool_calls": true, "tool_choice": "auto", "tools": [],
    })
}

fn created_at() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_secs())
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
    let (mut reasoning, mut content, mut calls) = (String::new(), String::new(), Vec::new());
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
        return openai_error("engine stopped", "server_error", None);
    };
    response_object(
        &id,
        model,
        "completed",
        output_items(&reasoning, &content, &calls),
        usage(&d),
    )
}

/// Handle streaming response with semantic events. Text streams as it arrives; reasoning and
/// tool calls are announced as whole items once known.
#[allow(unused_assignments)] // the macro's last expansion sets a flag nothing reads afterwards
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

    let mut seq = 0u64;
    let mut send = |conn: &mut TcpStream, event: &str, mut data: Value| -> std::io::Result<()> {
        data["type"] = json!(event);
        data["sequence_number"] = json!(seq);
        seq += 1;
        write!(conn, "event: {event}\ndata: {data}\n\n")?;
        conn.flush()
    };

    let skeleton = response_object(&id, model, "in_progress", Vec::new(), Value::Null);
    send(&mut conn, "response.created", json!({"response": skeleton}))?;
    send(
        &mut conn,
        "response.in_progress",
        json!({"response": skeleton}),
    )?;

    let mut split = ThinkSplit::new(thinking_enabled);
    let mut tool_parser = ToolCallParser::new();
    let (mut reasoning, mut content, mut calls) =
        (String::new(), String::new(), Vec::<ToolCall>::new());
    // Output index of the message item once its first text has been sent.
    let mut message_at: Option<usize> = None;
    let mut next_index = 0usize;
    let mut done_info = None;

    // Reasoning ends before the answer starts, so it can be emitted as one finished item the
    // moment the first answer text (or the end) arrives.
    let mut reasoning_sent = false;
    macro_rules! flush_reasoning {
        () => {
            if !reasoning_sent && !reasoning.is_empty() {
                let item = json!({"type": "reasoning", "id": format!("rs_{next_index}"), "summary": [{"type": "summary_text", "text": reasoning}]});
                send(&mut conn, "response.output_item.added", json!({"output_index": next_index, "item": {"type": "reasoning", "id": format!("rs_{next_index}"), "summary": []}}))?;
                send(&mut conn, "response.output_item.done", json!({"output_index": next_index, "item": item}))?;
                next_index += 1;
            }
            reasoning_sent = true;
        };
    }
    macro_rules! text_delta {
        ($text:expr) => {
            if !$text.is_empty() {
                flush_reasoning!();
                let at = match message_at {
                    Some(at) => at,
                    None => {
                        let at = next_index;
                        next_index += 1;
                        message_at = Some(at);
                        send(&mut conn, "response.output_item.added", json!({"output_index": at, "item": {"type": "message", "id": format!("msg_{at}"), "role": "assistant", "status": "in_progress", "content": []}}))?;
                        send(&mut conn, "response.content_part.added", json!({"item_id": format!("msg_{at}"), "output_index": at, "content_index": 0, "part": {"type": "output_text", "text": "", "annotations": []}}))?;
                        at
                    }
                };
                send(&mut conn, "response.output_text.delta", json!({"item_id": format!("msg_{at}"), "output_index": at, "content_index": 0, "delta": $text, "logprobs": []}))?;
                content.push_str(&$text);
            }
        };
    }

    for ev in rx {
        match ev {
            Event::Text(t) => {
                let (r, c) = split.push(&t, false);
                reasoning.push_str(&r);
                let (safe, found) = tool_parser.push(&c, false);
                calls.extend(found);
                text_delta!(safe);
            }
            Event::Done(d) => done_info = Some(d),
        }
    }
    let (r, c) = split.push("", true);
    reasoning.push_str(&r);
    let (safe, found) = tool_parser.push(&c, true);
    calls.extend(found);
    text_delta!(safe);
    flush_reasoning!();

    let Some(d) = done_info else { return Ok(()) };
    // A response with no text and no tool calls still carries an (empty) message.
    if message_at.is_none() && calls.is_empty() {
        let at = next_index;
        next_index += 1;
        message_at = Some(at);
        send(
            &mut conn,
            "response.output_item.added",
            json!({"output_index": at, "item": {"type": "message", "id": format!("msg_{at}"), "role": "assistant", "status": "in_progress", "content": []}}),
        )?;
        send(
            &mut conn,
            "response.content_part.added",
            json!({"item_id": format!("msg_{at}"), "output_index": at, "content_index": 0, "part": {"type": "output_text", "text": "", "annotations": []}}),
        )?;
    }
    let mut output = Vec::new();
    if !reasoning.is_empty() {
        output.push(json!({"type": "reasoning", "id": format!("rs_{}", output.len()), "summary": [{"type": "summary_text", "text": reasoning}]}));
    }
    if let Some(at) = message_at {
        let part = json!({"type": "output_text", "text": content, "annotations": []});
        let item = json!({"type": "message", "id": format!("msg_{at}"), "role": "assistant", "status": "completed", "content": [part]});
        send(
            &mut conn,
            "response.output_text.done",
            json!({"item_id": format!("msg_{at}"), "output_index": at, "content_index": 0, "text": content, "logprobs": []}),
        )?;
        send(
            &mut conn,
            "response.content_part.done",
            json!({"item_id": format!("msg_{at}"), "output_index": at, "content_index": 0, "part": part}),
        )?;
        send(
            &mut conn,
            "response.output_item.done",
            json!({"output_index": at, "item": item}),
        )?;
        output.push(item);
    }
    for call in &calls {
        let at = next_index;
        next_index += 1;
        let args = call.arguments.to_string();
        let item = json!({"type": "function_call", "id": format!("fc_{at}"), "status": "completed", "call_id": call.id, "name": call.name, "arguments": args});
        send(
            &mut conn,
            "response.output_item.added",
            json!({"output_index": at, "item": {"type": "function_call", "id": format!("fc_{at}"), "status": "in_progress", "call_id": call.id, "name": call.name, "arguments": ""}}),
        )?;
        send(
            &mut conn,
            "response.function_call_arguments.delta",
            json!({"item_id": format!("fc_{at}"), "output_index": at, "delta": args}),
        )?;
        send(
            &mut conn,
            "response.function_call_arguments.done",
            json!({"item_id": format!("fc_{at}"), "output_index": at, "name": call.name, "arguments": args}),
        )?;
        send(
            &mut conn,
            "response.output_item.done",
            json!({"output_index": at, "item": item}),
        )?;
        output.push(item);
    }
    send(
        &mut conn,
        "response.completed",
        json!({"response": response_object(&id, model, "completed", output, usage(&d))}),
    )?;
    Ok(())
}
