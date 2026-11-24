//! Multi-protocol HTTP front end: OpenAI, Anthropic, llama-native APIs.
//! One thread per connection; generation itself happens on the driver thread.

use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::Arc;
use std::sync::mpsc::{Sender, channel};

use serde_json::{Map, Value, json};

use crate::api;
use crate::chat::ChatTemplate;
use crate::sampler::SampleParams;
use crate::sched::Job;
use crate::tokenizer::Tokenizer;

pub struct Shared {
    pub model: String,
    pub tok: Tokenizer,
    pub template: ChatTemplate,
    /// Template variables applied unless the request overrides them (e.g. enable_thinking).
    pub default_kwargs: Map<String, Value>,
    pub defaults: SampleParams,
    pub jobs: Sender<Job>,
    /// Number of slots (for /props, /slots).
    pub n_slots: usize,
    /// Context size (for /props).
    pub ctx_size: usize,
}

pub fn serve(listener: TcpListener, shared: Arc<Shared>) {
    for conn in listener.incoming().flatten() {
        let shared = shared.clone();
        std::thread::spawn(move || {
            let _ = handle(conn, &shared);
        });
    }
}

fn respond_cors(conn: &mut TcpStream, status: &str, body: &Value) -> std::io::Result<()> {
    let body = body.to_string();
    write!(conn, "HTTP/1.1 {status}\r\nContent-Type: application/json\r\nAccess-Control-Allow-Origin: *\r\nAccess-Control-Allow-Methods: GET, POST, OPTIONS\r\nAccess-Control-Allow-Headers: *\r\nConnection: close\r\n\r\n{body}")
}

fn respond_options(conn: &mut TcpStream) -> std::io::Result<()> {
    write!(conn, "HTTP/1.1 204 No Content\r\nAccess-Control-Allow-Origin: *\r\nAccess-Control-Allow-Methods: GET, POST, OPTIONS, DELETE\r\nAccess-Control-Allow-Headers: *\r\nAccess-Control-Max-Age: 86400\r\nConnection: close\r\n\r\n")
}

struct Request {
    method: String,
    path: String,
    body: Vec<u8>,
}

fn parse_request(conn: &mut TcpStream) -> std::io::Result<Request> {
    let mut reader = BufReader::new(conn.try_clone()?);
    let mut line = String::new();
    reader.read_line(&mut line)?;
    let mut parts = line.split_whitespace();
    let method = parts.next().unwrap_or("").to_string();
    let full_path = parts.next().unwrap_or("").to_string();
    let path = full_path.split('?').next().unwrap_or("").to_string();

    let mut len = 0usize;
    loop {
        let mut h = String::new();
        if reader.read_line(&mut h)? == 0 || h.trim().is_empty() {
            break;
        }
        if let Some((k, v)) = h.split_once(':') {
            if k.eq_ignore_ascii_case("content-length") {
                len = v.trim().parse().unwrap_or(0);
            }
            // Accept and ignore Authorization, anthropic-version, x-api-key headers
        }
    }
    let mut body = vec![0u8; len];
    reader.read_exact(&mut body)?;

    Ok(Request { method, path, body })
}

fn handle(mut conn: TcpStream, sh: &Shared) -> std::io::Result<()> {
    conn.set_nodelay(true)?;
    let req = parse_request(&mut conn)?;

    // CORS preflight
    if req.method == "OPTIONS" {
        return respond_options(&mut conn);
    }

    // Route the request
    match (req.method.as_str(), req.path.as_str()) {
        // Health endpoints
        ("GET", "/health" | "/v1/health") => {
            respond_cors(&mut conn, "200 OK", &api::llama_native::health())
        }

        // Model info
        ("GET", "/v1/models" | "/models") => {
            respond_cors(&mut conn, "200 OK", &json!({
                "object": "list",
                "data": [{
                    "id": sh.model,
                    "object": "model",
                    "owned_by": "local"
                }]
            }))
        }

        // OpenAI Chat Completions
        ("POST", "/v1/chat/completions" | "/chat/completions") => {
            handle_chat_completions(conn, sh, &req.body)
        }

        // OpenAI legacy Completions
        ("POST", "/v1/completions" | "/completions") => {
            handle_completions(conn, sh, &req.body)
        }

        // OpenAI Responses API
        ("POST", "/v1/responses") => {
            handle_responses(conn, sh, &req.body)
        }

        // Anthropic Messages
        ("POST", "/v1/messages") => {
            handle_anthropic_messages(conn, sh, &req.body)
        }

        // Anthropic count_tokens
        ("POST", "/v1/messages/count_tokens") => {
            handle_anthropic_count_tokens(&mut conn, sh, &req.body)
        }

        // llama-native /completion
        ("POST", "/completion") => {
            handle_llama_completion(conn, sh, &req.body)
        }

        // llama-native /tokenize
        ("POST", "/tokenize") => {
            match serde_json::from_slice::<Value>(&req.body) {
                Ok(r) => respond_cors(&mut conn, "200 OK", &api::llama_native::tokenize(&r, &sh.tok)),
                Err(e) => respond_cors(&mut conn, "400 Bad Request", &openai_err(&format!("invalid JSON: {e}"))),
            }
        }

        // llama-native /detokenize
        ("POST", "/detokenize") => {
            match serde_json::from_slice::<Value>(&req.body) {
                Ok(r) => respond_cors(&mut conn, "200 OK", &api::llama_native::detokenize(&r, &sh.tok)),
                Err(e) => respond_cors(&mut conn, "400 Bad Request", &openai_err(&format!("invalid JSON: {e}"))),
            }
        }

        // llama-native /apply-template
        ("POST", "/apply-template") => {
            match serde_json::from_slice::<Value>(&req.body) {
                Ok(r) => match api::llama_native::apply_template(&r, &sh.template, &sh.default_kwargs) {
                    Ok(v) => respond_cors(&mut conn, "200 OK", &v),
                    Err(e) => respond_cors(&mut conn, "400 Bad Request", &openai_err(&e)),
                },
                Err(e) => respond_cors(&mut conn, "400 Bad Request", &openai_err(&format!("invalid JSON: {e}"))),
            }
        }

        // llama-native /props
        ("GET", "/props") => {
            respond_cors(&mut conn, "200 OK", &api::llama_native::props(&sh.model, &sh.defaults, sh.n_slots, sh.ctx_size))
        }

        // llama-native /slots
        ("GET", "/slots") => {
            respond_cors(&mut conn, "200 OK", &api::llama_native::slots(sh.n_slots))
        }

        // Image endpoints - return 501 Not Implemented
        ("POST", p) if p.starts_with("/v1/images") => {
            respond_cors(&mut conn, "501 Not Implemented", &json!({
                "error": {
                    "message": "Image generation is not supported by this model",
                    "type": "invalid_request_error",
                    "code": "model_not_supported"
                }
            }))
        }

        // 404 for unknown routes
        _ => respond_cors(&mut conn, "404 Not Found", &openai_err("not found")),
    }
}

fn openai_err(msg: &str) -> Value {
    json!({"error": {"message": msg, "type": "invalid_request_error"}})
}

fn anthropic_err(msg: &str) -> Value {
    json!({"type": "error", "error": {"type": "invalid_request_error", "message": msg}})
}

// --- Request handlers ---

fn handle_chat_completions(mut conn: TcpStream, sh: &Shared, body: &[u8]) -> std::io::Result<()> {
    let req: Value = match serde_json::from_slice(body) {
        Ok(r) => r,
        Err(e) => return respond_cors(&mut conn, "400 Bad Request", &openai_err(&format!("invalid JSON: {e}"))),
    };

    // Check for image content
    if has_image_content(&req, "messages") {
        return respond_cors(&mut conn, "400 Bad Request", &openai_err("image input is not supported; this model is text-only"));
    }

    let gen_req = match api::openai_chat::parse_request(&req, &sh.template, &sh.tok, &sh.default_kwargs, &sh.defaults) {
        Ok(r) => r,
        Err(e) => return respond_cors(&mut conn, "400 Bad Request", &openai_err(&e)),
    };

    let prompt = sh.tok.encode(&gen_req.prompt_text, true);
    let (tx, rx) = channel();

    if sh.jobs.send(Job {
        prompt,
        max_tokens: gen_req.max_tokens,
        sampling: gen_req.sampling.clone(),
        stop: gen_req.stop.clone(),
        tx,
    }).is_err() {
        return respond_cors(&mut conn, "503 Service Unavailable", &openai_err("engine stopped"));
    }

    if gen_req.stream {
        api::openai_chat::handle_stream(
            conn, rx, &sh.model,
            gen_req.thinking.is_enabled(),
            gen_req.has_tools,
            gen_req.stream_include_usage,
        )
    } else {
        let resp = api::openai_chat::handle_sync(rx, &sh.model, gen_req.thinking.is_enabled(), gen_req.has_tools);
        respond_cors(&mut conn, "200 OK", &resp)
    }
}

fn handle_completions(mut conn: TcpStream, sh: &Shared, body: &[u8]) -> std::io::Result<()> {
    let req: Value = match serde_json::from_slice(body) {
        Ok(r) => r,
        Err(e) => return respond_cors(&mut conn, "400 Bad Request", &openai_err(&format!("invalid JSON: {e}"))),
    };

    let gen_req = match api::openai_completions::parse_request(&req, &sh.tok, &sh.defaults) {
        Ok(r) => r,
        Err(e) => return respond_cors(&mut conn, "400 Bad Request", &openai_err(&e)),
    };

    // Handle prompt encoding
    let prompt = if let Some(ids) = &gen_req.prompt_ids {
        ids.clone()
    } else {
        sh.tok.encode(&gen_req.prompt_text, true)
    };

    // For echo, we need the decoded prompt text
    let echo_prefix = if gen_req.echo {
        if gen_req.prompt_ids.is_some() {
            sh.tok.decode(&prompt)
        } else {
            gen_req.prompt_text.clone()
        }
    } else {
        String::new()
    };

    let (tx, rx) = channel();

    if sh.jobs.send(Job {
        prompt,
        max_tokens: gen_req.max_tokens,
        sampling: gen_req.sampling.clone(),
        stop: gen_req.stop.clone(),
        tx,
    }).is_err() {
        return respond_cors(&mut conn, "503 Service Unavailable", &openai_err("engine stopped"));
    }

    if gen_req.stream {
        api::openai_completions::handle_stream(conn, rx, &sh.model, &echo_prefix)
    } else {
        let resp = api::openai_completions::handle_sync(rx, &sh.model, &echo_prefix);
        respond_cors(&mut conn, "200 OK", &resp)
    }
}

fn handle_responses(mut conn: TcpStream, sh: &Shared, body: &[u8]) -> std::io::Result<()> {
    let req: Value = match serde_json::from_slice(body) {
        Ok(r) => r,
        Err(e) => return respond_cors(&mut conn, "400 Bad Request", &openai_err(&format!("invalid JSON: {e}"))),
    };

    // Check for image input
    if has_image_in_responses_input(&req) {
        return respond_cors(&mut conn, "400 Bad Request", &openai_err("image input is not supported; this model is text-only"));
    }

    let gen_req = match api::openai_responses::parse_request(&req, &sh.template, &sh.tok, &sh.default_kwargs, &sh.defaults) {
        Ok(r) => r,
        Err(e) => return respond_cors(&mut conn, "400 Bad Request", &openai_err(&e)),
    };

    let prompt = sh.tok.encode(&gen_req.prompt_text, true);
    let (tx, rx) = channel();

    if sh.jobs.send(Job {
        prompt,
        max_tokens: gen_req.max_tokens,
        sampling: gen_req.sampling.clone(),
        stop: gen_req.stop.clone(),
        tx,
    }).is_err() {
        return respond_cors(&mut conn, "503 Service Unavailable", &openai_err("engine stopped"));
    }

    if gen_req.stream {
        api::openai_responses::handle_stream(conn, rx, &sh.model, gen_req.thinking.is_enabled(), gen_req.has_tools)
    } else {
        let resp = api::openai_responses::handle_sync(rx, &sh.model, gen_req.thinking.is_enabled(), gen_req.has_tools);
        respond_cors(&mut conn, "200 OK", &resp)
    }
}

fn handle_anthropic_messages(mut conn: TcpStream, sh: &Shared, body: &[u8]) -> std::io::Result<()> {
    let req: Value = match serde_json::from_slice(body) {
        Ok(r) => r,
        Err(e) => return respond_cors(&mut conn, "400 Bad Request", &anthropic_err(&format!("invalid JSON: {e}"))),
    };

    // Check for image content
    if has_anthropic_image_content(&req) {
        return respond_cors(&mut conn, "400 Bad Request", &anthropic_err("image input is not supported; this model is text-only"));
    }

    let gen_req = match api::anthropic::parse_request(&req, &sh.template, &sh.tok, &sh.default_kwargs, &sh.defaults) {
        Ok(r) => r,
        Err(e) => return respond_cors(&mut conn, "400 Bad Request", &anthropic_err(&e)),
    };

    let prompt = sh.tok.encode(&gen_req.prompt_text, true);
    let (tx, rx) = channel();

    if sh.jobs.send(Job {
        prompt,
        max_tokens: gen_req.max_tokens,
        sampling: gen_req.sampling.clone(),
        stop: gen_req.stop.clone(),
        tx,
    }).is_err() {
        return respond_cors(&mut conn, "503 Service Unavailable", &anthropic_err("engine stopped"));
    }

    if gen_req.stream {
        api::anthropic::handle_stream(conn, rx, &sh.model, gen_req.thinking.is_enabled(), gen_req.has_tools)
    } else {
        let resp = api::anthropic::handle_sync(rx, &sh.model, gen_req.thinking.is_enabled(), gen_req.has_tools);
        respond_cors(&mut conn, "200 OK", &resp)
    }
}

fn handle_anthropic_count_tokens(conn: &mut TcpStream, sh: &Shared, body: &[u8]) -> std::io::Result<()> {
    let req: Value = match serde_json::from_slice(body) {
        Ok(r) => r,
        Err(e) => return respond_cors(conn, "400 Bad Request", &anthropic_err(&format!("invalid JSON: {e}"))),
    };

    match api::anthropic::count_tokens(&req, &sh.template, &sh.tok, &sh.default_kwargs) {
        Ok(v) => respond_cors(conn, "200 OK", &v),
        Err(e) => respond_cors(conn, "400 Bad Request", &anthropic_err(&e)),
    }
}

fn handle_llama_completion(mut conn: TcpStream, sh: &Shared, body: &[u8]) -> std::io::Result<()> {
    let req: Value = match serde_json::from_slice(body) {
        Ok(r) => r,
        Err(e) => return respond_cors(&mut conn, "400 Bad Request", &openai_err(&format!("invalid JSON: {e}"))),
    };

    let gen_req = match api::llama_native::parse_completion_request(&req, &sh.tok, &sh.defaults) {
        Ok(r) => r,
        Err(e) => return respond_cors(&mut conn, "400 Bad Request", &openai_err(&e)),
    };

    let prompt = if let Some(ids) = &gen_req.prompt_ids {
        ids.clone()
    } else {
        sh.tok.encode(&gen_req.prompt_text, true)
    };

    let (tx, rx) = channel();

    if sh.jobs.send(Job {
        prompt,
        max_tokens: gen_req.max_tokens,
        sampling: gen_req.sampling.clone(),
        stop: gen_req.stop.clone(),
        tx,
    }).is_err() {
        return respond_cors(&mut conn, "503 Service Unavailable", &openai_err("engine stopped"));
    }

    if gen_req.stream {
        api::llama_native::handle_completion_stream(conn, rx, &sh.model, gen_req.return_tokens, &sh.tok)
    } else {
        let resp = api::llama_native::handle_completion_sync(rx, &sh.model, gen_req.return_tokens, &sh.tok);
        respond_cors(&mut conn, "200 OK", &resp)
    }
}

// --- Image detection helpers ---

fn has_image_content(req: &Value, key: &str) -> bool {
    if let Some(Value::Array(msgs)) = req.get(key) {
        for msg in msgs {
            if let Some(Value::Array(content)) = msg.get("content") {
                for part in content {
                    if part.get("type").and_then(Value::as_str) == Some("image_url") {
                        return true;
                    }
                }
            }
        }
    }
    false
}

fn has_image_in_responses_input(req: &Value) -> bool {
    if let Some(Value::Array(items)) = req.get("input") {
        for item in items {
            if item.get("type").and_then(Value::as_str) == Some("input_image") {
                return true;
            }
            if let Some(Value::Array(content)) = item.get("content") {
                for part in content {
                    if part.get("type").and_then(Value::as_str) == Some("input_image") {
                        return true;
                    }
                }
            }
        }
    }
    false
}

fn has_anthropic_image_content(req: &Value) -> bool {
    if let Some(Value::Array(msgs)) = req.get("messages") {
        for msg in msgs {
            if let Some(Value::Array(content)) = msg.get("content") {
                for block in content {
                    if block.get("type").and_then(Value::as_str) == Some("image") {
                        return true;
                    }
                }
            }
        }
    }
    false
}
