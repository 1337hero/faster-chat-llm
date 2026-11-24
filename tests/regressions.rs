//! CPU-only regression checks exercise the binary's real adapters and scheduler.
//! Only the GPU engine is replaced, with a deterministic A/B/C token source.
#![expect(
    dead_code,
    reason = "CPU checks import binary modules without GPU entry points"
)]
#[path = "../src/api/mod.rs"]
mod api;
#[path = "../src/chat.rs"]
mod chat;
#[path = "../src/gguf/mod.rs"]
mod gguf;
#[path = "../src/sampler.rs"]
mod sampler;
#[path = "../src/sched.rs"]
mod sched;
#[path = "../src/tokenizer.rs"]
mod tokenizer;

mod hip {
    pub const BMAX: usize = 16;
}
mod engine {
    pub struct RowOut {
        pub token: u32,
    }
    pub struct Engine {
        pub calls: u32,
    }
    impl Engine {
        pub fn reset(&mut self, _: usize) {}
        pub fn accept(&mut self, _: usize, _: usize) {}
        pub fn logits(&self, _: usize) -> Vec<f32> {
            unreachable!()
        }
        pub fn draft(&mut self, _: &[(usize, u32)], _: usize) -> Vec<Vec<u32>> {
            unreachable!()
        }
        pub fn step(&mut self, _: &[(usize, &[u32], bool)]) -> Vec<RowOut> {
            let token = 65 + self.calls;
            self.calls += 1;
            vec![RowOut { token }]
        }
    }
}
fn defaults() -> sampler::SampleParams {
    sampler::SampleParams {
        temperature: 0.0,
        top_k: 1,
        top_p: 1.0,
        min_p: 0.0,
        seed: 1,
    }
}
static NEXT_FIXTURE: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
fn fixture() -> gguf::Gguf {
    fn string(b: &mut Vec<u8>, s: &str) {
        b.extend((s.len() as u64).to_le_bytes());
        b.extend(s.as_bytes());
    }
    let mut b = Vec::new();
    b.extend(gguf::GGUF_MAGIC.to_le_bytes());
    b.extend(3u32.to_le_bytes());
    b.extend(0u64.to_le_bytes());
    b.extend(2u64.to_le_bytes());
    string(&mut b, "tokenizer.ggml.tokens");
    b.extend(9u32.to_le_bytes());
    b.extend(8u32.to_le_bytes());
    b.extend(256u64.to_le_bytes());
    for v in 0..=255u8 {
        string(&mut b, &tokenizer::byte_level_encode(&[v]));
    }
    string(&mut b, "tokenizer.ggml.merges");
    b.extend(9u32.to_le_bytes());
    b.extend(8u32.to_le_bytes());
    b.extend(0u64.to_le_bytes());
    b.resize(b.len().next_multiple_of(32), 0);
    let path = std::env::temp_dir().join(format!(
        "fcl-review-fixture-{}-{}.gguf",
        std::process::id(),
        NEXT_FIXTURE.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
    ));
    std::fs::write(&path, b).unwrap();
    let g = gguf::Gguf::open(&path).unwrap();
    std::fs::remove_file(path).unwrap();
    g
}
#[test]
fn whitespace_matches_documented_regex() {
    assert_eq!(tokenizer::pretokenise("a  b"), ["a", " ", " b"]);
}
#[test]
fn malformed_tool_call_does_not_panic() {
    let mut p = api::toolcall::ToolCallParser::new();
    let _ = p.push("<tool_call></function><function=x></tool_call>", true);
}
#[test]
fn oversized_token_ids_are_rejected() {
    let tok = tokenizer::Tokenizer::from_gguf(&fixture()).unwrap();
    let r = api::openai_completions::parse_request(
        &serde_json::json!({"prompt":[4294967295u64]}),
        &tok,
        &defaults(),
    );
    assert!(
        r.is_err(),
        "accepted token IDs: {:?}",
        r.unwrap().prompt_ids
    );
}
fn run_driver(max_tokens: usize, stop: Vec<String>, ctx: usize) -> (String, usize) {
    let tok = tokenizer::Tokenizer::from_gguf(&fixture()).unwrap();
    let (tx, rx) = std::sync::mpsc::channel();
    let (events, out) = std::sync::mpsc::channel();
    tx.send(sched::Job {
        prompt: vec![1],
        max_tokens,
        sampling: defaults(),
        stop,
        tx: events,
    })
    .unwrap();
    let driver = sched::Driver::new(
        engine::Engine { calls: 0 },
        tok,
        1,
        sched::Config {
            draft_n: 0,
            ctx,
            chunk: 16,
        },
    );
    let thread = std::thread::spawn(move || driver.run(rx));
    let mut text = String::new();
    let n;
    loop {
        match out.recv_timeout(std::time::Duration::from_secs(2)).unwrap() {
            sched::Event::Text(t) => text.push_str(&t),
            sched::Event::Done(d) => {
                n = d.completion_tokens;
                break;
            }
        }
    }
    drop(tx);
    thread.join().unwrap();
    (text, n)
}
#[test]
fn context_limit_flushes_held_text() {
    let (text, n) = run_driver(100, vec!["ZZZZ".into()], 3);
    assert_eq!(n, 3);
    assert_eq!(text, "ABC");
}
#[test]
fn zero_max_tokens_generates_nothing() {
    let (text, n) = run_driver(0, vec![], 16);
    assert_eq!((text, n), (String::new(), 0));
}
fn done() -> sched::Done {
    sched::Done {
        finish_reason: "stop",
        prompt_tokens: 1,
        cached_tokens: 0,
        completion_tokens: 1,
        prefill_s: 0.0,
        decode_s: 0.0,
        drafted: 0,
        accepted: 0,
    }
}
#[test]
fn streaming_tool_calls_have_indexes() {
    use std::io::Read;
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let mut client = std::net::TcpStream::connect(listener.local_addr().unwrap()).unwrap();
    client
        .set_read_timeout(Some(std::time::Duration::from_secs(2)))
        .unwrap();
    let (server, _) = listener.accept().unwrap();
    let (tx, rx) = std::sync::mpsc::channel();
    tx.send(sched::Event::Text(
        "<tool_call><function=f></function></tool_call>".into(),
    ))
    .unwrap();
    tx.send(sched::Event::Done(done())).unwrap();
    drop(tx);
    let t = std::thread::spawn(move || {
        api::openai_chat::handle_stream(server, rx, "test", false, true, false).unwrap()
    });
    let mut wire = String::new();
    client.read_to_string(&mut wire).unwrap();
    t.join().unwrap();
    let chunks: Vec<serde_json::Value> = wire
        .lines()
        .filter_map(|l| l.strip_prefix("data: "))
        .filter_map(|l| serde_json::from_str(l).ok())
        .collect();
    let call = chunks
        .iter()
        .find_map(|c| c.pointer("/choices/0/delta/tool_calls/0"))
        .unwrap();
    assert_eq!(call.get("index"), Some(&serde_json::json!(0)));
}
#[test]
fn zero_gguf_alignment_returns_an_error() {
    let mut b = Vec::new();
    b.extend(gguf::GGUF_MAGIC.to_le_bytes());
    b.extend(3u32.to_le_bytes());
    b.extend(0u64.to_le_bytes());
    b.extend(1u64.to_le_bytes());
    let key = "general.alignment";
    b.extend((key.len() as u64).to_le_bytes());
    b.extend(key.as_bytes());
    b.extend(4u32.to_le_bytes());
    b.extend(0u32.to_le_bytes());
    b.resize(64, 0);
    let path =
        std::env::temp_dir().join(format!("fcl-review-zero-align-{}.gguf", std::process::id()));
    std::fs::write(&path, b).unwrap();
    let result = gguf::Gguf::open(&path);
    std::fs::remove_file(path).unwrap();
    assert!(result.is_err());
}

#[test]
fn token_id_arrays_are_checked_in_both_adapters() {
    use serde_json::json;
    let tok = tokenizer::Tokenizer::from_gguf(&fixture()).unwrap();
    for prompt in [
        json!([-1]),
        json!([256]),
        json!([4294967296u64]),
        json!([1.5]),
        json!(["A", -1]),
        json!([true]),
    ] {
        let request = json!({"prompt": prompt});
        assert!(
            api::openai_completions::parse_request(&request, &tok, &defaults()).is_err(),
            "{request}"
        );
        assert!(
            api::llama_native::parse_completion_request(&request, &tok, &defaults()).is_err(),
            "{request}"
        );
    }
    let request = json!({"prompt": [0, 255]});
    assert_eq!(
        api::openai_completions::parse_request(&request, &tok, &defaults())
            .unwrap()
            .prompt_ids,
        Some(vec![0, 255])
    );
    assert_eq!(
        api::llama_native::parse_completion_request(&request, &tok, &defaults())
            .unwrap()
            .prompt_ids,
        Some(vec![0, 255])
    );
}

#[test]
fn native_zero_generation_limit_is_preserved() {
    let tok = tokenizer::Tokenizer::from_gguf(&fixture()).unwrap();
    let req = api::llama_native::parse_completion_request(
        &serde_json::json!({"prompt": "A", "n_predict": 0}),
        &tok,
        &defaults(),
    )
    .unwrap();
    assert_eq!(req.max_tokens, 0);
}

#[test]
fn whitespace_keeps_trailing_and_newline_behavior() {
    for (input, expected) in [
        ("a   b", vec!["a", "  ", " b"]),
        ("a\t\tb", vec!["a", "\t", "\tb"]),
        ("a  ", vec!["a", "  "]),
        ("a \nb", vec!["a", " \n", "b"]),
    ] {
        assert_eq!(tokenizer::pretokenise(input), expected, "{input:?}");
    }
}

#[test]
fn normal_length_completion_does_not_duplicate_text() {
    assert_eq!(run_driver(3, vec!["ZZZZ".into()], 16), ("ABC".into(), 3));
    assert_eq!(run_driver(3, vec![], 16), ("ABC".into(), 3));
}

#[test]
fn matched_stop_and_its_prefix_are_not_flushed() {
    assert_eq!(run_driver(20, vec!["BC".into()], 16).0, "A");
    assert_eq!(run_driver(20, vec!["ABC".into()], 16).0, "");
}
