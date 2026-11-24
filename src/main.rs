//! faster-chat-llm — a GGUF-native inference engine for Qwen3.8-27B on ROCm/gfx1201.

mod api;
mod chat;
mod engine;
mod gguf;
mod hip;
mod model;
mod sampler;
mod sched;
mod server;
mod tokenizer;

use std::collections::BTreeMap;
use std::process::ExitCode;

use gguf::Gguf;

fn usage() -> ExitCode {
    eprintln!(
        "faster-chat-llm — GGUF-native engine for Qwen3.8-27B (ROCm/gfx1201)

usage: faster-chat-llm <command> <gguf> [args]

  serve     OpenAI / Anthropic / llama-server compatible HTTP server
              --port 8996  --host 127.0.0.1  --alias NAME
              --tp 2            cards (tensor-parallel ranks)
              --ctx 262144      KV capacity in tokens, shared by all slots
              --slots 2         concurrent sequences
              --draft 7         MTP drafts per decode step (0 = off)
              --draft-vocab K   draft over the first K token ids (default 102400)
              --chunk 2048      prefill rows per step
              --no-think        default enable_thinking to false
  gen       greedy generation from a raw prompt
              --prompt TEXT | --ids 1,2,3 | --ids-file F    -n N    --tp N    --ctx N    --chunk N
              --draft N  --draft-vocab K     --profile    --dump (per-layer hidden state)
  score     teacher-force token ids and print top-k next-token log-probabilities
              --ids-file F  --prefix N  [--topk 20] [--tp N] [--ctx N]
  render    <user text> [--no-think]   print the chat prompt and its token ids
  inspect   header, metadata, tensor table, per-type totals
  meta      [prefix]   dump metadata keys
  tensor    <name>     one tensor's info and dequantised head
  dump      machine-readable tensor table (name, type, dims, offset)
  encode    <text|->   token ids for a string (- reads stdin)
  tokcheck  <jsonl>    diff token ids against a reference, one JSON object per line
"
    );
    ExitCode::from(2)
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        return usage();
    }
    let cmd = args[1].as_str();
    let path = &args[2];

    let gguf = match Gguf::open(path) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("open {path}: {e}");
            return ExitCode::FAILURE;
        }
    };

    match cmd {
        "inspect" => inspect(&gguf),
        "meta" => dump_meta(&gguf, args.get(3).map(|s| s.as_str())),
        "dump" => dump_tsv(&gguf),
        "encode" => match args.get(3).map(|s| s.as_str()) {
            // "-" reads the text from stdin (arguments are capped at 128 KiB)
            Some("-") => encode_cmd(&gguf, &std::io::read_to_string(std::io::stdin()).unwrap_or_default()),
            other => encode_cmd(&gguf, other.unwrap_or("")),
        },
        "tokcheck" => match args.get(3) {
            Some(path) => tokcheck(&gguf, path),
            None => usage(),
        },
        "gen" => generate(&gguf, &args[3..]),
        "score" => score(&gguf, &args[3..]),
        "serve" => serve(&gguf, &args[3..]),
        "render" => render(&gguf, &args[3..]),
        "tensor" => match args.get(3) {
            Some(name) => show_tensor(&gguf, name),
            None => usage(),
        },
        _ => usage(),
    }
}

fn inspect(g: &Gguf) -> ExitCode {
    let file_len = match g.file_len() {
        Ok(v) => v,
        Err(e) => {
            eprintln!("file length: {e}");
            return ExitCode::FAILURE;
        }
    };
    if let Err(e) = g.validate_layout() {
        eprintln!("layout check FAILED: {e}");
        return ExitCode::FAILURE;
    }

    println!("file            {}", g.path().display());
    println!("bytes           {file_len}");
    println!("gguf version    {}", g.version);
    println!("alignment       {}", g.alignment);
    println!("data offset     {}", g.data_offset);
    println!("metadata keys   {}", g.meta.len());
    println!("tensors         {}", g.tensors().len());
    println!("arch            {}", g.str_of("general.architecture").unwrap_or("?"));
    println!("name            {}", g.str_of("general.name").unwrap_or("?"));

    let mut by_type: BTreeMap<String, (u64, u64)> = BTreeMap::new();
    for t in g.tensors() {
        let e = by_type.entry(t.ty.name().to_string()).or_insert((0, 0));
        e.0 += 1;
        e.1 += t.n_bytes();
    }
    println!("\n{:<10} {:>8} {:>18}", "type", "count", "bytes");
    let mut total = 0u64;
    for (name, (count, bytes)) in &by_type {
        println!("{name:<10} {count:>8} {bytes:>18}");
        total += bytes;
    }
    println!("{:─<10} {:─>8} {:─>18}", "", "", "");
    println!("{:<10} {:>8} {:>18}", "total", g.tensors().len(), total);

    let expected = file_len.saturating_sub(g.data_offset);
    let pad = expected.saturating_sub(total);
    println!("\ntensor data span {expected} bytes, accounted {total}, slack {pad}");

    // shapes that matter for this model
    println!("\nkey tensors:");
    for t in g.tensors() {
        let is_key = !t.name.starts_with("blk.")
            || t.name.contains("nextn")
            || t.name.ends_with("attn_q.weight")
            || t.name.ends_with("attn_output.weight")
            || t.name.ends_with("ffn_gate.weight")
            || t.name.ends_with("ssm_a")
            || t.name.ends_with("ssm_conv1d.weight");
        if !is_key {
            continue;
        }
        let shape: Vec<String> = t.shape().iter().map(|d| d.to_string()).collect();
        println!(
            "  {:<44} {:<6} [{}]  {} B",
            t.name,
            t.dtype_name(),
            shape.join(", "),
            t.n_bytes()
        );
    }
    ExitCode::SUCCESS
}

fn dump_meta(g: &Gguf, prefix: Option<&str>) -> ExitCode {
    for (k, v) in &g.meta {
        if let Some(p) = prefix {
            if !k.starts_with(p) {
                continue;
            }
        }
        print!("{k:<52} {:<8}", v.kind());
        if let gguf::types::MetaValue::String(s) = v {
            if s.len() > 200 || s.contains('\n') {
                println!(" {} chars, first line:", s.len());
                println!("    {}", s.lines().next().unwrap_or(""));
                continue;
            }
        }
        println!(" {v}");
    }
    ExitCode::SUCCESS
}

fn show_tensor(g: &Gguf, name: &str) -> ExitCode {
    let t = match g.tensor(name) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("{e}");
            return ExitCode::FAILURE;
        }
    };
    let shape: Vec<String> = t.shape().iter().map(|d| d.to_string()).collect();
    println!("name   {}", t.name);
    println!("type   {}", t.dtype_name());
    println!("shape  [{}]", shape.join(", "));
    println!("elems  {}", t.n_elements);
    println!("bytes  {}", t.n_bytes());
    println!("offset {}", t.offset);

    if !t.ty.is_supported() {
        println!("\n(dequantisation not implemented for {})", t.ty);
        return ExitCode::SUCCESS;
    }
    let raw = match g.read_raw(name) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("read: {e}");
            return ExitCode::FAILURE;
        }
    };
    let mut deq = vec![0f32; t.n_elements as usize];
    gguf::dequant::dequantise(t.ty, &raw, &mut deq);
    let row = *t.dims.first().unwrap_or(&1) as usize;
    println!("\nfirst row ({row} values), first 12:");
    let shown: Vec<String> = deq.iter().take(12).map(|v| format!("{v:+.6}")).collect();
    println!("  {}", shown.join("  "));
    let (mut min, mut max, mut sum) = (f32::INFINITY, f32::NEG_INFINITY, 0f64);
    for &v in &deq {
        min = min.min(v);
        max = max.max(v);
        sum += v as f64;
    }
    println!(
        "\nall: min {min:+.6}  max {max:+.6}  mean {:+.6}  rms {:.6}",
        sum / deq.len() as f64,
        (deq.iter().map(|v| (*v as f64) * (*v as f64)).sum::<f64>() / deq.len() as f64).sqrt()
    );
    ExitCode::SUCCESS
}

/// Tab-separated tensor table, for diffing against an independent GGUF parser.
fn dump_tsv(g: &Gguf) -> ExitCode {
    println!("name\ttype\tdims\toffset\tbytes");
    for t in g.tensors() {
        let dims: Vec<String> = t.dims.iter().map(|d| d.to_string()).collect();
        println!(
            "{}\t{}\t{}\t{}\t{}",
            t.name,
            t.dtype_name(),
            dims.join(","),
            t.offset,
            t.n_bytes()
        );
    }
    ExitCode::SUCCESS
}

fn encode_cmd(g: &Gguf, text: &str) -> ExitCode {
    let tok = match tokenizer::Tokenizer::from_gguf(g) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("tokenizer: {e}");
            return ExitCode::FAILURE;
        }
    };
    let ids = tok.encode(text, true);
    println!("{}", ids.iter().map(|i| i.to_string()).collect::<Vec<_>>().join(","));
    println!("# vocab {} decoded {:?}", tok.vocab_size(), tok.decode(&ids));
    ExitCode::SUCCESS
}

/// Each line: {"text": "...", "ids": [...]}. Reports the first mismatch per line.
fn tokcheck(g: &Gguf, path: &str) -> ExitCode {
    let tok = match tokenizer::Tokenizer::from_gguf(g) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("tokenizer: {e}");
            return ExitCode::FAILURE;
        }
    };
    let body = match std::fs::read_to_string(path) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("read {path}: {e}");
            return ExitCode::FAILURE;
        }
    };
    let (mut cases, mut ok) = (0usize, 0usize);
    let mut failures = Vec::new();
    for (lineno, line) in body.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        // minimal JSON: expects exactly {"text": "...", "ids": [n, ...]}
        let text = match between(line, "\"text\": ", ", \"ids\"") {
            Some(t) => unescape_json(t),
            None => {
                failures.push(format!("line {}: no text field", lineno + 1));
                continue;
            }
        };
        let ids_part = match line.split("\"ids\": [").nth(1).and_then(|s| s.split(']').next()) {
            Some(s) => s,
            None => {
                failures.push(format!("line {}: no ids field", lineno + 1));
                continue;
            }
        };
        let want: Vec<u32> = ids_part
            .split(',')
            .filter_map(|s| s.trim().parse::<u32>().ok())
            .collect();
        cases += 1;
        let got = tok.encode(&text, true);
        if got == want {
            ok += 1;
        } else {
            let at = got.iter().zip(&want).position(|(a, b)| a != b);
            failures.push(format!(
                "line {}: text {:?} diverges at token {:?}: got {:?} want {:?}\n   got  {:?}\n   want {:?}",
                lineno + 1,
                truncate(&text, 60),
                at,
                at.map(|i| got.get(i).copied()),
                at.map(|i| want.get(i).copied()),
                &got[..got.len().min(24)],
                &want[..want.len().min(24)]
            ));
        }
    }
    println!("{ok}/{cases} cases match");
    for f in failures.iter().take(12) {
        println!("  {f}");
    }
    if ok == cases && cases > 0 {
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
    }
}

fn truncate(s: &str, n: usize) -> String {
    if s.chars().count() <= n {
        s.to_string()
    } else {
        s.chars().take(n).collect::<String>() + "…"
    }
}

fn between<'a>(s: &'a str, a: &str, b: &str) -> Option<&'a str> {
    let i = s.find(a)? + a.len();
    let j = s[i..].find(b)? + i;
    Some(&s[i..j])
}

fn unescape_json(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut it = s.chars();
    while let Some(c) = it.next() {
        if c != '\\' {
            out.push(c);
            continue;
        }
        match it.next() {
            Some('n') => out.push('\n'),
            Some('t') => out.push('\t'),
            Some('r') => out.push('\r'),
            Some('"') => out.push('"'),
            Some('\\') => out.push('\\'),
            Some('u') => {
                let hex: String = (0..4).filter_map(|_| it.next()).collect();
                if let Ok(v) = u32::from_str_radix(&hex, 16) {
                    if let Some(c) = char::from_u32(v) {
                        out.push(c);
                    }
                }
            }
            Some(other) => out.push(other),
            None => {}
        }
    }
    out
}

fn opt<'a>(args: &'a [String], key: &str) -> Option<&'a str> {
    args.iter().position(|a| a == key).and_then(|i| args.get(i + 1)).map(|s| s.as_str())
}

fn generate(g: &Gguf, args: &[String]) -> ExitCode {
    use std::time::Instant;
    let num = |k: &str, d: usize| opt(args, k).and_then(|v| v.parse().ok()).unwrap_or(d);
    let (n_gen, tp, ctx, chunk) = (num("-n", 16), num("--tp", 1), num("--ctx", 4096), num("--chunk", 2048).max(hip::BMAX));
    let hp = match model::HParams::from_gguf(g) {
        Ok(h) => h,
        Err(e) => {
            eprintln!("{e}");
            return ExitCode::FAILURE;
        }
    };
    let tok = tokenizer::Tokenizer::from_gguf(g).expect("tokenizer");
    let ids_file = opt(args, "--ids-file").map(|p| std::fs::read_to_string(p).expect("read ids file"));
    let ids: Vec<u32> = match (ids_file.as_deref().or(opt(args, "--ids")), opt(args, "--prompt")) {
        (Some(s), _) => s.split(|c: char| !c.is_ascii_digit()).filter_map(|v| v.parse().ok()).collect(),
        (_, Some(p)) => tok.encode(p, true),
        _ => return usage(),
    };
    eprintln!("prompt tokens: {}", if ids.len() <= 64 { format!("{ids:?}") } else { format!("{} ids", ids.len()) });

    let mut eng = match build_engine(g, &hp, tp, ctx, 1, num("--draft-vocab", 102400), chunk) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("load: {e}");
            return ExitCode::FAILURE;
        }
    };

    let profile = args.iter().any(|a| a == "--profile");
    if profile {
        eng.prof = Some(std::cell::RefCell::new(engine::Prof::new()));
    }
    let t1 = Instant::now();
    let mut next = 0u32;
    let n_chunks = ids.chunks(chunk).count();
    for (i, c) in ids.chunks(chunk).enumerate() {
        eng.dump = args.iter().any(|a| a == "--dump") && i + 1 == n_chunks;
        let out = eng.step(&[(0, c, true)]);
        next = out.last().unwrap().token;
    }
    eng.dump = false;
    eprintln!("prefill {} tokens in {:.3}s", ids.len(), t1.elapsed().as_secs_f64());
    if args.iter().any(|a| a == "--dump") {
        let last_chunk = (ids.len() - 1) % chunk + 1;
        let lg = eng.logits(if last_chunk > hip::BMAX { 0 } else { last_chunk - 1 });
        let mut top: Vec<(usize, f32)> = lg.iter().copied().enumerate().collect();
        top.sort_by(|a, b| b.1.total_cmp(&a.1));
        for (id, v) in top.iter().take(10) {
            println!("logit {id} {v:.5} {:?}", tok.decode(&[*id as u32]));
        }
    }

    if profile {
        eprintln!("prefill profile:");
        eng.prof.as_ref().unwrap().borrow().report();
        eng.prof = Some(std::cell::RefCell::new(engine::Prof::new()));
    }
    let t2 = Instant::now();
    let n_draft = num("--draft", 0).min(hip::BMAX - 1);
    let mut out_ids = vec![next];
    let (mut steps, mut drafted, mut accepted) = (0usize, 0usize, 0usize);
    while out_ids.len() < n_gen {
        steps += 1;
        if n_draft == 0 {
            let out = eng.step(&[(0, &[next], false)]);
            eng.accept(0, 1);
            next = out[0].token;
            out_ids.push(next);
            continue;
        }
        // Self-speculative step: draft with the MTP head, verify all rows in one trunk pass,
        // keep the longest prefix the trunk agrees with. Output is the trunk's own greedy choice.
        let drafts = eng.draft(&[(0, next)], n_draft).remove(0);
        let mut rows = vec![next];
        rows.extend(&drafts);
        let out = eng.step(&[(0, &rows, false)]);
        let mut a = 1;
        while a <= drafts.len() && out[a - 1].token == drafts[a - 1] {
            a += 1;
        }
        eng.accept(0, a);
        drafted += drafts.len();
        accepted += a - 1;
        out_ids.extend(out[..a].iter().map(|o| o.token));
        next = out[a - 1].token;
    }
    out_ids.truncate(n_gen);
    if n_draft > 0 {
        eprintln!("mtp: {steps} steps, acceptance {:.3} ({accepted}/{drafted}), mean tokens/step {:.2}", accepted as f64 / drafted as f64, (accepted + steps) as f64 / steps as f64);
    }
    if let Some(p) = &eng.prof {
        eprintln!("decode profile:");
        p.borrow().report();
    }
    let dt = t2.elapsed().as_secs_f64();
    eprintln!("decode {} tokens in {:.3}s = {:.2} tok/s", out_ids.len() - 1, dt, (out_ids.len() - 1) as f64 / dt);
    println!("ids {}", out_ids.iter().map(|i| i.to_string()).collect::<Vec<_>>().join(","));
    println!("text {:?}", tok.decode(&out_ids));
    ExitCode::SUCCESS
}

/// Load the model onto `tp` cards and build an engine with one sequence slot.
fn build_engine(g: &Gguf, hp: &model::HParams, tp: usize, ctx: usize, n_slots: usize, draft_vocab: usize, rows: usize) -> Result<engine::Engine, String> {
    // The KV pool is addressed in 16-slot tiles and 4096-slot attention segments.
    let ctx = ctx.next_multiple_of(hip::ATTN_SEG_SLOTS);
    let t0 = std::time::Instant::now();
    let n_dev = hip::init();
    assert!(tp <= n_dev, "need {tp} devices, HIP sees {n_dev}");
    let mut ranks = Vec::new();
    for rank in 0..tp {
        hip::set_device(rank);
        let s = hip::stream_create();
        let mut wave = [0i32];
        let wb = hip::DevBuf::<i32>::new(1);
        unsafe { hip::pk_wave_size(s, wb.ptr()) };
        wb.download(&mut wave, s);
        hip::sync(s);
        assert_eq!(wave[0], 32, "kernels assume wave32");
        let sh = model::Shard { rank, n: tp };
        let mut loader = model::Loader::new(g, s);
        let w = match loader.load(hp, sh, draft_vocab, |l| {
            if l % 16 == 15 {
                eprintln!("  rank {rank}: layer {} loaded", l + 1)
            }
        }) {
            Ok(w) => w,
            Err(e) => return Err(e),
        };
        let st = loader.stats;
        drop(loader);
        let (free, total) = hip::mem_info();
        eprintln!(
            "rank {rank}: {:.2} GB in {:.1}s (read {:.1}s, upload {:.1}s); VRAM free {:.2}/{:.2} GiB",
            st.bytes as f64 / 1e9,
            (st.read + st.upload).as_secs_f64(),
            st.read.as_secs_f64(),
            st.upload.as_secs_f64(),
            free as f64 / (1u64 << 30) as f64,
            total as f64 / (1u64 << 30) as f64
        );
        ranks.push(engine::Rank::new(rank, s, hp, sh, w, n_slots, ctx, rows));
    }
    let eng = engine::Engine::new(hp.clone(), ranks, n_slots, ctx, rows);
    eprintln!("startup {:.1}s", t0.elapsed().as_secs_f64());

    Ok(eng)
}

fn score(g: &Gguf, args: &[String]) -> ExitCode {
    let num = |k: &str, d: usize| opt(args, k).and_then(|v| v.parse().ok()).unwrap_or(d);
    let (prefix, topk, tp) = (num("--prefix", 1), num("--topk", 20), num("--tp", 1));
    let Some(path) = opt(args, "--ids-file") else { return usage() };
    let ids: Vec<u32> = std::fs::read_to_string(path)
        .expect("read ids file")
        .split(|c: char| !c.is_ascii_digit())
        .filter_map(|v| v.parse().ok())
        .collect();
    let ctx = num("--ctx", ids.len() + 16);
    let hp = model::HParams::from_gguf(g).expect("hparams");
    let mut eng = match build_engine(g, &hp, tp, ctx, 1, num("--draft-vocab", 102400), hip::BMAX) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("load: {e}");
            return ExitCode::FAILURE;
        }
    };
    let mut pos = 0;
    for c in ids.chunks(hip::BMAX) {
        eng.step(&[(0, c, true)]);
        for row in 0..c.len() {
            if pos + 1 >= prefix {
                let lg = eng.logits(row);
                let max = lg.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
                let lse = max + lg.iter().map(|&v| (v as f64 - max).exp()).sum::<f64>().ln();
                let mut top: Vec<(usize, f32)> = lg.iter().copied().enumerate().collect();
                top.select_nth_unstable_by(topk, |a, b| b.1.total_cmp(&a.1));
                top.truncate(topk);
                top.sort_by(|a, b| b.1.total_cmp(&a.1));
                let items: Vec<String> = top.iter().map(|(i, v)| format!("{i}:{:.5}", *v as f64 - lse)).collect();
                println!("{pos} {}", items.join(" "));
            }
            pos += 1;
        }
    }
    ExitCode::SUCCESS
}

fn chat_template(g: &Gguf) -> Result<chat::ChatTemplate, String> {
    chat::ChatTemplate::new(g.str_of("tokenizer.chat_template").ok_or("GGUF has no chat template")?.to_string())
}

fn think_kwargs(args: &[String]) -> serde_json::Map<String, serde_json::Value> {
    let mut m = serde_json::Map::new();
    if args.iter().any(|a| a == "--no-think") {
        m.insert("enable_thinking".into(), false.into());
    }
    m
}

fn render(g: &Gguf, args: &[String]) -> ExitCode {
    let Some(text) = args.first() else { return usage() };
    let tmpl = chat_template(g).expect("chat template");
    let tok = tokenizer::Tokenizer::from_gguf(g).expect("tokenizer");
    let msgs = serde_json::json!([{"role": "user", "content": text}]);
    match tmpl.render(&msgs, None, &think_kwargs(args)) {
        Ok(p) => {
            println!("{p}");
            println!("ids {}", tok.encode(&p, true).iter().map(|i| i.to_string()).collect::<Vec<_>>().join(","));
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("{e}");
            ExitCode::FAILURE
        }
    }
}

fn serve(g: &Gguf, args: &[String]) -> ExitCode {
    let num = |k: &str, d: usize| opt(args, k).and_then(|v| v.parse().ok()).unwrap_or(d);
    let (port, tp, ctx, n_slots, draft_n) = (num("--port", 8996), num("--tp", 2), num("--ctx", 262144), num("--slots", 2), num("--draft", 7));
    let chunk = num("--chunk", 2048).max(hip::BMAX);
    let host = opt(args, "--host").unwrap_or("127.0.0.1");
    let hp = model::HParams::from_gguf(g).expect("hparams");
    let tok = tokenizer::Tokenizer::from_gguf(g).expect("tokenizer");
    let template = chat_template(g).expect("chat template");
    let eng = match build_engine(g, &hp, tp, ctx, n_slots, num("--draft-vocab", 102400), chunk) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("load: {e}");
            return ExitCode::FAILURE;
        }
    };
    let f = |k: &str, d: f32| g.f32_of(&format!("general.sampling.{k}")).unwrap_or(d);
    let defaults = sampler::SampleParams {
        temperature: f("temp", 1.0),
        top_k: g.u64_of("general.sampling.top_k").unwrap_or(20) as usize,
        top_p: f("top_p", 0.95),
        min_p: 0.0,
        seed: 0,
    };
    let (jobs, rx) = std::sync::mpsc::channel();
    let shared = std::sync::Arc::new(server::Shared {
        model: opt(args, "--alias").unwrap_or("qwen3.8-27b").to_string(),
        tok: tokenizer::Tokenizer::from_gguf(g).expect("tokenizer"),
        template,
        default_kwargs: think_kwargs(args),
        defaults,
        jobs,
        n_slots,
        ctx_size: ctx,
    });
    let listener = match std::net::TcpListener::bind((host, port as u16)) {
        Ok(l) => l,
        Err(e) => {
            eprintln!("bind {host}:{port}: {e}");
            return ExitCode::FAILURE;
        }
    };
    eprintln!("listening on http://{host}:{port} (tp {tp}, ctx {ctx}, slots {n_slots}, draft {draft_n})");
    std::thread::spawn(move || server::serve(listener, shared));
    // The engine stays on this thread: every HIP call comes from one place.
    sched::Driver::new(eng, tok, n_slots, sched::Config { draft_n, ctx, chunk }).run(rx);
    ExitCode::SUCCESS
}
