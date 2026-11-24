# faster-chat-llm

A GGUF-native inference engine for AMD Radeon GPUs, built to serve models at speed on AMD Radeon AI PRO R9700 and RX 7900 XTX first.

It reads standard GGUF files directly (metadata, tokenizer, chat template, quantised tensors) and serves them over an OpenAI-compatible HTTP API. No offline conversion, no custom container format, no repacking step.

## Goals

**1. Radiance-class speed.** The target is the throughput Radiance reaches on the same hardware
(~280 tok/s c1 on an R9700 with speculative decoding), not the ~40-70 tok/s a generic GGUF engine
manages. Everything below is in service of that number.

That reference is a 4-bit model (MXFP4, with a merged LoRA). At Q8_0 the same cards read twice
the bytes per step: 14.2 GB per card, 26.4 ms at the measured 537 GB/s, so 280 tok/s would take
about 7.4 tokens out of every step. The model's own MTP head yields 2.2 to 5.6. With weights
as stored, the number is out of reach by arithmetic, not by engineering; reaching it means a
4-bit weight format or a much stronger drafter, both opt-in under goal 3.

**2. RDNA4 and RDNA3 are first-class, not "portable".** Two arch families get their own kernel
libraries, selected at load time:

- `libr4d` — gfx1200, gfx1201 (RX 9070 series, Radeon AI PRO R9700)
- `libr3d` — gfx11xx (RX 7900 XTX and the rest of RDNA3)

Per-family kernels, not one generic HIP path with `#ifdef`s sprinkled through it. A kernel that
only runs well on one family is a bug, not a limitation.

**3. Lossless by default.** Weights are used as stored: no requantisation, no repacking to a
faster format. Speculative decoding must be exact — the same tokens greedy decoding would have
produced, verified rather than assumed. Lossy levers (quantised KV cache, a smaller weight
format, an external drafter) are opt-in and labelled as such.

**4. Multi-GPU is the normal case.** A 27B Q8_0 is 29 GB, so it does not fit one 32 GB card
alongside a useful KV cache. Two-card tensor parallel over PCIe is the default deployment, and
the decode allreduce has to cost microseconds per step, not milliseconds.

**5. Long context, held.** 262K tokens unified across two slots, stable at full length — not just
at load time.

## Non-goals

- Portability off AMD/RDNA, or off Linux/ROCm.
- Vision and multimodal. Text first.
- Being a framework. One engine, a few model architectures, tuned hard for the cards above.

## Status

Working engine for one model: Qwen3.8-27B Q8_0 (`qwen35`, hybrid Gated DeltaNet + full
attention, one MTP layer) at tensor-parallel 2 on two Radeon AI PRO R9700 (gfx1201).

What is built and verified:

- **Loader.** Reads the GGUF directly and streams tensors to the cards; checked
  tensor-for-tensor against an independent parser. Q8_0 weights are used as stored.
- **Kernels.** Q8_0 matmul on the int8 matrix core, Gated DeltaNet, gated attention on the f16
  matrix core. A token computes the same bits alone, in a verify batch, or beside another
  sequence.
- **Two-card exchange that runs on the cards.** Decode-sized steps trade partial sums through
  coherent pinned host memory with no host round trip (8.5 µs per 20 KB hop, single process).
- **Lossless speculative decode** with the GGUF's own MTP head: drafts are verified against the
  full model, rollback is replay-based, and output is token-identical to non-speculative decode
  on every prompt checked.
- **262,144-token context** shared by two slots: needle retrieval 3/3 at 130,728 and at 259,231
  tokens, and two concurrent 131K-token requests both retrieve 3/3.
- **HTTP server** speaking OpenAI chat / completions / Responses, Anthropic Messages and
  llama-server's native endpoints, with tool-call parsing. Exercised with the official OpenAI
  and Anthropic Python SDKs: chat, streaming, tool-call round trips, Responses, thinking.

Measured on this hardware against llama.cpp serving the same GGUF on the same two cards
(tensor split + MTP drafts), both measured the same night:

| Metric | llama.cpp | faster-chat-llm | Ceiling | How measured |
|---|---:|---:|---:|---|
| Decode, 1 stream | 67.8 tok/s | 79.3 | ~100 at the panel's median tokens per step (67 to 170 by prompt); 37.9 without speculation | median of a fixed six-prompt panel, greedy, thinking off, 7 MTP drafts; draft acceptance 0.17 to 0.61 by prompt |
| Decode, 2 streams | 127.0 | 162.1 | ~200 (twice the 1-stream ceiling: a weight pass costs the same for 1 to 16 rows) | wall-clock aggregate |
| Prefill at 4K / 32K / 128K | 1555 / 1675 / 1104 to 1153 | 1890 / 1768 / 1346 to 1391 | not stated (no independent peak for the int8 matrix core) | cold unique prompt, prompt tokens / time to first token; real sizes ~2.8K / ~21.5K / 131K |
| Prefill at 259K | 801 | 1029 | | single run each |
| Decode at 131K context | 40.2 | 43.9 to 44.2 | | 400-token answers |
| KV capacity | 262,144 | 262,144 | 262,144 | needle retrieval 3/3 at 130,728 and 259,231 tokens; two concurrent 131K requests, both 3/3 |
| Start to ready | 12.5 to 14 s | 7 to 8 s | ~5 s read floor | warm page cache |
| Time to first token | 185 ms | 87 ms | | 19-token prompt; 42 / 262 / 942 tokens: 123 / 242 / 545 ms vs 176 / 294 / 669 |
| Peak memory | 54.2 GiB | 50.4 GiB | 68 GiB cap | VRAM on both cards + host anonymous memory; 25.8 GiB per card at full context |
| Quality | 99.50%, KL 0.000204 between two of its own backends | 99.67%, KL 0.000211 | | top-token agreement and top-20 KL vs upstream llama.cpp, teacher-forced over 2,991 positions |

Against the BF16 checkpoint, KL is 0.00031 (llama.cpp's Q8_0 path: 0.00032). Speculative
decode is token-identical to plain decode under greedy, and byte-identical under sampling
(temperature 0.8, two seeds).

Known gaps:

- Prefill throughput moves with host load (large batches are timed partly from the host): the
  same build has measured 10 to 15% apart an hour apart. Compare only same-hour runs.
- A stream's decode slows while the other slot prefills a long prompt (35 to 61 tok/s measured
  beside a 131K-token prefill).
- One model, one quantisation (Q8_0 + F32), one GPU family. No vision. Sampling other than
  greedy runs on the host. `logprobs` are not implemented.
- Goal 1 is not reachable at Q8_0 (see the arithmetic under it).

## Build

Requires ROCm 7.x with `hipcc` available (default `/opt/rocm`, override with `ROCM_PATH`).

```sh
cargo build --release
./target/release/faster-chat-llm serve <model.gguf> --port 8996
```

`cargo build --profile fast` is the iteration build: it drops LTO so a host-side edit relinks in
well under a second (the kernels are compiled the same either way).

Arch selection is currently hardcoded to `gfx1201` in `build.rs`; the `libr4d` / `libr3d` split
above is where that becomes a real choice.
