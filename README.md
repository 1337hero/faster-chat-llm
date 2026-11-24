# faster-chat-llm

A GGUF-native inference engine for AMD Radeon GPUs, built to serve models at speed on AMD Radeon AI PRO R9700 and RX 7900 XTX first.

It reads standard GGUF files directly (metadata, tokenizer, chat template, quantised tensors) and serves them over an OpenAI-compatible HTTP API. No offline conversion, no custom container format, no repacking step.

## Goals

**1. Radiance-class speed.** The target is the throughput Radiance reaches on the same hardware
(~280 tok/s c1 on an R9700 with speculative decoding), not the ~40-70 tok/s a generic GGUF engine
manages. Everything below is in service of that number.

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

Bring-up. The GGUF loader is verified tensor-for-tensor against an independent parser, the TP1
forward pass lands inside upstream llama.cpp's own cross-backend deviation envelope, and the
engine generates and serves. TP2, the MTP draft head, and the fast kernels are in progress.

Current model support: Qwen3.8-27B (`qwen35`, hybrid Gated DeltaNet + full attention, one MTP
layer). Measurement history and rejected ideas live in `PLAN.md` and `ledger.jsonl`; nothing here
is a performance claim yet.

## Build

Requires ROCm 7.x with `hipcc` available (default `/opt/rocm`, override with `ROCM_PATH`).

```sh
cargo build --release
./target/release/faster-chat-llm serve <model.gguf> --port 8996
```

Arch selection is currently hardcoded to `gfx1201` in `build.rs`; the `libr4d` / `libr3d` split
above is where that becomes a real choice.
