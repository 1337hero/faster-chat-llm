# Working on faster-chat-llm

This is a Rust 2024 application for Qwen inference on AMD gfx1201 GPUs, with HIP
kernels in C++. Keep the runtime small, typed, and measurable. Do not turn it into
a general inference framework to solve a local problem.

## Before changing code

- Read the affected modules and `CONTRIBUTING.md`. Parent `AGENTS.md` rules apply.
- Inspect `git status`. Preserve unrelated files and work from other sessions.
- Questions are read-only unless Mike explicitly asks for implementation.
- Reproduce a bug before fixing it. Keep a check that fails for the original reason.
- Do not commit, push, change shared services, or unload models without approval.

## Module boundaries

- `src/api/`: translate wire formats into `InternalRequest`; format JSON and SSE.
- `src/server.rs`: HTTP handling, prompt rendering/tokenization, job submission.
- `src/sched.rs`: slot admission, KV ownership, prefix reuse, generation termination.
- `src/engine.rs`: tensor-parallel execution, GPU ordering, speculative verification.
- `src/hip.rs`: HIP bindings, allocations, transfer contracts, kernel declarations.
- `src/model.rs`: model geometry, sharding, streamed weight loading.
- `src/tokenizer.rs` and `src/gguf/`: checkpoint parsing and tokenization parity.
- `kernels/kernels.cpp`: HIP kernels. Keep its ABI synchronized with `src/hip.rs`.

Keep protocol code out of the engine and device management out of API adapters.
The scheduler owns the engine; HTTP threads submit jobs through channels.

## Rust design

- Prefer concrete types, enums, and exhaustive matches to flags and stringly typed state.
- Use checked conversions at input boundaries. An integer cast is not validation.
- Validate token IDs against the vocabulary and signed kernel index range before
  GPU submission. Keep a defensive assertion at the engine boundary.
- Use named structs when tuples stop explaining their fields. Add abstractions only
  when they remove real duplication or enforce an invariant.
- Prefer borrowed slices for inputs. Clone to establish ownership, not to silence
  the borrow checker. Keep allocations out of decode loops unless justified.
- Use the standard library and existing dependencies first. Do not add async
  runtimes, error frameworks, or derive packages without a concrete need.
- Follow Rust naming and conversion conventions. Limit visibility to actual callers.
- Remove dead application code rather than retaining speculative public APIs.

## Errors and parsing

- Malformed HTTP, model output, and GGUF input return errors, not panics.
- Reserve `assert!`, `expect`, and `unwrap` in production for proven internal
  invariants or startup assumptions with useful diagnostics. Release panics abort.
- Check lengths, offsets, alignment, arithmetic overflow, and narrowing conversions
  before indexing, allocating, or entering FFI. Never silently discard invalid IDs.
- Search closing delimiters relative to their opening delimiter. Check UTF-8
  boundaries before slicing streamed text.
- Preserve the source error when adding context; do not replace it with a vague string.

## Unsafe code and HIP

- A safe API must not depend on an unenforced caller memory-safety promise.
- Every new or changed unsafe block gets a local `SAFETY:` explanation of the
  actual invariant. Every unsafe function documents its `# Safety` contract.
- `unsafe_op_in_unsafe_fn` is denied. Use explicit unsafe blocks inside unsafe functions.
- Asynchronous host/device copies are unsafe unless a wrapper keeps the required
  borrows alive through synchronization. Name the synchronization point at callers.
- Upload sources remain live and unmodified; download destinations remain live
  and inaccessible until completion. Keep device allocations alive as well.
- Typed transfers accept only sealed `DeviceScalar` types. Do not copy Rust
  ownership-bearing values, references, or validity-constrained types as device data.
- Initialize every downloaded byte, including padding between active batch rows.
- Track the selected device and stream ordering explicitly. Preserve cross-card
  ordering and buffer lifetimes when changing overlap or speculative execution.
- Do not add blanket `unsafe impl Send` or `Sync`. Explain why device ownership
  remains correct across threads before implementing either trait.

## Generation and protocol behavior

- Hold back possible stop-string suffixes, but flush unmatched text on normal
  termination. Never emit the matched stop string or send text twice.
- Test zero limits, context exhaustion, EOS, stop strings spanning chunks, and UTF-8
  splits. Admission and decode must agree on token accounting.
- Preserve tokenizer behavior against the model's reference, including whitespace,
  special tokens, and BPE boundaries. Plausible text output is not proof of parity.
- Synchronous JSON and streaming deltas have different schemas. Test serialized
  events, including indexes and termination, rather than only intermediate objects.

## Verification

Run these before calling work finished:

```bash
cargo fmt --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test
```

- Building needs ROCm/hipcc even for CPU-only tests; `ROCM_PATH` defaults to `/opt/rocm`.
- `tests/regressions.rs` exercises real parsers, adapters, and scheduler logic with a
  deterministic replacement for the GPU engine. It does not verify HIP execution.
- Test important behavior and risky boundaries. Avoid tests tied to private layout
  or tests that merely repeat implementation steps.
- Prove fixes against the old implementation when practical. A compilation failure
  is not a valid negative control for a runtime bug.
- Keep Clippy warning-free. Use a narrowly scoped `#[expect(..., reason = "...")]`
  when a lint is demonstrably wrong; do not suppress a lint group to get green output.
- Run long builds/tests with the background tool and wait for completion. No sleeping
  or polling. Group independent tool calls and run the relevant check after edits.
- Report the commands, results, and limits of verification. Distinguish CPU harness
  results from live GPU inference and full-context stability tests.

## Performance work

Hold model, prompts, sampling, thinking settings, and benchmark protocol fixed.
Record acceptance beside throughput for speculative decoding. Require output
correctness before accepting a speedup. Do not disrupt shared GPU workloads to run
benchmarks; follow the parent repository's device and process-hygiene rules.

## Basis for these rules

Adapted from the Rust review/best-practices skills and their references, with
application and HIP constraints taking precedence over reusable-library advice:

- [Rust API Guidelines](https://rust-lang.github.io/api-guidelines/)
- [Rust Style Guide](https://doc.rust-lang.org/style-guide/)
- [Effective Rust](https://www.lurklurk.org/effective-rust/)
- [Microsoft Pragmatic Rust Guidelines](https://microsoft.github.io/rust-guidelines/)
- [Rust Design Patterns](https://rust-unofficial.github.io/patterns/)

Do not apply those references mechanically. Keep the simplest design that preserves
this application's safety, model parity, and serving behavior.
