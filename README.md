# pageserve

A GPU LLM serving engine built from scratch: continuous batching, paged KV
cache, chunked prefill, a from-scratch decoder model (RMSNorm/RoPE/GQA/SwiGLU)
wired to a custom paged-attention op, served over an OpenAI-compatible API.
Reference model Qwen2.5-3B. Modal's `modal setup` auth is broken
Modal-side (support ticket open) — `server.py` (plain FastAPI/uvicorn, no
Modal/Docker/platform SDK) is the deployment path instead, verified live
end-to-end on a free Colab T4 GPU (`pageserve_live_server.ipynb`), tunneled
publicly with a zero-signup Cloudflare quick tunnel.

Sibling project to [NuclaDB](https://github.com/Rakshit-gen/NuclaDB) (vector
search engine) and [Inferoute](https://github.com/Rakshit-gen/inferoute) (LLM
gateway) — Inferoute routes to backends; this project *is* a backend.

## Status

- [x] Phase 1 — correctness baseline (naive HF `generate()` on Modal)
- [x] Phase 2 — continuous batching + paged KV cache
- [x] Phase 3 — chunked prefill
- [x] Phase 4 — CUDA graphs: capture/replay mechanism runs correctly on real GPU (T4, Colab); still only proven for a fixed start_pos, not across a real decode loop — see known gaps
- [x] Phase 5 — one custom Triton kernel (fused RMSNorm) — **verified on real GPU** (T4, Colab): matches `engine.model.rms_norm` to bf16-scale tolerance (max abs diff 0.0039)
- [x] Phase 6a — prefix caching (radix-style block hashing)
- [ ] Phase 6b — quantization, tensor parallelism (not started)
- [x] Phase 7 — OpenAI-compatible streaming API (`/v1/chat/completions`) — **verified live**: `server.py` on a real GPU, hit over a real public URL (Cloudflare tunnel) with `/health`, a non-streaming completion, and a streaming completion, all correct
- [ ] Phase 8 — benchmark vs real vLLM instance (harness written in `benchmarks/`, no numbers yet — nothing gets filled in without actually running it)

### What's actually verified, and how

Everything below has been run for real, on this machine, with real weights —
not just written and assumed correct:

- **Scheduler correctness** (`tests/test_scheduler.py`): admission, chunked
  prefill token-budget enforcement, and preemption-with-recompute all pass.
  Preemption caught a real bug on the first run (a stale-snapshot loop bug
  letting an already-preempted sequence get double-processed in the same
  iteration) — fixed, now covered by the test.
- **Paged attention numerics** (`tests/test_paged_attention.py`): paged,
  block-scattered attention output is bit-identical (to float32 tolerance)
  to plain contiguous causal attention, at block boundaries, partial last
  blocks, single tokens, and the actual incremental decode invariant
  (recomputing Q for only the new chunk matches a full recompute's tail).
- **Prefix caching** (`tests/test_prefix_cache.py`): shared prefixes reuse
  blocks, diverging suffixes don't, ref-counting keeps blocks alive until
  every referencing sequence releases them.
- **Concurrency** (`tests/test_runner.py`): 20 concurrent client threads
  submitting through one shared scheduler/runner, no deadlock, no
  cross-contamination between sequences.
- **The from-scratch model against real weights, CPU and GPU both:**
  `scripts/verify_model.py` / `scripts/verify_end_to_end.py` checked
  Qwen2.5-0.5B-Instruct on CPU first (max abs logit diff 5.8e-5, identical
  argmax; full chunked-prefill + continuous-batching pipeline matches HF's
  greedy `generate()` byte-for-byte). That CPU test caught two real bugs:
  an off-by-one between "tokens sampled" and "tokens fed through the
  model" during decode, and an apparent divergence that turned out to be
  HF's `generate()` silently applying this checkpoint's default
  `repetition_penalty: 1.1` even under `do_sample=False`.
  `pageserve_gpu_verify.ipynb` then reran both checks on the **real 3B
  target, on an actual GPU (T4, bf16)**: argmax matched exactly on the
  logit check (max abs diff 0.35, expected at bf16 precision — not a bug,
  bf16 just has ~3 decimal digits), and the full end-to-end generation
  matched HF's `generate()` exactly across 12 tokens, first try.
- **The Triton RMSNorm kernel and CUDA graph capture**, both exercised for
  the first time ever in that same GPU run — both passed. The kernel
  matched the PyTorch reference to bf16 tolerance; the graph capture/replay
  ran without error at the shape it was tested at (see gap below for what
  that does and doesn't prove).

### Known gaps (flagged, not hidden)

- **Not batched across sequences within one iteration.** The scheduler
  interleaves prefill/decode across sequences over *time* (real continuous
  batching), but `engine/runner.py` currently issues one `forward_step` call
  per sequence per iteration rather than concatenating all of a step's
  chunks into one GPU call. That's where continuous batching's actual
  throughput win comes from (bigger matmuls, not just better scheduling) —
  it's the natural next phase, not yet built.
- **CUDA graphs only proven at a fixed start_pos.** The Colab run captured
  and replayed at `start_pos=0` only — it did not test replaying across a
  real, advancing decode loop. That still needs start_pos/block_table to
  become GPU-resident tensors updated via `.copy_()` instead of the plain
  Python ints/lists `paged_attention.py` uses today; a graph captured at
  one start_pos can't correctly replay at the next one as written.
- **`server.py` verified live, but only on a free/ephemeral GPU so far.**
  `pageserve_live_server.ipynb` ran it for real on a Colab T4, tunneled
  publicly via Cloudflare, and hit `/health`, a non-streaming completion,
  and a streaming completion — all correct, real coherent output, clean
  SSE framing (`[DONE]` terminator, proper `chat.completion.chunk`
  shape). That session dies when the Colab tab disconnects, so this
  proves the code path works, not that there's a standing deployment.
  `app.py`'s Modal version has the same logic in Modal's decorator shell,
  kept in case Modal's auth gets fixed later. A persistent deployment
  still needs a rented GPU box (RunPod Pod or similar) to run `server.py`
  on continuously.
- **Streaming's token-boundary decode risk didn't trigger, but isn't
  hardened.** The observed stream decoded every chunk cleanly — no BPE
  split garbling — but that's one prompt's worth of luck, not a fix. The
  `ponytail:` comment in `server.py`/`app.py` about decoding each new span
  independently instead of keeping a boundary-safe trailing buffer still
  stands.
- **Phase 8's benchmark still has no numbers.** A live server now exists
  (ephemeral, Colab-based) — `benchmarks/bench_vs_vllm.py` could be run
  against it plus a real vLLM instance, but that comparison hasn't
  happened yet.

## Setup

Local (dev machine, no GPU — for the pure-Python/CPU checks below):
```
python3 -m venv .venv && source .venv/bin/activate
pip install modal torch transformers accelerate
```

On a rented GPU box (for `server.py` or `app.py`):
```
pip install -r requirements.txt
python3 server.py          # starts on :8000 — no Docker, no platform SDK
```

## Run

```
python3 tests/test_scheduler.py                          # pure Python, no GPU
python3 tests/test_paged_attention.py                     # pure Python, no GPU
python3 tests/test_prefix_cache.py                        # pure Python, no GPU
python3 tests/test_runner.py                              # pure Python, no GPU
python3 scripts/verify_model.py                           # CPU, downloads Qwen2.5-0.5B
python3 scripts/verify_end_to_end.py                      # CPU, downloads Qwen2.5-0.5B
python3 server.py                                         # GPU box only — starts the API on :8000
modal run app.py                                          # only if Modal auth ever gets fixed
```
