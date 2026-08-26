"""Phase 8: head-to-head benchmark harness against a real vLLM instance.

Hits both servers' OpenAI-compatible /v1/chat/completions with the same
prompt set and concurrency, measuring TTFT (time to first token), TPOT
(time per output token), and aggregate throughput. Prints results — does
not invent them. Nothing has been run yet: this needs both a deployed
pageserve endpoint (app.py's PageServe.web, once Modal is authenticated)
and a real vLLM instance on comparable hardware (same GPU type, same
model) to compare against. Do not fill in numbers here without actually
running it — see [[project_resume_latex]]-style discipline: every
benchmark on the resume must come from actually running the harness.
"""

import argparse
import statistics
import time

import httpx

PROMPTS = [
    "Explain how a hash table resolves collisions.",
    "Write a haiku about distributed systems.",
    "What's the difference between TCP and UDP?",
    "Summarize the CAP theorem in two sentences.",
    "Give three examples of idempotent HTTP methods.",
]


def run_one(base_url: str, prompt: str, max_tokens: int) -> dict:
    payload = {
        "model": "bench",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": True,
    }
    start = time.perf_counter()
    first_token_time = None
    num_tokens = 0

    with httpx.stream("POST", f"{base_url}/v1/chat/completions", json=payload, timeout=120) as resp:
        for line in resp.iter_lines():
            if not line.startswith("data: ") or line.endswith("[DONE]"):
                continue
            if first_token_time is None:
                first_token_time = time.perf_counter()
            num_tokens += 1

    end = time.perf_counter()
    ttft = (first_token_time - start) if first_token_time else None
    total = end - start
    tpot = (total - ttft) / max(num_tokens - 1, 1) if ttft else None
    return {"ttft_s": ttft, "tpot_s": tpot, "total_s": total, "num_chunks": num_tokens}


def bench(name: str, base_url: str, max_tokens: int) -> None:
    results = [run_one(base_url, p, max_tokens) for p in PROMPTS]
    ttfts = [r["ttft_s"] for r in results if r["ttft_s"] is not None]
    tpots = [r["tpot_s"] for r in results if r["tpot_s"] is not None]

    print(f"\n=== {name} ({base_url}) ===")
    print(f"  median TTFT: {statistics.median(ttfts):.3f}s" if ttfts else "  no successful runs")
    print(f"  median TPOT: {statistics.median(tpots)*1000:.1f}ms/token" if tpots else "")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pageserve-url", required=True, help="e.g. https://<workspace>--pageserve-pageserve-web.modal.run")
    parser.add_argument("--vllm-url", required=True, help="a real vLLM OpenAI-compatible server, same GPU/model")
    parser.add_argument("--max-tokens", type=int, default=128)
    args = parser.parse_args()

    bench("pageserve", args.pageserve_url, args.max_tokens)
    bench("vLLM", args.vllm_url, args.max_tokens)
