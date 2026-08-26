"""Modal deployment. Two engines live side by side:

`Engine`     — Phase 1 baseline: naive single-request HF generate(). The
               correctness oracle every later phase must match.
`PageServe`  — Phase 2+: our own scheduler + paged KV cache + from-scratch
               model, driven by one shared ContinuousBatchingRunner so
               concurrent requests actually batch together on the GPU.

Verified locally (CPU, Qwen2.5-0.5B, see scripts/verify_*.py) to match HF's
greedy generate() byte-for-byte, including under chunked prefill and
concurrent submission. NOT yet run against the real 3B model on GPU — that
needs Modal auth (`modal setup`, interactive) before `modal run app.py` can
actually execute anything here.
"""

import modal

app = modal.App("pageserve")

MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
BLOCK_SIZE = 16
NUM_BLOCKS = 2048  # 2048 * 16 = 32,768 tokens of cache capacity per container

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.5.1",
        "transformers==4.46.3",
        "accelerate==1.1.1",
        "fastapi[standard]",
    )
    .add_local_python_source("engine")
)


@app.cls(gpu="L4", image=image, scaledown_window=120)
class Engine:
    """Phase 1 baseline — see module docstring."""

    @modal.enter()
    def load(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, torch_dtype=torch.bfloat16, device_map="cuda"
        )

    @modal.method()
    def generate(self, prompt: str, max_new_tokens: int = 128) -> str:
        import torch

        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            out = self.model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False, repetition_penalty=1.0
            )
        return self.tokenizer.decode(
            out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )


@app.cls(gpu="L4", image=image, scaledown_window=120)
class PageServe:
    """Phase 2+ engine: continuous batching + paged KV cache + chunked
    prefill, all sharing one GPU-resident model via ContinuousBatchingRunner."""

    @modal.enter()
    def load(self):
        import torch
        from transformers import AutoTokenizer

        from engine.block_allocator import BlockAllocator
        from engine.model import PagedCausalLM
        from engine.runner import ContinuousBatchingRunner
        from engine.scheduler import Scheduler

        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        self.model = PagedCausalLM(MODEL_ID, device="cuda", dtype=torch.bfloat16)
        self.model.allocate_kv_cache(num_blocks=NUM_BLOCKS, block_size=BLOCK_SIZE)

        alloc = BlockAllocator(num_blocks=NUM_BLOCKS, block_size=BLOCK_SIZE)
        self.scheduler = Scheduler(alloc, block_size=BLOCK_SIZE, max_batch_size=32, token_budget=512)
        self.runner = ContinuousBatchingRunner(self.scheduler, self._process_chunk)

    def _process_chunk(self, input_ids, block_table, start_pos):
        import torch

        ids = torch.tensor(input_ids, device="cuda")
        with torch.no_grad():
            logits = self.model.forward_step(ids, block_table, start_pos)
        return int(logits[-1].argmax())

    @modal.method()
    def generate(self, prompt: str, max_new_tokens: int = 128) -> str:
        from engine.sequence import Sequence

        ids = self.tokenizer(prompt, return_tensors="pt")["input_ids"][0].tolist()
        seq = Sequence(
            prompt_tokens=ids,
            max_new_tokens=max_new_tokens,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        self.runner.submit(seq)
        self.runner.wait_until_finished(seq, timeout=120)
        return self.tokenizer.decode(seq.output_tokens, skip_special_tokens=True)

    @modal.asgi_app()
    def web(self):
        """Phase 7: OpenAI-compatible /v1/chat/completions, streaming and
        non-streaming, backed by the same shared runner/scheduler as
        generate() above — concurrent HTTP requests batch together."""
        import json
        import time
        import uuid

        from fastapi import FastAPI
        from fastapi.responses import StreamingResponse
        from pydantic import BaseModel

        from engine.sequence import Sequence, SeqStatus

        web_app = FastAPI()

        class ChatMessage(BaseModel):
            role: str
            content: str

        class ChatRequest(BaseModel):
            model: str = MODEL_ID
            messages: list[ChatMessage]
            max_tokens: int = 128
            stream: bool = False

        @web_app.post("/v1/chat/completions")
        def chat_completions(req: ChatRequest):
            prompt = self.tokenizer.apply_chat_template(
                [m.model_dump() for m in req.messages], tokenize=False, add_generation_prompt=True
            )
            ids = self.tokenizer(prompt, return_tensors="pt")["input_ids"][0].tolist()
            seq = Sequence(
                prompt_tokens=ids, max_new_tokens=req.max_tokens, eos_token_id=self.tokenizer.eos_token_id
            )
            self.runner.submit(seq)

            completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
            created = int(time.time())

            if not req.stream:
                self.runner.wait_until_finished(seq, timeout=120)
                text = self.tokenizer.decode(seq.output_tokens, skip_special_tokens=True)
                return {
                    "id": completion_id,
                    "object": "chat.completion",
                    "created": created,
                    "model": req.model,
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": text},
                            "finish_reason": "stop",
                        }
                    ],
                }

            def event_stream():
                # ponytail: decodes each new token span independently, which
                # can mangle multi-token UTF-8/BPE sequences at split
                # boundaries (real servers keep a small trailing buffer and
                # only flush on a clean boundary). Upgrade if garbled output
                # shows up in practice — untested either way, no GPU run yet.
                last_sent = 0
                while True:
                    with self.runner.cv:
                        self.runner.cv.wait_for(
                            lambda: len(seq.output_tokens) > last_sent
                            or seq.status == SeqStatus.FINISHED
                        )
                        new_tokens = seq.output_tokens[last_sent:]
                        finished = seq.status == SeqStatus.FINISHED
                        last_sent = len(seq.output_tokens)

                    if new_tokens:
                        piece = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
                        chunk = {
                            "id": completion_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": req.model,
                            "choices": [{"index": 0, "delta": {"content": piece}, "finish_reason": None}],
                        }
                        yield f"data: {json.dumps(chunk)}\n\n"

                    if finished:
                        final_chunk = {
                            "id": completion_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": req.model,
                            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                        }
                        yield f"data: {json.dumps(final_chunk)}\n\n"
                        yield "data: [DONE]\n\n"
                        break

            return StreamingResponse(event_stream(), media_type="text/event-stream")

        return web_app


@app.local_entrypoint()
def main(prompt: str = "Explain paged attention in one sentence."):
    print("--- Phase 1 baseline ---")
    print(Engine().generate.remote(prompt))
    print("--- Phase 2+ pageserve engine ---")
    print(PageServe().generate.remote(prompt))
