"""Standalone OpenAI-compatible server — no Modal, no Docker, no
platform-specific SDK. Run directly on any rented GPU box after `pip
install -r requirements.txt`:

    python3 server.py

Same engine/ code verified in pageserve_gpu_verify.ipynb (real 3B model,
real GPU) — this file is just a plain FastAPI/uvicorn shell around it,
identical request-handling logic to app.py's Modal version, so either
deployment target works off the same engine/ package untouched.

UNTESTED end to end: never actually started as a server or hit with a real
HTTP request. The engine/ internals are GPU-verified; this file (loading,
routing, streaming) is the new, unverified part.
"""

import json
import time
import uuid

import torch
import uvicorn
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from transformers import AutoTokenizer

from engine.block_allocator import BlockAllocator
from engine.model import PagedCausalLM
from engine.runner import ContinuousBatchingRunner
from engine.scheduler import Scheduler
from engine.sequence import Sequence, SeqStatus

MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
BLOCK_SIZE = 16
NUM_BLOCKS = 2048  # 2048 * 16 = 32,768 tokens of cache capacity

print(f"[pageserve] loading {MODEL_ID}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = PagedCausalLM(MODEL_ID, device="cuda", dtype=torch.bfloat16)
model.allocate_kv_cache(num_blocks=NUM_BLOCKS, block_size=BLOCK_SIZE)

_allocator = BlockAllocator(num_blocks=NUM_BLOCKS, block_size=BLOCK_SIZE)
_scheduler = Scheduler(_allocator, block_size=BLOCK_SIZE, max_batch_size=32, token_budget=512)


def _process_chunk(input_ids: list[int], block_table: list[int], start_pos: int) -> int:
    ids = torch.tensor(input_ids, device="cuda")
    with torch.no_grad():
        logits = model.forward_step(ids, block_table, start_pos)
    return int(logits[-1].argmax())


runner = ContinuousBatchingRunner(_scheduler, _process_chunk)
print("[pageserve] model loaded, runner started, ready for requests")

app = FastAPI()


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str = MODEL_ID
    messages: list[ChatMessage]
    max_tokens: int = 128
    stream: bool = False


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/v1/chat/completions")
def chat_completions(req: ChatRequest):
    prompt = tokenizer.apply_chat_template(
        [m.model_dump() for m in req.messages], tokenize=False, add_generation_prompt=True
    )
    ids = tokenizer(prompt, return_tensors="pt")["input_ids"][0].tolist()
    seq = Sequence(
        prompt_tokens=ids, max_new_tokens=req.max_tokens, eos_token_id=tokenizer.eos_token_id
    )
    runner.submit(seq)

    completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    created = int(time.time())

    if not req.stream:
        runner.wait_until_finished(seq, timeout=120)
        text = tokenizer.decode(seq.output_tokens, skip_special_tokens=True)
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
        # ponytail: decodes each new token span independently, which can
        # mangle multi-token UTF-8/BPE sequences at split boundaries. Same
        # cut corner as app.py's version — upgrade if garbled output shows
        # up in practice.
        last_sent = 0
        while True:
            with runner.cv:
                runner.cv.wait_for(
                    lambda: len(seq.output_tokens) > last_sent or seq.status == SeqStatus.FINISHED
                )
                new_tokens = seq.output_tokens[last_sent:]
                finished = seq.status == SeqStatus.FINISHED
                last_sent = len(seq.output_tokens)

            if new_tokens:
                piece = tokenizer.decode(new_tokens, skip_special_tokens=True)
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


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
