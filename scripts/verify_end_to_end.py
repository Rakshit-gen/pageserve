"""Ties Phase 1 (HF generate() as correctness oracle) to Phase 2 (our
scheduler + paged KV cache + from-scratch model): drives one sequence
through the real continuous-batching scheduler with a small token budget
(forcing chunked prefill across multiple steps), then decodes token by
token, and checks the generated ids match HF's own greedy generate() byte
for byte.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from engine.block_allocator import BlockAllocator
from engine.model import PagedCausalLM
from engine.scheduler import Scheduler
from engine.sequence import Sequence

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"


def main():
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    prompt = "The capital of France is"
    prompt_ids = tok(prompt, return_tensors="pt")["input_ids"][0].tolist()
    max_new = 12

    hf_model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float32)
    hf_model.eval()
    with torch.no_grad():
        # repetition_penalty=1.0 disables this checkpoint's default 1.1
        # penalty (baked into its generation_config.json) — without pinning
        # it, do_sample=False is *not* pure greedy, it's greedy-after-penalty,
        # which no longer matches a plain argmax and isn't what we're testing.
        hf_out = hf_model.generate(
            torch.tensor([prompt_ids]),
            max_new_tokens=max_new,
            do_sample=False,
            repetition_penalty=1.0,
        )
    hf_generated = hf_out[0][len(prompt_ids):].tolist()
    del hf_model

    block_size, num_blocks = 8, 64
    alloc = BlockAllocator(num_blocks=num_blocks, block_size=block_size)
    # token_budget=4 on a 5-token prompt forces 2 prefill chunks — exercising
    # chunked prefill, not just a one-shot forward.
    sched = Scheduler(alloc, block_size=block_size, max_batch_size=4, token_budget=4)

    model = PagedCausalLM(MODEL_ID, device="cpu", dtype=torch.float32)
    model.allocate_kv_cache(num_blocks=num_blocks, block_size=block_size)

    seq = Sequence(prompt_tokens=prompt_ids, max_new_tokens=max_new, eos_token_id=tok.eos_token_id)
    sched.add_request(seq)

    while sched.has_work():
        prev_prompt_processed = {s.seq_id: s.num_prompt_tokens_processed for s in sched.running}
        prev_output_len = {s.seq_id: len(s.output_tokens) for s in sched.running}

        batch = sched.step()

        for entry in batch:
            s = entry.seq
            before_prompt = prev_prompt_processed.get(s.seq_id, 0)
            before_output_len = prev_output_len.get(s.seq_id, 0)

            if before_prompt < s.prompt_len:
                start_pos = before_prompt
                input_ids = torch.tensor(s.prompt_tokens[before_prompt : before_prompt + entry.chunk])
            else:
                # output_tokens counts sampled tokens, but the KV cache only
                # holds tokens already fed through the model — the most
                # recently sampled token hasn't been fed back yet, so its
                # cache position is one behind len(output_tokens).
                start_pos = s.prompt_len + before_output_len - 1
                input_ids = torch.tensor([s.output_tokens[-1]])

            with torch.no_grad():
                logits = model.forward_step(input_ids, s.block_table, start_pos)

            if entry.produces_token:
                next_token = int(logits[-1].argmax())
                sched.append_token(s, next_token)

    print("HF generate():      ", hf_generated)
    print("pageserve engine:   ", seq.output_tokens)
    assert seq.output_tokens == hf_generated, "engine output diverges from HF's own greedy generate()"
    print("END-TO-END VERIFIED: chunked prefill + continuous batching + paged KV cache match HF exactly")


if __name__ == "__main__":
    main()
