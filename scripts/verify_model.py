"""Loads a real Qwen2 checkpoint through both HF's own model and our
from-scratch PagedCausalLM, and checks that logits agree to numerical
tolerance. Uses the 0.5B variant for a CPU-feasible smoke test — same
architecture family as the 3B deployment target, so this validates the
model code path itself, not just this specific checkpoint size. Run this
again against the 3B checkpoint on GPU before trusting it for real serving.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from engine.model import PagedCausalLM

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"


def main():
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    prompt = "The capital of France is"
    ids = tok(prompt, return_tensors="pt")["input_ids"][0]

    hf_model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float32)
    hf_model.eval()
    with torch.no_grad():
        hf_logits = hf_model(ids.unsqueeze(0)).logits[0]  # (seq_len, vocab)

    ours = PagedCausalLM(MODEL_ID, device="cpu", dtype=torch.float32)
    ours.allocate_kv_cache(num_blocks=8, block_size=16)
    block_table = list(range(8))
    with torch.no_grad():
        our_logits = ours.forward_step(ids, block_table, start_pos=0)

    hf_top = hf_logits.argmax(dim=-1)
    our_top = our_logits.argmax(dim=-1)
    print("HF top tokens:  ", hf_top.tolist())
    print("Ours top tokens:", our_top.tolist())

    max_abs_diff = (hf_logits - our_logits).abs().max().item()
    print(f"max abs logit diff: {max_abs_diff:.6f}")

    assert torch.equal(hf_top, our_top), "argmax token predictions diverge"
    assert max_abs_diff < 0.05, f"logits diverge too much: {max_abs_diff}"
    print("MODEL VERIFIED against real HF checkpoint")


if __name__ == "__main__":
    main()
