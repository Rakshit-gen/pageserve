"""Checks INT8-quantized PagedCausalLM against the unquantized version on
real weights (Qwen2.5-0.5B, CPU). Quantization is lossy by construction —
this does not assert exact argmax match (an 8-bit round can legitimately
flip a close call), it checks the divergence is small and reports exactly
how small, rather than silently passing either way.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from transformers import AutoTokenizer

from engine.model import PagedCausalLM

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"


def main():
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    prompt = "The capital of France is"
    ids = tok(prompt, return_tensors="pt")["input_ids"][0]

    fp32 = PagedCausalLM(MODEL_ID, device="cpu", dtype=torch.float32, quantize=False)
    fp32.allocate_kv_cache(num_blocks=8, block_size=16)
    with torch.no_grad():
        fp32_logits = fp32.forward_step(ids, list(range(8)), start_pos=0)
    del fp32

    quant = PagedCausalLM(MODEL_ID, device="cpu", dtype=torch.float32, quantize=True)
    quant.allocate_kv_cache(num_blocks=8, block_size=16)
    with torch.no_grad():
        quant_logits = quant.forward_step(ids, list(range(8)), start_pos=0)

    fp32_top = fp32_logits.argmax(dim=-1)
    quant_top = quant_logits.argmax(dim=-1)
    matches = (fp32_top == quant_top).sum().item()
    total = fp32_top.shape[0]

    max_abs_diff = (fp32_logits - quant_logits).abs().max().item()
    print(f"argmax matches: {matches}/{total}")
    print(f"max abs logit diff: {max_abs_diff:.4f}")

    assert matches >= total - 1, f"quantization changed argmax at more than 1 position: {matches}/{total}"
    print("QUANTIZATION VERIFIED (lossy but bounded, as expected)")


if __name__ == "__main__":
    main()
