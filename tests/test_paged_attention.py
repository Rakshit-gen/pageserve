"""Verifies paged attention is numerically identical to contiguous causal
attention regardless of how physical blocks are scattered. Runs on CPU, no
GPU/Modal/model weights needed — this is the highest-risk piece of Phase 2
and the one most worth verifying before ever touching real weights.
"""

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from engine.paged_attention import (
    causal_attention_incremental,
    paged_causal_attention,
    paged_causal_attention_incremental,
    reference_causal_attention,
    write_kv,
)


def _random_scrambled_cache(seq_len, num_kv_heads, head_dim, block_size, num_blocks, seed):
    g = torch.Generator().manual_seed(seed)
    k = torch.randn(seq_len, num_kv_heads, head_dim, generator=g)
    v = torch.randn(seq_len, num_kv_heads, head_dim, generator=g)

    num_blocks_needed = (seq_len + block_size - 1) // block_size
    physical_ids = list(range(num_blocks))
    random.Random(seed).shuffle(physical_ids)
    block_table = physical_ids[:num_blocks_needed]

    k_cache = torch.zeros(num_blocks, block_size, num_kv_heads, head_dim)
    v_cache = torch.zeros(num_blocks, block_size, num_kv_heads, head_dim)
    write_kv(k_cache, v_cache, block_table, block_size, start_pos=0, k=k, v=v)
    return k, v, k_cache, v_cache, block_table


def test_paged_matches_reference_exact_block_boundary():
    block_size, num_kv_heads, head_dim, num_heads = 8, 2, 16, 8
    seq_len = block_size * 2  # exactly 2 full blocks — the off-by-one danger zone
    q = torch.randn(seq_len, num_heads, head_dim)
    k, v, k_cache, v_cache, block_table = _random_scrambled_cache(
        seq_len, num_kv_heads, head_dim, block_size, num_blocks=10, seed=1
    )

    expected = reference_causal_attention(q, k, v)
    actual = paged_causal_attention(q, k_cache, v_cache, block_table, block_size, seq_len)
    assert torch.allclose(expected, actual, atol=1e-5), "paged attention diverges at block boundary"


def test_paged_matches_reference_partial_last_block():
    block_size, num_kv_heads, head_dim, num_heads = 8, 2, 16, 8
    seq_len = block_size * 2 + 3  # last block only partially filled
    q = torch.randn(seq_len, num_heads, head_dim)
    k, v, k_cache, v_cache, block_table = _random_scrambled_cache(
        seq_len, num_kv_heads, head_dim, block_size, num_blocks=10, seed=2
    )

    expected = reference_causal_attention(q, k, v)
    actual = paged_causal_attention(q, k_cache, v_cache, block_table, block_size, seq_len)
    assert torch.allclose(expected, actual, atol=1e-5), "paged attention diverges on partial last block"


def test_paged_matches_reference_single_token():
    block_size, num_kv_heads, head_dim, num_heads = 8, 2, 16, 8
    seq_len = 1
    q = torch.randn(seq_len, num_heads, head_dim)
    k, v, k_cache, v_cache, block_table = _random_scrambled_cache(
        seq_len, num_kv_heads, head_dim, block_size, num_blocks=10, seed=3
    )

    expected = reference_causal_attention(q, k, v)
    actual = paged_causal_attention(q, k_cache, v_cache, block_table, block_size, seq_len)
    assert torch.allclose(expected, actual, atol=1e-5)


def test_incremental_matches_full_recompute_slice():
    # If a step recomputes Q only for the new chunk, its output for those
    # positions must be identical to what a full contiguous recompute would
    # have produced for the same rows — this is the actual invariant
    # continuous batching depends on (decode steps don't redo old queries).
    num_heads, num_kv_heads, head_dim = 8, 2, 16
    total_len, start_pos = 20, 14
    chunk_len = total_len - start_pos

    q_full = torch.randn(total_len, num_heads, head_dim)
    k = torch.randn(total_len, num_kv_heads, head_dim)
    v = torch.randn(total_len, num_kv_heads, head_dim)

    full_out = reference_causal_attention(q_full, k, v)
    incremental_out = causal_attention_incremental(
        q_full[start_pos:], k, v, query_start_pos=start_pos
    )
    assert torch.allclose(full_out[start_pos:], incremental_out, atol=1e-5)


def test_paged_incremental_matches_reference_across_two_steps():
    # Simulates the real driver loop: write chunk 1 into a fresh cache,
    # attend; write chunk 2 (decode step), attend again. Compare each
    # step's output against the plain reference over the same growing
    # context.
    block_size, num_kv_heads, head_dim, num_heads, num_blocks = 4, 2, 16, 8, 20

    prompt_len, decode_len = 9, 3
    total_len = prompt_len + decode_len
    k_all = torch.randn(total_len, num_kv_heads, head_dim)
    v_all = torch.randn(total_len, num_kv_heads, head_dim)
    q_all = torch.randn(total_len, num_heads, head_dim)

    k_cache = torch.zeros(num_blocks, block_size, num_kv_heads, head_dim)
    v_cache = torch.zeros(num_blocks, block_size, num_kv_heads, head_dim)
    block_table: list[int] = []

    def ensure_blocks(total_tokens):
        needed = (total_tokens + block_size - 1) // block_size
        while len(block_table) < needed:
            block_table.append(len(block_table))  # simple bump allocator for the test

    # Step 1: prefill the whole prompt in one chunk.
    ensure_blocks(prompt_len)
    write_kv(k_cache, v_cache, block_table, block_size, start_pos=0, k=k_all[:prompt_len], v=v_all[:prompt_len])
    out1 = paged_causal_attention_incremental(
        q_all[:prompt_len], k_cache, v_cache, block_table, block_size, start_pos=0
    )
    expected1 = reference_causal_attention(q_all[:prompt_len], k_all[:prompt_len], v_all[:prompt_len])
    assert torch.allclose(out1, expected1, atol=1e-5)

    # Steps 2..: one decode token at a time.
    pos = prompt_len
    for _ in range(decode_len):
        ensure_blocks(pos + 1)
        write_kv(k_cache, v_cache, block_table, block_size, start_pos=pos, k=k_all[pos : pos + 1], v=v_all[pos : pos + 1])
        out = paged_causal_attention_incremental(
            q_all[pos : pos + 1], k_cache, v_cache, block_table, block_size, start_pos=pos
        )
        expected = reference_causal_attention(q_all[: pos + 1], k_all[: pos + 1], v_all[: pos + 1])[-1:]
        assert torch.allclose(out, expected, atol=1e-5), f"mismatch at decode step pos={pos}"
        pos += 1


if __name__ == "__main__":
    test_paged_matches_reference_exact_block_boundary()
    test_paged_matches_reference_partial_last_block()
    test_paged_matches_reference_single_token()
    test_incremental_matches_full_recompute_slice()
    test_paged_incremental_matches_reference_across_two_steps()
    print("all paged attention tests passed")
