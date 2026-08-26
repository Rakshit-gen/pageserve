"""Phase 2 core op: causal attention read from a paged (block-scattered) KV
cache instead of a contiguous tensor.

`reference_causal_attention` is the correctness oracle — plain contiguous
GQA causal attention. `paged_causal_attention` must produce identical
output regardless of how the same logical sequence's K/V happen to be
scattered across physical blocks; that equivalence is the entire point of
paging and is what tests/test_paged_attention.py checks.
"""

import math

import torch


def causal_attention_incremental(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    query_start_pos: int = 0,
) -> torch.Tensor:
    """q: (chunk_len, num_heads, head_dim) — queries for the new tokens only.
    k, v: (total_len, num_kv_heads, head_dim) — full context including the
    new tokens (total_len = query_start_pos + chunk_len). Query row i
    (absolute position query_start_pos + i) attends to all key positions
    <= query_start_pos + i.

    This is what continuous batching needs: a decode step (or a chunked
    prefill continuation) only recomputes Q for the new tokens, not the
    whole history — the KV cache is what makes that not recompute K/V for
    old tokens either.
    """
    chunk_len, num_heads, head_dim = q.shape
    total_len, num_kv_heads, _ = k.shape
    group = num_heads // num_kv_heads
    k_ = k.repeat_interleave(group, dim=1).transpose(0, 1)  # (num_heads, total_len, head_dim)
    v_ = v.repeat_interleave(group, dim=1).transpose(0, 1)
    q_ = q.transpose(0, 1)  # (num_heads, chunk_len, head_dim)

    scale = 1.0 / math.sqrt(head_dim)
    scores = torch.einsum("hqd,hkd->hqk", q_, k_) * scale

    query_pos = torch.arange(query_start_pos, query_start_pos + chunk_len, device=q.device).unsqueeze(1)
    key_pos = torch.arange(total_len, device=q.device).unsqueeze(0)
    disallowed = key_pos > query_pos  # (chunk_len, total_len)
    scores = scores.masked_fill(disallowed, float("-inf"))

    probs = torch.softmax(scores, dim=-1)
    out = torch.einsum("hqk,hkd->hqd", probs, v_)
    return out.transpose(0, 1)  # (chunk_len, num_heads, head_dim)


def reference_causal_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
) -> torch.Tensor:
    """Full-sequence special case of causal_attention_incremental: q, k, v
    all cover the same (seq_len, ..., head_dim) — the plain-attention
    correctness oracle used by the paged-attention tests."""
    return causal_attention_incremental(q, k, v, query_start_pos=0)


def write_kv(
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_table: list[int],
    block_size: int,
    start_pos: int,
    k: torch.Tensor,
    v: torch.Tensor,
) -> None:
    """Write k[i], v[i] (new tokens starting at logical position start_pos)
    into their physical block/offset per block_table. In-place."""
    for i in range(k.shape[0]):
        logical_pos = start_pos + i
        block_idx = logical_pos // block_size
        offset = logical_pos % block_size
        phys_block = block_table[block_idx]
        k_cache[phys_block, offset] = k[i]
        v_cache[phys_block, offset] = v[i]


def gather_kv(
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_table: list[int],
    block_size: int,
    seq_len: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reassemble the logical (seq_len, num_kv_heads, head_dim) K/V for a
    sequence from its scattered physical blocks."""
    num_blocks_needed = (seq_len + block_size - 1) // block_size
    parts_k = []
    parts_v = []
    for b in range(num_blocks_needed):
        phys_block = block_table[b]
        take = min(block_size, seq_len - b * block_size)
        parts_k.append(k_cache[phys_block, :take])
        parts_v.append(v_cache[phys_block, :take])
    return torch.cat(parts_k, dim=0), torch.cat(parts_v, dim=0)


def paged_causal_attention(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_table: list[int],
    block_size: int,
    seq_len: int,
) -> torch.Tensor:
    """Full-context special case (prefill from scratch): q covers the whole
    logical sequence. k_cache/v_cache: (num_blocks, block_size, num_kv_heads,
    head_dim)."""
    k, v = gather_kv(k_cache, v_cache, block_table, block_size, seq_len)
    return reference_causal_attention(q, k, v)


def paged_causal_attention_incremental(
    q_new: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_table: list[int],
    block_size: int,
    start_pos: int,
) -> torch.Tensor:
    """The actual per-step op the scheduler drives: q_new covers only this
    step's chunk (1 token on decode, up to token_budget on chunked prefill).
    Caller must have already written this chunk's K/V into the cache via
    write_kv before calling this."""
    chunk_len = q_new.shape[0]
    total_len = start_pos + chunk_len
    k, v = gather_kv(k_cache, v_cache, block_table, block_size, total_len)
    return causal_attention_incremental(q_new, k, v, query_start_pos=start_pos)
