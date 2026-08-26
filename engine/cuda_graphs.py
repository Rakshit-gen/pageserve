"""Phase 4: CUDA graph capture for the decode step.

Decode steps are tiny (1 new token) and GPU-bound-nothing — dominated by
Python/CUDA-launch overhead, not compute. Capturing the forward pass as a
CUDA graph replays all of its kernel launches as one op, which is where the
payoff actually is (prefill isn't graphed here: its shape varies per call,
which is exactly what graphs can't tolerate).

UNTESTED: CUDA graph capture requires a real CUDA device and cannot be
exercised on CPU at all — there is no meaningful fallback to test against
locally, unlike paged_attention.py or the scheduler.

Known gaps this depends on, neither built yet:
1. forward_step runs once per sequence per scheduler iteration (see
   engine/runner.py), not batched across sequences into one call —
   per-batch-size bucketing only pays off once decode is actually batched.
2. bigger problem: paged_attention.py's write_kv/gather_kv take start_pos
   and block_table as plain Python ints/lists, looped over in Python. CUDA
   graphs replay the exact same kernel launches every time — start_pos
   changes on every decode step, so a graph captured with start_pos=N is
   only valid to replay at start_pos=N, not N+1. Real graph-compatible
   paged decode needs start_pos and block_table to live as GPU tensors
   updated in place (`.copy_()`) between replays, which paged_attention.py
   doesn't support today. This module demonstrates the capture/replay
   mechanics correctly; it is not wired to be reusable across steps until
   that rework happens.
"""

import torch


class DecodeGraph:
    """Captures one CUDA graph for a fixed decode batch size. Call
    `capture()` once per bucket size after warmup, then `replay()` on every
    subsequent decode step of that size instead of re-dispatching Python."""

    def __init__(self, model, block_size: int):
        self.model = model
        self.block_size = block_size
        self.graph: torch.cuda.CUDAGraph | None = None
        self.static_input_ids: torch.Tensor | None = None
        self.static_start_pos: int | None = None
        self.static_block_table: list[int] | None = None
        self.static_logits: torch.Tensor | None = None

    def capture(self, batch_size: int, block_table: list[int], start_pos: int, warmup_iters: int = 3) -> None:
        device = self.model.device
        self.static_input_ids = torch.zeros(batch_size, dtype=torch.long, device=device)
        self.static_block_table = block_table
        self.static_start_pos = start_pos

        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(warmup_iters):
                self.model.forward_step(self.static_input_ids, self.static_block_table, self.static_start_pos)
        torch.cuda.current_stream().wait_stream(stream)

        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.static_logits = self.model.forward_step(
                self.static_input_ids, self.static_block_table, self.static_start_pos
            )

    def replay(self, input_ids: torch.Tensor) -> torch.Tensor:
        assert self.graph is not None, "capture() must run before replay()"
        self.static_input_ids.copy_(input_ids)
        self.graph.replay()
        return self.static_logits
