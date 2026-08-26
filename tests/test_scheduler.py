"""Real, runnable tests for the continuous batching scheduler — no GPU needed.

Covers the invariants that matter: every admitted sequence eventually
finishes (no livelock under preemption), chunked prefill respects the token
budget, and blocks are never over- or double-allocated.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engine.block_allocator import BlockAllocator
from engine.scheduler import Scheduler
from engine.sequence import Sequence, SeqStatus

EOS = 0


def run_to_completion(sched: Scheduler, max_steps: int = 10_000) -> int:
    steps = 0
    while sched.has_work():
        steps += 1
        assert steps <= max_steps, "scheduler livelocked"
        batch = sched.step()
        for entry in batch:
            if entry.produces_token:
                # fake model: always emit a non-EOS token until max_new_tokens hits
                sched.append_token(entry.seq, token_id=42)
    return steps


def test_single_sequence_runs_to_completion():
    alloc = BlockAllocator(num_blocks=100, block_size=16)
    sched = Scheduler(alloc, block_size=16, max_batch_size=4, token_budget=64)
    seq = Sequence(prompt_tokens=list(range(10)), max_new_tokens=5, eos_token_id=EOS)
    sched.add_request(seq)

    run_to_completion(sched)

    assert seq.status == SeqStatus.FINISHED
    assert len(seq.output_tokens) == 5
    assert alloc.num_free() == 100, "blocks not returned after finish"


def test_preemption_preserves_progress_eventually():
    # Only enough blocks for ONE sequence's full context at a time, forcing
    # preemption+recompute of whichever sequence loses the race.
    alloc = BlockAllocator(num_blocks=2, block_size=8)
    sched = Scheduler(alloc, block_size=8, max_batch_size=4, token_budget=64)

    seq_a = Sequence(prompt_tokens=list(range(6)), max_new_tokens=3, eos_token_id=EOS)
    seq_b = Sequence(prompt_tokens=list(range(6)), max_new_tokens=3, eos_token_id=EOS)
    sched.add_request(seq_a)
    sched.add_request(seq_b)

    run_to_completion(sched)

    assert seq_a.status == SeqStatus.FINISHED
    assert seq_b.status == SeqStatus.FINISHED
    assert len(seq_a.output_tokens) == 3
    assert len(seq_b.output_tokens) == 3
    assert alloc.num_free() == 2


def test_chunked_prefill_respects_token_budget():
    alloc = BlockAllocator(num_blocks=100, block_size=16)
    sched = Scheduler(alloc, block_size=16, max_batch_size=4, token_budget=4)
    seq = Sequence(prompt_tokens=list(range(10)), max_new_tokens=1, eos_token_id=EOS)
    sched.add_request(seq)

    batch1 = sched.step()
    assert len(batch1) == 1
    assert batch1[0].chunk == 4  # capped by token_budget, not full 10-token prompt
    assert not batch1[0].produces_token

    batch2 = sched.step()
    assert batch2[0].chunk == 4
    assert seq.num_prompt_tokens_processed == 8

    batch3 = sched.step()
    assert batch3[0].chunk == 2  # remaining prompt tokens
    assert batch3[0].produces_token  # prefill now complete


def test_allocator_never_double_allocates():
    alloc = BlockAllocator(num_blocks=3, block_size=8)
    a = alloc.allocate()
    b = alloc.allocate()
    c = alloc.allocate()
    assert len({a, b, c}) == 3
    assert alloc.num_free() == 0
    alloc.free(b)
    assert alloc.num_free() == 1
    d = alloc.allocate()
    assert d == b


if __name__ == "__main__":
    test_single_sequence_runs_to_completion()
    test_preemption_preserves_progress_eventually()
    test_chunked_prefill_respects_token_budget()
    test_allocator_never_double_allocates()
    print("all scheduler tests passed")
