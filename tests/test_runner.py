"""Concurrency test for the shared continuous-batching runner — pure
Python, no GPU/model needed. Several client threads submit sequences at
the same time against one shared scheduler/runner; this is what actually
exercises the lock/condition-variable logic, which is the riskiest new
code here (a deadlock or a lost wakeup would hang the test, not just fail
an assertion).
"""

import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engine.block_allocator import BlockAllocator
from engine.runner import ContinuousBatchingRunner
from engine.scheduler import Scheduler
from engine.sequence import Sequence

EOS = -1


def fake_process_chunk(input_ids, block_table, start_pos):
    # Deterministic fake "model": next token = last input token + 1.
    return input_ids[-1] + 1


def test_concurrent_clients_all_finish_correctly():
    alloc = BlockAllocator(num_blocks=200, block_size=8)
    sched = Scheduler(alloc, block_size=8, max_batch_size=8, token_budget=32)
    runner = ContinuousBatchingRunner(sched, fake_process_chunk)

    results = {}
    errors = []

    def client(client_id: int):
        try:
            seq = Sequence(
                prompt_tokens=[client_id * 100],
                max_new_tokens=5,
                eos_token_id=EOS,
            )
            runner.submit(seq)
            runner.wait_until_finished(seq, timeout=10)
            results[client_id] = list(seq.output_tokens)
        except Exception as e:  # noqa: BLE001 — surfaced via `errors` for the assertion below
            errors.append((client_id, e))

    threads = [threading.Thread(target=client, args=(i,)) for i in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=15)
        assert not t.is_alive(), "a client thread hung — deadlock or lost wakeup in the runner"

    runner.stop()

    assert not errors, f"client errors: {errors}"
    assert len(results) == 20
    for client_id, output in results.items():
        assert len(output) == 5, f"client {client_id} got {len(output)} tokens, expected 5"
        # fake model increments by 1 each step from the prompt token — every
        # client's own chain must stay internally consistent, proving no
        # cross-contamination between concurrently-batched sequences.
        expected_start = client_id * 100 + 1
        assert output == list(range(expected_start, expected_start + 5)), (
            f"client {client_id} output {output} suggests cross-contamination between sequences"
        )


if __name__ == "__main__":
    test_concurrent_clients_all_finish_correctly()
    print("all runner concurrency tests passed")
