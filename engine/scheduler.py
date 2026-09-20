"""Phase 2 + 3: continuous batching scheduler with chunked prefill.

Each call to step() builds one iteration's batch: some sequences contribute
a chunk of unprocessed prompt (prefill), others contribute one token
(decode) — mixed in the same iteration, capped by a per-step token budget
(chunked prefill), so a long prompt can't block concurrent decodes for
multiple full iterations.

Preemption is recompute-only (no CPU swap): when a running sequence needs a
block that doesn't exist, evict the most-recently-admitted other sequence,
free its blocks, and requeue it to redo prefill from scratch.
"""

from dataclasses import dataclass

from .block_allocator import BlockAllocator
from .sequence import Sequence, SeqStatus


@dataclass
class BatchEntry:
    seq: Sequence
    chunk: int  # number of new tokens this sequence contributes this step
    produces_token: bool  # True once its prompt is fully consumed (decode step)


class Scheduler:
    def __init__(
        self,
        allocator: BlockAllocator,
        block_size: int,
        max_batch_size: int,
        token_budget: int,
    ):
        self.allocator = allocator
        self.block_size = block_size
        self.max_batch_size = max_batch_size
        self.token_budget = token_budget
        self.waiting: list[Sequence] = []
        self.running: list[Sequence] = []

    def add_request(self, seq: Sequence) -> None:
        self.waiting.append(seq)

    def has_work(self) -> bool:
        return bool(self.waiting or self.running)

    def _ensure_blocks(self, seq: Sequence, tokens_after_step: int) -> bool:
        needed = (tokens_after_step + self.block_size - 1) // self.block_size
        while len(seq.block_table) < needed:
            if self.allocator.num_free() == 0:
                return False
            seq.block_table.append(self.allocator.allocate())
        return True

    def _preempt_victim_other_than(self, protected: list[Sequence]) -> bool:
        for i in range(len(self.running) - 1, -1, -1):
            victim = self.running[i]
            if any(victim is p for p in protected):
                continue
            del self.running[i]
            freed = victim.reset_for_recompute()
            self.allocator.free_many(freed)
            self.waiting.insert(0, victim)
            return True
        return False

    def _admit(self) -> None:
        while self.waiting and len(self.running) < self.max_batch_size:
            if self.allocator.num_free() < 1:
                break
            seq = self.waiting.pop(0)
            seq.block_table.append(self.allocator.allocate())
            seq.status = SeqStatus.RUNNING
            self.running.append(seq)

    def step(self) -> list[BatchEntry]:
        self._admit()
        batch: list[BatchEntry] = []
        batched_seqs: list[Sequence] = []
        tokens_used = 0

        for seq in list(self.running):
            if tokens_used >= self.token_budget:
                break
            if seq not in self.running:
                # preempted earlier in this same iteration by another
                # sequence's block request; it's back in the waiting
                # queue now and will be re-admitted on a later step().
                continue

            if seq.is_prefill_incomplete():
                remaining_prompt = seq.prompt_len - seq.num_prompt_tokens_processed
                chunk = min(remaining_prompt, self.token_budget - tokens_used)
                projected_total = seq.num_prompt_tokens_processed + chunk
            else:
                chunk = 1
                projected_total = seq.num_tokens + 1

            # A sequence already appended to `batch` this iteration must
            # never be picked as a preemption victim by a later sequence's
            # block request: it would silently null out that already-
            # returned BatchEntry's block_table out from under the caller.
            protected = [seq, *batched_seqs]
            ok = self._ensure_blocks(seq, projected_total)
            while not ok and self._preempt_victim_other_than(protected):
                ok = self._ensure_blocks(seq, projected_total)
            if not ok:
                continue  # stalled this iteration; retried next step

            if seq.is_prefill_incomplete():
                seq.num_prompt_tokens_processed += chunk
            produces_token = not seq.is_prefill_incomplete()

            batch.append(BatchEntry(seq=seq, chunk=chunk, produces_token=produces_token))
            batched_seqs.append(seq)
            tokens_used += chunk

        return batch

    def append_token(self, seq: Sequence, token_id: int) -> None:
        seq.append_token(token_id)
        if seq.status == SeqStatus.FINISHED:
            self.running.remove(seq)
            self.allocator.free_many(seq.block_table)
            seq.block_table = []
