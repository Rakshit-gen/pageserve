"""Phase 2/7: shared continuous-batching driver.

One Scheduler instance serves many concurrent requests: a single background
thread repeatedly calls scheduler.step() and drives the model; caller
threads (one per incoming request) just submit a Sequence and block until
it finishes. This is what actually makes "continuous batching" concurrent —
without a shared driver loop, each request would just run its own serial
generate() and never share a batch with anyone else.
"""

import threading
from typing import Callable

from .scheduler import Scheduler
from .sequence import Sequence, SeqStatus

ProcessChunk = Callable[[list[int], list[int], int], int]


class ContinuousBatchingRunner:
    def __init__(self, scheduler: Scheduler, process_chunk: ProcessChunk):
        """process_chunk(input_ids, block_table, start_pos) -> next_token_id.
        Must run the model forward for this chunk (writing its KV as a side
        effect) and return the argmax next-token id. Called for every
        chunk, including ones where produces_token is False (prefill still
        has to advance the KV cache) — the runner just ignores the return
        value in that case."""
        self.scheduler = scheduler
        self.process_chunk = process_chunk
        self.lock = threading.Lock()
        self.cv = threading.Condition(self.lock)
        self._stop = False
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def submit(self, seq: Sequence) -> None:
        with self.cv:
            self.scheduler.add_request(seq)
            self.cv.notify_all()

    def wait_until_finished(self, seq: Sequence, timeout: float | None = None) -> None:
        with self.cv:
            finished = self.cv.wait_for(lambda: seq.status == SeqStatus.FINISHED, timeout=timeout)
        if not finished:
            raise TimeoutError(f"sequence {seq.seq_id} did not finish within {timeout}s")

    def stop(self) -> None:
        with self.cv:
            self._stop = True
            self.cv.notify_all()

    def _loop(self) -> None:
        while True:
            with self.cv:
                while not self._stop and not self.scheduler.has_work():
                    self.cv.wait()
                if self._stop:
                    return
                prev_prompt_processed = {
                    s.seq_id: s.num_prompt_tokens_processed for s in self.scheduler.running
                }
                prev_output_len = {s.seq_id: len(s.output_tokens) for s in self.scheduler.running}
                batch = self.scheduler.step()

            for entry in batch:
                s = entry.seq
                before_prompt = prev_prompt_processed.get(s.seq_id, 0)
                before_output_len = prev_output_len.get(s.seq_id, 0)

                if before_prompt < s.prompt_len:
                    start_pos = before_prompt
                    input_ids = s.prompt_tokens[before_prompt : before_prompt + entry.chunk]
                else:
                    # output_tokens counts sampled tokens; the KV cache only
                    # holds tokens already fed through the model, one behind.
                    start_pos = s.prompt_len + before_output_len - 1
                    input_ids = [s.output_tokens[-1]]

                next_token = self.process_chunk(input_ids, s.block_table, start_pos)

                if entry.produces_token:
                    with self.cv:
                        self.scheduler.append_token(s, next_token)
                        self.cv.notify_all()
