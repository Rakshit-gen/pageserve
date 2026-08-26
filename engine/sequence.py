"""A single request's lifecycle state, tracked across scheduler iterations."""

from dataclasses import dataclass, field
from enum import Enum, auto
from itertools import count


class SeqStatus(Enum):
    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()


_ids = count()


@dataclass
class Sequence:
    prompt_tokens: list[int]
    max_new_tokens: int = 128
    eos_token_id: int | None = None
    seq_id: int = field(default_factory=lambda: next(_ids))
    output_tokens: list[int] = field(default_factory=list)
    block_table: list[int] = field(default_factory=list)
    status: SeqStatus = SeqStatus.WAITING
    num_prompt_tokens_processed: int = 0  # for chunked prefill

    @property
    def num_tokens(self) -> int:
        return len(self.prompt_tokens) + len(self.output_tokens)

    @property
    def prompt_len(self) -> int:
        return len(self.prompt_tokens)

    def is_prefill_incomplete(self) -> bool:
        return self.num_prompt_tokens_processed < self.prompt_len

    def num_blocks_needed(self, block_size: int) -> int:
        return (self.num_tokens + block_size - 1) // block_size

    def append_token(self, token_id: int) -> None:
        self.output_tokens.append(token_id)
        if self.eos_token_id is not None and token_id == self.eos_token_id:
            self.status = SeqStatus.FINISHED
        elif len(self.output_tokens) >= self.max_new_tokens:
            self.status = SeqStatus.FINISHED

    def reset_for_recompute(self) -> list[int]:
        """Preemption via recompute: drop generated progress, keep the request.

        Returns the block ids to free; caller is responsible for returning
        them to the allocator.
        """
        freed = self.block_table
        self.block_table = []
        self.output_tokens = []
        self.num_prompt_tokens_processed = 0
        self.status = SeqStatus.WAITING
        return freed
