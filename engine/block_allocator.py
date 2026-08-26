"""Phase 2: paged KV cache block allocator.

Fixed-size block free-list, like an OS page table. Sequences own a list of
block ids (their "block table"); the allocator only tracks which physical
blocks are free.
"""


class OutOfBlocksError(Exception):
    pass


class BlockAllocator:
    def __init__(self, num_blocks: int, block_size: int):
        self.block_size = block_size
        self.num_blocks = num_blocks
        self._free = list(range(num_blocks))

    def allocate(self) -> int:
        if not self._free:
            raise OutOfBlocksError("no free KV cache blocks")
        return self._free.pop()

    def free(self, block_id: int) -> None:
        self._free.append(block_id)

    def free_many(self, block_ids: list[int]) -> None:
        self._free.extend(block_ids)

    def num_free(self) -> int:
        return len(self._free)

    def can_allocate(self, n: int) -> bool:
        return len(self._free) >= n
