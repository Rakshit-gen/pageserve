"""Phase 6: prefix caching — reuse KV blocks across requests that share an
identical token prefix (e.g. a common system prompt), skipping prefill
recompute for the shared portion.

Blocks are content-addressed: hash(prev_block_hash, token_ids_in_block).
Two sequences whose leading blocks hash identically share the same physical
block id and a ref count, instead of each writing their own copy. Matching
stops at the first block-content miss — a shared prefix, not a shared
suffix or arbitrary subsequence, since KV for token i depends on everything
before it.
"""

from dataclasses import dataclass, field


def _block_hash(prev_hash: int, tokens: tuple[int, ...]) -> int:
    return hash((prev_hash, tokens))


@dataclass
class PrefixMatch:
    block_table: list[int]
    num_cached_tokens: int
    last_hash: int
    num_full_blocks_matched: int


@dataclass
class PrefixCache:
    block_size: int
    _hash_to_block: dict = field(default_factory=dict)
    _ref_count: dict = field(default_factory=dict)
    _hash_of_block: dict = field(default_factory=dict)

    def match(self, token_ids: list[int]) -> PrefixMatch:
        """Walk token_ids block by block; reuse cached physical blocks for
        every leading block whose content hash already exists, stopping at
        the first miss. Increments ref counts on every reused block."""
        block_table: list[int] = []
        prev_hash = 0
        num_cached_tokens = 0
        num_full_blocks = len(token_ids) // self.block_size

        for i in range(num_full_blocks):
            chunk = tuple(token_ids[i * self.block_size : (i + 1) * self.block_size])
            h = _block_hash(prev_hash, chunk)
            phys = self._hash_to_block.get(h)
            if phys is None:
                return PrefixMatch(block_table, num_cached_tokens, prev_hash, i)
            self._ref_count[phys] += 1
            block_table.append(phys)
            num_cached_tokens += self.block_size
            prev_hash = h

        return PrefixMatch(block_table, num_cached_tokens, prev_hash, num_full_blocks)

    def register_block(self, phys_block: int, prev_hash: int, chunk: tuple[int, ...]) -> int:
        """Record a newly-written full block as cacheable; returns its hash
        (pass as prev_hash for the next block in the same sequence)."""
        h = _block_hash(prev_hash, chunk)
        existing = self._hash_to_block.get(h)
        if existing is not None:
            self._ref_count[existing] += 1
            return h
        self._hash_to_block[h] = phys_block
        self._ref_count[phys_block] = 1
        self._hash_of_block[phys_block] = h
        return h

    def release(self, phys_block: int) -> bool:
        """Decrement ref count for a block this sequence no longer needs.
        Returns True if the block is now unreferenced and safe to free."""
        if phys_block not in self._ref_count:
            return True
        self._ref_count[phys_block] -= 1
        if self._ref_count[phys_block] <= 0:
            h = self._hash_of_block.pop(phys_block, None)
            if h is not None:
                self._hash_to_block.pop(h, None)
            self._ref_count.pop(phys_block, None)
            return True
        return False
