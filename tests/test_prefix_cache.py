"""Real, runnable tests for prefix caching — pure Python, no GPU needed."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engine.prefix_cache import PrefixCache


def test_shared_prefix_reuses_blocks_diverging_suffix_does_not():
    cache = PrefixCache(block_size=4)

    seq1_tokens = list(range(12))  # 3 full blocks: [0-3][4-7][8-11]
    # Simulate writing seq1's blocks fresh (physical ids 100, 101, 102).
    prev_hash = 0
    seq1_blocks = []
    for i, phys in enumerate((100, 101, 102)):
        chunk = tuple(seq1_tokens[i * 4 : (i + 1) * 4])
        prev_hash = cache.register_block(phys, prev_hash, chunk)
        seq1_blocks.append(phys)

    # seq2 shares seq1's first two blocks exactly, diverges on the third.
    seq2_tokens = seq1_tokens[:8] + [999, 999, 999, 999]
    match = cache.match(seq2_tokens)

    assert match.block_table == [100, 101]
    assert match.num_cached_tokens == 8
    assert match.num_full_blocks_matched == 2  # 3rd block must be recomputed


def test_ref_counting_keeps_shared_block_alive_until_both_release():
    cache = PrefixCache(block_size=4)
    chunk = tuple(range(4))
    h = cache.register_block(phys_block=5, prev_hash=0, chunk=chunk)

    # A second sequence's match() call increments the same block's ref count.
    match = cache.match(list(chunk))
    assert match.block_table == [5]

    assert cache.release(5) is False, "still referenced by the second sequence"
    assert cache.release(5) is True, "last reference released, block must free"

    # Cache entry is gone — a third sequence with the same prefix cannot
    # match a block that's been freed and (by the allocator) reused for
    # something else.
    match_after = cache.match(list(chunk))
    assert match_after.num_full_blocks_matched == 0


def test_no_match_when_prefix_completely_different():
    cache = PrefixCache(block_size=4)
    cache.register_block(phys_block=1, prev_hash=0, chunk=(0, 1, 2, 3))

    match = cache.match([9, 9, 9, 9, 10, 11, 12, 13])
    assert match.block_table == []
    assert match.num_cached_tokens == 0


if __name__ == "__main__":
    test_shared_prefix_reuses_blocks_diverging_suffix_does_not()
    test_ref_counting_keeps_shared_block_alive_until_both_release()
    test_no_match_when_prefix_completely_different()
    print("all prefix cache tests passed")
