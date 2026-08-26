"""Smoke check for Phase 1: greedy decode must be deterministic.

This is the invariant every later phase has to preserve — if continuous
batching or the paged KV cache ever makes output depend on what else is in
the batch, this test (run against that phase's engine) is what catches it.
"""

def test_greedy_decode_is_deterministic():
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from app import Engine

    engine = Engine()
    prompt = "The capital of France is"
    first = engine.generate.remote(prompt, max_new_tokens=16)
    second = engine.generate.remote(prompt, max_new_tokens=16)
    assert first == second, f"non-deterministic greedy decode: {first!r} != {second!r}"
    assert first.strip(), "empty generation"


if __name__ == "__main__":
    test_greedy_decode_is_deterministic()
    print("ok")
