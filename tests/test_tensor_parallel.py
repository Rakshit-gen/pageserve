"""Real, runnable tests for tensor-parallel primitives — 2 CPU processes
via torch.distributed's gloo backend (no GPU needed). Verifies the
sharded, communicating computation matches a plain single-process
reference exactly: column-parallel shards concatenate to the full output,
and column-parallel -> row-parallel (mimicking an attention/MLP up/down
projection pair) reproduces the full computation after the all-reduce.
"""

import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.multiprocessing as mp

from engine.tensor_parallel import column_parallel_linear, row_parallel_linear

WORLD_SIZE = 2


def _init_process_group(rank: int, world_size: int, sync_file: str):
    import torch.distributed as dist

    dist.init_process_group(
        backend="gloo", rank=rank, world_size=world_size, init_method=f"file://{sync_file}"
    )


def _column_parallel_worker(rank, world_size, sync_file, data_file, result_file):
    _init_process_group(rank, world_size, sync_file)
    data = torch.load(data_file, weights_only=True)
    x, weight = data["x"], data["weight"]

    local_out = column_parallel_linear(x, weight, rank, world_size)
    torch.save(local_out, f"{result_file}.{rank}")


def _column_then_row_worker(rank, world_size, sync_file, data_file, result_file):
    _init_process_group(rank, world_size, sync_file)
    data = torch.load(data_file, weights_only=True)
    x, w1, w2 = data["x"], data["w1"], data["w2"]

    hidden_shard = column_parallel_linear(x, w1, rank, world_size)
    out = row_parallel_linear(hidden_shard, w2, rank, world_size)
    torch.save(out, f"{result_file}.{rank}")


def test_column_parallel_shards_concatenate_to_full_output():
    torch.manual_seed(0)
    x = torch.randn(4, 16)
    weight = torch.randn(8, 16)  # out_features=8, divisible by world_size=2
    expected = x @ weight.T

    with tempfile.TemporaryDirectory() as tmp:
        sync_file = os.path.join(tmp, "sync")
        data_file = os.path.join(tmp, "data.pt")
        result_file = os.path.join(tmp, "result")
        torch.save({"x": x, "weight": weight}, data_file)

        mp.spawn(
            _column_parallel_worker,
            args=(WORLD_SIZE, sync_file, data_file, result_file),
            nprocs=WORLD_SIZE,
            join=True,
        )

        shard0 = torch.load(f"{result_file}.0", weights_only=True)
        shard1 = torch.load(f"{result_file}.1", weights_only=True)
        reconstructed = torch.cat([shard0, shard1], dim=-1)
        assert torch.allclose(reconstructed, expected, atol=1e-5)


def test_column_then_row_parallel_matches_full_computation():
    torch.manual_seed(1)
    x = torch.randn(4, 16)
    w1 = torch.randn(8, 16)  # column-parallel: 16 -> 8, sharded on the 8
    w2 = torch.randn(16, 8)  # row-parallel: 8 -> 16, sharded on the 8 (matches w1's output shard)
    expected = (x @ w1.T) @ w2.T

    with tempfile.TemporaryDirectory() as tmp:
        sync_file = os.path.join(tmp, "sync")
        data_file = os.path.join(tmp, "data.pt")
        result_file = os.path.join(tmp, "result")
        torch.save({"x": x, "w1": w1, "w2": w2}, data_file)

        mp.spawn(
            _column_then_row_worker,
            args=(WORLD_SIZE, sync_file, data_file, result_file),
            nprocs=WORLD_SIZE,
            join=True,
        )

        out0 = torch.load(f"{result_file}.0", weights_only=True)
        out1 = torch.load(f"{result_file}.1", weights_only=True)
        # every rank should hold the identical, fully-reduced result
        assert torch.allclose(out0, out1, atol=1e-5)
        assert torch.allclose(out0, expected, atol=1e-5)


if __name__ == "__main__":
    test_column_parallel_shards_concatenate_to_full_output()
    test_column_then_row_parallel_matches_full_computation()
    print("all tensor parallel tests passed")
