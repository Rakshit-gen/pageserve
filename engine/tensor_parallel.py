"""Phase 6b: tensor parallelism primitives (Megatron-style column/row
parallel linear layers).

Column-parallel splits a weight's *output* dimension across ranks — no
communication needed, each rank just owns a slice of attention heads (or
MLP intermediate channels) and computes its slice independently. Row-
parallel splits the *input* dimension and all-reduces the partial sums —
this is what the projection right after a column-parallel layer needs,
since its input is already sharded across ranks the same way.

Uses torch.distributed generically: gloo works on CPU (what
tests/test_tensor_parallel.py uses to actually verify this), nccl is what
a real multi-GPU deployment would use instead. Nothing here has been run
on more than one real GPU — this repo has never had access to two at once.
"""

import torch
import torch.distributed as dist


def column_parallel_linear(
    x: torch.Tensor, full_weight: torch.Tensor, rank: int, world_size: int
) -> torch.Tensor:
    """full_weight: (out_features, in_features). Returns this rank's shard
    of the output — (*, out_features // world_size). No communication:
    the caller decides whether/how to gather shards from other ranks."""
    out_features = full_weight.shape[0]
    assert out_features % world_size == 0, "output dim must divide evenly across ranks"
    shard = out_features // world_size
    local_weight = full_weight[rank * shard : (rank + 1) * shard]
    return x @ local_weight.T


def row_parallel_linear(
    x_shard: torch.Tensor, full_weight: torch.Tensor, rank: int, world_size: int
) -> torch.Tensor:
    """full_weight: (out_features, in_features). x_shard is this rank's
    slice of the input dimension — the output of a preceding column-
    parallel layer. Each rank computes a partial output from its shard,
    then all-reduces (sums) so every rank ends up with the true, full
    output."""
    in_features = full_weight.shape[1]
    assert in_features % world_size == 0, "input dim must divide evenly across ranks"
    shard = in_features // world_size
    local_weight = full_weight[:, rank * shard : (rank + 1) * shard]
    partial = x_shard @ local_weight.T
    dist.all_reduce(partial, op=dist.ReduceOp.SUM)
    return partial
