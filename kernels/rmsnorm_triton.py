"""Phase 5: one hand-written Triton kernel — fused RMSNorm.

Everything else in this repo either uses a real dependency where one exists
(flash-attn would replace paged_attention.py's attention math on a GPU
build) or is plain PyTorch. This is the one op written as a custom kernel,
scoped deliberately small: fusing the mean-of-squares reduction, rsqrt, and
elementwise scale into a single kernel launch instead of three separate
PyTorch ops (each of which is a full read+write pass over the tensor).

UNTESTED: Triton requires a CUDA device to compile and run, even for a
correctness smoke test — there is no CPU fallback. Do not trust this until
it's been run against engine.model.rms_norm's output on real GPU hardware.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _rmsnorm_kernel(
    x_ptr, weight_ptr, out_ptr,
    stride_row,
    n_cols,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    row_ptr = x_ptr + row_idx * stride_row
    x = tl.load(row_ptr + col_offsets, mask=mask, other=0.0).to(tl.float32)

    variance = tl.sum(x * x, axis=0) / n_cols
    inv_rms = 1.0 / tl.sqrt(variance + eps)
    x_normed = x * inv_rms

    weight = tl.load(weight_ptr + col_offsets, mask=mask, other=0.0).to(tl.float32)
    y = x_normed * weight

    out_row_ptr = out_ptr + row_idx * stride_row
    tl.store(out_row_ptr + col_offsets, y.to(x_ptr.dtype.element_ty), mask=mask)


def rms_norm_triton(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """x: (num_rows, hidden_size) on CUDA. Drop-in replacement for
    engine.model.rms_norm, fused into one kernel launch per call instead of
    pow/mean/rsqrt/mul as separate PyTorch ops."""
    assert x.is_cuda, "Triton kernels require a CUDA tensor — no CPU fallback"
    num_rows, n_cols = x.shape
    out = torch.empty_like(x)
    block_size = triton.next_power_of_2(n_cols)
    _rmsnorm_kernel[(num_rows,)](
        x, weight, out, x.stride(0), n_cols, eps, BLOCK_SIZE=block_size
    )
    return out
