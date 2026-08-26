"""Phase 6b: weight-only INT8 quantization.

Each linear layer's weight is quantized to int8 with a per-output-channel
symmetric scale, dequantized back to the compute dtype just before its
matmul. This is a memory-footprint optimization (roughly halves weight
storage vs bf16, quarters vs fp32) — decode is memory-bandwidth-bound on
small batches with big weights, so cutting weight bytes moved per step is
a real win even without a custom int8 GEMM kernel (which this repo
doesn't have; the matmul itself still runs in the compute dtype after
dequantizing).
"""

import torch


def quantize_per_channel(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """weight: (out_features, in_features). Returns (int8_weight, scale),
    scale shape (out_features, 1), such that
    dequantize(int8_weight, scale) ~= weight."""
    absmax = weight.float().abs().amax(dim=1, keepdim=True).clamp(min=1e-8)
    scale = absmax / 127.0
    q = torch.clamp((weight.float() / scale).round(), -127, 127).to(torch.int8)
    return q, scale


def dequantize(q: torch.Tensor, scale: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    return (q.to(torch.float32) * scale).to(dtype)


class QuantizedWeight:
    """Drop-in replacement for a plain weight tensor: stores int8 + scale,
    dequantizes on each access. `.get()` instead of using the tensor
    directly, so callers opt in explicitly rather than a quantized tensor
    silently behaving like a normal one."""

    def __init__(self, weight: torch.Tensor):
        self.dtype = weight.dtype
        self.q, self.scale = quantize_per_channel(weight)

    def get(self) -> torch.Tensor:
        return dequantize(self.q, self.scale, self.dtype)

    def to(self, device) -> "QuantizedWeight":
        self.q = self.q.to(device)
        self.scale = self.scale.to(device)
        return self

    @property
    def storage_bytes(self) -> int:
        return self.q.numel() * self.q.element_size() + self.scale.numel() * self.scale.element_size()
