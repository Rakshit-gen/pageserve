"""Real, runnable tests for INT8 weight quantization — pure math, no
model/GPU needed."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from engine.quantization import QuantizedWeight, dequantize, quantize_per_channel


def test_round_trip_error_is_bounded():
    torch.manual_seed(0)
    weight = torch.randn(64, 128, dtype=torch.float32)
    q, scale = quantize_per_channel(weight)

    assert q.dtype == torch.int8
    assert scale.shape == (64, 1)

    recovered = dequantize(q, scale, torch.float32)
    max_err = (weight - recovered).abs().max().item()
    # per-channel absmax/127 quantization step size is at most absmax/127;
    # error per element is bounded by half a quantization step.
    max_step = (weight.abs().amax(dim=1, keepdim=True) / 127.0).max().item()
    assert max_err <= max_step, f"quantization error {max_err} exceeds one step {max_step}"


def test_quantized_weight_storage_is_smaller():
    weight = torch.randn(2048, 2048, dtype=torch.float32)
    qw = QuantizedWeight(weight)
    fp32_bytes = weight.numel() * weight.element_size()
    assert qw.storage_bytes < fp32_bytes / 3, "int8 + per-channel scale should be well under 1/3 of fp32 size"


def test_dequantized_matmul_close_to_original():
    torch.manual_seed(1)
    x = torch.randn(8, 128)
    weight = torch.randn(64, 128)
    qw = QuantizedWeight(weight)

    expected = x @ weight.T
    actual = x @ qw.get().T
    rel_err = (expected - actual).abs().max().item() / expected.abs().max().item()
    assert rel_err < 0.05, f"quantized matmul diverges too much: {rel_err:.4f} relative error"


if __name__ == "__main__":
    test_round_trip_error_is_bounded()
    test_quantized_weight_storage_is_smaller()
    test_dequantized_matmul_close_to_original()
    print("all quantization tests passed")
