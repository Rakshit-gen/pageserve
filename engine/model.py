"""Phase 2 numerics: a from-scratch decoder-only transformer forward pass
(RMSNorm, RoPE, GQA attention, SwiGLU MLP — the Llama/Qwen2 architecture
family) wired to our own paged KV cache instead of HF's internal cache.

Loads real weights from a pretrained checkpoint rather than random init —
this reimplements the math against paged_attention.py, it does not invent a
new architecture. Config dimensions (hidden size, head counts, rope theta,
norm eps) all come from the checkpoint's own AutoConfig, never hardcoded.

Correctness must be checked against the real HF model before trusting this
for anything — see scripts/verify_model.py.
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .paged_attention import paged_causal_attention_incremental, write_kv
from .quantization import QuantizedWeight

Weight = torch.Tensor | QuantizedWeight


def _resolve(w: Weight) -> torch.Tensor:
    return w.get() if isinstance(w, QuantizedWeight) else w


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    dtype = x.dtype
    x = x.float()
    variance = x.pow(2).mean(-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps)
    return weight * x.to(dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def build_rope_cache(max_len: int, head_dim: int, theta: float, device, dtype):
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(max_len, device=device).float()
    freqs = torch.outer(t, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos().to(dtype), emb.sin().to(dtype)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: (chunk_len, num_heads, head_dim); cos/sin: (chunk_len, head_dim)
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    return x * cos + rotate_half(x) * sin


@dataclass
class LayerWeights:
    input_ln: torch.Tensor
    post_ln: torch.Tensor
    q_w: Weight
    k_w: Weight
    v_w: Weight
    o_w: Weight
    gate_w: Weight
    up_w: Weight
    down_w: Weight
    q_b: torch.Tensor | None = None
    k_b: torch.Tensor | None = None
    v_b: torch.Tensor | None = None


class PagedCausalLM:
    """Loads a real HF checkpoint's weights, runs forward passes through our
    own paged-attention implementation instead of HF's model code."""

    def __init__(
        self,
        model_id: str,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
        quantize: bool = False,
    ):
        from transformers import AutoConfig, AutoModelForCausalLM

        self.device = device
        self.dtype = dtype
        self.quantize = quantize
        self.config = AutoConfig.from_pretrained(model_id)
        self.num_heads = self.config.num_attention_heads
        self.num_kv_heads = self.config.num_key_value_heads
        self.hidden_size = self.config.hidden_size
        self.head_dim = getattr(self.config, "head_dim", self.hidden_size // self.num_heads)
        self.rms_eps = self.config.rms_norm_eps
        rope_params = getattr(self.config, "rope_parameters", None)
        self.rope_theta = (
            rope_params["rope_theta"] if rope_params else self.config.rope_theta
        )
        self.max_position_embeddings = self.config.max_position_embeddings

        hf_model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype)
        sd = hf_model.state_dict()

        self.embed_tokens = sd["model.embed_tokens.weight"].to(device)
        self.final_norm_w = sd["model.norm.weight"].to(device)
        self.lm_head_w = sd.get("lm_head.weight", self.embed_tokens).to(device)

        def load_weight(key: str) -> Weight:
            w = sd[key]
            if self.quantize:
                return QuantizedWeight(w).to(device)
            return w.to(device)

        self.layers: list[LayerWeights] = []
        for i in range(self.config.num_hidden_layers):
            p = f"model.layers.{i}."
            layer = LayerWeights(
                input_ln=sd[p + "input_layernorm.weight"].to(device),
                post_ln=sd[p + "post_attention_layernorm.weight"].to(device),
                q_w=load_weight(p + "self_attn.q_proj.weight"),
                k_w=load_weight(p + "self_attn.k_proj.weight"),
                v_w=load_weight(p + "self_attn.v_proj.weight"),
                o_w=load_weight(p + "self_attn.o_proj.weight"),
                gate_w=load_weight(p + "mlp.gate_proj.weight"),
                up_w=load_weight(p + "mlp.up_proj.weight"),
                down_w=load_weight(p + "mlp.down_proj.weight"),
                q_b=sd.get(p + "self_attn.q_proj.bias"),
                k_b=sd.get(p + "self_attn.k_proj.bias"),
                v_b=sd.get(p + "self_attn.v_proj.bias"),
            )
            for attr in ("q_b", "k_b", "v_b"):
                val = getattr(layer, attr)
                if val is not None:
                    setattr(layer, attr, val.to(device))
            self.layers.append(layer)

        del hf_model

        self.cos_cache, self.sin_cache = build_rope_cache(
            self.max_position_embeddings, self.head_dim, self.rope_theta, device, dtype
        )

    def allocate_kv_cache(self, num_blocks: int, block_size: int) -> None:
        shape = (len(self.layers), num_blocks, block_size, self.num_kv_heads, self.head_dim)
        self.k_cache = torch.zeros(shape, dtype=self.dtype, device=self.device)
        self.v_cache = torch.zeros(shape, dtype=self.dtype, device=self.device)
        self.block_size = block_size

    def forward_step(
        self,
        input_ids: torch.Tensor,
        block_table: list[int],
        start_pos: int,
    ) -> torch.Tensor:
        """Runs one scheduler-step chunk through every layer. Returns logits
        for every position in the chunk (chunk_len, vocab_size); the caller
        only needs the last row once a sequence's prefill completes."""
        chunk_len = input_ids.shape[0]
        x = F.embedding(input_ids, self.embed_tokens)

        cos = self.cos_cache[start_pos : start_pos + chunk_len]
        sin = self.sin_cache[start_pos : start_pos + chunk_len]

        for layer_idx, layer in enumerate(self.layers):
            residual = x
            h = rms_norm(x, layer.input_ln, self.rms_eps)

            q = F.linear(h, _resolve(layer.q_w), layer.q_b).view(chunk_len, self.num_heads, self.head_dim)
            k = F.linear(h, _resolve(layer.k_w), layer.k_b).view(chunk_len, self.num_kv_heads, self.head_dim)
            v = F.linear(h, _resolve(layer.v_w), layer.v_b).view(chunk_len, self.num_kv_heads, self.head_dim)

            q = apply_rope(q, cos, sin)
            k = apply_rope(k, cos, sin)

            write_kv(
                self.k_cache[layer_idx], self.v_cache[layer_idx],
                block_table, self.block_size, start_pos, k, v,
            )
            attn_out = paged_causal_attention_incremental(
                q, self.k_cache[layer_idx], self.v_cache[layer_idx],
                block_table, self.block_size, start_pos,
            )
            attn_out = attn_out.reshape(chunk_len, self.num_heads * self.head_dim)
            x = residual + F.linear(attn_out, _resolve(layer.o_w))

            residual = x
            h = rms_norm(x, layer.post_ln, self.rms_eps)
            gate = F.linear(h, _resolve(layer.gate_w))
            up = F.linear(h, _resolve(layer.up_w))
            x = residual + F.linear(F.silu(gate) * up, _resolve(layer.down_w))

        x = rms_norm(x, self.final_norm_w, self.rms_eps)
        return F.linear(x, self.lm_head_w)
