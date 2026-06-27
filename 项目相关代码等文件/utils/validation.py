"""Validation utilities for model components."""
import torch
import torch.nn.functional as F
from typing import Callable


def validate_rms_norm(norm_layer: torch.nn.Module, dim: int = 256) -> bool:
    """Validate RMSNorm: output std should be ~1.0."""
    x = torch.randn(2, 128, dim)
    out = norm_layer(x)
    std_mean = out.std(dim=-1).mean().item()
    assert 0.95 <= std_mean <= 1.05, f"RMSNorm std out of range: {std_mean}"
    print(f"✅ RMSNorm validation passed! std_mean={std_mean:.3f}")
    return True


def validate_swiglu(swiglu_layer: torch.nn.Module, dim: int = 256, hidden: int = 1024) -> bool:
    """Validate SwiGLU against manual implementation."""
    x = torch.randn(2, 128, dim)
    out_custom = swiglu_layer(x)

    # Manual SwiGLU using same weights
    w1 = swiglu_layer.w1.weight
    w2 = swiglu_layer.w2.weight
    w3 = swiglu_layer.w3.weight
    out_torch = (F.silu(x @ w1.T) * (x @ w3.T)) @ w2.T

    max_diff = (out_custom - out_torch).abs().max().item()
    assert max_diff < 1e-5, f"SwiGLU max diff too high: {max_diff}"
    print(f"✅ SwiGLU validation passed! max_diff={max_diff:.2e}")
    return True


def validate_rope(
    apply_rope_fn: Callable,
    dim: int = 64,
    max_seq_len: int = 128,
) -> bool:
    """Validate RoPE relative position property."""
    from model.rope import precompute_freqs_cis

    freqs_cis = precompute_freqs_cis(dim, max_seq_len)

    B, H = 2, 8
    # 使用完全相同的 query 和 key 内容
    q_same = torch.randn(B, 1, H, dim)
    k_same = torch.randn(B, 1, H, dim)

    # 位置 i 的 query 与位置 j 的 key 的内积
    pos_i = 0
    pos_j = 1
    q_i = q_same
    k_j = k_same
    q_i_rot, _ = apply_rope_fn(q_i, k_j, freqs_cis[pos_i:pos_i+1])
    _, k_j_rot = apply_rope_fn(q_i, k_j, freqs_cis[pos_j:pos_j+1])
    score_0_1 = (q_i_rot[0, 0] * k_j_rot[0, 0]).sum()

    pos_i = 100
    pos_j = 101
    q_i2 = q_same
    k_j2 = k_same
    q_i2_rot, _ = apply_rope_fn(q_i2, k_j2, freqs_cis[pos_i:pos_i+1])
    _, k_j2_rot = apply_rope_fn(q_i2, k_j2, freqs_cis[pos_j:pos_j+1])
    score_100_101 = (q_i2_rot[0, 0] * k_j2_rot[0, 0]).sum()

    diff = abs(score_0_1 - score_100_101).item()
    assert diff < 1e-4, f"RoPE relative position property violated: diff={diff}"
    print(f"✅ RoPE validation passed! relative position diff={diff:.2e}")
    return True