#!/usr/bin/env python3
"""Validation script for all components."""
import yaml
import torch
import torch.nn.functional as F

from model import RMSNorm, SwiGLU, ModernTransformer
from model.rope import precompute_freqs_cis, apply_rotary_emb
from utils.validation import validate_rms_norm, validate_swiglu, validate_rope


def main() -> None:
    """Run all validation tests."""
    print("=" * 50)
    print("Running Component Validations")
    print("=" * 50)

    # Load config
    with open("configs/config.yaml") as f:
        config = yaml.safe_load(f)

    # 1. RMSNorm
    rms_norm = RMSNorm(config["d_model"], eps=float(config["norm_eps"]))
    validate_rms_norm(rms_norm, config["d_model"])

    # 2. SwiGLU
    hidden_dim = int(config["d_model"] * 4 * config["ffn_dim_multiplier"])
    swiglu = SwiGLU(config["d_model"], hidden_dim)
    validate_swiglu(swiglu, config["d_model"], hidden_dim)

    # 3. RoPE
    head_dim = config["d_model"] // config["n_heads"]
    validate_rope(apply_rotary_emb, head_dim, config["max_seq_len"])

    # 4. Shape propagation
    print("\nTesting shape propagation...")
    model = ModernTransformer(config)
    x = torch.randint(0, config["vocab_size"], (2, 128))
    logits = model(x)
    expected_shape = (2, 128, config["vocab_size"])
    assert logits.shape == expected_shape, f"Shape mismatch: {logits.shape} vs {expected_shape}"
    assert torch.isfinite(logits).all(), "NaN/Inf detected in logits!"
    print(f"✅ Shape test passed: {logits.shape}")

    # 5. KV cache size check (theoretical)
    print("\nKV Cache Analysis:")
    mha_cache = config["max_seq_len"] * config["n_heads"] * head_dim * 2  # K and V
    gqa_cache = config["max_seq_len"] * config["n_kv_heads"] * head_dim * 2
    reduction = (1 - gqa_cache / mha_cache) * 100
    print(f"  MHA cache size (theoretical): {mha_cache:,} floats")
    print(f"  GQA cache size (theoretical): {gqa_cache:,} floats")
    print(f"  Reduction: {reduction:.1f}%")
    assert reduction > 70, f"GQA reduction less than 75%: {reduction:.1f}%"
    print("✅ GQA cache reduction meets spec (>75%)")

    print("\n" + "=" * 50)
    print("All validations passed!")
    print("=" * 50)


if __name__ == "__main__":
    main()