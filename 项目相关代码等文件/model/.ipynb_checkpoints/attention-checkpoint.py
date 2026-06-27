"""Grouped Query Attention (GQA) with RoPE and KV Cache support."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from .rope import apply_rotary_emb


class GQAAttention(nn.Module):
    """Grouped Query Attention layer with KV caching.

    Uses n_heads query heads and n_kv_heads key/value heads, where
    n_kv_heads divides n_heads.
    """

    def __init__(self, config: dict) -> None:
        super().__init__()
        self.n_heads = config["n_heads"]
        self.n_kv_heads = config["n_kv_heads"]
        self.head_dim = config["d_model"] // self.n_heads
        self.dropout = config["dropout"]

        assert self.n_heads % self.n_kv_heads == 0, "n_heads must be divisible by n_kv_heads"

        self.wq = nn.Linear(config["d_model"], self.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(config["d_model"], self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(config["d_model"], self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(self.n_heads * self.head_dim, config["d_model"], bias=False)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """Forward pass with optional KV caching.

        Args:
            x: Input tensor of shape (B, L, d_model).
            freqs_cis: Precomputed RoPE frequencies.
            mask: Attention mask.
            past_key_value: Cached (K, V) from previous steps.
            use_cache: Whether to return new cache for next step.

        Returns:
            Output tensor of shape (B, L, d_model) and optionally new cache.
        """
        B, L, _ = x.shape

        q = self.wq(x).view(B, L, self.n_heads, self.head_dim)
        k = self.wk(x).view(B, L, self.n_kv_heads, self.head_dim)
        v = self.wv(x).view(B, L, self.n_kv_heads, self.head_dim)

        # Apply RoPE
        q, k = apply_rotary_emb(q, k, freqs_cis)

        # Concatenate with past cache if provided
        if past_key_value is not None:
            k_cache, v_cache = past_key_value
            k = torch.cat([k_cache, k], dim=1)
            v = torch.cat([v_cache, v], dim=1)

        # Store new cache if requested
        new_cache = (k, v) if use_cache else None

        # Expand KV heads to match Q heads
        k = k.repeat_interleave(self.n_heads // self.n_kv_heads, dim=2)
        v = v.repeat_interleave(self.n_heads // self.n_kv_heads, dim=2)

        # Transpose to (B, H, L, D)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        # Determine if causal masking should be applied
        is_causal = (past_key_value is None) and (mask is None)

        # Use PyTorch's efficient attention
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=is_causal,
        )

        out = out.transpose(1, 2).reshape(B, L, -1)
        return self.wo(out), new_cache