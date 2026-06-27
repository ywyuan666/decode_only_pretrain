"""Rotary Position Embedding (RoPE) implementation."""
import torch


def precompute_freqs_cis(
    dim: int, max_seq_len: int, theta: float = 10000.0
) -> torch.Tensor:
    """Precompute complex exponential frequencies for RoPE.

    Args:
        dim: Head dimension (must be even).
        max_seq_len: Maximum sequence length.
        theta: Base for frequency computation.

    Returns:
        Tensor of shape (max_seq_len, dim // 2) with complex numbers.
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(max_seq_len)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis


def apply_rotary_emb(
    xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary embedding to query and key tensors.

    Args:
        xq: Query tensor of shape (B, L, H, D).
        xk: Key tensor of shape (B, L, H_kv, D).
        freqs_cis: Precomputed complex frequencies.

    Returns:
        Tuple of rotated query and key tensors.
    """
    B, L, H, D = xq.shape
    # Reshape to pair real and imaginary parts
    xq_ = torch.view_as_complex(xq.float().reshape(B, L, H, -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(B, L, xk.shape[2], -1, 2))
    # Broadcast freqs_cis to match shape
    freqs_cis = freqs_cis[:L].view(1, L, 1, -1)
    # Rotate by multiplication in complex plane
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)