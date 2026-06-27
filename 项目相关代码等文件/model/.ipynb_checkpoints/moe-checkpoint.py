"""Mixture of Experts (MoE) FFN layer."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from .ffn import SwiGLU


class MoEFFN(nn.Module):
    """Mixture of Experts with Top-1 gating.
    Can be switched to a dense FFN via `use_moe=False`.
    """
    def __init__(self, config: dict) -> None:
        super().__init__()
        self.n_experts = config["n_experts"]
        self.d_model = config["d_model"]
        hidden_dim = int(self.d_model * 4 * config["ffn_dim_multiplier"])
        multiple_of = config.get("multiple_of", 256)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.experts = nn.ModuleList(
            [SwiGLU(self.d_model, hidden_dim) for _ in range(self.n_experts)]
        )
        self.gate = nn.Linear(self.d_model, self.n_experts, bias=False)
        self.noise_std = 0.01
        self.aux_loss: Optional[torch.Tensor] = None

        # Dense FFN for non-MoE mode
        self.dense_ffn = SwiGLU(self.d_model, hidden_dim)

    def forward(self, x: torch.Tensor, use_moe: bool = False) -> torch.Tensor:
        if not use_moe:
            return self.dense_ffn(x)

        gate_logits = self.gate(x)
        if self.training:
            gate_logits = gate_logits + torch.randn_like(gate_logits) * self.noise_std

        top1_gate, top1_idx = gate_logits.topk(1, dim=-1)
        weights = F.softmax(top1_gate, dim=-1)
        out = torch.zeros_like(x)
        for i in range(self.n_experts):
            mask = (top1_idx == i).squeeze(-1)
            if mask.any():
                expert_out = self.experts[i](x[mask])
                out[mask] = expert_out * weights[mask]
        self.aux_loss = self._load_balancing_loss(gate_logits)
        return out

    def _load_balancing_loss(self, gate_logits: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(gate_logits, dim=-1)
        mean_probs = probs.mean(dim=(0, 1))
        target = torch.ones_like(mean_probs) / self.n_experts
        return F.mse_loss(mean_probs, target)