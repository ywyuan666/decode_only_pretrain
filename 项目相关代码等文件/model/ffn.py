"""SwiGLU Feed-Forward Network."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLU(nn.Module):
    """SwiGLU activation based FFN.

    SwiGLU(x) = (Swish(xW1) ⊙ xW3) W2
    where Swish = x * sigmoid(x)
    """

    def __init__(self, d_model: int, hidden_dim: int) -> None:
        super().__init__()
        self.w1 = nn.Linear(d_model, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, d_model, bias=False)
        self.w3 = nn.Linear(d_model, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.w2(F.silu(self.w1(x)) * self.w3(x))