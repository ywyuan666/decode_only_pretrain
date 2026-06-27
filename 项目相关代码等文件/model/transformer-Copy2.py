import torch
import torch.nn as nn
import torch.nn.functional as F
from .attention import GQAAttention, precompute_freqs_cis
from typing import Optional

# ==================== RMSNorm ====================
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return norm * self.weight

# ==================== SwiGLU ====================
class SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))

# ==================== MoE (with expert usage statistics) ====================
class MoE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_experts = config["num_experts"]
        self.top_k = config["top_k"]
        self.dim = config["dim"]
        self.hidden_dim = config["hidden_dim"]

        self.experts = nn.ModuleList([SwiGLU(self.dim, self.hidden_dim) for _ in range(self.num_experts)])
        self.gate = nn.Linear(self.dim, self.num_experts, bias=False)

        # 用于统计专家使用次数（不影响梯度）
        self.register_buffer("expert_counts", torch.zeros(self.num_experts))
        self.register_buffer("total_tokens", torch.tensor(0))

    def reset_expert_counts(self):
        """清零统计计数器"""
        self.expert_counts.zero_()
        self.total_tokens.zero_()

    def forward(self, x):
        B, T, C = x.shape
        gate_logits = self.gate(x)
        weights, selected = torch.topk(gate_logits, self.top_k, dim=-1)
        weights = F.softmax(weights, dim=-1)

        # 更新 expert 统计（仅在非训练模式或根据需要）
        if not self.training:
            with torch.no_grad():
                for i in range(self.num_experts):
                    mask_i = (selected == i).any(dim=-1)
                    self.expert_counts[i] += mask_i.sum().item()
                self.total_tokens += selected.numel()

        output = torch.zeros_like(x)
        for i in range(self.num_experts):
            mask = (selected == i).any(dim=-1)
            if mask.any():
                expert_out = self.experts[i](x[mask])
                # 加权求和（多个专家可能被选中，需要加权）
                expert_weight = weights[selected == i].view(-1, 1)
                output[mask] += expert_out * expert_weight

        return output

# ==================== DecoderLayer ====================
class DecoderLayer(nn.Module):
    def __init__(self, config, use_moe: bool = False):
        super().__init__()
        self.norm1 = RMSNorm(config["dim"])
        self.attn = GQAAttention(config)
        self.norm2 = RMSNorm(config["dim"])
        self.use_moe = use_moe

        if use_moe:
            self.ffn = MoE(config)
        else:
            self.ffn = SwiGLU(config["dim"], config["hidden_dim"])

    def forward(self, x, freqs_cis=None, kv_cache=None):
        x = x + self.attn(self.norm1(x), freqs_cis=freqs_cis, kv_cache=kv_cache)
        x = x + self.ffn(self.norm2(x))
        return x

# ==================== ModernTransformer ====================
class ModernTransformer(nn.Module):
    def __init__(self, config, use_moe: bool = False):
        super().__init__()
        self.config = config
        self.use_moe = use_moe
        self.embedding = nn.Embedding(config["vocab_size"], config["dim"])
        self.layers = nn.ModuleList([DecoderLayer(config, use_moe=use_moe) for _ in range(config["n_layers"])])
        self.norm = RMSNorm(config["dim"])
        self.head = nn.Linear(config["dim"], config["vocab_size"], bias=False)
        self.freqs_cis = precompute_freqs_cis(config["head_dim"], config["max_seq_len"])

    def reset_all_expert_counts(self):
        """清零所有 MoE 层的专家计数"""
        for layer in self.layers:
            if self.use_moe and hasattr(layer.ffn, 'reset_expert_counts'):
                layer.ffn.reset_expert_counts()

    def get_expert_usage(self):
        """获取所有 MoE 层的专家使用总次数，设备与模型一致"""
        total = None
        for layer in self.layers:
            if self.use_moe and hasattr(layer.ffn, 'expert_counts'):
                if total is None:
                    total = torch.zeros_like(layer.ffn.expert_counts)
                total += layer.ffn.expert_counts
        if total is None:
            total = torch.zeros(self.config.get("num_experts", 2))
        return total
    def forward(self, x, kv_cache=None):
        device = x.device
        self.freqs_cis = self.freqs_cis.to(device)
        x = self.embedding(x)

        for layer in self.layers:
            x = layer(x, freqs_cis=self.freqs_cis, kv_cache=kv_cache)

        x = self.norm(x)
        logits = self.head(x)
        return logits, kv_cache