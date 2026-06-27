"""Complete Modern Transformer model with KV cache support."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Union

from .rms_norm import RMSNorm
from .rope import precompute_freqs_cis
from .attention import GQAAttention
from .moe import MoEFFN


class DecoderLayer(nn.Module):
    def __init__(self, config: dict) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(config["d_model"], eps=config["norm_eps"])
        self.attn = GQAAttention(config)
        self.ffn_norm = RMSNorm(config["d_model"], eps=config["norm_eps"])
        self.ffn = MoEFFN(config) if config.get("n_experts") else None

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        attn_out, present_kv = self.attn(
            self.attn_norm(x),
            freqs_cis,
            mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
        )
        x = x + attn_out
        if self.ffn is not None:
            x = x + self.ffn(self.ffn_norm(x), use_moe=False)
        return x, present_kv


class ModernTransformer(nn.Module):
    def __init__(self, config: dict) -> None:
        super().__init__()
        self.config = config
        self.tok_emb = nn.Embedding(config["vocab_size"], config["d_model"])
        self.layers = nn.ModuleList([DecoderLayer(config) for _ in range(config["n_layers"])])
        self.norm = RMSNorm(config["d_model"], eps=config["norm_eps"])
        self.lm_head = nn.Linear(config["d_model"], config["vocab_size"], bias=False)
        self.lm_head.weight = self.tok_emb.weight
        nn.init.normal_(self.tok_emb.weight, mean=0.0, std=0.02)
        head_dim = config["d_model"] // config["n_heads"]
        freqs_cis = precompute_freqs_cis(head_dim, config["max_seq_len"])
        self.register_buffer("freqs_cis", freqs_cis)

    # 原始训练用 forward（不用缓存）
    def forward(self, x: torch.Tensor, labels: Optional[torch.Tensor] = None):
        B, L = x.shape
        x = self.tok_emb(x)
        freqs_cis = self.freqs_cis[:L]
        for layer in self.layers:
            x, _ = layer(x, freqs_cis, use_cache=False)
        x = self.norm(x)
        logits = self.lm_head(x)
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            return logits, loss
        return logits

    def get_aux_loss(self):
        loss = torch.tensor(0.0, device=next(self.parameters()).device)
        for layer in self.layers:
            if hasattr(layer.ffn, 'aux_loss') and layer.ffn.aux_loss is not None:
                loss = loss + layer.ffn.aux_loss
        return loss

    # ========== KV Cache 辅助方法 ==========
    def _forward_with_cache(self, input_ids, use_cache=False, past_key_values=None):
        B, L = input_ids.shape
        x = self.tok_emb(input_ids)
        if past_key_values is not None:
            cached_len = past_key_values[0][0].size(1)
            freqs_cis = self.freqs_cis[cached_len : cached_len + L]
        else:
            freqs_cis = self.freqs_cis[:L]

        new_kvs = [] if use_cache else None
        for i, layer in enumerate(self.layers):
            past_kv = past_key_values[i] if past_key_values is not None else None
            x, present_kv = layer(x, freqs_cis, past_key_value=past_kv, use_cache=use_cache)
            if use_cache and present_kv is not None:
                new_kvs.append(present_kv)
        x = self.norm(x)
        logits = self.lm_head(x)
        return logits, new_kvs

    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 50,
        temperature: float = 0.8,
        top_k: int = 40,
        top_p: float = 0.95,
        repetition_penalty: float = 1.0,
    ) -> torch.Tensor:
        device = next(self.parameters()).device
        if input_ids.device != device:
            input_ids = input_ids.to(device)

        self.eval()
        past_key_values = None
        generated = input_ids.clone()

        with torch.no_grad():
            for _ in range(max_new_tokens):
                if generated.size(1) > self.config['max_seq_len']:
                    generated = generated[:, -self.config['max_seq_len']:]
                    past_key_values = None

                if past_key_values is None:
                    logits, past_key_values = self._forward_with_cache(generated, use_cache=True)
                else:
                    last_token = generated[:, -1:]
                    logits, past_key_values = self._forward_with_cache(last_token, use_cache=True, past_key_values=past_key_values)

                logits = logits[:, -1, :] / temperature

                # ---------- 重复惩罚 ----------
                if repetition_penalty != 1.0:
                    for token_id in torch.unique(generated):
                        if token_id < logits.size(-1):
                            if logits[0, token_id] > 0:
                                logits[0, token_id] /= repetition_penalty
                            else:
                                logits[0, token_id] *= repetition_penalty

                # ---------- 硬性 n‑gram blocking：禁止最近 3 个 token ----------
                n_block = 3
                recent_ids = set(generated[0, -n_block:].tolist())
                for tid in recent_ids:
                    if tid < logits.size(-1):
                        logits[0, tid] = -float('Inf')

                # Top-K
                if top_k > 0:
                    top_k = min(top_k, logits.size(-1))
                    v, _ = torch.topk(logits, top_k)
                    logits[logits < v[:, [-1]]] = -float('Inf')

                # Top-P
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_mask = cumulative_probs > top_p
                    sorted_mask[:, 1:] = sorted_mask[:, :-1].clone()
                    sorted_mask[:, 0] = False
                    mask = sorted_mask.scatter(1, sorted_indices, sorted_mask)
                    logits[mask] = -float('Inf')

                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                generated = torch.cat([generated, next_token], dim=-1)

        return generated