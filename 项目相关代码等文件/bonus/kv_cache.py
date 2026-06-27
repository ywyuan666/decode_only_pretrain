import torch
import time
from typing import Optional

class KVCache:
    def __init__(self, max_batch_size: int, max_seq_len: int, n_heads: int, head_dim: int, device: torch.device, dtype: torch.float32):
        # 预分配张量（核心修复：不再用 torch.cat）
        self.k_cache = torch.zeros(
            max_batch_size, n_heads, max_seq_len, head_dim,
            device=device, dtype=dtype
        )
        self.v_cache = torch.zeros(
            max_batch_size, n_heads, max_seq_len, head_dim,
            device=device, dtype=dtype
        )
        self.seq_len = 0
        self.max_seq_len = max_seq_len

    def update(self, k: torch.Tensor, v: torch.Tensor):
        batch_size, n_heads, new_len, dim = k.shape
        total_len = self.seq_len + new_len

        self.k_cache[:batch_size, :, self.seq_len:total_len] = k
        self.v_cache[:batch_size, :, self.seq_len:total_len] = v
        self.seq_len = total_len

        return (
            self.k_cache[:batch_size, :, :self.seq_len],
            self.v_cache[:batch_size, :, :self.seq_len]
        )

    def reset(self):
        self.seq_len = 0

def test_kv_cache_speed(model, device, max_tokens=1000):
    model.eval()
    dummy = torch.randint(0, 1000, (1, 10)).to(device)

    # 必须加同步！
    torch.cuda.synchronize()
    t0 = time.time()

    kv_cache = KVCache(
        max_batch_size=1,
        max_seq_len=max_tokens + 10,
        n_heads=model.config.n_heads,
        head_dim=model.config.head_dim,
        device=device,
        dtype=torch.float32
    )

    x = dummy
    for _ in range(max_tokens):
        logits, _ = model(x, kv_cache=kv_cache)
        next_token = logits.argmax(-1)[:, -1:]
        x = next_token

    torch.cuda.synchronize()
    print(f"✅ KV Cache 加速版耗时：{time.time() - t0:.2f}s")