import torch
from model import ModernTransformer
import time

# ======================
# 1. 配置
# ======================
class Config:
    def __init__(self):
        self.dim = 512
        self.n_heads = 8
        self.n_kv_heads = 2
        self.head_dim = 64
        self.vocab_size = 50304
        self.n_layers = 6
        self.max_seq_len = 1024
        self.hidden_dim = 1408
        self.num_experts = 4
        self.top_k = 2

config = Config()
device = "cuda" if torch.cuda.is_available() else "cpu"

# ======================
# 2. KVCache 类（直接写在测试文件里，100%不报错）
# ======================
class KVCache:
    def __init__(self, max_batch_size: int, max_seq_len: int, n_heads: int, head_dim: int, device: torch.device, dtype: torch.float32):
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

# ======================
# 3. 初始化模型
# ======================
model = ModernTransformer(config).to(device)
model.eval()

# ======================
# 4. 运行测试（优化1：KV Cache 加速）
# ======================
max_tokens = 1000
dummy_input = torch.randint(0, 50304, (1, 10)).to(device)

torch.cuda.synchronize()
start = time.time()

kv_cache = KVCache(
    max_batch_size=1,
    max_seq_len=1024,
    n_heads=config.n_kv_heads,
    head_dim=config.head_dim,
    device=device,
    dtype=torch.float32
)

x = dummy_input
for _ in range(max_tokens):
    logits, _ = model(x[:, -1:], kv_cache=kv_cache)
    next_token = logits.argmax(-1)[:, -1:]
    x = torch.cat([x, next_token], dim=-1)

torch.cuda.synchronize()
end = time.time()

print("\n✅ 优化1 运行成功！")
print(f"生成 {max_tokens} token 耗时：{end-start:.2f} 秒")
print(f"✅ KV Cache 加速修复完成！")