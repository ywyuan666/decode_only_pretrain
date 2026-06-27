import torch
from model.attention import precompute_freqs_cis

# ====================== 配置 ======================
class Config:
    def __init__(self):
        self.dim = 512
        self.n_heads = 8
        self.n_kv_heads = 2
        self.head_dim = 64
        self.vocab_size = 50304
        self.n_layers = 2
        self.max_seq_len = 1024
        self.hidden_dim = 1408
        self.num_experts = 4
        self.top_k = 2

config = Config()
device = "cuda" if torch.cuda.is_available() else "cpu"

# ====================== 位置编码（必须传！） ======================
freqs_cis = precompute_freqs_cis(config.head_dim, config.max_seq_len).to(device)

# ====================== 导入 ======================
from model.transformer import DecoderLayer

print("=== 优化2：DecoderLayer 支持 use_moe 开关 ===")

# 1. 标准稠密模型（无MoE）
layer_dense = DecoderLayer(config, use_moe=False).to(device)
# 2. MoE模型
layer_moe = DecoderLayer(config, use_moe=True).to(device)

# 测试输入
x = torch.randn(2, 16, 512).to(device)

# 前向（传入 freqs_cis）
out_dense = layer_dense(x, freqs_cis=freqs_cis)
out_moe = layer_moe(x, freqs_cis=freqs_cis)

print("✅ 标准模型输出形状:", out_dense.shape)
print("✅ MoE 模型输出形状:", out_moe.shape)
print("\n🎉 优化2 完成！稠密 vs MoE 对比成功！")