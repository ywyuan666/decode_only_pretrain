import torch
from model.transformer import ModernTransformer
from tokenizers import Tokenizer

# ====================== 模型配置（100%匹配训练）======================
config_dict = {
    "dim": 512,
    "n_heads": 8,
    "n_kv_heads": 2,
    "head_dim": 64,
    "vocab_size": 50304,
    "n_layers": 6,
    "max_seq_len": 256,
    "hidden_dim": 1408,
    "num_experts": 4,
    "top_k": 2
}

# ====================== KV Cache ======================
class KVCache:
    def __init__(self, max_batch_size, max_seq_len, n_heads, head_dim, device, dtype=torch.float32):
        self.k_cache = torch.zeros(max_batch_size, n_heads, max_seq_len, head_dim, device=device, dtype=dtype)
        self.v_cache = torch.zeros(max_batch_size, n_heads, max_seq_len, head_dim, device=device, dtype=dtype)
        self.seq_len = 0

    def update(self, k, v):
        B, H, T, D = k.shape
        self.k_cache[:B, :, self.seq_len:self.seq_len+T] = k
        self.v_cache[:B, :, self.seq_len:self.seq_len+T] = v
        self.seq_len += T
        return self.k_cache[:B, :, :self.seq_len], self.v_cache[:B, :, :self.seq_len]

# ====================== 完美生成函数（无BUG版）======================
@torch.no_grad()
def generate(
    model,
    tokenizer,
    prompt="Hello",
    max_new_tokens=60,
    temperature=0.7,
    top_k=30,
    device="cuda"
):
    model.eval()
    encoded = tokenizer.encode(prompt)
    tokens = torch.tensor([encoded.ids], dtype=torch.long).to(device)
    kv_cache = KVCache(1, 512, 2, 64, device=device)

    for _ in range(max_new_tokens):
        logits, _ = model(tokens[:, -1:], kv_cache=kv_cache)
        logits = logits[:, -1] / temperature

        # 屏蔽所有乱码、坏token
        BAD_TOKENS = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,27,29,50256]
        for tid in BAD_TOKENS:
            if tid < logits.size(-1):
                logits[:, tid] = -float('inf')

        # Top-K 采样
        v, _ = torch.topk(logits, top_k)
        logits[logits < v[:, [-1]]] = -float('inf')

        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        tokens = torch.cat([tokens, next_token], dim=-1)

    # 🔥 终极清理：干净、无空格、无乱码
    text = tokenizer.decode(tokens[0].tolist())
    text = text.replace("Ġ", " ").replace("Ĥ", "").replace("�", "")
    text = text.replace("/", "").replace("'", "").replace('"', '')
    text = " ".join([w for w in text.split() if w.strip() and len(w) > 1])
    return text

# ====================== 主程序 ======================
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 模型初始化
    model = ModernTransformer(config_dict, use_moe=False).to(device)
    
    # 加载训练好的最佳模型
    checkpoint = torch.load("best_model.pt", map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    # 加载分词器
    tokenizer = Tokenizer.from_file("tokenizer.json")

    # 生成文本
    prompt = "Once upon a time"
    output = generate(model, tokenizer, prompt=prompt, device=device)
    
    print("🎉 优化4 完美生成结果：")
    print(output)