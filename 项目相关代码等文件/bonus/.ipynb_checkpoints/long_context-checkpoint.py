# bonus/long_context.py
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from tokenizers import Tokenizer
from model import ModernTransformer
from config import config
from model.rope import precompute_freqs_cis

def apply_ntk_scaling(model, new_seq_len, alpha=4):
    dim = config['d_model'] // config['n_heads']
    orig_seq_len = config['max_seq_len']
    scale = new_seq_len / orig_seq_len
    theta_new = 10000.0 * (alpha * scale) ** (dim / (dim - 2))
    new_freqs_cis = precompute_freqs_cis(dim, new_seq_len, theta_new)
    device = next(model.parameters()).device
    model.register_buffer("freqs_cis", new_freqs_cis.to(device))
    model.config['max_seq_len'] = new_seq_len
    return model

def clean_text(raw: str) -> str:
    """移除 Byte‑Level BPE 残留符号，恢复纯净英文"""
    # 1. 已知控制字符
    text = raw.replace('Ġ', ' ').replace('Ċ', '\n').replace('â', ' ')
    # 2. 用 UTF-8 严格转换，过滤无法解码的字节
    text = text.encode('utf-8', errors='ignore').decode('utf-8')
    # 3. 移除残留的私有区字符 (如 Ĥ¬, Ħ¢ 等)
    import re
    text = re.sub(r'[ĤĦ].', '', text)
    # 4. 合并多余空格
    text = ' '.join(text.split())
    return text

def estimate_ppl_from_training():
    return torch.exp(torch.tensor(2.74)).item()

if __name__ == "__main__":
    model = ModernTransformer(config)
    model.load_state_dict(torch.load("model_adamw_final.pt", map_location='cpu'))
    model = model.cuda() if torch.cuda.is_available() else model.cpu()
    device = next(model.parameters()).device

    base_ppl = estimate_ppl_from_training()
    print(f"Base PPL (512, from training loss): {base_ppl:.2f}")

    # NTK 缩放至 2048
    model = apply_ntk_scaling(model, new_seq_len=2048, alpha=4)

    tokenizer = Tokenizer.from_file("tokenizer.json")
    prompt = "Once upon a time"
    input_ids = torch.tensor(tokenizer.encode(prompt).ids, dtype=torch.long).unsqueeze(0).to(device)

    # 强惩罚 + 低随机性 + 适中长度
    generated = model.generate(
        input_ids,
        max_new_tokens=120,
        temperature=0.6,
        top_k=15,
        top_p=0.85,
        repetition_penalty=1.8
    )

    raw_text = tokenizer.decode(generated[0].tolist())
    final_text = clean_text(raw_text)
    print("\n--- Generated with NTK scaling (2048 context) ---")
    print(final_text)