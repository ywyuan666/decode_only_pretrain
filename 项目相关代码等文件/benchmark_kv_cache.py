#!/usr/bin/env python3
"""Benchmark KV Cache speedup."""
import time
import yaml
import torch
import importlib.util
from model.transformer import ModernTransformer
from utils.tokenizer import load_tokenizer

# 绕过 bonus/__init__.py 直接导入 kv_cache 模块
spec = importlib.util.spec_from_file_location("kv_cache", "bonus/kv_cache.py")
kv_cache = importlib.util.module_from_spec(spec)
spec.loader.exec_module(kv_cache)
generate_with_kv_cache = kv_cache.generate_with_kv_cache

def main():
    with open("configs/config.yaml") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ModernTransformer(config).to(device)
    # 加载一个训练好的checkpoint，如果没有训练完，可以先用随机权重测试，但结果可能不具代表性
    # 这里假设你已经有 checkpoints/final.pt
    model.load_state_dict(torch.load("checkpoints/final.pt", map_location=device))
    model.eval()
    tokenizer = load_tokenizer(config["tokenizer_path"])

    prompt = "Once upon a time"
    max_tokens = 100

    # ---------- 使用 KV Cache ----------
    start = time.time()
    out1 = generate_with_kv_cache(model, tokenizer, prompt, max_tokens, temperature=0.8)
    time_cache = time.time() - start
    print(f"With KV Cache: {time_cache:.2f}s, {max_tokens/time_cache:.1f} tokens/s")

    # ---------- 不使用 KV Cache (标准逐token生成) ----------
    input_ids = tokenizer.encode(prompt).ids
    input_tensor = torch.tensor([input_ids], device=device)
    generated = input_ids.copy()
    model.eval()
    start = time.time()
    with torch.no_grad():
        for _ in range(max_tokens):
            logits = model(input_tensor)
            probs = torch.softmax(logits[:, -1, :] / 0.8, dim=-1)
            next_token = torch.multinomial(probs, 1).item()
            generated.append(next_token)
            input_tensor = torch.cat([input_tensor, torch.tensor([[next_token]], device=device)], dim=1)
            if next_token == tokenizer.token_to_id("</s>"):
                break
    time_no_cache = time.time() - start
    print(f"Without KV Cache: {time_no_cache:.2f}s, {max_tokens/time_no_cache:.1f} tokens/s")

    speedup = time_no_cache / time_cache
    print(f"Speedup: {speedup:.2f}x")

if __name__ == "__main__":
    main()