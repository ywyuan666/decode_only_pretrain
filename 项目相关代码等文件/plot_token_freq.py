import torch
import matplotlib.pyplot as plt
from collections import Counter
from tokenizers import Tokenizer
from model import ModernTransformer
from config import config

def generate_text(model, prompt, max_tokens=100, temperature=0.7, top_k=25, top_p=0.9):
    tokenizer = Tokenizer.from_file("tokenizer.json")
    device = next(model.parameters()).device
    input_ids = torch.tensor(tokenizer.encode(prompt).ids, dtype=torch.long).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        for _ in range(max_tokens):
            if input_ids.size(1) > config["max_seq_len"]:
                input_ids = input_ids[:, -config["max_seq_len"]:]
            logits = model(input_ids)[:, -1, :] / temperature
            if top_k > 0:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float('Inf')
            if top_p > 0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                sorted_indices_to_remove[:, 0] = False
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = -float('Inf')
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=-1)
    return tokenizer.decode(input_ids[0].tolist())

def main():
    print("加载模型...")
    model = ModernTransformer(config)
    model.load_state_dict(torch.load("model_adamw_final.pt", map_location='cpu'))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    prompts = [
        "Once upon a time",
        "The little cat",
        "In a bright sunny day",
        "The brave knight",
        "A tiny mouse"
    ]
    all_text = []
    for prompt in prompts:
        generated = generate_text(model, prompt, max_tokens=50)
        all_text.append(generated)
        print(f"Prompt: {prompt}\nGenerated: {generated}\n")

    # 分词并统计频率
    tokenizer = Tokenizer.from_file("tokenizer.json")
    all_ids = []
    for text in all_text:
        all_ids.extend(tokenizer.encode(text).ids)

    counter = Counter(all_ids)
    common = counter.most_common(20)
    ids, counts = zip(*common)
    words = [tokenizer.decode([tid]).strip() for tid in ids]

    plt.figure(figsize=(12, 5))
    plt.bar(words, counts, color='steelblue')
    plt.xlabel('Token')
    plt.ylabel('Frequency')
    plt.title('Top 20 Most Frequent Tokens in Generated Text')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('token_freq.png', dpi=150)
    print("分词频率图已保存为 token_freq.png")

if __name__ == "__main__":
    main()