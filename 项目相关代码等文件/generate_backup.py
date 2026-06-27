# generate.py
import torch
import tiktoken
from model import ModernTransformer
from config import config

def generate(model, prompt, max_tokens=50, temperature=0.8, top_k=40, top_p=0.95):
    device = next(model.parameters()).device
    enc = tiktoken.get_encoding("gpt2")
    input_ids = torch.tensor(enc.encode(prompt), dtype=torch.long).unsqueeze(0).to(device)
    
    model.eval()
    with torch.no_grad():
        for _ in range(max_tokens):
            # 截断到最大长度
            if input_ids.size(1) > config['max_seq_len']:
                input_ids = input_ids[:, -config['max_seq_len']:]
            logits = model(input_ids)
            logits = logits[:, -1, :] / temperature
            
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
    
    return enc.decode(input_ids[0].tolist())

if __name__ == "__main__":
    import sys
    prompt = sys.argv[1] if len(sys.argv) > 1 else "Once upon a time"
    model = ModernTransformer(config)
    model.load_state_dict(torch.load("model_adamw_final.pt", map_location='cpu'))
    model = model.cuda() if torch.cuda.is_available() else model
    print(generate(model, prompt, max_tokens=80))