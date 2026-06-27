# generate_with_cache.py
import torch
from tokenizers import Tokenizer
from model import ModernTransformer
from config import config

def main():
    import sys
    prompt = sys.argv[1] if len(sys.argv) > 1 else "The little cat"

    tokenizer = Tokenizer.from_file("tokenizer.json")
    model = ModernTransformer(config)
    model.load_state_dict(torch.load("model_adamw_final.pt", map_location='cpu'))
    if torch.cuda.is_available():
        model = model.cuda()
    else:
        model = model.cpu()

    device = next(model.parameters()).device

    input_ids = torch.tensor(tokenizer.encode(prompt).ids, dtype=torch.long).unsqueeze(0).to(device)
    generated_ids = model.generate(
    input_ids,
    max_new_tokens=80,
    temperature=0.7,        # 从 0.8 降到 0.7，更确定
    top_k=25,               # 从 40 降到 25
    top_p=0.9,              # 从 0.95 降到 0.9
    repetition_penalty=1.5  # 从 1.2 提高到 1.5，更强惩罚
)
    text = tokenizer.decode(generated_ids[0].tolist())
    text = text.replace('Ġ', ' ').replace('Ċ', '\n').replace('â', ' ')
    text = text.replace('Ĥ¬', '').strip()
    print(text)

if __name__ == "__main__":
    main()