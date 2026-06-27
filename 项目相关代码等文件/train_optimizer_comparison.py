# train_optimizer_comparison.py
import os, time, math, torch, wandb
from torch.utils.data import DataLoader, Dataset
from tokenizers import Tokenizer
from model import ModernTransformer
from config import config

torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

class TinyStoriesDataset(Dataset):
    def __init__(self, split='train', block_size=512):
        self.block_size = block_size
        self.tokenizer = Tokenizer.from_file("tokenizer.json")
        self.eos_token = "<|endoftext|>"
        self.eos_token_id = self.tokenizer.token_to_id(self.eos_token)

        from datasets import load_dataset
        dataset = load_dataset("roneneldan/TinyStories", split=split)

        self.data = []
        for story in dataset["text"]:
            tokens = self.tokenizer.encode(story).ids
            tokens.append(self.eos_token_id)
            self.data.extend(tokens)
        self.data = torch.tensor(self.data, dtype=torch.long)

    def __len__(self):
        return (len(self.data) - 1) // self.block_size

    def __getitem__(self, idx):
        i = idx * self.block_size
        x = self.data[i:i+self.block_size]
        y = self.data[i+1:i+self.block_size+1]
        return x, y

def get_lr(step, warmup_steps, plateau_steps, total_steps, max_lr=3e-3):
    if step < warmup_steps:
        return max_lr * float(step) / max(1, warmup_steps)
    elif step < plateau_steps:
        return max_lr
    else:
        progress = float(step - plateau_steps) / max(1, total_steps - plateau_steps)
        return max(0.0, 0.5 * (1 + math.cos(math.pi * progress))) * max_lr

def train(opt_name='adamw', total_steps=2000, batch_size=4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wandb.init(project="optimizer_comparison", name=f"{opt_name}_{total_steps}steps", mode="offline")

    train_dataset = TinyStoriesDataset(block_size=config['max_seq_len'])
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=0, pin_memory=True
    )

    model = ModernTransformer(config).to(device)
    print(f"[{opt_name}] Model params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    if opt_name == 'adamw':
        opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.01, betas=(0.9, 0.95))
    elif opt_name == 'sophia':
        from sophia_opt import SophiaG
        opt = SophiaG(model.parameters(), lr=3e-3, weight_decay=0.1, betas=(0.965, 0.99), rho=0.01)
    elif opt_name == 'lion':
        from lion_pytorch import Lion
        opt = Lion(model.parameters(), lr=1e-3, weight_decay=0.1)
    else:
        raise ValueError(f"Unknown optimizer: {opt_name}")

    warmup_steps = 200
    plateau_steps = 500
    start_time = time.time()
    final_loss = None
    for step, (x, y) in enumerate(train_loader):
        if step >= total_steps:
            break
        lr = get_lr(step, warmup_steps, plateau_steps, total_steps)
        for param_group in opt.param_groups:
            param_group['lr'] = lr

        x, y = x.to(device), y.to(device)
        _, loss = model(x, y)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()

        if step % 50 == 0:
            elapsed = time.time() - start_time
            print(f"[{opt_name}] Step {step:4d} | loss: {loss.item():.4f} | lr: {lr:.2e} | time: {elapsed:.1f}s")
            wandb.log({"train/loss": loss.item(), "train/lr": lr}, step=step)
        final_loss = loss.item()

    print(f"[{opt_name}] Final loss at {total_steps} steps: {final_loss:.4f}")
    torch.save(model.state_dict(), f"model_{opt_name}_{total_steps}steps.pt")
    wandb.finish()
    return final_loss

if __name__ == "__main__":
    import sys
    opt = sys.argv[1] if len(sys.argv) > 1 else 'adamw'
    steps = int(sys.argv[2]) if len(sys.argv) > 2 else 2000
    train(opt, total_steps=steps)