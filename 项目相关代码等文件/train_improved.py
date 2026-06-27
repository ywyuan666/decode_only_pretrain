# train_improved.py
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


def get_lr(step, warmup_steps, plateau_steps, total_steps, max_lr=1e-3):
    """根据步数手动计算当前学习率（warmup → plateau → cosine decay）"""
    if step < warmup_steps:
        return max_lr * float(step) / max(1, warmup_steps)
    elif step < plateau_steps:
        return max_lr
    else:
        progress = float(step - plateau_steps) / max(1, total_steps - plateau_steps)
        return max(0.0, 0.5 * (1 + math.cos(math.pi * progress))) * max_lr


def train(optimizer_name='adamw', total_steps=10000, batch_size=4, resume=False, reset_lr=False):
    """
    total_steps: 本次训练要跑的步数（如果 reset_lr=True，表示额外训练步数）
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_name = f"{optimizer_name}_final"
    if resume and reset_lr:
        run_name += "_resetlr_longplateau"
    elif resume:
        run_name += "_resumed"
    wandb.init(project="modern_transformer", name=run_name, mode="offline")

    train_dataset = TinyStoriesDataset(block_size=config['max_seq_len'])
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=0, pin_memory=True
    )

    model = ModernTransformer(config).to(device)

    # 决定学习率峰值
    if reset_lr:
        max_lr = 1e-3          # 重置学习率时更温和，防止破坏已学知识
    else:
        max_lr = 3e-3

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=max_lr,
        weight_decay=0.01,
        betas=(0.9, 0.95)
    )

    start_step = 0
    ckpt_path = f"model_{optimizer_name}_final.pt"

    if resume and os.path.exists(ckpt_path):
        print(f"Loading checkpoint from {ckpt_path} ...")
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt)

        if reset_lr:
            print("Resetting optimizer & learning rate schedule. Starting from step 0 with fresh optimizer.")
            start_step = 0
        else:
            start_step = 20000
            print(f"Resumed from step {start_step} (no optimizer state loaded, lr schedule continues)")
    else:
        print("Starting from scratch.")

    print(f"Model params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    # 学习率调度参数
    warmup_steps = 200
    plateau_steps = int(total_steps * 0.5)  # 峰值学习率持续到一半步数，延长探索时间

    start_time = time.time()
    for step, (x, y) in enumerate(train_loader, start=start_step):
        if step >= total_steps:
            break

        lr = get_lr(step, warmup_steps, plateau_steps, total_steps, max_lr=max_lr)
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
            print(f"Step {step:4d} | loss: {loss.item():.4f} | lr: {lr:.2e} | time: {elapsed:.1f}s")
            wandb.log({"train/loss": loss.item(), "train/lr": lr}, step=step)

    # 保存最终模型
    torch.save(model.state_dict(), f"model_{optimizer_name}_final.pt")
    wandb.finish()


if __name__ == "__main__":
    import sys
    opt = sys.argv[1] if len(sys.argv) > 1 else 'adamw'
    resume = '--resume' in sys.argv
    reset_lr = '--reset_lr' in sys.argv

    extra_steps = 10000
    train(opt, total_steps=extra_steps, resume=resume, reset_lr=reset_lr)