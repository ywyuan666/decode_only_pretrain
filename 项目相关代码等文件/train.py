#!/usr/bin/env python3
"""Main training script with optimizer comparison."""
import os
import yaml
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import wandb
from typing import Dict, Any

from model.transformer import ModernTransformer
from utils.data import prepare_data, get_dataloader

# Optional optimizer imports
try:
    from lion_pytorch import Lion
    LION_AVAILABLE = True
except ImportError:
    LION_AVAILABLE = False

try:
    from sophia import SophiaG
    SOPHIA_AVAILABLE = True
except ImportError:
    SOPHIA_AVAILABLE = False


def get_optimizer(model: torch.nn.Module, config: Dict[str, Any]) -> torch.optim.Optimizer:
    """Factory for optimizers."""
    name = config.get("optimizer", "adamw").lower()
    lr = float(config["learning_rate"])
    wd = float(config["weight_decay"])

    if name == "adamw":
        return AdamW(model.parameters(), lr=lr, weight_decay=wd)
    elif name == "lion":
        if not LION_AVAILABLE:
            raise ImportError("Lion optimizer not installed. Run: pip install lion-pytorch")
        return Lion(model.parameters(), lr=lr/3, weight_decay=wd)
    elif name == "sophia":
        if not SOPHIA_AVAILABLE:
            raise ImportError("Sophia optimizer not installed.")
        return SophiaG(model.parameters(), lr=lr, weight_decay=wd)
    else:
        raise ValueError(f"Unknown optimizer: {name}")


def train(config_path: str = "configs/config.yaml") -> None:
    """Main training function."""
    # Load config
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Initialize wandb
    wandb.init(project="modern-transformer", config=config, mode="offline")

    # Prepare data (tokenizer + binary files)
    prepare_data(config)

    # Create dataloaders
    train_loader = get_dataloader(
        split="train",
        batch_size=config["batch_size"],
        max_seq_len=config["max_seq_len"],
        tokenizer_path=config["tokenizer_path"],
        bin_path=config["train_bin_path"],
    )

    # Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ModernTransformer(config).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Optimizer and scheduler
    optimizer = get_optimizer(model, config)
    scheduler = CosineAnnealingLR(optimizer, T_max=config["max_steps"])

    # ====================== 优化3：保存最佳模型 ======================
    best_loss = float("inf")
    best_model_path = config["best_model_path"]

    # Training loop
    model.train()
    step = 0
    while step < config["max_steps"]:
        for batch in train_loader:
            if step >= config["max_steps"]:
                break

            x, y = batch
            x, y = x.to(device), y.to(device)

            logits, _ = model(x)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                y.reshape(-1)
            )

            # Optimize
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"])
            optimizer.step()
            scheduler.step()

            # ====================== 保存最佳模型 ======================
            if loss.item() < best_loss:
                best_loss = loss.item()
                torch.save({
                    "step": step,
                    "model_state_dict": model.state_dict(),
                    "loss": best_loss
                }, best_model_path)
                print(f"✅ 最佳模型已保存 | loss={best_loss:.4f}")

            # Log
            if step % config.get("log_interval", 10) == 0:
                lr = scheduler.get_last_lr()[0]
                wandb.log({"loss": loss.item(), "lr": lr}, step=step)
                print(f"Step {step:4d} | loss: {loss.item():.4f}")

            step += 1

    # Save final checkpoint
    os.makedirs(config["save_dir"], exist_ok=True)
    torch.save(model.state_dict(), os.path.join(config["save_dir"], "final.pt"))
    wandb.finish()
    print("Training completed!")


if __name__ == "__main__":
    train()