#!/usr/bin/env python3
"""Training script with optional FP8 mixed precision."""
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

# FP8 training import
from bonus.fp8_training import FP8Trainer, FP8_AVAILABLE


def get_optimizer(model: torch.nn.Module, config: Dict[str, Any]) -> torch.optim.Optimizer:
    """Factory for optimizers."""
    name = config.get("optimizer", "adamw").lower()
    lr = config["learning_rate"]
    wd = config["weight_decay"]

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
    """Main training function with optional FP8 support."""
    # Load config
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Initialize wandb
    wandb.init(project="modern-transformer", config=config)

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

    # Optimizer and scheduler (will be used only if not using FP8Trainer)
    optimizer = get_optimizer(model, config)
    scheduler = CosineAnnealingLR(optimizer, T_max=config["total_steps"])

    # Check FP8 availability and configuration
    use_fp8 = config.get("use_fp8", False)
    if use_fp8 and not FP8_AVAILABLE:
        print("Warning: FP8 requested but not available. Falling back to standard training.")
        use_fp8 = False

    if use_fp8:
        print("Using FP8 mixed precision training.")
        fp8_trainer = FP8Trainer(model, config, use_fp8=True)
        # When using FP8Trainer, we need to handle scheduler and logging manually.
        # FP8Trainer.train_step() internally performs forward/backward/optimizer step.
    else:
        print("Using standard full-precision training.")

    # Training loop
    model.train()
    step = 0
    while step < config["total_steps"]:
        for x, y in train_loader:
            if step >= config["total_steps"]:
                break

            x, y = x.to(device), y.to(device)

            if use_fp8:
                # FP8 training step
                loss = fp8_trainer.train_step(x, y)
                # Note: FP8Trainer.train_step does not return auxiliary loss.
                # If auxiliary loss is needed, you can extend FP8Trainer.
                aux_loss_val = 0.0
                total_loss = loss  # already includes any extra losses inside trainer
                # Scheduler step (still needed)
                scheduler.step()
            else:
                # Standard training step
                logits = model(x)
                loss = F.cross_entropy(logits.view(-1, config["vocab_size"]), y.view(-1))
                aux_loss = model.get_aux_loss()
                total_loss = loss + 0.01 * aux_loss
                aux_loss_val = aux_loss.item()

                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"])
                optimizer.step()
                scheduler.step()

            # Logging
            if step % config["log_interval"] == 0:
                log_dict = {
                    "train/loss": loss.item() if use_fp8 else loss.item(),
                    "train/aux_loss": aux_loss_val if not use_fp8 else 0.0,
                    "train/lr": scheduler.get_last_lr()[0],
                    "step": step,
                }
                wandb.log(log_dict)
                print(f"Step {step:4d} | loss: {loss.item():.4f} | aux: {aux_loss_val:.4f}")

            step += 1

    # Save checkpoint
    os.makedirs("checkpoints", exist_ok=True)
    checkpoint_name = "fp8_final.pt" if use_fp8 else "final.pt"
    torch.save(model.state_dict(), f"checkpoints/{checkpoint_name}")
    wandb.finish()
    print(f"Training completed! Checkpoint saved to checkpoints/{checkpoint_name}")


if __name__ == "__main__":
    train()