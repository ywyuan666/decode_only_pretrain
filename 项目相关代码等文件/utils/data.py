"""Data loading and preprocessing utilities."""
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from typing import Optional, Iterator

from .tokenizer import train_bpe_tokenizer, load_tokenizer


class TinyStoriesDataset(Dataset):
    """TinyStories dataset tokenized and ready for training."""

    def __init__(
        self,
        split: str = "train",
        tokenizer_path: str = "data/tinystories_tokenizer.json",
        max_seq_len: int = 512,
        bin_path: Optional[str] = None,
    ) -> None:
        self.tokenizer = load_tokenizer(tokenizer_path)
        self.max_seq_len = max_seq_len

        if bin_path and os.path.exists(bin_path):
            # Load preprocessed binary file
            self.data = np.memmap(bin_path, dtype=np.int32, mode='r')
        else:
            # Tokenize and save
            self.data = self._load_and_tokenize(split)
            if bin_path:
                self._save_bin(bin_path)

    def _load_and_tokenize(self, split: str) -> np.ndarray:
        """Load dataset and tokenize all stories."""
        ds = load_dataset("roneneldan/TinyStories", split=split, trust_remote_code=True)
        all_tokens = []
        eos_id = self.tokenizer.token_to_id("</s>")
        for story in ds["text"]:
            tokens = self.tokenizer.encode(story).ids
            all_tokens.extend(tokens + [eos_id])
        return np.array(all_tokens, dtype=np.int32)

    def _save_bin(self, path: str) -> None:
        """Save tokenized data as binary file."""
        self.data.tofile(path)

    def __len__(self) -> int:
        return max(0, len(self.data) - self.max_seq_len)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        chunk = self.data[idx:idx + self.max_seq_len + 1]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y


def get_dataloader(
    split: str = "train",
    batch_size: int = 4,
    max_seq_len: int = 512,
    num_workers: int = 4,
    tokenizer_path: str = "data/tinystories_tokenizer.json",
    bin_path: Optional[str] = None,
) -> DataLoader:
    """Create a DataLoader for TinyStories."""
    dataset = TinyStoriesDataset(
        split=split,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        bin_path=bin_path,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )


def prepare_data(config: dict) -> None:
    """One-time data preparation: train tokenizer and save binary data."""
    # Train tokenizer if not exists
    if not os.path.exists(config["tokenizer_path"]):
        print("Training BPE tokenizer...")
        ds = load_dataset("roneneldan/TinyStories", split="train", trust_remote_code=True)

        def text_iterator() -> Iterator[str]:
            for story in ds["text"]:
                yield story

        train_bpe_tokenizer(
            text_iterator(),
            vocab_size=config["vocab_size"],
            save_path=config["tokenizer_path"],
        )

    # Create binary files
    for split, bin_path in [("train", config["train_bin_path"]), ("validation", config["val_bin_path"])]:
        if not os.path.exists(bin_path):
            print(f"Creating {split} binary file...")
            _ = TinyStoriesDataset(
                split=split,
                tokenizer_path=config["tokenizer_path"],
                max_seq_len=config["max_seq_len"],
                bin_path=bin_path,
            )
    print("Data preparation complete.")