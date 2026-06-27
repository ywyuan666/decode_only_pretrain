"""BPE Tokenizer training and loading."""
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
from tokenizers.processors import TemplateProcessing
from typing import Iterator


def train_bpe_tokenizer(
    text_iterator: Iterator[str],
    vocab_size: int = 8192,
    save_path: str = "data/tinystories_tokenizer.json",
) -> Tokenizer:
    """Train a Byte-Level BPE tokenizer.

    Args:
        text_iterator: Iterator yielding raw text.
        vocab_size: Desired vocabulary size.
        save_path: Path to save the tokenizer.

    Returns:
        Trained tokenizer.
    """
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.post_processor = TemplateProcessing(
        single="<s> $A </s>",
        pair="<s> $A </s> $B:1 </s>:1",
        special_tokens=[
            ("<s>", 0),
            ("</s>", 1),
            ("<pad>", 2),
            ("<unk>", 3),
        ],
    )

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<s>", "</s>", "<pad>", "<unk>"],
        min_frequency=2,
    )

    tokenizer.train_from_iterator(text_iterator, trainer)
    tokenizer.save(save_path)
    return tokenizer


def load_tokenizer(path: str = "data/tinystories_tokenizer.json") -> Tokenizer:
    """Load a saved tokenizer."""
    return Tokenizer.from_file(path)