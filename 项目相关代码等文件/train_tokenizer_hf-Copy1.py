from tokenizers import Tokenizer, models, trainers, pre_tokenizers
from datasets import load_dataset

print("Loading TinyStories...")
dataset = load_dataset("roneneldan/TinyStories", split="train")

# 将所有故事写入一个文本文件（每行一个故事）
with open("tinystories_text.txt", "w", encoding="utf-8") as f:
    for story in dataset["text"]:
        f.write(story + "\n")

print("Training BPE tokenizer (vocab_size=8192)...")
tokenizer = Tokenizer(models.BPE())
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

trainer = trainers.BpeTrainer(
    vocab_size=8192,
    special_tokens=["<|endoftext|>"],   # 与 GPT-2 风格一致
    min_frequency=2
)

tokenizer.train(["tinystories_text.txt"], trainer)
tokenizer.save("tokenizer.json")
print("Saved tokenizer.json")