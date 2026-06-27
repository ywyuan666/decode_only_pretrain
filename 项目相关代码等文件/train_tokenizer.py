import tiktoken
from datasets import load_dataset

print("加载 TinyStories 训练集...")
dataset = load_dataset("roneneldan/TinyStories", split="train")

# 将所有故事写入一个文本文件（每行一个故事）
text_file = "tinystories_text.txt"
with open(text_file, "w", encoding="utf-8") as f:
    for story in dataset["text"]:
        f.write(story + "\n")

print(f"已保存 {len(dataset)} 个故事到 {text_file}")

# 训练 BPE tokenizer
print("开始训练 tokenizer（vocab_size=8192，可能需要几分钟）...")
enc = tiktoken.Encoding.train(
    text_file,
    model_name="tinystories_8192",
    vocab_size=8192,
    pat_str=None,              # 使用 GPT-2 的默认正则分割
    num_threads=8,             # 根据 CPU 核心数调整
)

# 保存为 tokenizer.json
enc.save("tokenizer.json")
print("训练完成！已保存为 tokenizer.json")

# 验证
enc2 = tiktoken.Encoding.from_file("tokenizer.json")
test_text = "Once upon a time"
tokens = enc2.encode_ordinary(test_text)
print(f"测试编码 '{test_text}' -> {tokens}")
print(f"词表大小: {enc2.n_vocab}")