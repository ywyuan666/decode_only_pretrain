from datasets import load_dataset
dataset = load_dataset("Anthropic/hh-rlhf", split="train[:100]")
for sample in dataset:
    chosen = sample["chosen"]
    rejected = sample["rejected"]