#!/usr/bin/env python3
"""Optimizer comparison script."""
import os
import yaml
import torch
import time
import subprocess

optimizers = ["adamw", "sophia", "lion"]
results = {}

for opt in optimizers:
    print(f"\n{'='*50}")
    print(f"Testing optimizer: {opt}")
    print('='*50)

    # 修改config.yaml中的optimizer字段
    with open("configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    config["optimizer"] = opt
    config["total_steps"] = 500  # 确保每次训练步数一致

    # 保存临时配置
    temp_config_path = f"configs/temp_config_{opt}.yaml"
    with open(temp_config_path, "w") as f:
        yaml.dump(config, f)

    # 记录开始时间
    start_time = time.time()
    # 调用训练脚本
    subprocess.run(["python", "train.py", "--config", temp_config_path], check=True)
    elapsed_time = time.time() - start_time

    # 读取wandb离线日志中最后的loss (也可以直接从train.py输出中抓取，这里简化)
    # 我们将在训练日志中手动记录
    print(f"Optimizer {opt} completed in {elapsed_time:.2f} seconds.")
    results[opt] = {
        "time": elapsed_time,
        "config": temp_config_path,
    }

print("\nSummary:")
for opt, data in results.items():
    print(f"{opt}: {data['time']:.2f}s")