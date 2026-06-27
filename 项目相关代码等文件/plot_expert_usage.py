import torch
import matplotlib.pyplot as plt
from model import ModernTransformer   # 你的模型文件为 model.py

def main():
    # 配置必须与训练时完全一致，尤其是 n_heads, n_kv_heads, head_dim 等
    config = {
        "dim": 256,
        "n_heads": 8,
        "n_kv_heads": 2,
        "head_dim": 32,                # 256 / 8
        "vocab_size": 8192,
        "n_layers": 4,
        "max_seq_len": 512,
        "num_experts": 2,
        "top_k": 1,                    # 确保与你的 MoE 一致
        "hidden_dim": 1024,            # 确保与你的 MoE 一致
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ModernTransformer(config, use_moe=True).to(device)
    model.eval()

    # 如果你想加载训练好的权重，取消下面注释
    # model.load_state_dict(torch.load("model_adamw_final.pt", map_location=device), strict=False)

    # 测试随机 batch 统计 expert 分配
    total_usage = torch.zeros(config["num_experts"], device=device)
    num_batches = 20
    for _ in range(num_batches):
        # 随机 token 序列，模拟真实输入
        input_ids = torch.randint(0, config["vocab_size"], (4, 256), device=device)
        model.reset_all_expert_counts()
        with torch.no_grad():
            _ = model(input_ids)
        total_usage += model.get_expert_usage()  # 已经在同一设备

    # 转移到 CPU 并归一化
    total_usage = total_usage.cpu().numpy()
    total_usage = total_usage / total_usage.sum()

    # 绘图
    experts = [f"Expert {i}" for i in range(config["num_experts"])]
    plt.bar(experts, total_usage, color=['#2ca02c', '#ff7f0e'])
    plt.ylabel("Token Fraction")
    plt.title("MoE Expert Utilization Distribution")
    for i, v in enumerate(total_usage):
        plt.text(i, v + 0.01, f"{v*100:.1f}%", ha='center')
    plt.ylim(0, 1.0)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig("expert_usage.png", dpi=150)
    print("Expert 利用率图已保存为 expert_usage.png")

if __name__ == "__main__":
    main()