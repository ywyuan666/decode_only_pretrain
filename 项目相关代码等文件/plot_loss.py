import json
import os
import glob
import matplotlib.pyplot as plt

def load_wandb_history(run_dir):
    """从 wandb 离线 run 文件夹中读取 history（jsonl）"""
    history_file = os.path.join(run_dir, "files", "wandb-history.jsonl")
    if not os.path.exists(history_file):
        # 可能直接在 run 目录下
        for root, dirs, files in os.walk(run_dir):
            for f in files:
                if f == "wandb-history.jsonl":
                    history_file = os.path.join(root, f)
                    break
    steps, losses = [], []
    with open(history_file, 'r') as f:
        for line in f:
            data = json.loads(line)
            if 'train/loss' in data or 'loss' in data or 'train_loss' in data:
                # 你的日志中 key 为 'train/loss'
                loss_val = data.get('train/loss', data.get('loss', data.get('train_loss')))
                step_val = data.get('step', data.get('_step', len(steps)))
                if loss_val is not None:
                    losses.append(loss_val)
                    steps.append(step_val)
    return steps, losses

def main():
    # 找到最新的 wandb run（你可以手动指定具体目录）
    wandb_root = "wandb"
    runs = []
    for root, dirs, files in os.walk(wandb_root):
        if "wandb-history.jsonl" in files:
            runs.append(root)
    if not runs:
        print("未找到 wandb 日志，将使用预置数据绘图。")
        # 如果没有日志，你可以手动填入训练时打印的 loss 数据
        steps = [0, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500,
                 600, 700, 800, 900, 1000, 2000, 4000, 8000, 12000, 16000, 20000, 24000, 28000, 32000, 36000, 40000]
        losses = [9.06, 6.14, 5.90, 5.27, 5.31, 4.80, 4.77, 4.89, 4.67, 4.81, 4.41,
                  4.51, 4.31, 4.24, 4.17, 4.06, 3.94, 3.81, 3.30, 3.00, 2.85, 2.80, 2.76, 2.72, 2.71, 2.70, 2.71]
    else:
        # 取最新的一个 run
        latest_run = sorted(runs)[-1]
        steps, losses = load_wandb_history(latest_run)
        print(f"从 {latest_run} 加载了 {len(steps)} 个数据点")

    plt.figure(figsize=(10, 5))
    plt.plot(steps, losses, color='#1f77b4', linewidth=1.5, label='Training Loss')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve (AdamW, Modern Transformer)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('loss_curve.png', dpi=150)
    print("损失曲线已保存为 loss_curve.png")

if __name__ == "__main__":
    main()