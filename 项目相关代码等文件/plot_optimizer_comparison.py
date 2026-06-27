import matplotlib.pyplot as plt
import numpy as np

# 数据（请替换为你实际记录的最终值）
optimizers = ['AdamW', 'Sophia', 'Lion']
final_loss = [3.86, 5.84, 5.67]          # 2k steps 的最终 loss
time_sec   = [74.22, 75.02, 73.46]       # 训练耗时（秒）

x = np.arange(len(optimizers))
width = 0.35

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# 左图：Final Loss
bars1 = ax1.bar(x, final_loss, width, color=['#2ca02c', '#d62728', '#ff7f0e'])
ax1.set_ylabel('Final Loss')
ax1.set_title('Final Loss after 2000 Steps')
ax1.set_xticks(x)
ax1.set_xticklabels(optimizers)
ax1.grid(axis='y', alpha=0.3)
# 给柱子添加数值标签
for bar, val in zip(bars1, final_loss):
    ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.1,
             f'{val:.2f}', ha='center', va='bottom')

# 右图：Training Time
bars2 = ax2.bar(x, time_sec, width, color=['#2ca02c', '#d62728', '#ff7f0e'])
ax2.set_ylabel('Training Time (sec)')
ax2.set_title('Training Time (2000 Steps)')
ax2.set_xticks(x)
ax2.set_xticklabels(optimizers)
ax2.grid(axis='y', alpha=0.3)
for bar, val in zip(bars2, time_sec):
    ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
             f'{val:.1f}s', ha='center', va='bottom')

plt.tight_layout()
plt.savefig('optimizer_comparison.png', dpi=150)
print("优化器对比图已保存为 optimizer_comparison.png")