# 一键复现脚本 `reproduce.sh`

将以下内容保存为 `reproduce.sh`，并在项目根目录下执行即可自动完成全部实验与图表生成。

```bash
#!/bin/bash
echo "======================================"
echo " Modern Transformer – 完整复现脚本"
echo "======================================"

# 0. 环境检查
echo "[0/9] 检查依赖..."
python -c "import torch; print('PyTorch', torch.__version__)"
python -c "import tokenizers; print('tokenizers OK')"
python -c "import wandb; print('wandb OK')"
python -c "import matplotlib; print('matplotlib OK')"

# 1. 组件验证
echo "[1/9] 运行组件级验证..."
python validate.py

# 2. 训练（AdamW，自动从断点续训）
echo "[2/9] 开始训练 AdamW（支持断点续训）..."
python train_improved.py adamw --resume --reset_lr

# 3. 优化器对比（短训练）
echo "[3/9] 运行优化器对比实验（各2000步）..."
python train_optimizer_comparison.py adamw 2000
python train_optimizer_comparison.py sophia 2000
python train_optimizer_comparison.py lion 2000

# 4. KV Cache 加速测试
echo "[4/9] 测试 KV Cache 加速..."
if [ -f model_adamw_final.pt ]; then
    python bonus/kv_cache.py
else
    echo "[!] model_adamw_final.pt 未找到，跳过 KV Cache 测试（请先完成训练）"
fi

# 5. 长上下文外推（NTK）
echo "[5/9] 测试长上下文外推..."
if [ -f model_adamw_final.pt ]; then
    python bonus/long_context.py
else
    echo "[!] model_adamw_final.pt 未找到，跳过长上下文测试"
fi

# 6. FP8 混合精度对比
echo "[6/9] 运行 FP8 训练对比..."
python bonus/fp8_training.py

# 7. 绘制 Expert 利用率直方图
echo "[7/9] 绘制 MoE 专家利用率直方图..."
if [ -f model_adamw_final.pt ]; then
    python plot_expert_usage.py
else
    echo "[!] 未找到训练好的模型，使用随机模型绘制 expert 利用率图（仅供参考）"
    python plot_expert_usage.py
fi

# 8. 绘制生成文本词频图
echo "[8/9] 绘制生成文本词频分布..."
if [ -f model_adamw_final.pt ]; then
    python plot_token_freq.py
else
    echo "[!] 未找到训练好的模型，跳过词频图绘制"
fi

# 9. 生成最终文本样例
echo "[9/9] 生成最终文本样例..."
if [ -f model_adamw_final.pt ]; then
    python generate.py "Once upon a time"
    python generate_with_cache.py "The little cat"
else
    echo "[!] model_adamw_final.pt 未找到，跳过生成演示"
fi

echo ""
echo "======================================"
echo " 全部实验完成！请查看 VALIDATION.md"
echo "======================================"
```

### 使用方法

1. 赋予执行权限：
   ```bash
   chmod +x reproduce.sh
   ```
2. 运行：
   ```bash
   ./reproduce.sh
   ```

### 依赖文件清单

请确保以下文件与 `reproduce.sh` 位于同一项目目录：

- `validate.py`
- `train_improved.py`
- `train_optimizer_comparison.py`
- `generate.py`
- `generate_with_cache.py`
- `plot_expert_usage.py`
- `plot_token_freq.py`
- `bonus/kv_cache.py`
- `bonus/long_context.py`
- `bonus/fp8_training.py`
- `sophia_opt.py`（`train_optimizer_comparison.py` 需要）
- `config.py`、`model/` 文件夹、`tokenizer.json`

> **注意**：部分脚本（如 `plot_expert_usage.py`、`plot_token_freq.py`）依赖 `matplotlib`，请提前通过 `pip install matplotlib` 安装。

