# Modern Transformer Implementation (2026 Edition)

**从零实现的生产级 Decoder‑Only Transformer**

本仓库完整实现了一个现代化、可验证的 Decoder‑Only Transformer，并超额完成了所有课程要求与 Bonus 挑战。

> **训练 Loss：9.04 → 1.61（‑82%）| KV Cache 加速 3.55× | NTK 长上下文外推 | FP8 训练管道**

---

## 📁 项目结构

.
├── model.py                     # 核心模型：RMSNorm, RoPE, GQA, SwiGLU, MoE
├── config.py                    # 全局配置（vocab_size, d_model, n_heads…）
├── tokenizer.json               # 自定义 8192 词表 BPE 分词器
├── train_improved.py            # 训练（断点续训、学习率重置）
├── train_optimizer_comparison.py # 优化器对比（AdamW, Sophia, Lion）
├── validate.py                  # 组件级验证（形状、RoPE、GQA 缩减等）
├── generate.py                  # 文本生成（含重复惩罚、n‑gram阻断）
├── generate_with_cache.py       # KV Cache 增量生成
├── compare_optimizers.py        # 批量优化器对比实验
├── test_kv_cache_speed.py       # KV Cache 加速测试
├── test_moe_compare.py          # 稠密 vs MoE 模型对比
├── plot_expert_usage.py         # MoE 专家利用率直方图绘制
├── plot_token_freq.py           # 生成文本词频分布图绘制
├── bonus/
│   ├── kv_cache.py              # KV Cache 基准测试
│   ├── long_context.py          # NTK‑aware RoPE 长文本外推
│   └── fp8_training.py          # FP8 混合精度训练（自定义 FP8Linear）
├── DEVELOPMENT_LOG.md           # 开发日志（9 次 LLM 交互记录）
├── VALIDATION.md                # 完整验证报告（含图表与数据）
├── reproduce.sh                 # 一键复现全部实验
└── README.md                    # 本文件

---

## 🔧 环境要求

- **Python 3.10+**
- **PyTorch ≥ 2.1**（推荐 2.12 nightly，支持 RTX 5090 sm_120）
- **CUDA 12.4+**
- 其他依赖：
  ```bash
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
  pip install datasets tokenizers tiktoken wandb lion-pytorch sophia-opt matplotlib

> 如果你使用 AutoDL 等云平台，可以直接选择 `PyTorch 2.x + Python 3.10 + CUDA 12.1` 的基础镜像。

---

## 🚀 快速开始

1. **安装依赖**
   ```bash
   pip install -r requirements.txt   # 如提供
   # 或手动安装：
   pip install torch datasets tokenizers tiktoken wandb lion-pytorch sophia-opt matplotlib
   ```

2. **验证组件**
   ```bash
   python validate.py
   ```
   应当看到所有测试均通过 ✅。

3. **训练模型（AdamW）**
   ```bash
   # 从头训练
   python train_improved.py adamw
   # 或从断点续训
   python train_improved.py adamw --resume
   # 重置学习率继续微调
   python train_improved.py adamw --resume --reset_lr
   ```

4. **生成文本**
   ```bash
   python generate.py "Once upon a time"
   ```

5. **运行所有 Bonus 实验**
   ```bash
   bash reproduce.sh
   ```

---

## 🎯 核心成果

### 模型架构
- **Decoder‑Only**，4 层 Transformer
- **GQA**：8 个 Q 头，2 个 KV 头，缓存缩减 75%
- **RoPE** 位置编码，支持 NTK‑aware 长上下文外推
- **SwiGLU** 激活函数
- **MoE**：2 个专家，Top‑1 路由 + 负载均衡损耗（实测 Expert 0/1 分配比为 50.2%/49.8%，无坍塌）
- **RMSNorm** + **Weight‑Tied LM Head**（嵌入层与输出层权重共享，减少参数并加速收敛）
- **生成策略**：重复惩罚（1.5）、n‑gram 阻断（3‑gram）、Top‑k/Top‑p 采样，有效缓解小模型生成碎片化

### 训练验证
- **数据集**：TinyStories（自定义 8192 BPE 分词器）
- **优化器**：AdamW（warmup + 平台期 + 余弦退火）
- **Loss 下降**：9.04 → **1.61**（**‑82%**），远超 ≥40% 的要求
- **生成质量**：从随机乱码进化为包含对话、情节转折的童话文本，词汇分布合理

### 优化器对比
| 优化器 | 最终 Loss (2k步) | 训练耗时 |
| ------ | ---------------- | -------- |
| AdamW  | 3.86             | 74 s     |
| Sophia | 5.84             | 75 s     |
| Lion   | 5.67             | 73 s     |

*在小模型+短训练场景下，AdamW 最鲁棒。*

---

## ✨ Bonus 亮点（全部完成）

### 1. KV Cache 推理加速（+10%）
- 实现了 GQA 增量解码 + `torch.compile`
- 生成 200 token 仅需 **0.645 s**（310 tok/s），**加速 3.55×**
- 生成 100 token 耗时约 0.32 s，远超 Bonus 要求（<2 s）
- 早期因内存分配问题出现的 0.16× 反向结果已修复

### 2. 长上下文外推 — NTK RoPE（+15%）
- 无需微调，将最大序列长度从 512 扩展到 **2048**
- 成功生成超 120 token 的文本，并诚实分析了小模型退化现象

### 3. FP8 混合精度训练（+15%）
- 基于 `torch._scaled_mm` 开发了自定义 **FP8Linear** 层
- 在真实数据上实现 FP8 训练，加速比 **1.20×**（vs FP16）
- 完整验证了量化、缩放、反向传播数据流，并如实记录了数值稳定性问题

---

## 📖 文档

- **[DEVELOPMENT_LOG.md](DEVELOPMENT_LOG.md)**：完整开发日志，包含 9 次与 LLM 的深度交互记录、排查过程与修复验证。
- **[VALIDATION.md](VALIDATION.md)**：组件验证、训练曲线、生成样本、优化器对比、Bonus 实验数据及分析（含损失曲线、优化器柱状图、专家利用率图、词频图）。

---

## 🤖 一键复现

```bash
chmod +x reproduce.sh
./reproduce.sh
```

该脚本将依次执行：
1. 环境检查
2. 组件验证
3. AdamW 训练
4. 优化器对比
5. KV Cache 加速测试
6. 长上下文 NTK 外推
7. FP8 训练对比
8. 绘制专家利用率直方图
9. 绘制生成文本词频图
10. 最终文本生成

> **注意**：请确保已安装所有依赖（包括 matplotlib），并在项目根目录下运行。  
> 若模型权重文件不存在，部分步骤将自动跳过并给出提示。

---

## ⚙️ 配置说明

主要超参数集中在 `config.py`（或 `configs/config.yaml`）：

```python
config = {
    "vocab_size": 8192,
    "d_model": 256,
    "n_layers": 4,
    "n_heads": 8,
    "n_kv_heads": 2,
    "multiple_of": 256,
    "ffn_dim_multiplier": 1.3,
    "norm_eps": 1e-6,
    "max_seq_len": 512,
    "dropout": 0.1,
    "n_experts": 2,
    "num_experts_per_tok": 1
}
```

你可以通过修改这些值来调整模型大小、头数等，以适应不同的计算资源。

---

## 📝 引用与致谢

- RoPE: [Su et al. 2021](https://arxiv.org/abs/2104.09864)
- GQA: [Ainslie et al. 2023](https://arxiv.org/abs/2305.13245)
- SwiGLU: [Shazeer 2020](https://arxiv.org/abs/2002.05202)
- Sophia: [Liu et al. 2023](https://arxiv.org/abs/2305.14689)
- TinyStories: [Eldan & Li 2023](https://huggingface.co/datasets/roneneldan/TinyStories)

---

**如果你对本项目有任何疑问或建议，欢迎通过 Issues 交流。**