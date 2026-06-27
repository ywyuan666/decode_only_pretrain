# Project 1: Modern Transformer Implementation from Scratch  
## *Decoder-Only Architecture with Cutting-Edge Components (2026 Edition)*

---



## Project Overview

**Objective**: 

- Implement a **production-grade decoder-only transformer** with modern architectural innovations. 
- Validate correctness through testing, training on real data, and comparative analysis. 
- Document the *entire development journey* of your "vibe coding" engineering.

**Core Philosophy**:  

> *"Don't just code—prove it works. Every line must be verified, every component validated, every decision justified."*

---



## 🔑 Required Components

| Component                   | Specification                                                | Why It Matters                                               |
| --------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| **Architecture**            | Decoder-only stack (min. 4 layers)                           | Foundation of all SOTA LLMs                                  |
| **Attention**               | **MHA at least /MLA or GQA prefered**<br>- Causal masking with RoPE | Reduces KV cache by 75% vs MHA<br>Industry standard since LLaMA 2 (2023) |
| **Position Encoding**( opt) | **RoPE (Rotary PE)**<br>- θ = 10000^(-2i/d)<br>GQA- Applied *inside* attention (not input) | Enables length extrapolation<br>Zero parameters, superior to sinusoidal |
| **FFN**                     | **GeLU/SwiGLU**<br>- hidden_dim = 4×d_model<br>- `SwiGLU(x) = (xW₁ ⊗ σ(xW₃))W₂` | 5-10% better than ReLU/GELU<br>Used in PaLM, LLaMA, GPT-4    |
| **Experts** (opt)           | **Simplified MoE**<br>- 2 experts (FFN variants)<br>- Top-1 gating with noise<br>- Load balancing loss | Introduces sparsity<br>Foundation for Mixtral, DeepSeek-MoE  |
| **Optimizer**(opt)          | **Implement & Compare**:<br>- AdamW (baseline)<br>- **Sophia** (curvature-aware)<br>- **Lion** (sign-based) | Sophia: 2× faster convergence on LLMs<br>Lion: 50% less memory, better generalization |
| **Norm**(opt)               | RMSNorm (not LayerNorm)<br>- No bias parameters<br>- ε = 1e-6 | Faster, matches LLaMA/Qwen implementations                   |
| **Output** (opt)            | Weight-tied LM Head<br>- Final RMSNorm → Linear → Logits     | Parameter efficiency + better convergence                    |

---

## 📐 Reference Technical Specifications 

```python
# REQUIRED CONFIG (non-negotiable)
config = {
    "vocab_size": 8192,      # Trained on TinyStories tokenizer
    "d_model": 256,          # Hidden dimension
    "n_layers": 4,           # Minimum stack depth
    "n_heads": 8,            # Total attention heads
    "n_kv_heads": 2,         # GQA: shared KV heads
    "multiple_of": 256,      # SwiGLU hidden dim alignment
    "ffn_dim_multiplier": 1.3,
    "norm_eps": 1e-6,
    "max_seq_len": 512,
    "dropout": 0.1,
    "n_experts": 2,          # MoE configuration
    "num_experts_per_tok": 1
}
```

---



## ✅ Reference Validation & Verification Protocol 

### Phase 1: Component-Level Verification
| Component      | Test                          | Pass Criteria                                            |
| -------------- | ----------------------------- | -------------------------------------------------------- |
| **RoPE**       | Rotate vectors at pos 0 vs 10 | `cos(θ₁₀)·x - sin(θ₁₀)·y` matches manual calc            |
| **MLA**        | KV cache size check           | `(seq_len × n_kv_heads × d_head)` = 75% smaller than MHA |
| **SwiGLU**     | Output vs PyTorch F.silu      | Max diff < 1e-5                                          |
| **MoE Gating** | Expert assignment histogram   | Load imbalance < 15% (with noise)                        |
| **RMSNorm**    | Output std ≈ 1.0              | `output.std(dim=-1).mean() ∈ [0.95, 1.05]`               |

### Phase 2: Integration Validation
```python
# REQUIRED TEST: Shape propagation
x = torch.randint(0, 8192, (2, 128))  # (B=2, L=128)
model = ModernTransformer(config)
logits = model(x)  # MUST output (2, 128, 8192)

assert logits.shape == (2, 128, 8192), "Shape mismatch!"
assert torch.all(torch.isfinite(logits)), "NaN/Inf detected!"
```

### Phase 3: Training Validation
1. **Dataset**: TinyStories (15MB, 20k stories)
2. **Training**:
   - 500 steps, batch_size=4, lr=3e-4
   - Log loss every 50 steps
3. **Pass Criteria**:
   - Loss decreases by ≥40% in 500 steps
   - Generate coherent text at step 500:
     ```
     Input: "Once upon a time"
     Output: "there was a little cat who loved to play..."
     ```

### Phase 4: Optimizer Comparison
| Optimizer  | Steps to Loss<2.0 | Final Loss | Memory (GB) |
| ---------- | ----------------- | ---------- | ----------- |
| AdamW      | 320               | 1.85       | 1.8         |
| **Sophia** | **180**           | **1.72**   | 2.1         |
| **Lion**   | **210**           | 1.78       | **1.2**     |

---



## 🌈 "Vibe Coding" Documentation Requirement (MANDATORY)

Students **must submit a development log** showing:

### 📝 Required Log Sections
1. **Prompt Engineering Log**
   ```markdown
   ## Iteration 3: Fixing RoPE Implementation
   **Prompt to LLM**:
   "My RoPE rotation isn't preserving relative position. 
   Here's my code: [code snippet]. 
   Why does Q₀·K₁ ≠ Q₁₀₀·K₁₀₁? Show corrected rotation math."
   
   **LLM Response**:
   "You're rotating after reshaping heads! Rotate BEFORE head splitting. 
   Also missing: negative frequencies for odd dimensions."
   
   **Fix Applied**:
   - Moved RoPE before head reshaping
   - Added conjugate pairs for complex rotation
   - Verified: Q₀·K₁ == Q₁₀₀·K₁₀₁ (within 1e-6)
   ```
   
2. **Debugging Journey**
   ```markdown
   ## Bug: MoE Expert Collapse
   **Symptom**: Expert 0 gets 95% tokens, Expert 1 gets 5%
   **Diagnosis**:
   - Added gating histogram logging
   - Discovered: No noise in gating → deterministic routing
   **Fix**:
   - Added Gaussian noise (std=1e-2) to logits
   - Implemented auxiliary load loss (α=1e-2)
   **Validation**:
   - After fix: Expert 0=52%, Expert 1=48%
   - Loss stabilized (no spikes)
   ```

3. **Validation Evidence**
   - Screenshots of tensorboard loss curves
   - Terminal output of shape checks
   - Side-by-side text generation samples (step 0 vs step 500)
   - Optimizer comparison table with metrics

4. **Architecture Decisions**
   ```markdown
   ## Why GQA over MQA?
   **Tested**: 
   - MQA (1 KV head): 35% faster but 8% worse perplexity
   - GQA (2 KV heads): 25% faster, 2% worse perplexity
   **Decision**: Chose GQA - best speed/quality tradeoff
   **Evidence**: [Link to ablation study notebook]
   ```

---

## 📦 Deliverables Checklist

- [ ] **Source Code** (`model.py`, `train.py`, `utils/`)
  - Modular components (RoPE, GQA, SwiGLU, MoE as separate classes)
  - Type hints + docstrings
- [ ] **Development Log** (`DEVELOPMENT_LOG.md`)
  - Minimum 5 iterations with prompts/responses/fixes
  - Validation evidence for each component
  - Optimizer comparison results
- [ ] **Validation Report** (`VALIDATION.md`)
  - Component test results (tables + screenshots)
  - Training metrics (loss curve, sample generations)
  - Shape propagation proof
- [ ] **Reproduction Script** (`reproduce.sh`)
  ```bash
  python train.py --config tiny_config.yaml --steps 500
  python validate.py --checkpoint step_500.pt
  ```

---



## 🌟 Bonus Challenges (For Extra Credit)

| Challenge                 | Points | Validation                                 |
| ------------------------- | ------ | ------------------------------------------ |
| **FP8 Mixed Precision**   | +15%   | 40% faster training, <0.5% accuracy drop   |
| **KV Cache Optimization** | +10%   | Generate 100 tokens in <2s (CPU)           |
| **Speculative Decoding**  | +20%   | 2.5× tokens/sec vs baseline                |
| **Long Context (2K)**     | +15%   | Coherent generation beyond training length |

---

## 💡 Critical Success Factors

1. **Verification > Implementation**  
   *"A component that isn't tested doesn't exist."*
   
2. **Document the struggle**  
   Show failed attempts, debugging logs, LLM prompts/responses
   
3. **Quantify everything**  
   "Loss decreased" ❌ → "Loss: 3.21 → 1.87 (-42%)" ✅
   
4. **Prove correctness**  
   Shape checks → Numerical checks → Functional checks → Training validation

---

## 📚 Starter Resources
- RoPE Paper: [Su et al. 2021](https://arxiv.org/abs/2104.09864)
- GQA Paper: [Ainslie et al. 2023](https://arxiv.org/abs/2305.13245)
- SwiGLU: [Shazeer 2020](https://arxiv.org/abs/2002.05202)
- Sophia Optimizer: [Liu et al. 2023](https://arxiv.org/abs/2305.14689)
- TinyStories Dataset: [HuggingFace](https://huggingface.co/datasets/roneneldan/TinyStories)

---

> **Final Note**:  
> *"This isn't about writing perfect code on the first try.  
> It's about building a **verifiable, trustworthy system** through relentless validation.  
> Show us your thinking—not just your code."*  
>
> **Submission Deadline**: 4 weeks  
> **Grading Weight**: 40% of final grade  
