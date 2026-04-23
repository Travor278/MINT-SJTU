# Reading Note: Evo-1
**Lightweight Vision-Language-Action Model with Preserved Semantic Alignment**
arXiv 2511.04555 · CVPR 2026 · MINT-SJTU (赵波组)

---

## 核心问题

现有轻量 VLA（如 SmolVLA、OpenVLA）端到端训练时，机器人数据的梯度会反传进预训练 VLM backbone，导致 **semantic drift**——VLM 原本学到的语义表示被破坏，注意力图从清晰聚焦退化为发散混乱。

这个问题在论文 Figure 2 里有直观展示：OpenVLA 的 Prismatic-7B backbone 在训练后注意力已经乱掉，而 Evo-1 的 InternVL3-1B 保持了清晰的语义聚焦。

---

## 方法

### 架构：Cross-Modulated Diffusion Transformer

Action expert 用 **纯 cross-attention DiT**（不是 self+cross 交替），noisy action sequence 作为 query，VLM 表示 + 机器人状态作为 KV，逐步去噪。

用 **flow matching** 而不是 DDPM：
- 插值：`A_τ = τ·A_t + (1-τ)·ε`，τ ~ Beta(0.02, 0.98)
- 学 velocity field，loss 比较预测 flow 和目标 flow

### Integration Module

取 VLM **第 14 层**的中间表示（而非最后一层）+ 机器人状态，拼接后送入 action expert 的 KV。

消融里试了几个变体（加 self-attention、不同层特征轮换等），都比这个基础设计差——说明保持信息传播的一致性比加复杂结构更重要。

### 两阶段训练（核心贡献）

**Stage 1**：冻结 VLM backbone，只训 integration module + action expert
→ 让 action expert 在不污染 VLM 的前提下对齐到多模态表示空间

**Stage 2**：解冻 backbone，全量联合微调
→ 此时 action expert 已经稳定，反传梯度不会造成剧烈语义漂移

直觉：先让学生（action expert）适应老师（VLM）的语言，再一起深化，而不是一开始就强行互相改造。

---

## 关键结果

| 基准 | Evo-1 (0.77B) | π0 (3.5B) | SmolVLA (2.25B) | OpenVLA (7B) |
|---|---|---|---|---|
| LIBERO avg | **94.8%** | 94.2% | — | 79.2% |
| MetaWorld avg | **80.6%** | 47.9% | 68.2% | — |
| RoboTwin avg | **37.8%** | 30.9% | — | — |
| 显存占用 | **2.3 GB** | — | — | ~16 GB |
| 推理频率 | **16.4 Hz** | — | — | — |

0.77B 参数，在三个基准上全面超过 π0（3.5B）和 SmolVLA（2.25B）。

---

## 与我工作的关联

**VLC_m**：我研究了 Qwen2-VL-7B 的注意力漂移（Wasserstein-1 drift signal），Evo-1 也在研究 VLA 训练中的注意力语义崩溃——问题的底层相同，只是一个是推理期干预，一个是训练期保护。

**Pick-and-Place**：我用 OpenVLA-OFT，Evo-1 的 Figure 2 直接对比了 Evo-1 vs OpenVLA 的注意力质量，可以用我自己的任务可视化验证这个差距。

---

## 开放问题

1. **Stage 2 冻结多少层合适？** 论文没有分析部分解冻 backbone 的效果——只训最后几层而不是全量 fine-tune，是否能在保留语义和适应任务之间更好地权衡？

2. **第 14 层的选择依据？** 用 VLM 的中间层而非最后层，论文没有给出选层的消融。InternVL3 共 24 层，第 14 层约在 60% 深度——是否有理论依据还是经验选定？

3. **flow matching vs DDPM？** 论文用 flow matching 但没有专门消融这个设计选择，换成 DDPM 是否有明显差距？
