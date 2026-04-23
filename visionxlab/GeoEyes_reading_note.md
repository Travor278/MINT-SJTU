# Reading Note: GeoEyes
**On-Demand Visual Focusing for Evidence-Grounded Understanding of Ultra-High-Resolution Remote Sensing Imagery**
arXiv 2602.14201 · VisionXLab, SJTU · Feb 2026

---

## 问题与动机

超高分辨率（UHR）遥感图像（典型尺寸 8500×8500）包含海量 token，直接送入 MLLM 计算量不可接受，主流做法是给模型配一个 zoom_in 工具。但现有方案（如 DeepEyes）存在 **Tool Usage Homogenization**——几乎对所有问题都调用 zoom，不管是否需要，导致计算浪费和错误累积。

根本矛盾：不同问题对 zoom 的需求异构（全局计数 vs 细粒度颜色识别），而 outcome-only 的奖励无法区分"该不该 zoom"。

---

## 方法：AdaZoom-GRPO（两阶段）

### Stage 1 — Cold-Start SFT on UHR-CoZ

自建数据集 UHR-CoZ（25,467 样本），用 GLM-4.5V 自动生成 zoom 轨迹：
- 模型迭代判断当前证据是否足够 → 若不够，输出 `zoom_in(bbox, reason)`
- 覆盖三种 regime：无需 zoom / 单次 zoom / 多步渐进 zoom

目的：让模型具备基本的工具调用能力，避免 RL 阶段的探索崩溃。

### Stage 2 — AdaZoom-GRPO（RL 精调）

用 GRPO 优化，奖励函数由四项组成：

**R_tool（自适应效率奖励）**
- 每个任务类别有基准步数 N_base(C)
- 实例难度用 P_α 缩放（基础模型表现越好，允许步数越少）
- 超出配额后指数衰减：`R_tool = P_α · exp(−γ · ΔN)`
- 解决"该不该 zoom"的异构问题

**R_cof（Chain-of-Focus 奖励）**
- 要求连续 zoom 窗口满足几何包含关系：`b_{t+1} ⊂ b_t`
- 正确 zoom-in → +β；回退（coarse 恢复）→ 0；漂移 → −β
- 关键：用方向性包含而非 IoU——IoU 在大尺度变化时失效（消融：41.92% vs 46.08%）

**R_proc（必要性验证奖励）**
- 对细粒度问题，若没有 zoom 动作就给出确定答案则惩罚
- 防止模型"凭空幻觉"

**R_acc + R_fmt**：标准正确性和格式奖励

---

## 关键结果

| 模型 | XLRS-Bench 均值 |
|---|---|
| GeoEyes | **54.23%** |
| DeepEyes（基线） | 50.0% |
| Qwen3-VL-235B（大27倍） | 51.1% |

zoom 调用率：GeoEyes 68.4%（选择性）vs DeepEyes 100%（饱和）

---

## 我的理解与问题

**理解到位的地方**

R_cof 的设计很精妙——用几何包含而非 IoU 评估 zoom 序列的质量，是因为标准 IoU 在比较大尺寸框和小尺寸框时会低估有效缩放（两个面积差10倍的框 IoU 可以很低但 zoom 完全合理）。这和 LRS-VQA 的 coarse-to-fine 思路一脉相承，但 GeoEyes 把"coarse-to-fine 是否正确执行"本身编入奖励信号，而不只是作为架构设计。

**与 LRS-VQA 的关系**

LRS-VQA 用 attention distillation 学习 text-guided 的关注区域，是确定性的两阶段 pipeline；GeoEyes 把"聚焦"变成了模型自主决策的 RL 问题——更灵活，但训练成本高得多。两者方向相同，复杂度不同。

**开放问题**

1. AdaZoom-GRPO 训练需要 GLM-4.5V 生成轨迹，如果 annotation 有偏差（如 GLM 总是倾向于多 zoom），RL 阶段能纠正吗？
2. zoom_in 接受归一化 bbox，但实际遥感图像中目标极小（几十像素），坐标误差会放大——模型对 bbox 精度的鲁棒性如何？
3. 论文没有分析多步 zoom 的推理延迟，"选择性 zoom"节省的计算是否抵消了 RL 推理链本身的开销？
