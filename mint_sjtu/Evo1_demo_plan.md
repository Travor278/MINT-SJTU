# Evo-1 复现记录

论文：**Evo-1: Lightweight Vision-Language-Action Model with Preserved Semantic Alignment**  
官方仓库：https://github.com/MINT-SJTU/Evo-1  
日期：2026-04-27

---

LIBERO 跑了三个 seed，均值 93.5%±1.3%，和官方 94.8% 差一点点，在合理范围内。MetaWorld MT50 跑出 80.7% difficulty average，和官方 80.6% 基本对齐。Level 2 做了 layer 13 的 attention 可视化，用 InternVL3 base 做了对照，能看到 fine-tune 之后 attention 确实更集中，但 fail case 也集中，所以"attention 聚焦"和"成功率"之间的关系比想象的复杂。

这次没有跑 OpenVLA baseline，所以 Level 2 的 attention 对比只是 Evo-1 final 对 InternVL3 base，不是论文 Figure 2 的完整跨模型复刻。

---

## 已完成

| 问题 | 实验 |
|---|---|
| LIBERO checkpoint 在本地能不能跑近官方成功率？ | Level 1：LIBERO 推理，3 seeds，1200 episodes |
| fine-tune 之后 attention 还保留语义聚焦吗？ | Level 2：layer 13 attention 可视化 + InternVL3 base 对照 |
| MetaWorld checkpoint 能复现 MT50 难度分组成功率吗？ | Level 3：MetaWorld MT50，500 episodes |

---

## 环境和配置

| 组件 | 配置 |
|---|---|
| 系统 | WSL Ubuntu 22.04 |
| GPU | NVIDIA GeForce RTX 5070 Laptop GPU, 8 GB |
| Server env | Python 3.10, PyTorch 2.7.0+cu128, Transformers 4.39.0, FlashAttention 2.8.3 |
| Client env | Python 3.8.13, PyTorch 1.11.0+cu113, robosuite 1.4.0, mujoco 3.2.3 |
| LIBERO checkpoint | `MINT-SJTU/Evo1_LIBERO` |
| MetaWorld checkpoint | `MINT-SJTU/Evo1_MetaWorld` |
| VLM | `OpenGVLab/InternVL3-1B` 本地副本 |
| 渲染 | LIBERO 和 MetaWorld 都用 `MUJOCO_GL=osmesa` |

产物路径：

| 类型 | 路径 |
|---|---|
| Level 1 详细报告 | `Evo-1_reproduction/reports/LEVEL1_REPRODUCTION_REPORT.md` |
| Level 1 日志 | `Evo-1_reproduction/level1/log_file/` |
| Level 1 多 seed 汇总 | `Evo-1_reproduction/reports/level1_multiseed_summary.md` |
| Level 1 视频 | `Evo-1_reproduction/level1/video_log_file/Evo1_libero_all/` |
| Level 2 图与 CSV | `Evo-1_reproduction/figures/` |
| Level 3 日志 | `Evo-1_reproduction/level3_metaworld_mt50_10ep.log` |

### Level 1

覆盖 `libero_spatial`、`libero_object`、`libero_goal`、`libero_10` 四套。每套 10 个任务，每个任务 10 个 episode，一次完整跑 400 episodes。Evo-1 server 加载 LIBERO checkpoint，LIBERO client 发图像、状态和指令，接收 action chunk 执行。正式结果用 seed=42/43/44 各跑一次，共 1200 episodes。

### Level 2

复用 Level 1 的 rollout 视频，没有重新跑仿真。流程：

```
rollout log + videos
  → 挑 success/failure 样本
  → 采帧
  → Evo-1 embedder + output_attentions=True
  → 提 layer 13 text-to-image attention
  → 存 overlay 和指标
```

因为 FlashAttention 不返回 attention weights，probe 时关掉了 FlashAttention。固定帧实验用 frame 30/60/90；frame sweep 对每个 case 多采几个帧，按下面这个指标选展示帧：

```
focus_score = top5_mass - 0.02 * entropy
```

### Level 3

MT50 全 50 个任务，每任务 10 episodes，共 500 episodes。Server 和 Level 1 共用 Evo-1 推理环境，client 用独立的 metaworld 环境。WSL 下 egl 渲染不稳定，正式跑用 `MUJOCO_GL=osmesa`。

---

## Level 1：LIBERO 结果

seed=42 的单次完整结果：

| Suite | 本地 | 官方 |
|---|---:|---:|
| `libero_spatial` | 88/100 = 88.0% | 92.7% |
| `libero_object` | 97/100 = 97.0% | 97.7% |
| `libero_goal` | 94/100 = 94.0% | 96.3% |
| `libero_10` | 91/100 = 91.0% | 92.3% |
| Overall | **370/400 = 92.5%** | **94.8%** |

后来又补了 seed=43 和 44，主要想看稳定性：

| Seed | Overall | `libero_spatial` | `libero_object` | `libero_goal` | `libero_10` |
|---:|---:|---:|---:|---:|---:|
| 42 | 370/400 = 92.5% | 88.0% | 97.0% | 94.0% | 91.0% |
| 43 | 380/400 = 95.0% | 93.0% | 98.0% | 95.0% | 94.0% |
| 44 | 372/400 = 93.0% | 86.0% | 97.0% | 96.0% | 93.0% |
| Mean ± std | **93.5% ± 1.3%** | 89.0% ± 3.6% | 97.3% ± 0.6% | 95.0% ± 1.0% | 92.7% ± 1.5% |

三次均值 93.5% 和官方 94.8% 差 1.3 个点，官方结果落在一个标准差附近。

平均步数：

| Suite | Average Steps |
|---|---:|
| `libero_spatial` | 7.40 |
| `libero_object` | 10.16 |
| `libero_goal` | 8.00 |
| `libero_10` | 18.50 |

失败主要集中在空间关系任务和长指令组合任务，比如 `libero_spatial` Task 2/6 都只有 7/10，`libero_10` Task 7/8 是 8/10。

---

## Level 2：Attention 可视化

### 固定帧分析

在每个 suite 各选一个 success 和 fail 样本，在同一帧索引提取 layer 13 attention。frame 60 大概在执行中段，通常比起始帧更能看出模型在关注什么。

![Level 2 frame 60 success/failure attention panel](Evo-1_reproduction/figures/level2_success_failure_panel_f060.png)

**图 1.** frame 60 的 success/failure attention 对比。每行对应一个 LIBERO suite，绿色成功，红色失败。

固定帧统计：

| Frame | Status | n | Max prob | Top-5 mass | Entropy |
|---:|---|---:|---:|---:|---:|
| 30 | success | 8 | 0.0446 | 0.1263 | 5.1427 |
| 30 | fail | 8 | 0.0385 | 0.1200 | 5.1704 |
| 60 | success | 8 | 0.0394 | 0.1261 | 5.1316 |
| 60 | fail | 8 | 0.0409 | 0.1205 | 5.1362 |
| 90 | success | 8 | 0.0405 | 0.1164 | 5.1681 |
| 90 | fail | 8 | 0.0395 | 0.1229 | 5.1524 |

success 和 fail 的均值差异很小。Attention probe 可以稳定给出逐样本的可视化解释，但单靠这个数字不能支撑"成功样本 attention 明显更好"的结论。

### Frame sweep 分析

不同任务关键动作发生的时间点不同，固定帧不一定抓到最有代表性的状态。Frame sweep 对每个 case 采多个候选帧，按 focus score 选一帧展示。

![Level 2 frame sweep best success/failure panel](Evo-1_reproduction/figures/level2_best_success_failure_panel.png)

**图 2.** Frame sweep 选出的 success/failure attention。这比固定帧更适合定性分析。

能看到多个成功样本里 Evo-1 的 attention 确实覆盖了目标物、容器或任务关键区域。失败样本也可能出现集中的 attention，但聚焦的位置有时偏了，或者被末端执行器、遮挡、视角变化干扰。

---

## InternVL3 Base 对照

为了验证 fine-tuning 后语义有没有保留，比较了两个版本：

```
Evo-1 final：加载 LIBERO fine-tuned checkpoint
InternVL3 base：同一个 InternVL3-1B backbone，不加载 Evo-1 checkpoint
```

注意这不等价于 OpenVLA baseline，只是在验证 Evo-1 fine-tune 之后语义 grounding 有没有被破坏。

![Evo-1 final vs InternVL3 base attention panel](Evo-1_reproduction/figures/level2_evo1_vs_internvl3_base_panel.png)

**图 3.** 同一批帧上 `original / Evo-1 final / InternVL3 base` 对比。

8 个 best-by-case 样本的统计：

| Group | n | Evo-1 focus | Base focus | Delta focus | Evo-1 top-5 | Base top-5 | Delta top-5 |
|---|---:|---:|---:|---:|---:|---:|---:|
| success | 4 | 0.0355 | 0.0132 | +0.0224 | 0.1374 | 0.1140 | +0.0234 |
| fail | 4 | 0.0505 | 0.0127 | +0.0379 | 0.1521 | 0.1138 | +0.0383 |
| all | 8 | 0.0430 | 0.0129 | +0.0301 | 0.1447 | 0.1139 | +0.0308 |

Evo-1 final 比 InternVL3 base attention 更集中，支持一个有限的结论：LIBERO fine-tuning 没有抹掉 InternVL3 的任务相关视觉响应，在这些 rollout 帧上还增强了任务条件化聚焦。

但 fail 组的 focus 提升同样很明显，说明 attention 集中不等于执行成功。失败还可以来自动作预测、时序决策、局部错误 grounding 或状态估计问题。

---

## Level 3：MetaWorld MT50 结果

MT50 全 50 任务，每任务 10 episodes，500 episodes 总计。原始成功率 420/500 = 84.0%，按官方难度分组：

| Difficulty | 本地 | 官方 |
|---|---:|---:|
| easy | 89.3% | 89.2% |
| medium | 76.4% | — |
| hard | 75.0% | 77.2% |
| very hard | 82.0% | 79.2% |
| **Difficulty average** | **80.7%** | **80.6%** |

和官方基本一致。这次没做多 seed 均值，是单次完整跑。

低分任务：

| Task | Success |
|---|---:|
| `handle-pull-side-v3` | 1/10 |
| `reach-v3` | 2/10 |
| `soccer-v3` | 3/10 |
| `hammer-v3` | 4/10 |
| `pick-out-of-hole-v3` | 4/10 |
| `handle-pull-v3` | 5/10 |
| `pick-place-v3` | 5/10 |

这些任务主要是精细接触、拉取或小目标抓取，不是简单的语义理解问题，更容易受接触动力学和初始状态采样影响。

---

## 说明

1. Level 1 有三个 seed；Level 3 只是单次完整评估，没有方差数据。
2. Attention overlay 是解释性证据，没法当作失败原因的因果证明。
3. Baseline 对比用的是 InternVL3 base，不是 OpenVLA-OFT，不能直接对应原论文 Figure 2。
4. 当前指标是全图 attention 分布；要得到更强的量化结论，得有目标物、容器和末端执行器的 region mask。

---

## 结论

LIBERO 三个 seed 均值 93.5%±1.3%，接近官方 94.8%；MetaWorld MT50 难度分组平均 80.7%，和官方 80.6% 基本一致。Attention 分析表明 Evo-1 fine-tune 之后语义 grounding 没有被破坏，在选定的 rollout 帧上 attention 比 InternVL3 base 更集中。

总结：两个主要仿真 benchmark 的结果在本地都可以复现；attention 证据和论文的 semantic preservation 叙事一致，但不包括 OpenVLA baseline 的完整对比。
