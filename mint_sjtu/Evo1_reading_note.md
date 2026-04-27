# Reading Note: Evo-1

**Evo-1: Lightweight Vision-Language-Action Model with Preserved Semantic Alignment**  
arXiv 2511.04555v2, 2025-12-05 · CVPR 2026 · MINT-SJTU / 赵波组  
Paper: https://arxiv.org/abs/2511.04555  
Code: https://github.com/MINT-SJTU/Evo-1

---

## 总结

Evo-1 不是一篇靠"大模型更大、数据更多"取胜的 VLA 论文，而是抓住了一个很实际的问题：**把预训练 VLM 直接端到端训成机器人策略时，动作监督会污染 VLM 原来的语义空间**。它的解法是一个轻量架构加两阶段训练，让 action expert 先适应 VLM 表示，再进行全模型微调，从而在小参数量、低显存和较高控制频率下拿到不错的仿真和真实机器人结果。

这篇论文的贡献不是"发明了一个非常复杂的新模型"，而是把 VLA 里几个常见模块重新组合得很干净，并且围绕 **semantic alignment preservation** 给出了问题诊断、方法、消融、效率和真实机器人验证。它的卖点是工程上有用、故事线清楚、结果性价比高。

---

## 已有问题

现有 VLA 一般从 VLM backbone 出发，再接 action head 学机器人动作。问题是机器人数据通常比互联网图文数据小得多、分布也窄得多，如果直接端到端训练，动作损失的梯度会一路反传进 VLM，使 VLM 原本学到的图文语义表示发生漂移。

论文把这个现象称为 **semantic drift**。直观表现是：训练后模型的视觉-语言 attention 不再稳定聚焦在任务相关物体上，而是变得发散、混乱，最后影响泛化。

论文 Figure 2 用 Evo-1 的 InternVL3-1B 和 OpenVLA 的 Prismatic-7B 做对比：

- Evo-1 训练后仍能在图像上形成较清晰的任务相关关注区域。
- OpenVLA 的 attention map 更容易丢掉语义结构，注意力区域看起来更散。

这个问题和我自己关注的 **VLM attention drift** 很接近：本质都是预训练多模态模型在新任务分布下表示空间被扰动。区别是 Evo-1 更偏训练期的保护机制，让动作学习不要一上来就冲击 VLM backbone。

---

## 解决方式

Evo-1 的轻量化不只是把 backbone 换小，同时改了三层东西：

1. **backbone 选择**：用 InternVL3-1B，而不是 7B 级别 VLM。
2. **动作建模方式**：用 cross-modulated DiT / flow matching 生成连续 action chunk。
3. **训练调度**：两阶段训练，先冻结 VLM，再全量微调。

如果只换一个小 VLM，可能会遇到两个问题：一是小模型容量不够，二是端到端动作训练仍然可能破坏语义空间。Evo-1 的核心说法是：轻量模型想要有效，不能只缩参数，还要让感知语义和控制学习之间的梯度关系更稳定。

---

## 架构拆解

整体数据流：

```text
multi-view RGB + language instruction
        -> InternVL3-1B backbone
        -> layer-14 fused VLM representation
        -> integration module + robot state
        -> cross-modulated DiT action expert
        -> 50-step action chunk
```

### Vision-Language Backbone

Evo-1 使用 **InternVL3-1B**：

- 视觉编码器：InternViT-300M，从 InternViT-6B 蒸馏而来。
- 语言模型：Qwen2.5-0.5B。
- 输入图像统一 resize 到 `448 x 448`。
- 图像 token 通过 `<img>` placeholder 插入语言 token 序列，由统一 decoder 处理。

论文强调 InternVL3 是 native multimodal VLM，不是把纯文本 LLM 后接视觉投影模块再临时对齐。因此它本身的跨模态表示更紧凑，也更适合轻量 VLA。

重要实现点：Evo-1 **只保留语言分支前 14 层**。论文理由是中间层往往比最后层更适合视觉-语言对齐和 visuomotor control。复现 Level 2 也确认了这一点：原始 Qwen2 language model 有 24 层，Evo-1 截断到前 14 层，所以论文里的"layer 14"对应代码里的零起始 `language_model.model.layers.13.self_attn`。

### Cross-Modulated Diffusion Transformer

动作头是条件去噪模型，用 flow matching 学连续动作轨迹，预测未来动作 chunk，默认 horizon `H = 50`。

训练时 action expert 接收：noisy action sequence 作为 query；VLM fused representation 和 robot state 作为 key/value；diffusion/flow timestep embedding。

```text
A_tau = tau * A_t + (1 - tau) * epsilon
v_theta(A_tau, z_t, s_t) -> target flow
```

`A_t` 是真实动作序列，`epsilon` 是噪声，`tau` 从分布中采样并 clamp 到稳定范围。模型学习从 noisy action 指向真实 action 的 velocity field。

DiT **主要依赖 stacked cross-attention**，不是交替 self/cross attention。论文的解释是：动作 token 每层都直接从 VLM 表示和机器人状态取条件信息，保持条件传播的一致性。

### Integration Module

最终采用的 Module A 很简洁：

- 取 VLM 第 14 层表示 `z_t`。
- 与 robot state `s_t` 拼接。
- 作为 action expert 所有 cross-attention block 的 KV。
- noisy action sequence 作为 query。

论文没选更复杂的层级注入或 cross/self 交替结构。结论是：对这个轻量 VLA，**信息条件的一致传播比结构堆复杂更重要**。

---

## 核心贡献——两阶段训练

### Stage 1: Action Expert Alignment

第一阶段冻结整个 VLM backbone，只训：integration module 和 action expert。

目的不是马上让整个模型表现最好，而是先让随机初始化的动作模块学会读懂 VLM 表示。这样动作损失不会在一开始就把预训练 VLM 的语义空间拉乱。

直觉上：先固定老师的语言，让学生学会听懂；不要一上来让老师和学生一起乱改教材。

### Stage 2: Full-Scale Fine-Tuning

第二阶段从 Stage 1 checkpoint 出发，解冻 VLM backbone，进行全模型联合微调。

这个时候 action expert 已经有稳定的感知-动作对齐，反传进 backbone 的梯度会更温和，论文认为这样可以在适应机器人任务的同时保留 VLM 的语义能力。

Meta-World 的训练设置：

| 阶段 | VLM | 训练模块 | steps | lr | batch |
|---|---|---|---:|---:|---:|
| Stage 1 | frozen | integration + action expert | 10k | 1e-5 | 16 |
| Stage 2 | unfrozen | full model | 65k | 1e-5 | 16 |

训练在 `8 x NVIDIA A100` 上分布式运行。所以"轻量部署"指的是推理，不是训练。

---

## 实验结果

### Simulation Benchmarks

| Benchmark | Evo-1 | 对比 | 结论 |
|---|---:|---|---|
| Meta-World avg | **80.6%** | SmolVLA 68.2%, π0 47.9% | 明显最高 |
| LIBERO avg | **94.8%** | π0 94.2%, GR00T N1 93.9%, OpenVLA 76.5% | 略高于 π0，长程任务也稳 |
| RoboTwin avg | **37.8%** | π0 30.9%, RDT 25.8% | 双臂子集最高 |

### Real-World Experiments

真实系统：6-DoF xArm6，固定环境相机 + wrist camera，LeRobot 2.1 数据格式，30Hz，任务包括 Pick and Place Can、Pour Foam from Cup 等。

| Model | Success |
|---|---:|
| Evo-1 | **78%** |
| π0 | 73% |
| OpenVLA-OFT | 55% |
| SmolVLA | 50% |

### Inference Efficiency

| Model | Params | GPU memory | Frequency | Real-world success |
|---|---:|---:|---:|---:|
| Evo-1 | 0.77B | **2.3GB** | **16.4Hz** | **78%** |

性能和效率一起看是 Evo-1 最亮的结果。机器人领域不只看 success rate，实时控制能力和边缘设备部署同样重要。

---

## Ablation 和说服力

### Integration Module Ablation

| Module | 设计 | 备注 |
|---|---|---|
| A | 第 14 层 VLM feature + state 作为所有 DiT 层 KV | 最终采用，条件信息最一致 |
| B | cross-attention 后插 self-attention | 更复杂，但打断条件传播 |
| C | 不同 DiT 层注入不同 VLM 层 feature | 层级更花，但不一定稳定 |
| D | 把 noisy action 也拼进 KV | 条件混合更强，但可能污染结构 |

Module A 最好，更复杂的结构没有赢。Evo-1 的优势可能来自"少而稳定"的信息流设计。

### Training Paradigm Ablation

对比 single-stage 和 two-stage：

- attention visualization：two-stage 的 attention map 更聚焦，single-stage 更混乱。
- benchmark performance：two-stage 在 Meta-World 各难度上更稳定。

这组消融很关键，因为它支撑了论文标题里的 **preserved semantic alignment**。没有这个实验，Evo-1 就只是普通轻量 action head 工程优化。

---

## 代码谱系判断

看完 `Evo1.py` 和 `flow_matching.py`，Evo-1 不是凭空发明了一套 VLA 范式，而是把几条已有路线重新组合到一个轻量、语义保护导向的实现里：

```text
Diffusion Policy  → 用去噪/扩散生成连续 action chunk
DiT               → 用 Transformer 替代 U-Net 做 denoising backbone
Flow Matching     → 预测 noise→data 的 velocity field，不是 DDPM noise
π0 / SmolVLA      → VLM backbone + flow-matching action expert
Evo-1             → 小 VLM + 纯 cross-attention action expert + 两阶段训练保护 VLM 语义
```

Evo-1 的新意不在"第一次提出 action flow matching"，而在几个具体取舍：

- **小 backbone**：InternVL3-1B，语言模型截到前 14 层，取中间层语义而不是最后层输出。
- **纯 cross-attention action expert**：`query = noisy action tokens`，`key/value = fused VLM tokens + state token`；相比 SmolVLA 式 self/cross 交替，更强调条件信息的连续传播。
- **flow matching 的动作生成**：训练时构造 `A_t = (1-t)*noise + t*action_gt`，学 `velocity = action_gt - noise`；推理时从随机 action 开始积分。
- **训练/推理分离**：`forward()` 是训练 velocity field，`get_action()` 是采样/积分。
- **语义保护是主线**：Stage 1 冻结 VLM，Stage 2 再联合微调。这是 Evo-1 相比 π0 / SmolVLA 更突出的论文叙事。

**Evo-1 是一篇组合创新很强的 VLA 工程论文，底层范式来自 Diffusion Policy、DiT、Flow Matching 和 π0/SmolVLA，重点在轻量化和 semantic drift 控制。**

---

## 代码主线

```text
LeRobotDataset
  -> images / prompt / state / action chunk / masks

InternVL3Embedder
  -> images + prompt -> fused_tokens

FlowmatchingActionHead
  -> fused_tokens + state + noisy action -> velocity

EVO1
  -> 组装 VLM embedder 和 action head，路由训练/推理

train.py
  -> 两阶段训练、loss、optimizer、checkpoint
```

### `Evo1.py`

`EVO1` 是顶层 wrapper。用 `actions_gt is None` 区分训练和推理：有 `actions_gt` 就训练 velocity field，没有就积分出 action chunk。Stage 1/2 由 `set_finetune_flags()` 控制，Stage 2 不是 LoRA，是直接更新 VLM + action head，VLM 在代码里截断到前 14 层，`lm_head` 也被换成 `Identity()`。

### `internvl3_embedder.py`

`__init__()` 加载 InternVL3-1B 后直接 `layers = layers[:14]`，"第 14 层"不是动态选择，是硬裁剪后的最后一层。`_prepare_and_fuse_embeddings()` 先 tokenize prompt，找 `<IMG_CONTEXT>` 位置，用 vision encoder 输出的 `vit_embeds` 替换占位符，再过截断的 language model，得到 `fused_hidden`。

### `flow_matching.py`

训练路径：

```text
noise ~ U(-1, 1)
t ~ Beta(2, 2)
A_t = (1-t) * noise + t * action_gt
target_velocity = action_gt - noise
```

模型输入 `A_t`、`t`、`fused_tokens` 和 `state`，预测 `pred_velocity`，MSE 对齐 `target_velocity`。

推理路径：

```text
action = random noise
for i in num_inference_timesteps:
    pred = model(action, t, fused_tokens, state)
    action = action + dt * pred
```

cross-attention：`query = noisy action tokens`，`key/value = fused VLM tokens + state token`。`time_emb` 加在 FFN 前，不是标准 DiT AdaLN，是简化版时间条件注入。

### `train.py`

训练循环核心：

```text
batch -> images / prompt / state / action_gt
images + prompt -> get_vl_embeddings(...)
model(fused_tokens, state, actions_gt, action_mask) -> pred_velocity, noise
target_velocity = action_gt - noise
loss = MSE(pred_velocity, target_velocity)
backward -> grad clip -> AdamW step -> LR scheduler
```

### `lerobot_dataset_pretrain_mp.py`

只准备训练样本，不做 tokenization。state/action 按数据集 min/max 归一化到 `[-1, 1]`，pad 到统一维度，`action_mask` 屏蔽不同 embodiment 中 padded action 维度。

### 疑问

14 层是硬编码，没有 layer selection 的消融数据，不知道为什么偏偏是 14。`time_emb` 注入方式较朴素，不是完整 DiT AdaLN，设计理由不明。`get_action()` 里有很多 debug `print`，部署时有没有关掉不清楚。train loop 里 `pred_velocity` 有显式 mask，`target_velocity` 没有——构造阶段应该已经处理，但没完全确认。

---

## 这篇论文不足的地方

**Semantic drift 主要还是 qualitative evidence**：attention visualization 比较主观，更强的证据应该包括 VQA 能力训练前后的下降量，或 attention drift 的量化指标（Wasserstein distance / entropy / object-region mass）。这正好是可以补的方向。

**第 14 层选择没有充分消融**：论文说中间层更适合跨模态对齐，但没展示 layer 8/10/12/14/16 的成功率对比，也没有不同层 attention drift 和动作成功率的对应关系。

**Stage 2 是否一定要全量解冻？** 论文选了直接全量微调，但 LoRA、分层学习率、只解冻后几层可能在小数据真实机器人 fine-tune 上更稳，这没有被充分讨论。

**"no robot pretraining"需要小心理解**：Evo-1 不用 OXE/DROID 这种大规模通用机器人预训练，但它仍然需要 benchmark-specific demonstrations（Meta-World 是 50 tasks × 50 demos）。不是"没有机器人数据也能做机器人控制"。

**RoboTwin 只选了 4 个任务**，结果很好，但不是完整 benchmark 覆盖，不能过度解读。

Baseline 公平性也需要留意——很多数字来自原论文，不一定在完全相同的训练预算、数据质量、evaluation seed 下重跑，主表是方向性证据，不是绝对公平的竞技场。

---

## 总体判断

这篇论文的风格是 **clean engineering paper**，不是理论型或范式颠覆型论文。能中 CVPR，靠的是把一个有现实价值的问题讲清楚，并且用小模型、高效率、真实机器人结果做出了很强的性价比证明。

如果要把它作为切入点做自己的工作，最自然的方向不是复刻架构，而是补它没做深的部分：量化 semantic drift、做不同解冻策略和层选择的消融、在自己的 Pick-and-Place 数据上比较 Evo-1 和 OpenVLA-OFT、研究 attention drift 能不能作为失败预测或 fine-tune early stopping 信号。
