# Reading Note: Evo-1

**Evo-1: Lightweight Vision-Language-Action Model with Preserved Semantic Alignment**  
arXiv 2511.04555v2, 2025-12-05 · CVPR 2026 · MINT-SJTU / 赵波组  
Paper: https://arxiv.org/abs/2511.04555  
Code: https://github.com/MINT-SJTU/Evo-1

---

## 总结

Evo-1 不是一篇靠“大模型更大、数据更多”取胜的 VLA 论文，而是抓住了一个很实际的问题：**把预训练 VLM 直接端到端训成机器人策略时，动作监督会污染 VLM 原来的语义空间**。它的解法是一个轻量架构加两阶段训练，让 action expert 先适应 VLM 表示，再进行全模型微调，从而在小参数量、低显存和较高控制频率下拿到不错的仿真和真实机器人结果。

我的理解是：这篇论文的贡献不是“发明了一个非常复杂的新模型”，而是把 VLA 里几个常见模块重新组合得很干净，并且围绕 **semantic alignment preservation** 给出了问题诊断、方法、消融、效率和真实机器人验证。它的卖点是工程上有用、故事线清楚、结果性价比高。

---

## 这篇论文到底在解决什么问题？

现有 VLA 一般从 VLM backbone 出发，再接 action head 学机器人动作。问题是机器人数据通常比互联网图文数据小得多、分布也窄得多，如果直接端到端训练，动作损失的梯度会一路反传进 VLM，使 VLM 原本学到的图文语义表示发生漂移。

论文把这个现象称为 **semantic drift**。直观表现是：训练后模型的视觉-语言 attention 不再稳定聚焦在任务相关物体上，而是变得发散、混乱，最后影响泛化。

论文 Figure 2 用 Evo-1 的 InternVL3-1B 和 OpenVLA 的 Prismatic-7B 做对比：

- Evo-1 训练后仍能在图像上形成较清晰的任务相关关注区域。
- OpenVLA 的 attention map 更容易丢掉语义结构，注意力区域看起来更散。

这个问题和我自己的 **VLC_m / Qwen2-VL attention drift** 很接近：本质都是预训练多模态模型在新任务分布下的表示空间被扰动。区别是：

- VLC_m 更偏推理期或分析期的 drift signal。
- Evo-1 更偏训练期的保护机制，让动作学习不要一上来就冲击 VLM backbone。

---

## 为什么不是简单地“换个小 backbone”？

Evo-1 的轻量化不只是把 backbone 换小。它同时改了三层东西：

1. **backbone 选择**：用 InternVL3-1B，而不是 7B 级别 VLM。
2. **动作建模方式**：用 cross-modulated DiT / flow matching 生成连续 action chunk。
3. **训练调度**：两阶段训练，先冻结 VLM，再全量微调。

如果只换一个小 VLM，可能会遇到两个问题：一是小模型容量不够，二是端到端动作训练仍然可能破坏语义空间。Evo-1 的核心说法是：轻量模型想要有效，不能只缩参数，还要让感知语义和控制学习之间的梯度关系更稳定。

---

## 架构拆解

整体数据流可以写成：

```text
multi-view RGB + language instruction
        -> InternVL3-1B backbone
        -> layer-14 fused VLM representation
        -> integration module + robot state
        -> cross-modulated DiT action expert
        -> 50-step action chunk
```

### 1. Vision-Language Backbone

Evo-1 使用 **InternVL3-1B**：

- 视觉编码器：InternViT-300M，从 InternViT-6B 蒸馏而来。
- 语言模型：Qwen2.5-0.5B。
- 输入图像统一 resize 到 `448 x 448`。
- 图像 token 通过 `<img>` placeholder 插入语言 token 序列，由统一 decoder 处理。

论文强调 InternVL3 是 native multimodal VLM，不是把一个纯文本 LLM 后接视觉投影模块再临时对齐。因此它本身的跨模态表示更紧凑，也更适合轻量 VLA。

一个重要实现点：Evo-1 **只保留语言分支前 14 层**。论文理由是中间层往往比最后层更适合视觉-语言对齐和 visuomotor control。我们 Level 2 复现也确认了这一点：原始 Qwen2 language model 有 24 层，本地 Evo-1 截断到前 14 层，所以论文里的 “layer 14” 对应代码里的 zero-based `language_model.model.layers.13.self_attn`。

### 2. Cross-Modulated Diffusion Transformer

动作头是一个条件去噪模型，用 flow matching 学连续动作轨迹。它不是 autoregressive 地一个动作一个动作吐出，而是预测一个未来动作 chunk，论文默认 horizon 是 `H = 50`。

训练时，action expert 接收：

- noisy action sequence，作为 query。
- VLM fused representation 和 robot state，作为 key/value。
- diffusion/flow timestep embedding。

论文公式大致是：

```text
A_tau = tau * A_t + (1 - tau) * epsilon
v_theta(A_tau, z_t, s_t) -> target flow
```

这里 `A_t` 是真实动作序列，`epsilon` 是噪声，`tau` 从分布中采样并 clamp 到稳定范围。模型学习的是从 noisy action 指向真实 action 的 velocity field。

值得注意的是，它的 DiT **主要依赖 stacked cross-attention**，而不是像一些动作专家那样交替使用 self-attention 和 cross-attention。论文的解释是：动作 token 每层都直接从 VLM 表示和机器人状态里取条件信息，可以保持条件传播的一致性。

### 3. Integration Module

Integration module 的作用是把 VLM 的中间语义表示和机器人 proprioceptive state 连接起来。

最终采用的 Module A 很朴素：

- 取 VLM 第 14 层表示 `z_t`。
- 与 robot state `s_t` 拼接。
- 作为 action expert 所有 cross-attention block 的 KV。
- noisy action sequence 作为 query。

论文没有选择更复杂的层级注入或 cross/self 交替结构。它的结论是：对于这个轻量 VLA，**信息条件的一致传播比结构堆复杂更重要**。

---

## 两阶段训练是核心贡献

### Stage 1: Action Expert Alignment

第一阶段冻结整个 VLM backbone，只训练：

- integration module
- action expert

目的不是马上让整个模型表现最好，而是先让随机初始化的动作模块学会读懂 VLM 表示。这样动作损失不会在一开始就把预训练 VLM 的语义空间拉乱。

直觉上可以理解为：先固定老师的语言，让学生学会听懂；不要一上来让老师和学生一起乱改教材。

### Stage 2: Full-Scale Fine-Tuning

第二阶段从 Stage 1 checkpoint 出发，解冻 VLM backbone，进行全模型联合微调。

这个时候 action expert 已经有了稳定的感知-动作对齐，反传进 backbone 的梯度会更温和，论文认为这样可以在适应机器人任务的同时保留 VLM 的语义能力。

补充材料里给了 Meta-World 的训练设置：

| 阶段 | VLM | 训练模块 | steps | lr | batch |
|---|---|---|---:|---:|---:|
| Stage 1 | frozen | integration + action expert | 10k | 1e-5 | 16 |
| Stage 2 | unfrozen | full model | 65k | 1e-5 | 16 |

论文说明实验在 `8 x NVIDIA A100` 上分布式训练。也就是说，它虽然主打“轻量部署”，但训练本身并不是笔记本级别的小实验。

---

## 实验结果

### 1. Simulation Benchmarks

论文主表覆盖三个仿真基准：

| Benchmark | Evo-1 | 对比对象 | 结论 |
|---|---:|---|---|
| Meta-World avg | **80.6%** | SmolVLA 68.2%, π0 47.9% | Evo-1 明显最高 |
| LIBERO avg | **94.8%** | π0 94.2%, GR00T N1 93.9%, OpenVLA 76.5% | Evo-1 略高于 π0，长程任务也稳 |
| RoboTwin avg | **37.8%** | π0 30.9%, RDT 25.8% | Evo-1 在双臂子集上最高 |

更细一点看：

- **Meta-World**：50 个 manipulation tasks，每个任务 50 条 demonstrations，按 easy / medium / hard / very hard 分组，评估 10 trials，并报告 5 次独立运行平均。
- **LIBERO**：40 个任务，分为 Spatial / Object / Goal / Long 四套，每个任务 10 trials，报告 5 次独立运行平均。
- **RoboTwin**：选取 4 个代表性双臂任务，每个任务 50 条 demonstrations，easy/hard 两档，每个任务 100 evaluation trials。

所以“只有三个 benchmark”这个说法只对主仿真实验成立，而且这三个 benchmark 内部并不小。尤其 Meta-World 是 50 任务，LIBERO 是 40 任务，RoboTwin 是双臂操作子集。

### 2. Real-World Experiments

论文还做了真实机器人实验，这一点容易被主表掩盖。

真实系统：

- 机器人：6-DoF xArm6。
- 传感器：固定环境相机 + wrist camera。
- 数据格式：LeRobot 2.1。
- 观测频率：30Hz。
- 任务：Pick and Place Can、Pour Foam from Cup、Hand Delivery、Can Stacking。

真实机器人平均成功率：

| Model | Success |
|---|---:|
| Evo-1 | **78%** |
| π0 | 73% |
| OpenVLA-OFT | 55% |
| SmolVLA | 50% |

论文还展示了 SO-100 级别桌面机械臂上的部署案例。严格说这不一定是强定量实验，但对“轻量可部署”这个 claim 有加分。

### 3. Inference Efficiency

Evo-1 最亮的结果其实是“性能和效率一起看”：

| Model | Params | GPU memory | Frequency | Real-world success |
|---|---:|---:|---:|---:|
| Evo-1 | 0.77B | **2.3GB** | **16.4Hz** | **78%** |

这解释了为什么它能成为 CVPR 论文：机器人领域不只看最终 success rate，还很看重是否能实时控制、能否在消费级 GPU 或边缘设备上部署。

---

## Ablation 和说服力

### 1. Integration Module Ablation

论文比较了四种 module：

| Module | 设计 | 我的理解 |
|---|---|---|
| A | 第 14 层 VLM feature + state 作为所有 DiT 层 KV | 最终采用，条件信息最一致 |
| B | cross-attention 后插 self-attention | 更复杂，但打断条件传播 |
| C | 不同 DiT 层注入不同 VLM 层 feature | 层级更花，但不一定稳定 |
| D | 把 noisy action 也拼进 KV | 条件混合更强，但可能污染结构 |

结论是 Module A 最好。这个结果有点反直觉：更复杂的结构没有赢，说明 Evo-1 的优势可能来自“少而稳定”的信息流设计。

### 2. Training Paradigm Ablation

论文比较了 single-stage 端到端训练和 two-stage 训练。

结果分两部分：

- attention visualization：two-stage 的 attention map 更聚焦，single-stage 更混乱。
- benchmark performance：two-stage 在 Meta-World 各难度上更稳定。

这一组消融很关键，因为它支撑了论文标题里的 **preserved semantic alignment**。如果没有这个实验，Evo-1 就更像普通轻量 action head 工程优化。

---

## 代码细读后的谱系判断

看完 `Evo1.py` 和 `flow_matching.py` 后，我觉得 Evo-1 不是凭空发明了一套 VLA 范式，而是把几条已有路线重新组合到一个轻量、语义保护导向的实现里：

```text
Diffusion Policy
  -> 用去噪/扩散生成连续 action chunk

DiT
  -> 用 Transformer 替代 U-Net，作为 denoising / generative backbone

Flow Matching / Rectified Flow
  -> 不预测 DDPM noise，而是预测从噪声到真实数据的 velocity field

π0 / SmolVLA
  -> VLM backbone + flow-matching action expert，生成连续机器人动作

Evo-1
  -> 小 VLM + 纯 cross-attention action expert + 两阶段训练保护 VLM 语义
```

所以 Evo-1 的新意不在“第一次提出 action flow matching”，而在几个具体取舍：

- **小 backbone**：用 InternVL3-1B，并且在代码里直接把 language model 截到前 14 层，取中间层语义而不是最后层输出。
- **纯 cross-attention action expert**：`query = noisy action tokens`，`key/value = fused VLM tokens + state token`；相比 SmolVLA 式 self/cross 交替，它更强调条件信息的连续传播。
- **flow matching 的动作生成**：训练时构造 `A_t = (1-t)*noise + t*action_gt`，学 `velocity = action_gt - noise`；推理时从随机 action 开始，用 `action = action + dt * pred` 积分得到 action chunk。
- **训练/推理分离**：`forward()` 是训练 velocity field，`get_action()` 是采样/积分；这更像 diffusion model 的 `forward` vs `sample`，而不是普通 Transformer 只靠 `train()/eval()` 区分。
- **语义保护是主线**：Stage 1 冻结 VLM，只训 action expert；Stage 2 再联合微调。这是 Evo-1 相比 π0 / SmolVLA 更突出的论文叙事。

一句话：**Evo-1 是一篇组合创新很强的 VLA 工程论文。底层范式来自 Diffusion Policy、DiT、Flow Matching 和 π0/SmolVLA，但它把重点放在轻量化和 semantic drift 控制上。**

---

## 代码主线理解

代码主路径已经基本清楚，可以按五个文件串起来：

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

### 1. `Evo1.py`

`EVO1` 是顶层 wrapper。它初始化 `InternVL3Embedder` 和 `FlowmatchingActionHead`，并用 `actions_gt is None` 区分训练和推理：

- 有 `actions_gt`：走 `FlowmatchingActionHead.forward()`，训练 velocity field。
- 没有 `actions_gt`：走 `FlowmatchingActionHead.get_action()`，从随机 action 积分出 action chunk。

Stage 1 / Stage 2 由 `set_finetune_flags()` 控制：

- Stage 1：`finetune_vlm=False`，冻结 VLM，只训 action head。
- Stage 2：`finetune_vlm=True` 且 `finetune_action_head=True`，全量微调 Evo-1。这里不是 LoRA / adapter，而是直接更新 VLM + action head；但 VLM 已经在代码里截断到前 14 层，`lm_head` 也被换成 `Identity()`。

### 2. `internvl3_embedder.py`

这个文件负责把 `images + prompt` 变成 action head 的 `fused_tokens`。

关键点：

- `__init__()` 加载 InternVL3-1B 后直接执行 `layers = layers[:14]`，所以“第 14 层”不是动态选择，而是硬裁剪后的最后一层。
- `<IMG_CONTEXT>` 只是图像占位符 token，不是真正视觉特征。
- `_prepare_and_fuse_embeddings()` 先把 prompt tokenize 成 token embeddings，再找到 `<IMG_CONTEXT>` 的位置，用 vision encoder 输出的 `vit_embeds` 替换这些占位符。
- 最后把混合后的 `inputs_embeds` 送入截断后的 language model，得到 `fused_hidden`，训练时返回完整 `[B, seq_len, 896]`。

所以 `fused_tokens` 不是“机械臂语言”，而是 **图像 token + 文本 token 经 InternVL3 融合后的多模态 hidden states**。

### 3. `flow_matching.py`

这是 action expert 的主体。它不是直接回归 clean action，而是学习一个 velocity field。

训练路径：

```text
noise ~ U(-1, 1)
t ~ Beta(2, 2)
A_t = (1-t) * noise + t * action_gt
target_velocity = action_gt - noise
```

模型输入 `A_t`、`t`、`fused_tokens` 和 `state`，预测 `pred_velocity`，用 MSE 对齐 `target_velocity`。

推理路径：

```text
action = random noise
for i in num_inference_timesteps:
    pred = model(action, t, fused_tokens, state)
    action = action + dt * pred
```

cross-attention 体现在：

```text
query = noisy action tokens
key/value = fused VLM tokens + state token
```

`time_emb` 不是加在 attention 前，而是加在 FFN 前；这是一种简化时间条件注入方式，不是标准 DiT AdaLN。可以理解成：attention 负责读语义上下文，time embedding 负责告诉动作更新当前处在 flow 的哪个时间点。

### 4. `train.py`

训练循环核心很短：

```text
batch -> images / prompt / state / action_gt
images + prompt -> get_vl_embeddings(..., return_cls_only=False)
model(fused_tokens, state, actions_gt, action_mask) -> pred_velocity, noise
target_velocity = action_gt - noise
loss = MSE(pred_velocity, target_velocity)
backward -> grad clip -> AdamW step -> LR scheduler
```

这里 `grad clip` 是防止梯度爆炸，`AdamW step` 真正更新参数，`LR scheduler` 做 warmup + cosine decay。

### 5. `lerobot_dataset_pretrain_mp.py`

`LeRobotDataset` 不做 tokenization，也不生成 fused tokens。它只准备训练样本：

```text
LeRobot parquet + videos + task metadata
  -> 当前 timestamp 的视频帧
  -> task prompt
  -> 当前 state
  -> 未来 H 步 action chunk
  -> image/state/action masks
```

state/action 会按数据集 min/max 归一化到 `[-1, 1]`，再 pad 到统一维度。`action_mask` 用来屏蔽不同 embodiment 中不存在的 padded action 维度。

### 代码层面的保留疑点

- 14 层是硬编码，缺少 layer selection ablation。
- `time_emb` 注入方式较朴素，不是完整 DiT AdaLN 设计。
- `get_action()` 有不少 debug `print`，部署时可能影响频率。
- train loop 里逐样本提 VLM fused tokens，效率可能一般。
- loss 中显式 mask 了 `pred_velocity`，但没有同样显式 mask `target_velocity`；虽然噪声构造阶段已处理 mask，但这里仍值得复核。

---

## 这篇论文的不足和我读下来保留的怀疑

### 1. Semantic drift 主要还是 qualitative evidence

论文用了 attention map 说明 semantic alignment preserved，但 attention visualization 本身比较主观。更强的证据应该包括：

- 图文检索或 VQA 能力训练前后的下降量。
- attention drift 的量化指标，比如 Wasserstein distance / entropy / object-region mass。
- 与 downstream success rate 的相关性分析。

这正好和我的 VLC_m 方向能接上：我可以把自己的 drift signal 用在 Evo-1 / OpenVLA-OFT 对比上，把论文的视觉证据变成量化证据。

### 2. 第 14 层选择还不够彻底

论文说中间层更适合跨模态对齐，但没有充分展示 “为什么就是第 14 层”。我更想看到：

- layer 8 / 10 / 12 / 14 / 16 的成功率对比。
- 不同层 attention drift 和动作成功率的对应关系。
- 是否不同任务类型需要不同层。

### 3. Stage 2 是否一定要全量解冻？

目前论文的两阶段设计是 Stage 2 full fine-tuning。但也许更稳的方案是：

- 只解冻后几层。
- LoRA / adapter fine-tune。
- 分层学习率，backbone lr 更小。
- 冻结 vision encoder，只调 language decoder 或 integration。

这对小数据真实机器人 fine-tune 可能更重要，因为真实机器人数据更少，更容易过拟合。

### 4. “no robot pretraining”需要小心表述

Evo-1 不用 OXE / DROID 这种大规模 robot pretraining，但它仍然需要 benchmark-specific demonstrations。比如 Meta-World 是 50 tasks x 50 demos，RoboTwin 每个任务 50 demos。

所以它不是“没有机器人数据也能做机器人控制”，而是“没有大规模通用机器人预训练，也能在下游 demonstrations 上训练出强策略”。

### 5. RoboTwin 只选了 4 个任务

RoboTwin 结果很好，但它不是完整 RoboTwin 全任务覆盖，而是 4 个代表性双臂任务。这个结果能证明双臂潜力，但不能过度解读成完整双臂通用能力。

### 6. Baseline 公平性仍需细看

很多 baseline 数字来自原论文或官方结果，不一定全都在完全相同训练预算、数据质量、prompt 格式、evaluation seed 下重跑。VLA 论文里这很常见，但读的时候要记住：主表是方向性证据，不是绝对公平的竞技场。

---

## 我会怎么评价这篇论文

### 优点

- 论文问题很清楚：轻量 VLA 不仅要小，还要保护预训练 VLM 的语义空间。
- 方法简单，工程可实现，不依赖巨量 robot pretraining。
- 三个仿真 benchmark、真实机器人、效率分析、消融和开源材料构成了比较完整的证据链。
- 0.77B + 2.3GB + 16.4Hz 对真实部署有实际意义。

### 弱点

- attention preservation 的量化还不够。
- 第 14 层选择、flow matching vs DDPM、full unfreeze vs partial unfreeze 都还可以更细。
- RoboTwin 是代表性子集，不是完整 benchmark。
- baseline 公平性需要结合原实验设置继续核对。

### 总体判断

这篇论文的风格是 **clean engineering paper**，不是理论型或范式颠覆型论文。它能中 CVPR，靠的是把一个有现实价值的问题讲清楚，并且用小模型、高效率、真实机器人结果做出了很强的性价比证明。

如果我要把它继续做成自己的工作切入点，最自然的方向不是复刻架构，而是补它没做深的部分：

1. 量化 semantic drift。
2. 做不同解冻策略和层选择的消融。
3. 在自己的真实 Pick-and-Place 数据上比较 Evo-1 和 OpenVLA-OFT。
4. 研究 attention drift 是否能作为失败预测或 fine-tune early stopping 信号。

---
