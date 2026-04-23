# Evo-1 复现计划

官方仓库：https://github.com/MINT-SJTU/Evo-1
HuggingFace checkpoints：Evo1_LIBERO、Evo1_MetaWorld

---

## 环境要求

```bash
# 最低 GPU：4GB 显存（只需 2.3GB 推理）
# Python 3.10, CUDA

git clone https://github.com/MINT-SJTU/Evo-1
cd Evo-1
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
```

---

## Level 0 — 代码结构通读（无 GPU，半天）

**目标**：理解 integration module + DiT action expert 的代码组织，能向别人解释。

```bash
# 重点看这几个文件
Evo-1/
  model/
    evo1.py          ← 主模型，integration module 在这里
    action_expert.py ← cross-modulated DiT
  train/
    stage1.py        ← 冻结 backbone 的训练逻辑
    stage2.py        ← 全量 fine-tune
  eval/
    eval_libero.py
    eval_metaworld.py
```

画出数据流图：obs image → InternVL3-1B (frozen) → layer 14 特征 → integration module → DiT (flow matching) → action

**输出**：一张手画/Mermaid 的架构图，标清各模块参数量

---

## Level 1 — LIBERO 推理验证（GPU 4GB+，1-2天）

**目标**：加载官方 checkpoint，跑通 LIBERO 推理，复现 94.8% 的数字。

```bash
# 下载 checkpoint
huggingface-cli download MINT-SJTU/Evo1_LIBERO --local-dir ./checkpoints/libero

# 跑评测（按官方 eval 脚本）
python eval/eval_libero.py \
    --checkpoint ./checkpoints/libero \
    --task libero_spatial \
    --episodes 20
```

对比官方数字：
- libero_spatial: 92.7%
- libero_object: 97.7%
- libero_goal: 96.3%
- libero_long: 92.3%

**输出**：自己跑出来的成功率表格，与论文 Table 1 对照

---

## Level 2 — 注意力语义对比可视化（GPU，2-3天）

**目标**：复现论文 Figure 2——Evo-1 vs OpenVLA 的注意力图对比，用自己的 Pick-and-Place 场景验证。

```python
import torch
from transformers import AutoModel

# 加载 Evo-1 的 InternVL3-1B backbone（stage2 训练后）
evo1_backbone = AutoModel.from_pretrained("./checkpoints/libero")

# 加载 OpenVLA-OFT 的 vision backbone（你已有）
openvla_backbone = ...

attn_storage = []
def attn_hook(module, input, output):
    attn_storage.append(output[1].detach().cpu())

# 同一张 obs 图，分别提取 layer 14 attention
for name, mod in evo1_backbone.named_modules():
    if "layer.14" in name and "attention" in name:
        mod.register_forward_hook(attn_hook)

# forward，可视化 attn heatmap
import matplotlib.pyplot as plt
# reshape attn [H, N, N] → patch grid，叠加到原图上
```

**输出**：Evo-1 vs OpenVLA 注意力热力图对比，与 Figure 2 并排

---

## Level 3 — MetaWorld 对比实验（GPU，3-5天）

**目标**：在 MetaWorld 的 hard/very-hard 任务上跑 Evo-1 vs 基线，复现 Table 2 数字。

```bash
pip install metaworld

huggingface-cli download MINT-SJTU/Evo1_MetaWorld --local-dir ./checkpoints/metaworld

python eval/eval_metaworld.py \
    --checkpoint ./checkpoints/metaworld \
    --difficulty hard \
    --episodes 50
```

重点复现：
- Easy: 89.2%
- Hard: 77.2%
- Very Hard: 79.2%

**输出**：四档难度成功率表格

---

## Level 4 — 接入 Pick-and-Place 做 fine-tune（完整，1-2周）

**目标**：在自己的 Franka 任务数据上 fine-tune Evo-1，与 OpenVLA-OFT 对比。

```bash
# 用官方 server-client 推理接口
# Stage 1: 只训 integration module + action expert
python train/stage1.py \
    --data ./data/pick_and_place \
    --backbone_ckpt ./checkpoints/libero \
    --epochs 20

# Stage 2: 全量 fine-tune
python train/stage2.py \
    --stage1_ckpt ./outputs/stage1 \
    --epochs 10
```

**输出**：Evo-1 vs OpenVLA-OFT 在自己任务上的成功率对比 + 推理延迟对比（Evo-1 只需 2.3GB）

---

## 现实优先级

没有 GPU：做 Level 0（代码通读 + 架构图），附在邮件里说明你读懂了代码
有 4GB GPU：Level 1 直接可跑，1天能出结果
有更大 GPU：Level 2 注意力可视化最有说服力，直接对应论文核心 claim
