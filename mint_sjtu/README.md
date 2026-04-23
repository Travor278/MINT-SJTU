# MINT-SJTU 具身智能方向调研与复现

赵波组（Machine Intelligence & Interaction Lab，上海交通大学）开源项目学习记录。  
主要方向：VLA 模型、具身操作、离线强化学习。

---

## 主复现项目：Evo-1

**Evo-1: Lightweight Vision-Language-Action Model with Preserved Semantic Alignment**  
CVPR 2026 · [arXiv 2511.04555](https://arxiv.org/abs/2511.04555) · [GitHub](https://github.com/MINT-SJTU/Evo-1)

### 核心贡献

现有轻量 VLA 端到端训练时，机器人数据梯度反传进预训练 VLM backbone，导致语义表示崩溃（attention 从聚焦退化为发散）。Evo-1 提出两阶段训练：

- **Stage 1**：冻结 VLM backbone，只训 integration module + action expert（DiT flow matching），让 action expert 先对齐到多模态表示空间
- **Stage 2**：解冻 backbone 全量联合微调

额外引入 cross-modulated DiT，取 VLM 第 14 层中间表示作为 KV，保留语义完整性。

0.77B 参数，2.3GB 显存，在 LIBERO（94.8%）、MetaWorld（80.6%）、RoboTwin（37.8%）全面超过 π0（3.5B）和 SmolVLA（2.25B）。

### 复现计划

**环境要求（WSL2 / Linux）**

```bash
git clone https://github.com/MINT-SJTU/Evo-1.git && cd Evo-1
conda create -n Evo1 python=3.10 -y && conda activate Evo1
pip install -r requirements.txt
MAX_JOBS=8 pip install -v flash-attn --no-build-isolation  # 必须，跳过会降成功率

# LIBERO 评测环境（单独 conda env，需 python 3.8）
conda create -n libero python=3.8.13 -y && conda activate libero
cd LIBERO_evaluation && git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
cd LIBERO && pip install -r requirements.txt
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
pip install -e . && pip install websockets huggingface_hub
```

**Level 0 — 代码通读（无 GPU，半天）**

梳理数据流：`obs image → InternVL3-1B (layer 14) → integration module → cross-modulated DiT → action`

重点文件：
```
model/evo1.py          # integration module
model/action_expert.py # cross-attention DiT
train/stage1.py        # 冻结 backbone 训练
train/stage2.py        # 全量 fine-tune
```

**Level 1 — LIBERO 推理验证（GPU 4GB+，1-2天）**

```bash
huggingface-cli download MINT-SJTU/Evo1_LIBERO --local-dir ./checkpoints/libero

# 终端1
conda activate Evo1 && python scripts/Evo1_server.py
# 终端2
conda activate libero && cd LIBERO_evaluation && python libero_client_4tasks.py
```

对照论文 Table 1：libero_spatial 92.7% / libero_object 97.7% / libero_goal 96.3% / libero_long 92.3%

**Level 2 — 注意力语义对比可视化（GPU，2-3天）**

提取 Evo-1 两阶段训练前后的 layer 14 attention map，与 OpenVLA-OFT 对比，复现论文 Figure 2。
用自己的 Pick-and-Place obs 图验证"两阶段训练保留语义一致性"这个核心 claim。

```python
attn_storage = []
def hook(m, inp, out): attn_storage.append(out[1].detach().cpu())
for name, mod in model.named_modules():
    if "layer.14" in name and "attention" in name:
        mod.register_forward_hook(hook)
```

**Level 3 — MetaWorld 成功率复现（GPU，3-5天）**

```bash
huggingface-cli download MINT-SJTU/Evo1_MetaWorld --local-dir ./checkpoints/metaworld
python eval/eval_metaworld.py --checkpoint ./checkpoints/metaworld --difficulty hard --episodes 50
```

目标：Easy 89.2% / Medium 76.8% / Hard 77.2% / Very Hard 79.2%

**Level 4 — Pick-and-Place fine-tune（有机器人环境，1-2周）**

在 Franka Panda 采集的数据上跑两阶段 fine-tune，对比 OpenVLA-OFT 的成功率和推理延迟（Evo-1 只需 16.4Hz / 2.3GB）。

---

## 其他可关注项目与 PR 机会

### RoboClaw — 具身 AI 框架
[GitHub](https://github.com/MINT-SJTU/RoboClaw) · 395 stars · 更新至 2026.03

**项目定位**：模块化具身智能助手，支持感知-推理-执行链路。正在向社区征集贡献。

**PR 机会**（官方 README 明确征集）：
- ROS2 execution layer 集成 ← **与自己 ROS2/MoveIt2 经验直接对口**
- 仿真器适配（MuJoCo/Gazebo）
- 评测框架

**可做的切入点**：把 ROS2 + MoveIt2 的控制接口封装成 RoboClaw 的 execution adapter，提交 PR。

### Evo-RL — 真实机器人离线强化学习
[GitHub](https://github.com/MINT-SJTU/Evo-RL) · 529 stars · 更新至 2026.03

**项目定位**：SO-101 / AgileX PiPER 上的 offline RL pipeline，包含数据采集、价值函数训练、策略优化。

**PR 机会**：
- HuggingFace 模型权重上传工具（README 标注 "coming soon"）
- HuggingFace 数据集格式转换脚本（标注 "coming soon"）
- 自定义价值函数的文档补充

**可做的切入点**：实现数据集 → LeRobot 格式的转换脚本，和 Evo-1 的训练 pipeline 打通，提交 PR。

### LeRobot-Anything-U-Arm — 跨本体遥操作
[GitHub](https://github.com/MINT-SJTU/LeRobot-Anything-U-Arm) · 244 stars

**项目定位**：基于 LeRobot 的跨本体遥操作框架，支持多种机械臂。

**PR 机会**：
- 新机械臂适配（Franka Panda 接口）← 与 Pick-and-Place 项目对口
- 遥操作数据质量过滤脚本

---

## 参考资料

- [Evo-1 论文](Evo1_2511.04555.pdf)
- [Evo-1 reading note](Evo1_reading_note.md)
- [Evo-1 demo 计划](Evo1_demo_plan.md)
- [MINT-SJTU VLA SOTA 排行榜](https://github.com/MINT-SJTU/Evo-SOTA.io)
