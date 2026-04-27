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

### 复现记录

**环境（WSL2 / Linux）**

```bash
git clone https://github.com/MINT-SJTU/Evo-1.git && cd Evo-1
conda create -n Evo1 python=3.10 -y && conda activate Evo1
pip install -r requirements.txt
MAX_JOBS=8 pip install -v flash-attn --no-build-isolation  # 必须装，跳过会掉成功率

# LIBERO 评测环境（单独 conda env，需 python 3.8）
conda create -n libero python=3.8.13 -y && conda activate libero
cd LIBERO_evaluation && git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
cd LIBERO && pip install -r requirements.txt
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
pip install -e . && pip install websockets huggingface_hub
```

**Level 1 — LIBERO 推理**

三个 seed（42/43/44），共 1200 episodes，均值 **93.5% ± 1.3%**，接近官方 94.8%。

```bash
huggingface-cli download MINT-SJTU/Evo1_LIBERO --local-dir ./checkpoints/libero

# 终端1
conda activate Evo1 && python scripts/Evo1_server.py
# 终端2
conda activate libero && cd LIBERO_evaluation && python libero_client_4tasks.py
```

对照论文 Table 1：libero_spatial 92.7% / libero_object 97.7% / libero_goal 96.3% / libero_long 92.3%

**Level 2 — Attention 可视化**

提取 Evo-1 的 layer 13 attention（论文里的"layer 14"，零起始索引就是 13），用 InternVL3 base 做对照。结果见 `Evo-1_reproduction/figures/`，fine-tune 后 attention 确实更集中，但 fail case 也集中，说明 attention 聚焦和成功率没有直接因果关系。

```python
# hook 目标
language_model.model.layers.13.self_attn
```

**Level 3 — MetaWorld 成功率**

MT50 全 50 任务，500 episodes，难度分组平均 **80.7%**，和官方 80.6% 基本一致。

```bash
huggingface-cli download MINT-SJTU/Evo1_MetaWorld --local-dir ./checkpoints/metaworld
# 启动 server 后用 MetaWorld_evaluation/mt50_evo1_client_prompt.py 跑
```

**Level 4 — 自定义数据 fine-tune（待做）**

在 Franka Panda 采集的数据上跑两阶段 fine-tune，对比 OpenVLA-OFT 的成功率和推理延迟（Evo-1 只需 16.4Hz / 2.3GB）。

## 参考资料

- [Evo-1 论文](Evo1_2511.04555.pdf)
- [Evo-1 reading note](Evo1_reading_note.md)
- [Evo-1 源码导览](Evo1_source_map.md)
- [Evo-1 复现记录](Evo1_demo_plan.md)
- [MINT-SJTU VLA SOTA 排行榜](https://github.com/MINT-SJTU/Evo-SOTA.io)
