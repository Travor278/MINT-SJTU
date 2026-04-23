# Interleave-VLA Demo 计划

官方代码：https://github.com/Interleave-VLA/Interleave-VLA
HuggingFace checkpoint：Interleave-OpenVLA（见 repo README）

---

## Level 0 — 指令格式可视化（无 GPU，1天）

**目标**：展示 interleaved instruction 的构造过程，证明理解了论文核心思路。

**做法**：
1. 从 Pick-and-Place 任务轨迹里取一帧，用 OWLv2 或手动裁剪目标物体图片
2. 写脚本展示 tokenizer 对两种指令的编码差异：

```python
# text-only
tokens_text = tokenizer("pick up the red cube and place it in the box")

# interleaved（示意，实际用 <BOI>/<EOI> special tokens）
tokens_interleaved = tokenizer.encode_interleaved(
    "pick up ", image_cube, " and place it in ", image_box
)

print(f"text-only token count: {len(tokens_text)}")
print(f"interleaved token count: {len(tokens_interleaved)}")  # 多 512 个 patch tokens
```

3. 可视化两种 token 序列的结构图（文字段 vs 图像段标色）

**输出**：一张 token 序列结构对比图 + 简短说明

---

## Level 1 — 注意力对比可视化（无 GPU，2-3天）

**目标**：复现论文 Figure 3 的三种失败模式，在自己的场景上验证。

**做法**：
1. 加载 OpenVLA-OFT（你已有）
2. 构造一个歧义性任务（场景里有两个相似物体）
3. 提取最后几层 cross-attention 对 observation patch 的权重
4. 可视化 heatmap，观察是否出现 attentional bias / diffused attention

```python
# hook 提取 attention weights
def get_attn_hook(storage):
    def hook(module, input, output):
        storage.append(output[1].detach().cpu())  # attn weights
    return hook

handles = []
for layer in model.language_model.model.layers[-4:]:
    h = layer.self_attn.register_forward_hook(get_attn_hook(attn_storage))
    handles.append(h)

# run forward, then visualize
```

**输出**：failure mode 热力图，与论文 Figure 3 对比

---

## Level 2 — Interleave-OpenVLA 推理（需要 GPU，3-5天）

**目标**：加载官方 checkpoint，对比 text-only vs interleaved 的推理输出。

**环境要求**：
- GPU 16GB+（checkpoint 约 7B 参数）
- `pip install -e .`（按官方 repo）

**做法**：
```bash
# 安装
git clone https://github.com/Interleave-VLA/Interleave-VLA
cd Interleave-VLA/openvla
pip install -e .

# 下载 checkpoint
huggingface-cli download <interleave-openvla-checkpoint>
```

```python
from interleave_openvla import InterleaveOpenVLA

model = InterleaveOpenVLA.from_pretrained(checkpoint_path)

# text-only
action_text = model.predict("pick up the red cube", obs_image)

# interleaved
action_interleaved = model.predict_interleaved(
    ["pick up ", crop_cube, " and place it in ", crop_box],
    obs_image
)

print("text action:       ", action_text)
print("interleaved action:", action_interleaved)
```

**输出**：两种模式下的 action 向量对比，可视化轨迹差异

---

## Level 3 — SimplerEnv 成功率对比（需要 GPU + SimplerEnv，1周）

**目标**：复现论文 Table 1 的部分数字，在 novel object 设置下跑成功率。

**环境要求**：
```bash
pip install simpler-env
# 参考 https://github.com/simpler-env/SimplerEnv
```

**做法**：
```bash
# text-only baseline
python eval_simpler.py --model openvla --task novel_object --episodes 50

# interleaved
python eval_simpler.py --model interleave_openvla --task novel_object --episodes 50
```

对比指标：
- In-domain success rate
- Novel object success rate（论文：30.2% → 55.7%）
- 注意力热力图（有/无 instruction image）

**输出**：小表格 + 视频对比，能直接对应论文 Table 1

---

## Level 4 — 接入自己的 Pick-and-Place（完整复现，2周）

**目标**：把 Interleave-VLA 接到 Franka + ROS 2 的实际任务里，构造真实交错指令。

**做法**：
1. 从机械臂 camera feed 实时裁剪目标物体（用 OWLv2 或手动标注）
2. 构造交错指令，送入 Interleave-OpenVLA
3. 对比同一任务下 text-only OpenVLA vs Interleave-OpenVLA 的成功率
4. 记录 novel object（没出现在训练集里的物体）的成功率差距

**输出**：真实机器人对比视频，是附在邮件里最有力的 demo

---

## 现实建议

没有 GPU 的情况下，**Level 0 + Level 1 已经足够作为 reading note 的配套材料**——Level 1 的注意力可视化用 OpenVLA-OFT（你已有）就能跑，不需要新 checkpoint。

有 GPU 后优先做 Level 2，能跑通推理就可以出结果。
