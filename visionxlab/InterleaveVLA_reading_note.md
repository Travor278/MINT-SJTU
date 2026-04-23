# Reading Note: Interleave-VLA
**Enhancing Robot Manipulation with Interleaved Image-Text Instructions**
arXiv 2505.02152 · ICLR 2026

---

## 问题与动机

text-only VLA（如 OpenVLA）在 out-of-domain 任务上泛化差。论文通过可视化注意力发现三种具体失败模式：

**1. Attentional Bias（注意力偏置）**
模型聚焦于语义相关但错误的对象。例如指令"抓红色罐子"，模型把注意力压在 Red Bull 而非可口可乐上，因为训练中"红色"和某类容器高度共现。

**2. Diffused Attention（注意力弥散）**
注意力均匀散布在整个场景，模型不确定目标在哪里，无法 ground 语言描述到具体物体。新类别任务中最常见。

**3. Attention Leakage（注意力泄漏）**
模型定位到了正确目标，但注意力区域扩散到背景，抓取动作因此不精确。

这三种失败都来源于同一个根因：**文本指令的语义歧义**——语言描述对"哪个物体"天然有不确定性，而视觉图像是精确的。

---

## 方法：交错图文指令

### 核心思路

把指令从纯文字改成图文交错格式：
```
text-only: "pick up the red cube and place it in the box"
interleaved: "pick up <BOI>[red_cube图像]<EOI> and place it in <BOI>[box图像]<EOI>"
```
instruction image 直接指向目标，消除语义歧义。

### Tokenizer 适配（轻量改动）

只需在词表里加两个特殊 token：`<BOI>`（image 起始）和 `<EOI>`（image 结束）。整个序列变成：
```
<BOI> patch_1 … patch_256 <EOI> text <BOI> patch_257 … patch_512 <EOI> text
```
VLA backbone 不需要任何结构改动，只改 tokenizer 和 input processor。

### 数据集：Interleaved X-Embodiment（210k episodes）

从 Open X-Embodiment 自动生成，三步流程：
1. **Qwen2.5** 解析指令，提取目标物体名称
2. **OWLv2** 在轨迹帧上做 open-vocabulary 检测，裁剪物体图像（82.6% 精度）
3. **Qwen2.5-VL + SAM** 验证 + 精化分割（95.6% 联合精度）

额外混入互联网图片（搜索引擎抓取的物体图），增加指令图的多样性。

### 训练

沿用原始 VLA 的 flow matching objective 和超参，唯一变化是 input 格式。消融证明这很重要——"Partial"（训练时用交错，推理时用文字）比完整 Interleave-VLA 差约 15pp，说明 interleaved evaluation 不可省。

---

## 关键结果

### SimplerEnv（仿真）

| 设置 | Text-VLA | Interleave-VLA |
|---|---|---|
| In-domain | 69.2% | **71.0%** |
| Novel objects | 30.2% | **55.7%** (+25.5pp) |
| Novel categories | 21.0% | **53.0%** (+32pp, **2.5×**) |

### 真实机械臂（FANUC）

| 设置 | Text-VLA | Interleave-VLA |
|---|---|---|
| Out-of-domain lift | ~25% | **71%** |
| Out-of-domain pick-place | ~15% | **38%** |

---

## 与我的工作的关联

我当前的 Pick-and-Place 系统用 OpenVLA-OFT + 纯文字指令。Interleave-OpenVLA 提供了可以直接加载的 HuggingFace checkpoint，复现路径是：

1. 把 Pick-and-Place 任务的目标物体从轨迹帧里裁剪出来，构造交错指令
2. 加载 Interleave-OpenVLA checkpoint，对比 text-only vs interleaved 的行为差异
3. 可视化两种指令下模型对目标 patch 的注意力分布

这也是论文最核心的 insight 的直接验证。

---

## 我的问题与思考

**1. Instruction image 的质量依赖**
论文用 OWLv2 裁剪目标物体，精度 82.6%（+SAM 后 95.6%）。但如果指令图裁错了（背景残留、目标不完整），模型是否反而引入 noise？论文没有分析 instruction image 质量对下游成功率的敏感性。

**2. Multi-object 泛化**
实验中每条指令通常只有 1-2 个 instruction image。现实操作可能需要多个参照物（"把 A 放到 B 和 C 之间"），序列长度会暴增（每张图 256 patches），计算代价如何？

**3. 为什么 Partial 比 Full 差这么多**
训练时用交错、推理时用文字，性能差 ~15pp。这说明模型在训练中学到的是"条件在图像上"的 policy，而不是通用的语义理解——和 text-only VLA 的问题不同，但同样是过拟合，只是过拟合到了图像模态。这个现象值得深究。
