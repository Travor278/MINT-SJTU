# SJTU RA 申请材料

赵波组（MINT-SJTU，上海交通大学）论文调研与复现记录。  
方向：轻量 VLA、具身操作、真实机器人离线 RL。

---

## Evo-1 复现（Level 1–3 已完成）

**Evo-1: Lightweight Vision-Language-Action Model with Preserved Semantic Alignment**  
CVPR 2026 · [arXiv 2511.04555](https://arxiv.org/abs/2511.04555) · [GitHub](https://github.com/MINT-SJTU/Evo-1)

LIBERO 三个 seed 均值 **93.5% ± 1.3%**（官方 94.8%），MetaWorld MT50 **80.7%** difficulty average（官方 80.6%）。

| 文件 | 内容 |
|---|---|
| [mint_sjtu/README.md](mint_sjtu/README.md) | 复现记录总览（环境、命令、各 level 结果） |
| [mint_sjtu/Evo1_reading_note.md](mint_sjtu/Evo1_reading_note.md) | Evo-1 reading note：两阶段训练 + 语义对齐保持机制 |
| [mint_sjtu/Evo1_demo_plan.md](mint_sjtu/Evo1_demo_plan.md) | 完整复现记录（结果、attention 分析、问题说明） |
| [mint_sjtu/Evo1_source_map.md](mint_sjtu/Evo1_source_map.md) | 源码导览 |
| [mint_sjtu/Evo-1_reproduction/](mint_sjtu/Evo-1_reproduction/) | 产物归档（日志、视频、attention 图） |

论文 PDF：[mint_sjtu/Evo1_2511.04555.pdf](mint_sjtu/Evo1_2511.04555.pdf)
