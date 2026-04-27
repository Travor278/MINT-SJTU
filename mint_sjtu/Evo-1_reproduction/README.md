# Evo-1 复现文件夹

WSL Ubuntu 22.04 里跑出来的产物同步到 D 盘，方便在 Windows 侧查看和归档。

## 文件结构

- `reports/`
  - `LEVEL1_REPRODUCTION_REPORT.md`：Level 1 环境、命令、LIBERO 完整结果
  - `LEVEL2_ATTENTION_NOTES.md`：Level 2 hook 目标、捕获方法、产物说明
- `code/`
  - 从 WSL `/home/Travor/work/Evo-1` 里 patch 过的文件和新写的 probe 脚本
  - 这是快照，不是完整的独立仓库，不能单独运行
- `level1/`
  - `log_file/`：LIBERO 评估日志（三个 seed）
  - `video_log_file/Evo1_libero_all/`：Level 1 正式视频
  - `video_log_file/Evo1_libero_all_smoke_20260424_001635/`：smoke test 视频
- `level2/level2_attention_outputs/`
  - 单 case 和 success/failure batch 的 attention 产物
  - 包括 manifests、per-case metadata、attention tensors、overlay 图和汇总面板
- `figures/`：Level 2 最终展示图

## Level 1 结果

跑了 3 个 seed（42/43/44），共 1200 episodes。

seed=42 单次完整结果：

| Suite | 成功率 |
|---|---:|
| `libero_spatial` | 88 / 100 |
| `libero_object` | 97 / 100 |
| `libero_goal` | 94 / 100 |
| `libero_10` | 91 / 100 |
| **Overall** | **370 / 400 = 92.5%** |

三个 seed 均值：**93.5% ± 1.3%**（官方 94.8%）

## Level 2 说明

hook 挂在：

```python
language_model.model.layers.13.self_attn
```

Evo-1 保留了 InternVL3 / Qwen 的前 14 层，所以论文里说的"layer 14"在代码里是零起始的 index 13。FlashAttention 不返回 attention weights，probe 时要关掉。

batch 对比面板（success vs failure）存在：

```
level2/level2_attention_outputs/batch_success_failure/level2_success_failure_panel.png
```

## Level 3 结果

MetaWorld MT50，500 episodes，难度分组平均 **80.7%**（官方 80.6%）。

## 注意：checkpoint 没有同步进来

模型文件太大，没有放进这个 git 工作区，实际位置：

- WSL Evo-1 仓库：`/home/Travor/work/Evo-1`
- LIBERO checkpoint：`/home/Travor/work/Evo-1/checkpoints/libero`
- MetaWorld checkpoint：`/home/Travor/work/Evo-1/checkpoints/metaworld`
- InternVL3 本地模型：`/home/Travor/work/Evo-1/checkpoints/internvl3-1b`
- Windows HF 缓存：`D:\Code\Work\SJTU_hf`
