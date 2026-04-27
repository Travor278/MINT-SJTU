# Evo-1 源码导览

本地仓库位置：`D:\Code\Work\SJTU\mint_sjtu\Evo-1`

GitHub 仓库：`https://github.com/MINT-SJTU/Evo-1`

当前分支：`main`，commit `d27d17a update readme`

## 总体结构

- `README.md`：官方安装、训练、评估和推理命令。
- `Evo_1/`：Evo-1 核心实现，包含训练逻辑和 websocket 推理服务。
- `Evo_1/scripts/train.py`：主训练入口，用 Accelerate + DeepSpeed。
- `Evo_1/scripts/Evo1.py`：顶层 `EVO1` 模型封装。
- `Evo_1/scripts/Evo1_server.py`：websocket 推理服务端，加载 checkpoint 和归一化统计量。
- `Evo_1/scripts/Evo1_client_xarm6.py`：xArm6 真实机器人推理客户端示例。
- `Evo_1/scripts/Evo1_client_aloha.py`：ALOHA 双臂推理客户端示例。
- `Evo_1/model/internvl3/internvl3_embedder.py`：InternVL3 图像/语言 embedding 路径。
- `Evo_1/model/action_head/flow_matching.py`：flow-matching 动作头。
- `Evo_1/dataset/lerobot_dataset_pretrain_mp.py`：LeRobot 格式数据集加载器及归一化处理。
- `Evo_1/dataset/config.yaml`：数据集路径和相机视角映射配置。
- `MetaWorld_evaluation/`：Meta-World websocket 客户端和任务定义。
- `LIBERO_evaluation/`：LIBERO websocket 客户端。
- `so100_evo1/`：附带的 LeRobot 代码树，以及 SO100/SO101 工作流的标定示例。

## 几个分支

- `main`：官方核心版本，包含仿真评估、训练、xArm6/ALOHA 示例。这次复现用的就是这个。
- `origin/evo1-flash`：根据 README 说明，训练更快、显存占用更低。
- `origin/evo1-lerobot`：完整 LeRobot 集成；要研究 SO100/SO101 就看这个。

## 主执行路径

训练流程：

1. 在 `Evo_1/dataset/config.yaml` 配置数据。
2. `accelerate launch` 启动 `Evo_1/scripts/train.py`。
3. Stage 1 只训 integration module 和 action head。
4. Stage 2 全模型微调，从 Stage 1 checkpoint 恢复。

推理流程：

1. 启动 `Evo_1/scripts/Evo1_server.py`。
2. 服务端从 checkpoint 路径加载模型（支持环境变量 `EVO1_CKPT_DIR`）。
3. 客户端通过 websocket 发 JSON observation。
4. observation 包含图像列表、图像 mask、机器人状态、动作 mask 和任务 prompt。
5. 服务端返回 action chunk。

仿真评估：

1. 从 `Evo_1` 启动 Evo-1 服务端。
2. 从 `MetaWorld_evaluation/mt50_evo1_client_prompt.py` 跑 Meta-World，或从 `LIBERO_evaluation/libero_client_4tasks.py` 跑 LIBERO。

## 环境配置

- 核心环境：Python 3.10，主要依赖在 `Evo_1/requirements.txt`。
- 关键依赖：`torch==2.5.1`、`torchvision==0.20.1`、`transformers==4.39.0`、`accelerate`、`deepspeed`、`websockets`、`opencv-python`、`flash-attn`。
- DeepSpeed 配置：`Evo_1/ds_config.json`，ZeRO stage 2 + bf16。
- `flash-attn` 安装时要设 `MAX_JOBS`，文档特别强调这一步，实测不装会影响成功率。
- LIBERO client 需要单独 Python 3.8 环境，和 server 环境完全隔离。

## 阅读顺序

1. 论文和 README：先把概念和发布的代码对齐。
2. `Evo_1/scripts/Evo1.py`：看 VLM embedding 和 action head 怎么连接。
3. `Evo_1/model/internvl3/internvl3_embedder.py`：图像预处理、prompt 处理、embedding 提取。
4. `Evo_1/model/action_head/flow_matching.py`：action chunk 生成、flow-matching 的 loss 和 sampling。
5. `Evo_1/dataset/lerobot_dataset_pretrain_mp.py`：LeRobot v2.1 数据结构、相机映射、mask、归一化和缓存。
6. `Evo_1/scripts/train.py`：配置、数据集、模型冻结、优化器参数组、checkpoint、resume 和分布式训练。
7. `Evo_1/scripts/Evo1_server.py` + 客户端脚本：推理时的数据格式和部署细节。
8. `MetaWorld_evaluation/` 和 `LIBERO_evaluation/`：benchmark 客户端和服务端协议。
9. 理解基础实现后再比较 `main`、`evo1-flash`、`evo1-lerobot` 的差异。

## 待探索

- 14 层是硬编码（`layers = layers[:14]`），论文没有给 layer selection 的消融数据，不知道为什么偏偏是 14。
- `time_emb` 注入方式是加在 FFN 前，不是标准 DiT AdaLN，这么设计的原因不太清楚。
- `get_action()` 里有很多 debug `print`，实际部署 16.4Hz 是怎么测出来的，这些 print 关了没有？
- train loop 里 `target_velocity` 在 loss 里好像没有 mask，但 `pred_velocity` 有；构造阶段应该已经处理了，但没有完全确认。
