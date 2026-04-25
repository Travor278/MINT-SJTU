# Evo-1 源码导览

本地仓库位置：`D:\SJTU\mint_sjtu\Evo-1`

GitHub 仓库：`https://github.com/MINT-SJTU/Evo-1`

当前分支：`main`

当前克隆到的最新提交：`d27d17a update readme`

## 总体结构

- `README.md`：官方安装、训练、评估和推理命令。
- `Evo_1/`：Evo-1 的核心实现，包含训练逻辑和 websocket 推理服务。
- `Evo_1/scripts/train.py`：主训练入口，使用 Accelerate + DeepSpeed。
- `Evo_1/scripts/Evo1.py`：顶层 `EVO1` 模型封装。
- `Evo_1/scripts/Evo1_server.py`：websocket 推理服务端，负责加载 checkpoint 和归一化统计量。
- `Evo_1/scripts/Evo1_client_xarm6.py`：xArm6 真实机器人推理客户端示例。
- `Evo_1/scripts/Evo1_client_aloha.py`：ALOHA 双臂推理客户端示例。
- `Evo_1/model/internvl3/internvl3_embedder.py`：InternVL3 图像/语言 embedding 路径。
- `Evo_1/model/action_head/flow_matching.py`：flow-matching 动作头。
- `Evo_1/dataset/lerobot_dataset_pretrain_mp.py`：LeRobot 格式数据集加载器，以及归一化/统计量处理逻辑。
- `Evo_1/dataset/config.yaml`：数据集路径和相机视角映射配置。
- `MetaWorld_evaluation/`：Meta-World websocket 客户端和任务定义。
- `LIBERO_evaluation/`：LIBERO websocket 客户端。
- `so100_evo1/`：随仓库附带的 LeRobot 代码树，以及 SO100/SO101 工作流的标定示例。

## 值得研究的分支

- `main`：官方核心版本，包含仿真评估、训练、xArm6/ALOHA 示例。
- `origin/evo1-flash`：根据 README 的说明，这是更快训练、降低 GPU 显存占用的分支。
- `origin/evo1-lerobot`：完整 LeRobot 集成分支；如果要研究 SO100/SO101，优先看这个分支。

## 主执行路径

训练流程：

1. 在 `Evo_1/dataset/config.yaml` 中配置数据。
2. 通过 `accelerate launch` 启动 `Evo_1/scripts/train.py`。
3. Stage 1 只训练 integration module 和 action head。
4. Stage 2 开启完整 VLM + action head 微调，并从 Stage 1 的 checkpoint 恢复。

推理流程：

1. 启动 `Evo_1/scripts/Evo1_server.py`。
2. 服务端从硬编码或默认 checkpoint 路径加载模型。
3. 客户端通过 websocket 发送 JSON observation。
4. observation 包含图像列表、图像 mask、机器人状态、动作 mask 和任务 prompt。
5. 服务端返回一个 action chunk。

仿真评估流程：

1. 从 `Evo_1` 启动 Evo-1 服务端。
2. 从 `MetaWorld_evaluation/mt50_evo1_client_prompt.py` 运行 Meta-World 客户端，或从 `LIBERO_evaluation/libero_client_4tasks.py` 运行 LIBERO 客户端。

## 环境要点

- README 中核心环境使用的 Python 版本是 `python=3.10`。
- 主要依赖部分固定在 `Evo_1/requirements.txt`。
- 关键依赖包括：`torch==2.5.1`、`torchvision==0.20.1`、`transformers==4.39.0`、`accelerate`、`deepspeed`、`websockets`、`opencv-python`、`flash-attn`。
- DeepSpeed 配置文件是 `Evo_1/ds_config.json`，使用 ZeRO stage 2 和 bf16。
- README 特别提醒：安装 `flash-attn` 时合理设置 `MAX_JOBS` 对机器人运动稳定性很重要。

## 建议深挖顺序

1. 论文和 README：先把概念模型和发布出来的代码对齐。
2. `Evo_1/scripts/Evo1.py`：理解 VLM embedding 和 action head 是怎么连接起来的。
3. `Evo_1/model/internvl3/internvl3_embedder.py`：查看图像预处理、prompt 处理和 embedding 提取逻辑。
4. `Evo_1/model/action_head/flow_matching.py`：理解 action chunk 的生成，以及 flow-matching 的 loss/sampling。
5. `Evo_1/dataset/lerobot_dataset_pretrain_mp.py`：理解预期的 LeRobot v2.1 数据结构、相机映射、mask、归一化和缓存。
6. `Evo_1/scripts/train.py`：串起配置、数据集、模型冻结、优化器参数组、checkpoint、resume 逻辑和分布式训练。
7. `Evo_1/scripts/Evo1_server.py` 加客户端脚本：追踪推理时的数据格式和部署假设。
8. `MetaWorld_evaluation/` 和 `LIBERO_evaluation/`：把 benchmark 客户端和服务端协议对应起来。
9. 在理解基础实现后，再比较 `main`、`evo1-flash` 和 `evo1-lerobot` 的差异。

## 下一轮可以先回答的问题

- 我们最先关心哪个目标：读懂论文、用自定义数据训练、跑 Meta-World/LIBERO 评估、xArm6/ALOHA 推理，还是 SO100/SO101 LeRobot 部署？
- 实际运行环境是纯 Windows、WSL/Linux，还是一台 CUDA Linux 机器？
- 现在要先搭环境，还是先做代码级架构阅读？
