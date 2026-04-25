# Evo-1 Level 1 LIBERO Reproduction Report

Date: 2026-04-24

## Status

Level 1 is reproduced end to end on `Ubuntu2204`.

- Server loads from local checkpoints in offline mode.
- LIBERO client completed all four suites with `10 episodes/task`.
- Output includes the official run log and 400 episode videos.
- GPU memory was released after the run.

## Machine And Environments

Host side:

- WSL distro: `Ubuntu2204`
- Workspace: `/home/Travor/work/Evo-1`
- GPU: `NVIDIA GeForce RTX 5070 Laptop GPU`, 8 GB VRAM

`Evo1` conda environment:

- Python: 3.10
- PyTorch: `2.7.0+cu128`
- Transformers: `4.39.0`
- FlashAttention: `2.8.3`

`libero` conda environment:

- Python: 3.8.13
- PyTorch: `1.11.0+cu113`
- robosuite: `1.4.0`
- mujoco: `3.2.3`
- LIBERO: PyPI package fallback, with local compatibility patch for `torch.load`

## Local Materials

Evo-1 LIBERO checkpoint:

```text
/home/Travor/work/Evo-1/checkpoints/libero
```

Required files present:

- `checkpoint.json`
- `config.json`
- `norm_stats.json`
- `mp_rank_00_model_states.pt`

InternVL3 local model:

```text
/home/Travor/work/Evo-1/checkpoints/internvl3-1b
```

LIBERO assets:

```text
/home/Travor/tools/conda-envs/libero/lib/python3.8/site-packages/libero/libero/assets
```

Artifacts:

```text
/home/Travor/work/Evo-1/LIBERO_evaluation/log_file/Evo1_libero_all.txt
/home/Travor/work/Evo-1/LIBERO_evaluation/video_log_file/Evo1_libero_all
```

There are 400 episode videos in `video_log_file/Evo1_libero_all`.

## Local Code Changes

`Evo_1/scripts/Evo1_server.py`

- `EVO1_CKPT_DIR`: checkpoint directory override.
- `EVO1_PORT`: websocket port override.
- `EVO1_VLM_DIR`: local InternVL3 directory override to avoid online Hugging Face access.

`LIBERO_evaluation/libero_client_4tasks.py`

- Sets `MUJOCO_GL=osmesa` before importing LIBERO.
- `EVO1_SERVER_URL`: websocket endpoint override.
- `EVO1_LIBERO_EPISODES`: episode count override.

Installed package patch:

- In `libero` site-packages, changed `torch.load(init_states_path, weights_only=False)` to `torch.load(init_states_path)` for compatibility with `torch==1.11.0`.

## Reproduction Commands

Start server:

```bash
cd ~/work/Evo-1/Evo_1
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
EVO1_CKPT_DIR=~/work/Evo-1/checkpoints/libero \
EVO1_VLM_DIR=~/work/Evo-1/checkpoints/internvl3-1b \
EVO1_PORT=9000 \
~/miniforge3/bin/conda run -n Evo1 python scripts/Evo1_server.py
```

Run Level 1 evaluation:

```bash
cd ~/work/Evo-1/LIBERO_evaluation
CUDA_VISIBLE_DEVICES="" \
MUJOCO_GL=osmesa \
EVO1_SERVER_URL=ws://127.0.0.1:9000 \
EVO1_LIBERO_EPISODES=10 \
~/miniforge3/bin/conda run -n libero python libero_client_4tasks.py
```

## Results

| Suite | Success | Success Rate | Paper / README Reference |
|---|---:|---:|---:|
| `libero_spatial` | 88/100 | 88.0% | 92.7% |
| `libero_object` | 97/100 | 97.0% | 97.7% |
| `libero_goal` | 94/100 | 94.0% | 96.3% |
| `libero_10` / long | 91/100 | 91.0% | 92.3% |
| Overall | 370/400 | 92.5% | 94.8% reported overall LIBERO |

Average steps from the run:

| Suite | Average Steps |
|---|---:|
| `libero_spatial` | 7.40 |
| `libero_object` | 10.16 |
| `libero_goal` | 8.00 |
| `libero_10` | 18.50 |

## Lower-Scoring Tasks

Task-level summaries below list only tasks with failures.

`libero_spatial`

| Task | Result |
|---|---:|
| Task 1 | 9/10 |
| Task 2 | 7/10 |
| Task 4 | 9/10 |
| Task 6 | 7/10 |
| Task 8 | 9/10 |
| Task 9 | 9/10 |
| Task 10 | 8/10 |

`libero_object`

| Task | Result |
|---|---:|
| Task 3 | 9/10 |
| Task 6 | 8/10 |

`libero_goal`

| Task | Result |
|---|---:|
| Task 2 | 9/10 |
| Task 4 | 8/10 |
| Task 7 | 8/10 |
| Task 10 | 9/10 |

`libero_10`

| Task | Result |
|---|---:|
| Task 3 | 8/10 |
| Task 5 | 9/10 |
| Task 7 | 8/10 |
| Task 8 | 8/10 |
| Task 9 | 9/10 |
| Task 10 | 9/10 |

## Notes

- Direct WSL access to `huggingface.co` was unavailable. Model and asset downloads were completed from Windows through the existing local proxy, then copied into WSL ext4 paths.
- The server was verified with a dummy websocket request before running LIBERO; it returned a `50 x 24` action chunk.
- Smoke test completed before the formal run: 39/40 episodes succeeded.
- Formal run peak steady server VRAM was around 2.2 GB.
- Final GPU usage after stopping the server: 0 MiB.

## Level 2 Handoff

The Level 1 environment is ready for Level 2. The next step is to add an attention capture path for the local InternVL3 model and confirm the actual retained language-layer module names. The Evo-1 code truncates the language model to the first 14 layers, so the target "layer 14" is expected to correspond to retained index `13`, not a literal `layer.14` module.
