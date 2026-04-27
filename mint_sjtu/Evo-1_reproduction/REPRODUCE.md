# Evo-1 Reproduction Checklist

This note records the exact local reproduction state for the Evo-1 LIBERO, Level 2 attention, and MetaWorld runs.

Date: 2026-04-26

## Source

Official repository:

```text
https://github.com/MINT-SJTU/Evo-1
```

Local official repo commit used in WSL:

```text
d27d17a83ebde66125b8a3df6cf5a26fc663660a
```

Local patches are stored under:

```text
mint_sjtu/Evo-1_reproduction/code/
```

The most important runtime patches are:

- `Evo_1/scripts/Evo1_server.py`: supports `EVO1_CKPT_DIR`, `EVO1_VLM_DIR`, and `EVO1_PORT`.
- `LIBERO_evaluation/libero_client_4tasks.py`: supports `EVO1_SERVER_URL`, `EVO1_LIBERO_EPISODES`, `EVO1_LIBERO_SEED`, and `EVO1_LIBERO_RUN_NAME`.
- `MetaWorld_evaluation/mt50_evo1_client_prompt.py`: supports `EVO1_METAWORLD_*` runtime environment variables.
- Level 2 probe scripts capture layer 13 attention with FlashAttention disabled.

## Environment Snapshot

### Evo1 server env

```text
python 3.10.20
torch 2.7.0+cu128
cuda 12.8
gpu NVIDIA GeForce RTX 5070 Laptop GPU
transformers 4.39.0
flash_attn 2.8.3
```

### LIBERO client env

```text
python 3.8.13
torch 1.11.0+cu113
robosuite 1.4.0
mujoco 3.2.3
websockets 13.1
```

### MetaWorld client env

```text
python 3.10.20
mujoco 3.8.0
websockets 16.0
metaworld unknown
```

## Checkpoint Hashes

```text
22dea26e5f3bdb1bcdf4a7faa323b7ff705248573274608b2c8c2a527ab166a9  checkpoints/libero/config.json
0c77c9f8c26ff75733ee7aee07f7e09950fd21e83f5862be812d1e031e365a04  checkpoints/libero/norm_stats.json
1bc3abd53046648ef72970f338753fadf5acf3f444117a7e3b36e1ff93f71ef0  checkpoints/libero/mp_rank_00_model_states.pt
f2e10705162e1d61745ea8d05dfb9e9d7560c024960b8292d5dee97b4ab8e554  checkpoints/metaworld/config.json
429a5a36ac3290ec835fb58f5887ca763f9316314ee4cc452629fb99a7618985  checkpoints/metaworld/norm_stats.json
38ce9c71ab10ea6ac1e51e451ad277288f269eb341a8e559c20c4f29f999c4d3  checkpoints/metaworld/mp_rank_00_model_states.pt
```

Checkpoint directory sizes:

```text
1.5G checkpoints/libero
1.5G checkpoints/metaworld
1.8G checkpoints/internvl3-1b
```

## Level 1 LIBERO

Start the server:

```bash
source ~/miniforge3/etc/profile.d/conda.sh
conda activate ~/tools/conda-envs/Evo1
cd ~/work/Evo-1/Evo_1

HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
EVO1_CKPT_DIR=~/work/Evo-1/checkpoints/libero \
EVO1_VLM_DIR=~/work/Evo-1/checkpoints/internvl3-1b \
EVO1_PORT=9000 \
python scripts/Evo1_server.py
```

Run the LIBERO client:

```bash
source ~/miniforge3/etc/profile.d/conda.sh
conda activate ~/tools/conda-envs/libero
cd ~/work/Evo-1/LIBERO_evaluation

CUDA_VISIBLE_DEVICES="" \
MUJOCO_GL=osmesa \
PYOPENGL_PLATFORM=osmesa \
EVO1_SERVER_URL=ws://127.0.0.1:9000 \
EVO1_LIBERO_EPISODES=10 \
EVO1_LIBERO_SEED=42 \
EVO1_LIBERO_RUN_NAME=Evo1_libero_all \
python libero_client_4tasks.py
```

Primary completed run:

```text
seed 42
400 episodes
370/400 = 92.5%
```

Additional completed seeds used distinct run names:

```text
EVO1_LIBERO_SEED=43 EVO1_LIBERO_RUN_NAME=Evo1_libero_seed43
EVO1_LIBERO_SEED=44 EVO1_LIBERO_RUN_NAME=Evo1_libero_seed44
```

Multi-seed result:

```text
seed 42: 370/400 = 92.5%
seed 43: 380/400 = 95.0%
seed 44: 372/400 = 93.0%
mean +/- std: 93.5% +/- 1.3%
```

Artifacts:

```text
mint_sjtu/Evo-1_reproduction/level1/log_file/Evo1_libero_all.txt
mint_sjtu/Evo-1_reproduction/level1/log_file/Evo1_libero_seed43.txt
mint_sjtu/Evo-1_reproduction/level1/log_file/Evo1_libero_seed44.txt
mint_sjtu/Evo-1_reproduction/reports/level1_multiseed_summary.md
mint_sjtu/Evo-1_reproduction/reports/level1_multiseed_summary.csv
```

The WSL-side seed 43 and seed 44 video directories each contain 400 mp4 files:

```text
~/work/Evo-1/LIBERO_evaluation/video_log_file/Evo1_libero_seed43/
~/work/Evo-1/LIBERO_evaluation/video_log_file/Evo1_libero_seed44/
```

Summarize the logs with:

```bash
python mint_sjtu/Evo-1_reproduction/scripts/summarize_libero_runs.py \
  mint_sjtu/Evo-1_reproduction/level1/log_file/Evo1_libero_all.txt \
  mint_sjtu/Evo-1_reproduction/level1/log_file/Evo1_libero_seed43.txt \
  mint_sjtu/Evo-1_reproduction/level1/log_file/Evo1_libero_seed44.txt \
  --csv mint_sjtu/Evo-1_reproduction/reports/level1_multiseed_summary.csv
```

## Level 2 Attention

The primary report uses:

```text
layer index 13
frame 60 fixed-frame panel
frame sweep best-by-case panel
Evo-1 final vs InternVL3 base panel
```

Figures and metrics are archived in:

```text
mint_sjtu/Evo-1_reproduction/figures/
```

## Level 3 MetaWorld

Start the server:

```bash
source ~/miniforge3/etc/profile.d/conda.sh
conda activate ~/tools/conda-envs/Evo1
cd ~/work/Evo-1/Evo_1

HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
EVO1_CKPT_DIR=~/work/Evo-1/checkpoints/metaworld \
EVO1_VLM_DIR=~/work/Evo-1/checkpoints/internvl3-1b \
EVO1_PORT=9100 \
python scripts/Evo1_server.py
```

Run the MetaWorld client:

```bash
source ~/miniforge3/etc/profile.d/conda.sh
conda activate ~/tools/conda-envs/metaworld
cd ~/work/Evo-1/MetaWorld_evaluation

MUJOCO_GL=osmesa \
PYOPENGL_PLATFORM=osmesa \
EVO1_SERVER_URL=ws://127.0.0.1:9100 \
EVO1_METAWORLD_EPISODES=10 \
EVO1_METAWORLD_EPISODE_HORIZON=400 \
EVO1_METAWORLD_TARGET_LEVEL=all \
EVO1_METAWORLD_SHOW_WINDOW=0 \
EVO1_METAWORLD_SAVE_VIDEO=0 \
EVO1_METAWORLD_SAVE_IMAGE=0 \
EVO1_METAWORLD_INSPECT=0 \
python -u mt50_evo1_client_prompt.py
```

Completed run:

```text
seed 4042
500 episodes
420/500 = 84.0% raw success
80.7% difficulty average
```

Archived log:

```text
mint_sjtu/Evo-1_reproduction/level3_metaworld_mt50_10ep.log
```
