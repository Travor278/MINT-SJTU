# Evo-1 Reproduction Sync

This folder is a D-drive snapshot of the Evo-1 Level 1 reproduction and Level 2 attention-probe work produced in WSL `Ubuntu2204`.

## Contents

- `reports/`
  - `LEVEL1_REPRODUCTION_REPORT.md`: Level 1 environment, commands, and formal LIBERO results.
  - `LEVEL2_ATTENTION_NOTES.md`: Level 2 hook target, capture method, and generated artifact notes.
- `code/`
  - Minimal patched files and new probe scripts copied from `/home/Travor/work/Evo-1`.
  - This is a snapshot for review and archiving, not a standalone full checkout.
- `level1/`
  - `log_file/`: LIBERO evaluation logs.
  - `video_log_file/Evo1_libero_all/`: formal Level 1 videos.
  - `video_log_file/Evo1_libero_all_smoke_20260424_001635/`: smoke-test videos.
- `level2/level2_attention_outputs/`
  - Single-case and success/failure batch attention captures.
  - Includes manifests, per-case metadata, attention tensors, overlays, and summary panels.

## Level 1 Result

Formal run: 400 episodes.

- `libero_spatial`: 88 / 100
- `libero_object`: 97 / 100
- `libero_goal`: 94 / 100
- `libero_10`: 91 / 100
- Overall: 370 / 400 = 92.5%

## Level 2 Status

The attention hook target was confirmed as:

```text
language_model.model.layers.13.self_attn
```

Evo-1 keeps the first 14 InternVL3/Qwen language layers, so the paper-style "layer 14" target corresponds to zero-based index `13` in the retained module list.

The batch attention panel is available at:

```text
level2/level2_attention_outputs/batch_success_failure/level2_success_failure_panel.png
```

## Large Files Not Copied Here

Model checkpoints were not copied into this git working tree to avoid turning the repo into a multi-GB artifact store.

Existing relevant locations:

- WSL Evo-1 checkout: `/home/Travor/work/Evo-1`
- WSL LIBERO checkpoint: `/home/Travor/work/Evo-1/checkpoints/libero`
- WSL InternVL3 local model: `/home/Travor/work/Evo-1/checkpoints/internvl3-1b`
- Windows HF staging cache: `D:\Code\Work\SJTU_hf`
