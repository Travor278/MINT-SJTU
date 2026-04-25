# Evo-1 Level 2 Attention Notes

Date: 2026-04-24

## Goal

Level 2 aims to inspect whether Evo-1 preserves semantic attention after fine-tuning. The immediate target is the InternVL3 language-model attention around the retained 14th layer.

## Confirmed Module Target

The local InternVL3 model is:

```text
InternVLChatModel
language_model: Qwen2ForCausalLM
```

The original language model has 24 decoder layers. Evo-1 truncates it to the first 14 layers in:

```text
Evo_1/model/internvl3/internvl3_embedder.py
```

Therefore the "layer 14" target corresponds to retained index `13`:

```text
language_model.model.layers.13.self_attn
```

Important: flash attention does not expose normal attention weights. The Level 2 probe disables flash attention by setting `use_flash_attn=False` when loading the model.

## Added Probe Entry Point

Script:

```text
Evo_1/scripts/level2_attention_probe.py
```

This script:

- Loads the Evo-1 LIBERO checkpoint.
- Loads local InternVL3 from `checkpoints/internvl3-1b`.
- Applies the Evo-1 checkpoint state dict, so the captured VLM weights are the fine-tuned Evo-1 weights.
- Runs one image/prompt through the embedder.
- Saves raw layer attention, mean-head attention, token-0 attention, and a quick 16x16 image-token heatmap.

Default command:

```bash
cd ~/work/Evo-1/Evo_1
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
~/miniforge3/bin/conda run -n Evo1 \
  python scripts/level2_attention_probe.py \
  --output-dir ~/work/Evo-1/level2_attention_outputs/probe_spatial_task1_ep1_frame0
```

By default it uses the first frame from:

```text
LIBERO_evaluation/video_log_file/Evo1_libero_all/libero_spatial/task1_episode1.mp4
```

The prompt is:

```text
pick up the black bowl between the plate and the ramekin and place it on the plate
```

## Probe Output

Output directory:

```text
level2_attention_outputs/probe_spatial_task1_ep1_frame0
```

Files:

- `layer13_attention.pt`
- `layer13_attention_mean.png`
- `layer13_token0_attention.png`
- `layer13_image1_token0_heatmap.png`
- `layer13_image1_overlay.png`
- `input_image1.png`
- `metadata.json`

Captured shapes:

```text
attention: [1, 14, 1024, 1024]
attention_mean_heads: [1024, 1024]
cls_to_tokens: [1024]
embedding: [1, 1024, 896]
```

Image-token mapping:

```text
num_image_token: 256
image token locations: 6..261
grid: 16 x 16
```

## Batch Success/Failure Probe

Batch script:

```text
Evo_1/scripts/level2_batch_attention_probe.py
```

Figure-panel script:

```text
Evo_1/scripts/level2_make_figure_panel.py
```

Command used:

```bash
cd ~/work/Evo-1/Evo_1
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
~/miniforge3/bin/conda run -n Evo1 \
  python scripts/level2_batch_attention_probe.py \
  --max-per-status 2 \
  --output-dir ~/work/Evo-1/level2_attention_outputs/batch_success_failure
```

This parsed all 400 Level 1 episodes and selected 16 comparison cases:

- 2 successes and 2 failures per suite.
- Suites: `libero_spatial`, `libero_object`, `libero_goal`, `libero_10`.
- Raw full attention is not saved by default for batch mode; compact tensors and visualizations are saved.

Batch output:

```text
level2_attention_outputs/batch_success_failure
```

Key files:

- `manifest.csv`
- `manifest.json`
- `layer13_overlay_contact_sheet.png`
- `level2_success_failure_panel.png`
- `layer13_attention_summary.csv`

Each case directory contains:

- `input_image1.png`
- `layer13_attention.pt`
- `layer13_attention_mean.png`
- `layer13_image1_token0_heatmap.png`
- `layer13_image1_overlay.png`
- `layer13_image1_text_to_image_heatmap.png`
- `layer13_image1_text_to_image_overlay.png`
- `layer13_token0_attention.png`
- `metadata.json`

The contact sheet uses text-to-image attention overlays, which are more informative than token-0 overlays.

The figure panel uses one success/failure pair per suite and places original frames next to text-to-image attention overlays:

```bash
cd ~/work/Evo-1/Evo_1
~/miniforge3/bin/conda run -n Evo1 \
  python scripts/level2_make_figure_panel.py \
  --batch-dir ~/work/Evo-1/level2_attention_outputs/batch_success_failure \
  --output ~/work/Evo-1/level2_attention_outputs/batch_success_failure/level2_success_failure_panel.png
```

## Next Technical Step

The current probe proves that attention capture and first success/failure overlays work. A first figure-style success/failure panel has also been generated. To approximate Figure 2 more closely, the next step is to probe more informative mid-trajectory frames instead of only frame 0.

Recommended sequence:

1. Inspect `layer13_overlay_contact_sheet.png` and `level2_success_failure_panel.png`.
2. Probe additional frame indices within the same videos, because frame 0 may precede the most informative interaction.
3. Build a polished figure panel from the best frames.
4. Add cleaner object-centric captions or crop panels if needed.
5. For a true paper-style comparison, add either a Stage 1 checkpoint or an OpenVLA-OFT baseline. At present we only have the final Evo-1 LIBERO checkpoint.
