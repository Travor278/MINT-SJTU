# VLA-IAP · Interaction-Aligned Visual Token Pruning

Training-free visual token pruning for Vision-Language-Action models.
Reproduces and extends **VLA-IAP** (arXiv 2603.22991, March 2026).

## Core idea

At 70% token retention:
- Success rate drop ≈ −1.5 pp on LIBERO-Spatial
- Inference latency reduction ≈ −31%

Three priors rank each visual patch:

| Prior | Signal | Module |
|---|---|---|
| Geometric | Sobel edge density | `priors.geometric_prior` |
| Semantic | Cross-modal attention (vis ↔ lang) | `priors.semantic_prior` |
| Motion | Second-order temporal difference | `priors.MotionPrior` |

IoU between semantic and motion masks switches between conservative / aggressive selection mode.

## Quick start

```bash
pip install -e .

# Offline demo — no GPU required, runs in seconds
python demo_offline.py
# → heatmap_demo.png  heatmap_motion.png  pareto_curve.png

# Smoke test — verifies all modules
python smoke_test.py

# Full evaluation (requires GPU + OpenVLA-OFT + LIBERO)
python eval_libero.py --tasks 5 --episodes 20 --sweep --video --out results.json
python visualize.py --results results.json
```

## Plug-and-play hook

```python
from vla_iap import VLAIAPHook

hook = VLAIAPHook(model, retention=0.70)
hook.install()

# run inference as normal — pruning happens automatically
hook.reset_episode()   # call at the start of each rollout
hook.uninstall()
```

## File structure

```
vla_iap/
  priors.py       geometric / semantic / motion priors
  selector.py     IoU-based token selection
  hook.py         VLAIAPHook — forward pre-hook on vision tower
  attn_patch.py   AttentionMaskPatcher — -inf bias on pruned attn columns
demo_offline.py   CPU-only demo, produces PNG outputs
smoke_test.py     unit tests (no model download)
eval_libero.py    LIBERO-Spatial evaluation loop
visualize.py      Pareto curve + comparison video
```
