# Ablation: v1 → v2 (40-case / loss-bug era)

These scripts document the historical path from the original BraTS pretrain
(v1, `checkpoints/brats_pretrain`) through the DiceCELoss fix on a 40-case
subset (v2, `checkpoints/brats_v2_loss_fix`). They are **not** part of the
primary v3 workflow.

## What went wrong / what this folder is for

- Early training effectively under-weighted the ET channel (loss-bug era).
- v2 retrained on the same 40-case / 10-val split after fixing DiceCELoss over
  all channels. That improved ET/TC but regressed WT on held-out data.
- Scaling to **v3** (`checkpoints/brats_scale_full`, 289 train / 20 val × 20
  epochs, cosine warmup + grad clip) removed the WT trade-off.

Keep this folder for paper ablation reproducibility only. Do not use these
checkpoints for new inference — use v3 via `python -m pipeline.cli run`.

## Primary model (main path under `src/`)

```
checkpoints/brats_scale_full/best_model.pt
PYTHONPATH=src python3 -m segmentation.compare_v3 --skip-inference-extras
PYTHONPATH=src python3 -m pipeline.cli run --input … --out outputs/<case>/
```

## Held-out metrics (left in place under `results/` / `checkpoints/`)

| Model | Checkpoint dir | Held-out eval |
|-------|----------------|---------------|
| v1 | `checkpoints/brats_pretrain` | `checkpoints/brats_eval_heldout/{raw,lcc}/` |
| v2 | `checkpoints/brats_v2_loss_fix` | `checkpoints/brats_eval_heldout_v2/{raw,lcc}/` |
| v3 | `checkpoints/brats_scale_full` | `checkpoints/brats_eval_heldout_v3/{raw,lcc}/` |

## Scripts here

| File | Purpose |
|------|---------|
| `compare_v1_v2.py` | Pairwise Dice / ellipsoid comparison for the 40-case models |

Run from repo root:

```bash
PYTHONPATH=src python3 experiments/ablation_v1_v2/compare_v1_v2.py --skip-inference
```

A thin shim remains at `src/segmentation/compare_v1_v2.py` for old imports.
Shared bootstrap helpers live in `segmentation.report_metrics`.
