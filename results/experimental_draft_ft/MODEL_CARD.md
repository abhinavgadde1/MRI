# Experimental draft fine-tune — MODEL CARD

**Experimental checkpoint trained on unverified machine-assisted drafts (draft_unverified). Agreement with drafts is not accuracy. Not for clinical use. Not the default CLI/API checkpoint.**

## Training data

- Base: `checkpoints/brats_scale_full/best_model.pt` (v3).
- Drafts: `batch2_01`, `07`, `10`, `12` `claude_draft_label.nii.gz` (`draft_unverified`).
- Channel policy: `01`/`12` → WT loss only; `07`/`10` → ET+TC+WT.
- BraTS replay ≥3:1 from `splits/scaleup/cfg_c_all_train.txt` (never held-out).
- Domain augs: thick-slice + T1c→T1 dropout (`batch2_augment`).

## Not the default

CLI, API, and UI continue to use `brats_scale_full`. This checkpoint lives only under `checkpoints/experimental_draft_ft/`.

## Do not use clinically

Drafts are unverified machine-assisted labels. Metrics vs drafts are agreement, not accuracy.
