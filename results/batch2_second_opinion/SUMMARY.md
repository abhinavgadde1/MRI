# Batch2 second opinion — SUMMARY

**Model drafts only — neither HD-GLIO nor v3 outlines are verified reference masks.**  
No outline in this folder is a verified reference mask. Existing v3 drafts and `corrected_label` files were left untouched. Protected paths (`checkpoints/brats_scale_full`, `splits/heldout.txt`, `data/processed/real_patients/`) were not modified.

Dark SWI signal inside a contour is a **flag for review**, not a diagnosis. SWI magnitude was converted and registered for the 10 native-T1c studies (see SWI section below). Dark SWI inside a contour is a review flag, not a diagnosis.

---

## PART A — Feasibility (short)

See `FEASIBILITY.md` for sources. Headline:

| Item | Result |
|---|---|
| License | Apache-2.0 (PyPI / GitHub NeuroAI-HD/HD-GLIO) |
| Weights | Official Zenodo `record/4014850` → `~/hd_glio_params` |
| Runtime | Docs require CUDA ≥4 GB; CPU “not supported” |
| This Mac | Apple Silicon, no CUDA; isolated `.venv-hdglio` (Py 3.11) |
| Patches | Darwin: `weights_only=False`, serial preprocess, CUDA `empty_cache` guard |
| Lean settings | `do_tta=False`, `step_size=1.0` → ~8–35 min/case on CPU |
| Alternative | Official nnU-Net v2 BraTS weights in a separate `.venv-nnunet` if CUDA becomes available |

Inputs used: HD-BET-style brain-masked registered 1 mm T1 / T1c / T2 / FLAIR (non-brain = 0), modality order T1→T1c→T2→FLAIR. Outputs resampled with nearest-neighbor onto each study’s labeling `T1c.nii.gz` when needed (smoke cases already matched voxel-for-voxel; all 10 drafts verified same shape+affine as T1c).

HD-GLIO labels mapped as: **1→BraTS 4 (ET)**, **2→BraTS 2 (edema)**; no separate necrotic class (TC ≡ ET for this mapping).

---

## PART B — Scan QC verdicts (19 usable studies)

All **19** studies: **`usable_with_caution`** (none `ready`, none `do not use`).

Common drivers: poor T2→T1 NCC (threshold 0.45), thick-slice / &lt;40-slice series, mixed / non-axial native orientations (esp. T1c PIR), T1c contrast sanity failures, and (9 studies) no verified native T1c.

| study_id | native_T1c | SWI in inventory | NCC T2 | NCC FLAIR | NCC T1c | T1c>T1 | verdict |
|---|---|---|---:|---:|---:|---|---|
| batch2_01 | yes | yes* | 0.414 | 0.804 | 0.823 | no | usable_with_caution |
| batch2_02 | yes | yes* | 0.265 | 0.748 | 0.741 | no | usable_with_caution |
| batch2_03 | yes | yes* | 0.384 | 0.815 | 0.744 | yes | usable_with_caution |
| batch2_04 | no | yes* | 0.310 | 0.754 | 1.000† | no | usable_with_caution |
| batch2_05 | yes | yes* | 0.387 | 0.778 | 0.719 | no | usable_with_caution |
| batch2_06 | yes | yes* | −0.059 | 0.110 | 0.276 | yes | usable_with_caution |
| batch2_07 | yes | yes* | 0.232 | 0.773 | 0.727 | yes | usable_with_caution |
| batch2_08 | yes | yes* | 0.327 | 0.729 | 0.802 | no | usable_with_caution |
| batch2_09 | no | yes* | 0.227 | 0.745 | 1.000† | no | usable_with_caution |
| batch2_10 | yes | yes* | 0.320 | 0.805 | 0.754 | no | usable_with_caution |
| batch2_11 | no | yes* | 0.207 | 0.748 | 1.000† | no | usable_with_caution |
| batch2_12 | yes | yes* | 0.277 | 0.791 | 0.767 | no | usable_with_caution |
| batch2_13 | yes | yes* | 0.144 | 0.805 | 0.739 | no | usable_with_caution |
| batch2_14 | no | yes* | 0.217 | 0.314 | 1.000† | no | usable_with_caution |
| batch2_15 | no | yes* | 0.205 | 0.798 | 1.000† | no | usable_with_caution |
| batch2_16 | no | yes* | 0.317 | 0.738 | 1.000† | no | usable_with_caution |
| batch2_17 | no | yes* | 0.152 | 0.777 | 1.000† | no | usable_with_caution |
| batch2_18 | no | no | −0.290 | 0.267 | 1.000† | no | usable_with_caution |
| batch2_20 | no | yes* | 0.192 | 0.765 | 1.000† | no | usable_with_caution |

\*SWI/SWIp listed in DICOM inventory but **not** converted to NIfTI → `swi_registered_to_t1c=False`.  
†NCC T1c = 1.000 indicates T1c stand-in (= T1 copy) for non-native-T1c studies.

Full flags: `scan_qc.csv`, `scan_qc.md`, `alignment_overlays/`.

---

## PART C — HD-GLIO vs v3 agreement (10 native-T1c studies)

> **Channel-order fix (2026-11-14):** v3 volumes/Dice above use regenerated `v3_draft_label.nii.gz` with BraTS order (FLAIR,T1,T1c,T2). Prior PART C used stale wrong-order v3 drafts. HD-GLIO segs were not re-inferred.

Ran on all 10 native-T1c IDs (QC had no additional `ready`+native-T1c studies). Lean CPU inference only (no second pass with HD-GLIO-AUTO preprocessing on native series — tool’s own co-registration stack is CUDA/Docker-oriented; we used the same brain-masked registered 1 mm volumes as v3).

| study | v3 WT mL | HD-GLIO WT mL | WT Dice | agreement class |
|---|---:|---:|---:|---|
| batch2_01 | 71.39 | 38.48 | 0.698 | partial |
| batch2_02 | 14.20 | 8.43 | 0.518 | partial |
| batch2_03 | 121.70 | 51.42 | 0.578 | partial |
| batch2_05 | 68.04 | 46.11 | 0.762 | both_agree |
| batch2_06 | 4.59 | 1.40 | 0.020 | disagree |
| batch2_07 | 119.31 | 62.36 | 0.594 | partial |
| batch2_08 | 1.42 | 0.00 | 0.000 | one_finds_no_tumor |
| batch2_10 | 48.92 | 38.40 | 0.645 | partial |
| batch2_12 | 19.51 | 13.09 | 0.714 | both_agree |
| batch2_13 | 4.04 | 3.47 | 0.770 | both_agree |

Classes: **both_agree** (WT Dice &gt; 0.7) = 3; **partial** (0.3–0.7) = 5; **disagree** (&lt; 0.3) = 1; **one_finds_no_tumor** = 1.


Per-region Dice/volumes/centroids: `agreement.csv`. Review montages: `<study_id>/review_montage.png`.

### Correct first (disagreement / partial / one-sided)

All second-opinion studies need human review before any correction. Suggested order by urgency (after channel-order fix):

1. **batch2_08** — HD-GLIO finds no tumor; v3 WT ~1.4 mL (one-sided).
2. **batch2_06** — still **disagree** (WT Dice 0.02); HD-GLIO much smaller than v3.
3. **batch2_02, 03, 01, 07, 10** — **partial** WT overlap (Dice 0.52–0.70); volumes still differ.
4. **batch2_05, 12, 13** — **both_agree** on WT (Dice &gt; 0.7) but still unverified drafts.

### Most reliable (agreement)

After regenerating v3 with correct channel order: **3/10** studies reach WT Dice &gt; 0.7 (`batch2_05`, `12`, `13`); **5/10** are partial. Treat every outline as a draft pending expert correction; do not prefer HD-GLIO or v3 automatically.

---

## Tests

`make test` → **228 passed** (includes `tests/test_batch2_second_opinion.py` for QC helpers + HD-GLIO label conversion on synthetic volumes). No training was run.

## SWI second-opinion check (native-T1c)

Philips `SWIp_fast` series contain interleaved **magnitude** (`SW_M`) and **phase** (`SW_P`). We converted **magnitude only**, rigid-registered to T1c (Mattes MI, same as other sequences). Dark-focus column is a **heuristic** (lowest 5% SWI inside either contour vs surrounding brain); **not a diagnosis**.

| study | SWI chosen | NCC→T1c | SWI dark focus inside either contour |
|---|---|---:|---|
| batch2_01 | magnitude | 0.543 | no |
| batch2_02 | magnitude | 0.527 | no |
| batch2_03 | magnitude | 0.466 | no |
| batch2_05 | magnitude | 0.460 | no |
| batch2_06 | magnitude | -0.110 | not_assessable |
| batch2_07 | magnitude | 0.504 | no |
| batch2_08 | magnitude | 0.555 | no |
| batch2_10 | magnitude | 0.477 | no |
| batch2_12 | magnitude | 0.583 | no |
| batch2_13 | magnitude | 0.552 | no |

