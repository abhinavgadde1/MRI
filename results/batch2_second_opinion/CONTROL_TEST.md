# HD-GLIO control test (BraTS heldout)

**Model drafts only — not reference masks.**

## Setup

- Source: `splits/heldout.txt` (read-only), seed **42**, n=5 with forced small/median/large WT.
- Inputs: cached BraTS NIfTIs under `data/processed/brats_nifti/` (already skull-stripped and co-registered).
- **HD-BET: skipped** — re-running skull-strip on already-stripped BraTS volumes would only risk crop/edge damage.
- Affine: HDF5 restack uses an **identity** RAS affine (`conversion_meta.json`); verified per case.
- Lean settings (same as batch2): `step_size=1.0`, `do_tta=False`.
- Label map: HD-GLIO 1→BraTS 4 (ET), 2→BraTS 2 (edema); TC≡ET for HD-GLIO (no necrotic class).
- Published CE Dice reference (Kickingereder 2019, approx.): **0.91**.

## Selected cases

| study_id | role | GT WT mL |
|---|---|---:|
| BraTS20_Training_028 | small | 20.9 |
| BraTS20_Training_058 | median | 91.6 |
| BraTS20_Training_166 | large | 223.0 |
| BraTS20_Training_327 | random | 183.8 |
| BraTS20_Training_257 | random | 82.0 |

## Metrics vs ground truth (HD-GLIO lean) and v3 LCC

| study | role | wall_s | HD Dice ET | HD Dice TC | HD Dice WT | HD HD95 WT | v3 Dice ET | v3 Dice TC | v3 Dice WT | orient OK |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| BraTS20_Training_028 | small | 443 | 0.249 | 0.438 | 0.644 | 8.544 | 0.628 | 0.820 | 0.842 | yes |
| BraTS20_Training_058 | median | 383 | 0.087 | 0.130 | 0.493 | 17.916 | 0.583 | 0.351 | 0.655 | yes |
| BraTS20_Training_166 | large | 441 | 0.037 | 0.036 | 0.835 | 10.050 | 0.935 | 0.957 | 0.926 | yes |
| BraTS20_Training_327 | random | 527 | 0.136 | 0.742 | 0.800 | 14.353 | 0.733 | 0.736 | 0.956 | yes |
| BraTS20_Training_257 | random | 477 | 0.011 | 0.038 | 0.821 | 33.548 | 0.788 | 0.748 | 0.947 | yes |

Mean HD-GLIO Dice ET=0.104, WT=0.719.

## Orientation

Per-case overlays: `outputs/batch2_second_opinion/control_test/<id>/orientation_overlay.png` (green=GT WT, red=HD-GLIO). Shape and affine matched the GT/T1c grid for every case when `orient OK=yes` — checked, not assumed.

## Below published performance — possible causes (likelihood order)

1. **Darwin/CPU patches / no CUDA** — weights load and run, but the path is unvalidated vs official CUDA; ET Dice collapsed (~0.10 mean) while WT is only middling.
2. **Preprocessing mismatch** — BraTS already stripped (HD-BET skipped); HD-GLIO training assumed HD-BET + its intensity pipeline.
3. **Lean settings** (`step_size=1.0`, no TTA) — tested below; **not** the main driver on the probe case.
4. **Orientation / identity affine** — ruled out for these five: shape+affine matched GT; overlays written under `control_test/<id>/orientation_overlay.png`.

### Cheapest test: default settings on one case

- Case: `BraTS20_Training_028` (`step_size=0.5`, TTA on).
- Wall-clock: lean **443 s** → default **12051 s** (~3.3 h).
- Dice ET: lean 0.249 → default 0.254 (no meaningful gain).
- Dice WT: lean 0.644 → default 0.616 (slightly worse).
- Conclusion: switching off lean settings does **not** recover published CE performance on this Mac install.

## Verdict

**Patched install trustworthy: unclear**

Mean WT Dice 0.719 is usable as a coarse second look, but mean ET Dice 0.104 is far below published CE Dice ~0.91 and far below v3 on the same cases (v3 mean WT ~0.865). Orientation checks passed; default settings did not fix ET. Treat batch2 HD-GLIO enhancing outlines as **unreliable** until a CUDA/official-runtime control matches published numbers.

Artifacts: `outputs/batch2_second_opinion/control_test/`.
