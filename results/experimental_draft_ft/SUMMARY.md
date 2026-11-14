# Experimental draft fine-tune SUMMARY

**Experimental checkpoint trained on unverified machine-assisted drafts (draft_unverified). Agreement with drafts is not accuracy. Not for clinical use. Not the default CLI/API checkpoint.**

## 1. BraTS held-out Dice (LCC, n=20 fixed subset)

| model | ET | TC | WT | mean |
|---|---:|---:|---:|---:|
| v3 | 0.569 | 0.650 | 0.841 | 0.687 |
| experimental_draft_ft | 0.568 | 0.648 | 0.841 | 0.686 |
| Δ(ft−v3) WT |  |  | -0.001 |  |

## 2. Leave-one-study-out (held-out draft study)

Agreement with drafts is **not accuracy** (drafts are unverified).

| study | model | WT Dice vs draft | WT Dice vs HD-GLIO | centroid dist to draft (mm) | WT mL |
|---|---|---:|---:|---:|---:|
| batch2_01 | v3 | 0.594 | 0.698 | 10.6 | 71.39 |
| batch2_01 | experimental_loso | 0.587 | 0.690 | 6.7 | 72.54 |
| batch2_07 | v3 | 0.420 | 0.594 | 6.9 | 119.31 |
| batch2_07 | experimental_loso | 0.425 | 0.593 | 28.2 | 113.03 |
| batch2_10 | v3 | 0.702 | 0.645 | 7.3 | 48.92 |
| batch2_10 | experimental_loso | 0.704 | 0.636 | 7.4 | 48.28 |
| batch2_12 | v3 | 0.758 | 0.714 | 13.7 | 19.51 |
| batch2_12 | experimental_loso | 0.762 | 0.719 | 14.2 | 18.96 |

## 3. Non-draft batch2 studies (inference only — no accuracy claim)

| study | model | WT mL | centroid mm |
|---|---|---:|---|
| batch2_02 | v3 | 14.20 | (165.5, 127.8, 106.7) |
| batch2_02 | experimental_draft_ft | 15.19 | (165.5, 127.9, 106.7) |
| batch2_03 | v3 | 121.70 | (97.4, 141.9, 89.9) |
| batch2_03 | experimental_draft_ft | 122.07 | (97.3, 141.9, 90.0) |
| batch2_05 | v3 | 68.04 | (126.7, 86.4, 97.2) |
| batch2_05 | experimental_draft_ft | 70.70 | (128.8, 77.7, 100.1) |
| batch2_06 | v3 | 4.59 | (58.4, 39.5, 70.9) |
| batch2_06 | experimental_draft_ft | 4.82 | (58.5, 39.6, 70.9) |
| batch2_08 | v3 | 1.42 | (136.1, 141.5, 62.8) |
| batch2_08 | experimental_draft_ft | 2.94 | (136.3, 141.5, 62.7) |
| batch2_13 | v3 | 4.04 | (127.8, 102.9, 46.5) |
| batch2_13 | experimental_draft_ft | 4.18 | (127.8, 102.9, 46.5) |

## 4. Plain statement

Agreement with the Claude drafts is **not accuracy**, because the drafts are unverified (`draft_unverified`). This checkpoint is experimental and is not the default model for CLI/API/UI.

