# BraTS scale-up plan (benchmark-driven; do not launch until confirmed)

## Available data

- Cached NIfTI subjects: **369**
- Held-out (`splits/heldout.txt`): **60**
- **Available for train/val: 309**

## Code changes already made (not yet used for a multi-day run)

- Default LR: **linear warmup (5% of steps) → cosine annealing** (`scheduler_type=cosine_warmup`)
- **Gradient clip** `max_norm=1.0`
- DiceCELoss(`include_background=True`) retained
- `learning_rate` logged every epoch in `metrics.csv`
- Default **`cache_rate=0.0`** (lazy `Dataset`) — 16 GB RAM; do not enable MONAI `CacheDataset` for 150+ full volumes on this laptop
- Optional `PersistentDataset` via `cache_dir` if disk cache is desired later

## Benchmark (real 2 epochs; 150 train + 20 val; MPS; `caffeinate -i -s`)

Config: batch_size=1, lr=1e-4, SegResNet, no AMP, seed=42, lists under `splits/scaleup/`.

| Epoch | epoch_seconds | LR (end) | val Dice mean |
|------:|--------------:|---------:|--------------:|
| 1 | **570.4 s** (9.5 min) | 5.46e-5 | 0.080 |
| 2 | **1169.9 s** (19.5 min) | 1.00e-6 | 0.093 |

- **Peak RSS** (`/usr/bin/time -l`): **1.94 GB**
- **Wall clock** for whole job: **19870 s ≈ 5.5 h** (matches overnight calendar time)
- Epoch time was **not stable**: epoch 2 was ~2× the reported `epoch_seconds` of epoch 1, and batch timestamps show multi-hour gaps (machine slept / throttled despite `caffeinate`). `user` CPU time was only ~282 s vs ~5.5 h wall — mostly idle/suspended.

**Note:** With `--epochs 2`, cosine `T_max` spans only 300 steps, so LR hits `eta_min` by end of epoch 2. The full run must set `max_epochs` to the real budget so warmup+cosine stretch across all epochs.

## Projections (val=20 fixed; scale train time ∝ n_train/150)

Using **healthy** epoch-1 rate (570 s @ 150 train) — assumes machine stays awake:

| Config | Train×epochs | min/epoch | Total | Fits in 4 d? |
|--------|-------------:|----------:|------:|:------------:|
| (a) | 150 × 30 | 9.5 | **7.3 h (0.30 d)** | yes |
| (b) | 250 × 25 | 15.3 | **9.8 h (0.41 d)** | yes |
| (c) | 289 × 20 | 17.6 | **9.0 h (0.37 d)** | yes |

Pessimistic (if overnight throttle repeats, ~2.8 h/epoch @ 150-scale → ~5.4 h/epoch @ 289): 289×20 ≈ **4.5 d** — eats the margin. **Sleep prevention is mandatory.**

## Recommendation

**Config (c): 289 train + 20 val × 20 epochs** (`splits/scaleup/cfg_c_all_*.txt`)

- Strongest paper claim (uses almost all non-held-out cases)
- Projected **~9 h** if awake; leaves **≫1 day** for held-out eval / ellipsoid / LCC / `make paper`
- Output dir suggestion: `checkpoints/brats_v3_scale289/`

**Sleep prevention (required):**

1. Keep Mac on **AC power**
2. Launch with `nohup caffeinate -d -i -m -s ... &` after `sudo pmset -a disablesleep 1` (see `docs/SLEEP_PREVENTION.md`)
3. **Please run interactively** (needs your password):

```bash
sudo pmset -a disablesleep 1
# after training finishes:
sudo pmset -a disablesleep 0
```

`sudo -n` failed in this environment — the agent will not attempt unattended sudo.

## Disk / cache

- Free disk ≈ **32 GB**; `brats_nifti` already **4.3 GB** — no extra case materialization needed
- Keep `cache_rate=0`; do not RAM-cache full volumes on 16 GB

## Waiting for your confirmation

No multi-day run has been launched. Reply with go-ahead (and whether you enabled `disablesleep`) to start (c) or name an alternate config.
