# Sleep / idle prevention for multi-day BraTS training (macOS)

## Context

A prior overnight `caffeinate -i -s` training run was mostly suspended: ~5.5 h wall
clock vs ~5 min actual CPU (`user` time). Laptop defaults still show `sleep=1`,
`displaysleep=5`, `disksleep=10` (`pmset -g`). Idle/system assertions alone are not
enough if the machine still enters sleep (closed lid, display sleep policies, etc.).

## Chosen method for the scale-up run

**Layer 1 (required, needs your password once):**

```bash
sudo pmset -a disablesleep 1
# verify:
pmset -g | grep -i disablesleep   # expect: disablesleep 1
# after the full run finishes:
sudo pmset -a disablesleep 0
```

**Layer 2 (always wrap training):** `caffeinate -d -i -m -s`

- `-d` — prevent display sleep
- `-i` — prevent idle sleep
- `-m` — prevent disk sleep
- `-s` — prevent system sleep (AC power)

```bash
nohup caffeinate -d -i -m -s env PYTHONPATH=src python3 -m segmentation.train ... \
  > checkpoints/<run>/train.log 2>&1 &
```

**Also keep the Mac on AC power** for the full run.

## Practical setup (lid / display / System Settings)

| Setting | What to do |
| --- | --- |
| Power | Plugged into AC the whole run |
| Lid | **Keep the lid open.** `disablesleep 1` stops idle/system sleep policies, but closing the lid on many MacBooks still triggers clamshell / display-off paths that can pause or throttle GPU/CPU work. Do not rely on closed-lid “clamshell mode” for this job. |
| Display | Leave the built-in display on (or plugged into an external display). `caffeinate -d` helps; do not force the display off via hot corners / Lock Screen if that also locks the session in a way that stalls training. |
| System Settings → Lock Screen | Set “Turn display off …” / “Start Screen Saver …” to **Never** (or a very long time) while the run is active. Screen lock alone is usually OK if the process keeps running, but display-off + lid-close is what burned the overnight bench. |
| System Settings → Battery / Energy | Disable Low Power Mode; prefer “Prevent automatic sleeping when display is off when plugged into power adapter” if your macOS version shows that toggle. |
| After the run | `sudo pmset -a disablesleep 0` so normal sleep returns |

**Bottom line:** `disablesleep 1` alone is **not** a guarantee with a closed lid. For this run: **AC power + lid open + disablesleep 1 + `caffeinate -d -i -m -s`.**

## Heartbeat

Training appends timestamped lines to `checkpoints/<run>/heartbeat.log` at epoch
start/end and every `heartbeat_interval_s` (default 300 s) during an epoch. If the
machine dies mid-run, the last heartbeat time is when it stopped.

## Verification during a run

```bash
pmset -g | grep -i disablesleep
pmset -g assertions | head -40   # should list caffeinate
pgrep -fl 'segmentation.train|caffeinate'
tail -f checkpoints/<run>/heartbeat.log
```
