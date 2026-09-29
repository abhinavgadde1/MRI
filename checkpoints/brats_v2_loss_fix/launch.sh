#!/bin/zsh
cd /Users/cherrygadde/MRI
export PYTHONPATH=src
exec caffeinate -i python3 -m segmentation.train \
  --output-dir checkpoints/brats_v2_loss_fix \
  --train-list splits/train.txt \
  --val-list checkpoints/brats_pretrain/train_run_val_cases.txt \
  --heldout-list splits/heldout.txt \
  --epochs 20 \
  --batch-size 1 \
  --lr 0.0001 \
  --model segresnet \
  --device mps \
  --log-every 1
