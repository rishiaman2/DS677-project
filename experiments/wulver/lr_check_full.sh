#!/bin/bash
# Full-length learning-rate check (decided 2026-11-13, before any test-set run; see
# experiments/decisions.md). The 30-epoch sweep put the best rate at the low edge of the grid
# for both models, so this repeats the comparison at the full 300-epoch schedule, still on the
# 45,000 / 5,000 validation split, with each grid extended one step lower. The best validation
# accuracy here decides the rate. Run from the repo root, then: python experiments/summarize_results.py
set -e
T=experiments/wulver/train.sbatch
for LR in 0.025 0.05 0.1; do
  sbatch --job-name=sweepfull_resnet18_lr$LR --time=03:00:00 $T experiments/configs/resnet18.yaml 0 \
    run_name=sweep_full_resnet18_lr$LR optim.lr=$LR data.val_size=5000
done
for LR in 1.375e-4 2.75e-4 5.5e-4; do
  sbatch --job-name=sweepfull_vitlite_p4_lr$LR --time=02:30:00 $T experiments/configs/vitlite_p4.yaml 0 \
    run_name=sweep_full_vitlite_p4_lr$LR optim.lr=$LR data.val_size=5000
done
