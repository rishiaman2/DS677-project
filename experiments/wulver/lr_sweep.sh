#!/bin/bash
# Short learning-rate sweeps on the 45,000 / 5,000 validation split; the test set is never used.
# 30 epochs each (about 6 minutes per job at the pilot's measured speed), with the schedule
# scaled down proportionally. Run from the repo root, then: python experiments/summarize_results.py
set -e
T=experiments/wulver/train.sbatch
for LR in 0.05 0.1 0.2; do
  sbatch --job-name=sweep_resnet18_lr$LR --time=00:45:00 $T experiments/configs/resnet18.yaml 0 \
    run_name=sweep_resnet18_lr$LR optim.lr=$LR data.val_size=5000 \
    train.epochs=30 train.mix_off_epoch=18 train.eval_every=5 optim.warmup_epochs=1
done
for LR in 2.75e-4 5.5e-4 1.1e-3; do
  sbatch --job-name=sweep_vitlite_p4_lr$LR --time=00:45:00 $T experiments/configs/vitlite_p4.yaml 0 \
    run_name=sweep_vitlite_p4_lr$LR optim.lr=$LR data.val_size=5000 \
    train.epochs=30 train.mix_off_epoch=18 train.eval_every=5 optim.warmup_epochs=1 optim.cooldown_epochs=1
done
