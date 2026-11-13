#!/bin/bash
# Core runs: 3 seeds each for the ResNet-18 baseline and the three patch sizes, plus one
# ViT-B/16 fine-tune. Time limits are about 2.5x the hours measured by the pilot (job 1541542,
# results/pilot/summary.md); shorter requests get scheduled sooner. A run that hits its limit
# resumes from its last checkpoint when submitted again. Run from the repo root.
set -e
submit () { sbatch --job-name="$1_s$2" --time="$3" experiments/wulver/train.sbatch "experiments/configs/$1.yaml" "$2"; }
for SEED in 0 1 2; do
  submit resnet18   $SEED 03:00:00   # pilot: 1.07 h
  submit vitlite_p8 $SEED 02:30:00   # pilot: 0.89 h
  submit vitlite_p4 $SEED 02:30:00   # pilot: 0.90 h
  submit vitlite_p2 $SEED 07:00:00   # pilot: 2.64 h
done
submit vitb16_finetune 0 01:30:00    # pilot: 0.30 h
