#!/bin/bash
# Core runs: 3 seeds each for the ResNet-18 baseline and the three patch sizes, plus one
# ViT-B/16 fine-tune. Time limits are generous; tighten them using results/pilot/summary.md.
# The course QoS allows up to 72 hours per job. Run from the repo root.
set -e
submit () { sbatch --job-name="$1_s$2" --time="$3" experiments/wulver/train.sbatch "experiments/configs/$1.yaml" "$2"; }
for SEED in 0 1 2; do
  submit resnet18   $SEED 08:00:00
  submit vitlite_p8 $SEED 03:00:00
  submit vitlite_p4 $SEED 05:00:00
  submit vitlite_p2 $SEED 20:00:00
done
submit vitb16_finetune 0 04:00:00
