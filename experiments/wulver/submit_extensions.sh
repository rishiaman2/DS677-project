#!/bin/bash
# Extensions, only after the core runs are done.
#   RQ4: conv stem instead of patches (64 tokens), 300 epochs.
#   RQ2 control: 2x2 patches for 65 epochs, about the same compute as 4x4 patches for 300 epochs.
set -e
T=experiments/wulver/train.sbatch
for SEED in 0 1 2; do
  sbatch --job-name=vitlite_convstem_s$SEED --time=03:00:00 $T experiments/configs/vitlite_convstem.yaml $SEED
  sbatch --job-name=vitlite_p2_65ep_s$SEED --time=02:00:00 $T experiments/configs/vitlite_p2.yaml $SEED \
    run_name=vitlite_p2_65ep train.epochs=65 train.mix_off_epoch=38 optim.warmup_epochs=2 optim.cooldown_epochs=2 \
    train.eval_every=5 train.ckpt_every=5
done
