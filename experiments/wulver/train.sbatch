#!/bin/bash -l
#SBATCH --job-name=train
#SBATCH --output=experiments/wulver/logs/%x.%j.out
#SBATCH --partition=course_gpu
#SBATCH --account=2026-fall-ds-677-ikoutis-rv44
#SBATCH --qos=course
#SBATCH --gres=gpu:a100_10g:1
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
# One training run per job. Submit from the repo root:
#   sbatch --job-name=vitlite_p4_s0 --time=05:00:00 experiments/wulver/train.sbatch experiments/configs/vitlite_p4.yaml 0
# Extra arguments after the seed become --set overrides (key=value). If a job hits its time limit,
# submit the same command again: train.py resumes from results/<run>/seed<k>/last.pt.
CONFIG=$1
SEED=$2
shift 2
source experiments/wulver/env.sh
export HF_HUB_OFFLINE=1
echo "job $SLURM_JOB_ID on $(hostname): $CONFIG seed $SEED $*"
nvidia-smi -L
if [ $# -gt 0 ]; then
  python src/train.py --config "$CONFIG" --seed "$SEED" --set "$@"
else
  python src/train.py --config "$CONFIG" --seed "$SEED"
fi
