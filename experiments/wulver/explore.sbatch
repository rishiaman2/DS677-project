#!/bin/bash -l
#SBATCH --job-name=explore_pilot
#SBATCH --output=experiments/wulver/logs/%x.%j.out
#SBATCH --partition=course_gpu
#SBATCH --account=2026-fall-ds-677-ikoutis-rv44
#SBATCH --qos=course
#SBATCH --gres=gpu:a100_10g:1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
# Milestone 2 Wulver exploration: what hardware a course GPU job gets, how fast it is,
# and how long each model in the plan would take. Submit from the repo root:
#   sbatch experiments/wulver/explore.sbatch
source experiments/wulver/env.sh
export HF_HUB_OFFLINE=1   # the ViT-B/16 checkpoint was cached on the login node by prepare_data.py

echo "== node and allocation"
hostname; date
echo "job $SLURM_JOB_ID, partition $SLURM_JOB_PARTITION, cpus $SLURM_CPUS_PER_TASK"

echo "== environment (same checks as the course guide's verify job)"
which python
python - <<'PY'
import importlib
for name in ["torch", "torchvision", "transformers", "datasets", "yaml", "sklearn", "numpy"]:
    try:
        m = importlib.import_module(name)
        print(f"  {name:<13} {getattr(m, '__version__', 'n/a')}")
    except Exception as e:
        print(f"  {name:<13} IMPORT FAILED: {type(e).__name__}: {e}")
import torch
print("  cuda available:", torch.cuda.is_available(), "| torch.version.cuda:", torch.version.cuda)
PY

echo "== nvidia-smi"
nvidia-smi
nvidia-smi -L

echo "== GPU check (device properties and measured matmul throughput)"
python experiments/wulver/gpu_check.py

echo "== pilots: 60 training steps per model, same code path as the real runs"
for cfg in resnet18 vitlite_p8 vitlite_p4 vitlite_p2 vitlite_convstem vitb16_finetune; do
  python src/train.py --config experiments/configs/$cfg.yaml --pilot-steps 60 || echo "pilot failed: $cfg"
done

echo "== summary"
python experiments/wulver/summarize_pilot.py | tee results/pilot/summary.md
