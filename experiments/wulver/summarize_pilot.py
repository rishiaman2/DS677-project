"""Turn the pilot measurements into a compute budget table (markdown) for the Wulver report.

Reads results/pilot/<run>.json written by `src/train.py --pilot-steps`, and prints measured
images/second and peak memory, the projected hours per full run, and the projected total for
the planned number of runs. SU cost uses the course_gpu rate: 2 + max(CPUs, memory GB / 4) per
hour, which is 6 SU/hour for the 4-CPU jobs in this repo (override with --su-per-hour).
"""
import argparse
import json
from pathlib import Path

PLANNED_RUNS = {  # run_name: (number of runs, tier)
    "resnet18": (3, "core"),
    "vitlite_p8": (3, "core"),
    "vitlite_p4": (3, "core"),
    "vitlite_p2": (3, "core"),
    "vitb16_finetune": (1, "core"),
    "vitlite_convstem": (3, "extension"),
}
ALLOCATION_SU = 2500


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pilot-dir", default="results/pilot")
    p.add_argument("--su-per-hour", type=float, default=6.0)
    args = p.parse_args()

    gpu = Path(args.pilot_dir) / "gpu_check.json"
    if gpu.exists():
        g = json.loads(gpu.read_text())
        print(f"GPU: {g.get('device_name')}, {g.get('total_memory_GB')} GB, {g.get('streaming_multiprocessors')} SMs "
              f"({g.get('fraction_of_full_A100_SMs')} of a full A100), measured bf16 matmul {g.get('bf16_matmul_TFLOPs')} TFLOP/s\n")

    print("| Run | Params (M) | Batch | Images/s (measured) | Peak GPU memory GB (measured) | Epochs | Hours per run (projected) | Runs | Hours total (projected) |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    totals = {"core": 0.0, "extension": 0.0}
    for f in sorted(Path(args.pilot_dir).glob("*.json")):
        if f.name == "gpu_check.json":
            continue
        r = json.loads(f.read_text())
        n, tier = PLANNED_RUNS.get(r["run_name"], (1, "other"))
        hours = r["estimated_hours_full_run"]
        totals[tier] = totals.get(tier, 0.0) + n * hours
        print(f"| {r['run_name']} | {r['params']/1e6:.2f} | {r['batch_size']} | {r['images_per_second']} | "
              f"{r['peak_gpu_memory_GB']} | {r['epochs_in_config']} | {hours} | {n} | {n*hours:.1f} |")
    print()
    for tier in ("core", "extension"):
        h = totals.get(tier, 0.0)
        su = h * args.su_per_hour
        print(f"{tier.capitalize()} total: {h:.1f} GPU-hours, about {su:.0f} SU ({100*su/ALLOCATION_SU:.1f}% of the {ALLOCATION_SU} SU allocation)")


if __name__ == "__main__":
    main()
