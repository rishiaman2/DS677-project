# Exploring the Wulver HPC Cluster

Milestone 2 report, DS 677 Deep Learning. Rishabh Verma (UCID rv44).

This report follows the course guide *Getting Started on Wulver*, step by step, and then uses what it set up to measure what my project will cost. Every value in [brackets] is copied from a job's `.out` file or a command's output.

## 1. Connecting (guide steps 0 to 3)

- Connected through the NJIT VPN (Cisco Secure Client, `vpn.njit.edu`, Duo approval), then `ssh rv44@wulver.njit.edu`. Landed on login node [login01 or login02].
- Worked only in my course directory, `/course/2026/fall/ds/677/ikoutis/rv44`, and ran no computation on the login node.
- Course account for every job: `2026-fall-ds-677-ikoutis-rv44`. GPU jobs: partition `course_gpu`, QoS `course`, `--gres=gpu:a100_10g:1`.

## 2. The ds677 environment (guide steps 4 to 6)

| Step | What I did | Evidence |
| --- | --- | --- |
| Load conda | `module load Miniforge3` | |
| Build by path, not by name | `sbatch install_ds677.sh`, which runs `conda env create -f ds677.yml -p .../rv44/envs/ds677` as a CPU job | Job [id], finished in [mm:ss] |
| libstdc++ hook | Added `activate.d/libstdcxx.sh` to the environment | |
| Verify on a GPU | `sbatch verify_ds677.sh` | Job [id] ended with `=== All checks passed ===`; torch [version], CUDA [version], device [name] |
| Size on disk | `du -sh .../rv44/envs/ds677` | [size] |

## 3. Running jobs through SLURM (guide steps 7 and 8)

| Job | Job ID | What it did | Elapsed | State |
| --- | --- | --- | --- | --- |
| `explore_pilot` (`experiments/wulver/explore.sbatch`) | [id] | Environment check, `nvidia-smi`, GPU benchmark, 60-step pilot of every model in my plan | [hh:mm:ss] | [COMPLETED] |

- Watched it with `squeue -u $USER` (state [PD, then R]) and `tail -F experiments/wulver/logs/explore_pilot.[id].out`.
- After it finished: `sacct -j [id] --format=JobID,JobName%20,State,Elapsed,MaxRSS`.

## 4. What a course GPU job actually gets

From the `nvidia-smi` section and `experiments/wulver/gpu_check.py` in that log:

| Property | Value |
| --- | --- |
| GPU (`nvidia-smi -L`) | [name and MIG profile] |
| GPU memory | [GB] |
| Streaming multiprocessors | [count] ([fraction] of a full A100's 108) |
| Measured bf16 matmul throughput | [TFLOP/s] |
| Measured TF32 matmul throughput | [TFLOP/s] |
| CPUs allocated | [4] |

## 5. What my project will cost

From `results/pilot/summary.md` (measured images per second and peak memory over 60 training steps per model, projected to full runs):

[paste the table and the two total lines here]

## 6. Problems and what I learned

- [Any problem hit, such as a long PD wait, a CXXABI import error or an out-of-memory error, and the fix.]
- [Measured throughput compared with the 5 to 10 TFLOP/s I assumed when planning.]
- [Whether the core budget fits my 2,500 SU allocation, and any change to batch sizes, seeds or time limits as a result.]
