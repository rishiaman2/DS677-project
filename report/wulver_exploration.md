# Exploring the Wulver HPC Cluster

Milestone 2 report, DS 677 Deep Learning. Rishabh Verma (UCID rv44). November 7, 2026.

I followed the course guide *Getting Started on Wulver* step by step, then used the environment it produced to measure what my project will cost. The whole core plan comes to about 17 GPU-hours, roughly 101 of my 2,500 service units (4%). Every number below comes from a job log or command output.

## 1. Connecting (guide steps 0 to 3)

- Connected through the NJIT VPN (Cisco Secure Client), then `ssh rv44@wulver.njit.edu` with Duo approval. Landed on login node `login02`.
- Worked only in my course directory, `/course/2026/fall/ds/677/ikoutis/rv44`. The project repository is cloned there as `DS677-project`.
- Every job uses my course account `2026-fall-ds-677-ikoutis-rv44` with QoS `course`. CPU jobs use partition `course`; GPU jobs use `course_gpu` with `--gres=gpu:a100_10g:1`.

## 2. Building the ds677 environment (guide steps 4 to 6)

| Step | What I did | Result |
| --- | --- | --- |
| Load conda | `module load Miniforge3` | |
| Build by path, as a job | `sbatch install_ds677.sh`, which runs `conda env create -f ds677.yml -p .../rv44/envs/ds677` on the `course` CPU partition | Job 1541509, COMPLETED in 3 min 52 s |
| libstdc++ hook | Added `etc/conda/activate.d/libstdcxx.sh` to the environment | |
| Verify on a GPU | `sbatch verify_ds677.sh` | Job 1541515, COMPLETED in 2 min 8 s, ended with `=== All checks passed ===` |

The verify job reported Python 3.11.18, torch 2.15.0+cu132, torchvision 0.29.2+cu132, transformers 5.22.2, numpy 1.26.4 and scikit-learn 1.9.1, with `cuda.is_available() : True` on an `NVIDIA A100-PCIE-40GB MIG 2g.10gb`.

## 3. Running jobs through SLURM (guide steps 7 and 8)

| Job | Job ID | What it did | Elapsed | State |
| --- | --- | --- | --- | --- |
| `ds677_install` | 1541509 | Built the environment (CPU job) | 00:03:52 | COMPLETED |
| `ds677_verify` | 1541515 | Imports and GPU smoke test | 00:02:08 | COMPLETED |
| `explore_pilot` | 1541542 | Environment check, `nvidia-smi`, GPU benchmark, 60-step pilot of every model in my plan | 00:03:48 | COMPLETED |

I followed each job with `squeue -u $USER` and `tail -F <job>.<id>.out`, and checked the final state and run time with `sacct`. The exploration job ran on compute node `n0802` with 4 CPUs. Its script is `experiments/wulver/explore.sbatch` in the repository.

## 4. What a course GPU job actually gets

| Property | Value |
| --- | --- |
| GPU | NVIDIA A100-PCIE-40GB, MIG profile 2g.10gb |
| GPU memory | 10.47 GB |
| Streaming multiprocessors | 28, which is 26% of a full A100's 108 |
| Compute capability | 8.0 |
| Driver / CUDA | Driver 580.159.04 (CUDA 13.0); PyTorch built for CUDA 13.2 |
| Measured bf16 matmul throughput | 65.4 TFLOP/s |
| Measured TF32 matmul throughput | 37.7 TFLOP/s |

The bf16 figure is about 81% of this slice's theoretical peak (28/108 of an A100's 312 TFLOP/s, or 80.9 TFLOP/s).

## 5. What my project will cost

Each model ran 60 real training steps, using the same code, data pipeline and batch size as the full runs. Speed and peak memory are measured. Hours are projected from that speed to the full run (50,000 images times the number of epochs).

| Run | Params (M) | Batch | Images/s (measured) | Peak GPU memory GB (measured) | Epochs | Hours per run (projected) | Runs | Hours total (projected) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| resnet18 | 11.17 | 128 | 3878.9 | 0.47 | 300 | 1.07 | 3 | 3.2 |
| vitb16_finetune | 85.81 | 32 | 230.7 | 3.97 | 5 | 0.3 | 1 | 0.3 |
| vitlite_convstem | 3.86 | 128 | 4182.7 | 0.65 | 300 | 1.0 | 3 | 3.0 |
| vitlite_p2 | 3.76 | 128 | 1576.6 | 2.06 | 300 | 2.64 | 3 | 7.9 |
| vitlite_p4 | 3.72 | 128 | 4615.6 | 0.57 | 300 | 0.9 | 3 | 2.7 |
| vitlite_p8 | 3.75 | 128 | 4677.0 | 0.21 | 300 | 0.89 | 3 | 2.7 |

Core total: 16.8 GPU-hours, about 101 SU (4.0% of the 2,500 SU allocation).
Extension total (conv stem): 3.0 GPU-hours, about 18 SU (0.7%).

The same measured speeds put the learning-rate sweeps at about 0.5 GPU-hours and the compute-matched 2x2 extension at about 1.7. Everything in the plan together comes to roughly 22 GPU-hours, or 132 SU (5.3%). These projections leave out evaluation passes and job start-up, so real runs will take somewhat longer.

## 6. Problems I hit and how I solved them

- **No environment yet.** My first `source experiments/wulver/env.sh` failed with `EnvironmentLocationNotFound`, because `envs/ds677` did not exist. Building it with the guide's install and verify jobs fixed this.
- **CIFAR-10 download.** The ds677 environment has `datasets` 1.1.1, which tries to fetch an old loading script for `uoft-cs/cifar10` that no longer exists. My data script fell back to torchvision's copy of CIFAR-10. It produced the expected 50,000 training and 10,000 test images, with exactly 5,000 and 1,000 per class.
- **Pretrained checkpoint warnings.** Loading `google/vit-base-patch16-224-in21k` reported the classifier as MISSING and the pooler as UNEXPECTED. This is expected: the checkpoint ships without a classification head, so a new 10-class head is trained for CIFAR-10, and the unused pooler is ignored.

## 7. What I learned and what changes in my plan

- **The GPU slice is bigger than I assumed.** I planned for a one-seventh slice of an 80 GB A100 sustaining 5 to 10 TFLOP/s. The job got a 2g.10gb slice (28 SMs). Measured speed times counted FLOPs (training taken as 3x the forward pass) works out to about 11 TFLOP/s for ViT-Lite-7/2 and 24 TFLOP/s for the ViT-B/16 fine-tune.
- **The small models are limited by the CPUs, not the GPU.** ViT-Lite-7/8 and ViT-Lite-7/4 ran at almost the same speed (4,677 and 4,616 images/s), even though 4x4 patches need 4x the compute (0.13 vs 0.51 GFLOPs per image). The bottleneck is most likely the 4 CPU workers doing data augmentation. The whole plan costs only about 5% of my allocation, so I will keep 4 CPUs. Requesting more CPUs is the first thing to try if wall-clock time becomes a problem.
- **Memory is not a constraint.** The largest model, ViT-B/16 at batch 32, peaked at 3.97 GB of the 10.47 GB available.
- **The budget leaves room.** The core plan uses 4% of my allocation and the longest run is about 2.6 hours, far under the 72-hour limit. Extra seeds or the extensions are affordable if time allows.
