# Why Vision Transformers Struggle Without Pretraining

> **Claim.** We claim that a small Vision Transformer trained from scratch on CIFAR-10 stays at least 1 accuracy point below a ResNet-18 at every patch size we test (8x8, 4x4 and 2x2 pixels), and we will test it against a ResNet-18 baseline trained with the same data, augmentation and 300-epoch budget on CIFAR-10.
>
> **Status:** approved by Akshay Rangamani on November 12, 2026.

DS 677 Deep Learning, Fall 2026. Model Review and Demo track. Rishabh Verma (solo project).

## 1. What this project asks

A Vision Transformer (ViT) cuts an image into square patches and treats each patch like a word. Unlike a CNN, it has no built-in preference for nearby pixels, so it has to learn locality from data. With millions of pretraining images it does. With CIFAR-10's 50,000 training images, it is much harder.

This project trains a small ViT (ViT-Lite-7, about 3.7M parameters, built with the Hugging Face ViT code) from scratch on CIFAR-10. It asks whether choosing the right patch size is enough to catch up with a ResNet-18, and it looks inside the layers to see what the ViT fails to learn without pretraining. The pretrained `google/vit-base-patch16-224-in21k` checkpoint is the reference for what pretraining buys.

## 2. Research questions

1. **RQ1 (the claim).** Can any patch size (8x8, 4x4, 2x2) let a ViT trained from scratch reach a ResNet-18 on CIFAR-10?
2. **RQ2.** How do accuracy and compute change with patch size? Going from 4x4 to 2x2 patches costs 4.6x the compute per image.
3. **RQ3.** What does the from-scratch ViT fail to learn, layer by layer? Measured with mean attention distance, linear probes per layer and layer knockout, and compared with the pretrained ViT-B/16.
4. **RQ4 (extension).** Does a small convolutional stem in front of the attention layers work better than smaller patches?
5. **Reference.** How far above every from-scratch model does the fine-tuned, pretrained ViT-B/16 sit?

## 3. Models

Measured with `python experiments/measure_cost.py` (PyTorch FLOP counter, one image, attention included):

| Model | Input | Tokens | Params (M) | Forward GFLOPs/image | Role |
| --- | --- | --- | --- | --- | --- |
| ResNet-18 (CIFAR stem) | 32x32 | n/a | 11.17 | 1.111 | Baseline |
| ViT-Lite-7/8 | 32x32 | 17 | 3.75 | 0.128 | Patch sweep, replication anchor |
| ViT-Lite-7/4 | 32x32 | 65 | 3.72 | 0.509 | Patch sweep, replication anchor |
| ViT-Lite-7/2 | 32x32 | 257 | 3.76 | 2.361 | Patch sweep (new point) |
| ViT-Lite-7 + conv stem | 32x32 | 65 | 3.86 | 0.586 | Extension (RQ4) |
| ViT-B/16, pretrained on ImageNet-21k | 224x224 | 197 | 85.81 | 35.126 | Reference |

ViT-Lite-7 is the configuration from Hassani et al. (2021): 7 layers, 4 heads, width 256, MLP width 512. Its counts match their reported 3.72M parameters and 0.26 GMACs for ViT-Lite-7/4.

## 4. Experiment plan

| Project Guide component | In this project |
| --- | --- |
| A. Model and claim | Hugging Face ViT at two scales, and the claim above |
| B. Baseline | CIFAR-style ResNet-18, learning rate tuned on a 5,000-image validation split, same augmentation and 300 epochs |
| C. Replication anchor | Reproduce ViT-Lite-7/8 (89.10%) and ViT-Lite-7/4 (93.57%) from Hassani et al. (2021); second anchor: ViT-B/16 fine-tune against 98.95% (Dosovitskiy et al., 2021, at 384 px) |
| D. Perturbation study | Sensitivity sweep over patch size (8x8, 4x4, 2x2), everything else held fixed |
| E. Analysis and outlook | Accuracy against compute, attention distance, layer probes, layer knockout, conv-stem extension |

- **Seeds.** 3 per main run (0, 1, 2), reported as mean and standard deviation.
- **Recipe.** Shared by every from-scratch model; see `experiments/configs/`. The transformers use the released CCT CIFAR-10 recipe: AdamW at learning rate 5.5e-4 with weight decay 0.06, batch 128, cosine schedule with 10 warmup epochs, RandAugment, Mixup and CutMix, and label smoothing 0.1.
- **Test set.** Learning-rate sweeps use only the validation split. Final runs train on all 50,000 images and report the final-epoch test accuracy, never a best epoch.

## 5. Repository layout

```text
README.md                         claim at the top, plan, how to run
report/                           Wulver exploration report, report drafts and final PDF source
src/models.py                     ViT-Lite-7 (patch 8/4/2), conv-stem variant, ResNet-18, ViT-B/16
src/data.py                       CIFAR-10 loading and augmentation
src/train.py                      training, evaluation, checkpoint/resume, pilot timing
src/analysis/                     attention distance, probes, knockout (Milestone 3)
experiments/configs/              one YAML per model
experiments/prepare_data.py       downloads CIFAR-10 (and caches the ViT-B/16 checkpoint)
experiments/measure_cost.py       parameter and FLOP counts in section 3
experiments/wulver/               environment, job scripts, submission scripts, logs
demo/                             self-contained demo notebook (Milestone 3)
```

## 6. How to run on Wulver

This follows the course guide *Getting Started on Wulver*. Everything lives in the course directory `/course/2026/fall/ds/677/ikoutis/rv44`, and jobs use the course's `ds677` conda environment at `envs/ds677` inside it (built with the guide's `install_ds677.sh` and checked with `verify_ds677.sh`). It already contains every package this repo needs. If the repo or environment lives somewhere else, set `REPO_DIR` or `ENV_PREFIX` in `experiments/wulver/env.sh`.

```bash
# 0. on the login node, inside the course directory
cd /course/2026/fall/ds/677/ikoutis/rv44
git clone https://github.com/rishiaman2/DS677-project.git
cd DS677-project

# 1. download CIFAR-10 and cache the ViT-B/16 checkpoint (needs internet, so login node; about a minute)
source experiments/wulver/env.sh
python experiments/prepare_data.py --out data --cache-vit

# 2. exploration and pilot: environment check, GPU slice, measured speed, projected budget
sbatch experiments/wulver/explore.sbatch
tail -F experiments/wulver/logs/explore_pilot.<jobid>.out
cat results/pilot/summary.md

# 3. learning-rate sweeps on the validation split, then set the chosen rates in the configs
bash experiments/wulver/lr_sweep.sh

# 4. core runs (3 seeds x 4 models, plus the ViT-B/16 fine-tune)
bash experiments/wulver/submit_core.sh

# 5. extensions, only after the core runs
bash experiments/wulver/submit_extensions.sh

# check on jobs
squeue -u $USER
sacct -u $USER --format=JobID,JobName%28,State,Elapsed,MaxRSS
```

A job that hits its time limit can be submitted again with the same command; `train.py` resumes from the last checkpoint.

## 7. Status

- [x] Repository, code and configs (smoke-tested on CPU with synthetic data)
- [x] Instructor meeting (November 6, 2026); claim revised from the feedback
- [x] Revised claim approved (November 12, 2026)
- [x] Wulver exploration and pilot (`report/wulver_exploration.md`): core plan about 17 GPU-hours, 101 SU
- [ ] Learning-rate sweeps
- [ ] Core runs: ResNet-18, ViT-Lite-7/8, 7/4, 7/2 (3 seeds each), ViT-B/16 fine-tune
- [ ] Analyses: attention distance, probes, knockout
- [ ] Extensions: conv stem, compute-matched 2x2 run
- [ ] Report, demo notebook, video (Milestone 3, Dec 13)

## 8. References

- Dosovitskiy et al. (2021). [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929).
- Hassani et al. (2021). [Escaping the Big Data Paradigm with Compact Transformers](https://arxiv.org/abs/2104.05704), and the [released CIFAR-10 config](https://github.com/SHI-Labs/Compact-Transformers/blob/main/configs/pretrained/cct_7-3x1_cifar10_300epochs.yml).
- Raghu et al. (2021). [Do Vision Transformers See Like Convolutional Neural Networks?](https://arxiv.org/abs/2108.08810)
- Nguyen et al. (2024). [An Image is Worth More Than 16x16 Patches: Exploring Transformers on Individual Pixels](https://arxiv.org/abs/2406.09415).
- Hugging Face: [google/vit-base-patch16-224-in21k](https://huggingface.co/google/vit-base-patch16-224-in21k), [uoft-cs/cifar10](https://huggingface.co/datasets/uoft-cs/cifar10).
