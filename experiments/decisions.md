# Decision log

Choices made during the project, recorded when they were made, so the report can state what was decided before which results were seen.

## 2026-11-13: learning rates

**30-epoch sweep** (`experiments/wulver/lr_sweep.sh`; seed 0; 45,000 train / 5,000 validation images; test set unused):

| Model | Learning rate | Validation accuracy (%) |
| --- | --- | --- |
| ResNet-18, SGD | 0.05 | 93.38 |
| ResNet-18, SGD | 0.1 | 92.86 |
| ResNet-18, SGD | 0.2 | 92.62 |
| ViT-Lite-7/4, AdamW | 2.75e-4 | 75.56 |
| ViT-Lite-7/4, AdamW | 5.5e-4 (published CCT recipe) | 67.84 |
| ViT-Lite-7/4, AdamW | 1.1e-3 | 64.18 |

**Observation.** For both models the best rate was the lowest value tried, so the grid did not bracket the optimum. Short training runs are known to favor smaller learning rates than the full schedule would (short-horizon bias, Wu et al., 2018), and the published ViT-Lite recipe uses 5.5e-4 for 300 epochs. After 30 epochs ViT-Lite-7/4 reached 75.6% against ResNet-18's 93.4%, a first sign of how much slower the ViT trains from scratch.

**Decision, made before any test-set run.** Repeat the comparison at the full 300-epoch schedule on the same validation split, with each grid extended one step lower: ResNet-18 at 0.025, 0.05 and 0.1; ViT-Lite-7/4 at 1.375e-4, 2.75e-4 and 5.5e-4 (`experiments/wulver/lr_check_full.sh`). The highest full-length validation accuracy decides each model's rate, and the ViT rate applies to every ViT-Lite-7 config. If the best is again at an edge of the grid, it is used as is and reported, without extending the grid further. The ViT-B/16 fine-tune keeps its fixed recipe and is not swept.
