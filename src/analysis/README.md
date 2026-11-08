# Analyses (Milestone 3)

All three run on trained checkpoints with forward passes only, so each takes minutes.

| Script (planned) | Question | What it measures |
| --- | --- | --- |
| `attention_distance.py` | RQ3: does the from-scratch ViT learn local attention in its early layers? | For every head in every layer, the attention-weighted mean pixel distance between a patch and the patches it attends to, over 1,000 test images, divided by image width so 32 px and 224 px models compare (the measure in Figure 7 of Dosovitskiy et al., 2021). Needs the model built with `attn: eager`. |
| `probes.py` | RQ3: where does class information appear? | Logistic-regression accuracy on the frozen CLS token after each block (pooled features after each ResNet stage). |
| `knockout.py` | RQ3: which layers does the network depend on? | Test accuracy when one block is skipped at a time (the residual path carries the input through). |
