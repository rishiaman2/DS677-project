# Demo (Milestone 3)

Planned: a self-contained notebook that

1. loads the trained ResNet-18, ViT-Lite-7 (patch 8, 4, 2) and fine-tuned ViT-B/16 checkpoints,
2. classifies a few CIFAR-10 test images (or an uploaded image resized to 32x32) with each model,
3. shows how each patch size cuts the same image into tokens, and
4. plots the ViT's attention from the CLS token over the patches, layer by layer.
