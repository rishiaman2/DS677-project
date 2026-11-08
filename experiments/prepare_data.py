"""Download CIFAR-10 once and save it as uint8 numpy arrays; optionally cache the ViT-B/16 checkpoint.

Run on the Wulver LOGIN node (compute nodes may not have internet):
    python experiments/prepare_data.py --out data --cache-vit

Writes data/cifar10_train.npz (50,000 images) and data/cifar10_test.npz (10,000 images),
each with x: uint8 (N, 32, 32, 3) and y: int64 (N,).
Source: the Hugging Face dataset uoft-cs/cifar10, with torchvision's copy as a fallback.
"""
import argparse
from pathlib import Path

import numpy as np


def from_hf():
    from datasets import load_dataset

    ds = load_dataset("uoft-cs/cifar10")
    out = {}
    for split in ("train", "test"):
        d = ds[split]
        img_col = "img" if "img" in d.column_names else "image"
        x = np.stack([np.asarray(im.convert("RGB"), dtype=np.uint8) for im in d[img_col]])
        y = np.asarray(d["label"], dtype=np.int64)
        out[split] = (x, y)
    return out


def from_torchvision(root: Path):
    from torchvision.datasets import CIFAR10

    out = {}
    for split, train in (("train", True), ("test", False)):
        d = CIFAR10(root=str(root), train=train, download=True)
        out[split] = (d.data.astype(np.uint8), np.asarray(d.targets, dtype=np.int64))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data")
    p.add_argument("--cache-vit", action="store_true", help="also download google/vit-base-patch16-224-in21k into the HF cache")
    args = p.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        splits = from_hf()
        source = "huggingface uoft-cs/cifar10"
    except Exception as e:  # network or package problem: fall back to torchvision's mirror
        print(f"Hugging Face download failed ({e}); falling back to torchvision")
        splits = from_torchvision(out_dir / "raw")
        source = "torchvision CIFAR10"

    for split, (x, y) in splits.items():
        np.savez(out_dir / f"cifar10_{split}.npz", x=x, y=y)
        counts = np.bincount(y, minlength=10).tolist()
        print(f"{split}: x {x.shape} {x.dtype}, y {y.shape}, per-class counts {counts}")
    print(f"source: {source}")

    if args.cache_vit:
        from transformers import ViTForImageClassification

        ViTForImageClassification.from_pretrained("google/vit-base-patch16-224-in21k", num_labels=10)
        print("cached google/vit-base-patch16-224-in21k")


if __name__ == "__main__":
    main()
