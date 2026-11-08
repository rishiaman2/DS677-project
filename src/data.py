"""CIFAR-10 loading and augmentation.

The data lives in data/cifar10_{train,test}.npz (created by experiments/prepare_data.py) as
uint8 arrays x of shape (N, 32, 32, 3) and int labels y. Augmentation follows the released
CCT CIFAR-10 config (Hassani et al., 2021) and is identical for every from-scratch model:
random resized crop (scale 0.8 to 1.0), horizontal flip, RandAugment, and batch-level
Mixup/CutMix (applied in train.py).
"""
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2

CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)
VIT_IN21K_MEAN = (0.5, 0.5, 0.5)  # preprocessor of google/vit-base-patch16-224-in21k
VIT_IN21K_STD = (0.5, 0.5, 0.5)


def load_split(data_dir: str, split: str):
    path = Path(data_dir) / f"cifar10_{split}.npz"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run: python experiments/prepare_data.py --out {data_dir}")
    d = np.load(path)
    x = torch.from_numpy(d["x"]).permute(0, 3, 1, 2).contiguous()  # uint8 (N, 3, 32, 32)
    y = torch.from_numpy(d["y"]).long()
    return x, y


class TensorImages(Dataset):
    def __init__(self, x: torch.Tensor, y: torch.Tensor, transform):
        self.x, self.y, self.transform = x, y, transform

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.transform(self.x[i]), self.y[i]


def build_transform(dcfg: dict, train: bool):
    size = dcfg.get("image_size", 32)
    if dcfg.get("normalize", "cifar") == "vit_in21k":
        mean, std = VIT_IN21K_MEAN, VIT_IN21K_STD
    else:
        mean, std = CIFAR_MEAN, CIFAR_STD
    ops = []
    if train:
        ops += [
            v2.RandomResizedCrop(32, scale=tuple(dcfg.get("crop_scale", [0.8, 1.0])), antialias=True),
            v2.RandomHorizontalFlip(),
        ]
        if dcfg.get("randaugment_magnitude"):
            ops.append(v2.RandAugment(num_ops=2, magnitude=int(dcfg["randaugment_magnitude"])))
    if size != 32:
        ops.append(v2.Resize((size, size), antialias=True))
    ops += [v2.ToDtype(torch.float32, scale=True), v2.Normalize(mean, std)]
    return v2.Compose(ops)


def make_loaders(dcfg: dict, batch_size: int, eval_batch_size: int, seed: int):
    """Returns (train_loader, eval_loader, eval_name).

    If data.val_size > 0, a fixed 5,000-image validation split (split seed 0, independent of the
    training seed) is held out and used for evaluation; the test set is never touched. This is the
    mode for learning-rate sweeps. With val_size 0 the model trains on all 50,000 images and is
    evaluated on the 10,000 test images; the reported number is the final epoch, never a best epoch.
    """
    x, y = load_split(dcfg["dir"], "train")
    val_size = int(dcfg.get("val_size", 0))
    if val_size > 0:
        perm = torch.randperm(len(y), generator=torch.Generator().manual_seed(0))
        val_idx, train_idx = perm[:val_size], perm[val_size:]
        train_x, train_y, eval_x, eval_y, eval_name = x[train_idx], y[train_idx], x[val_idx], y[val_idx], "val"
    else:
        train_x, train_y = x, y
        eval_x, eval_y = load_split(dcfg["dir"], "test")
        eval_name = "test"

    if dcfg.get("train_limit"):  # small subsets for smoke tests only
        train_x, train_y = train_x[: dcfg["train_limit"]], train_y[: dcfg["train_limit"]]

    workers = int(dcfg.get("num_workers", 4))
    common = dict(num_workers=workers, pin_memory=torch.cuda.is_available(), persistent_workers=workers > 0)
    train_loader = DataLoader(
        TensorImages(train_x, train_y, build_transform(dcfg, train=True)),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        generator=torch.Generator().manual_seed(seed),
        **common,
    )
    eval_loader = DataLoader(
        TensorImages(eval_x, eval_y, build_transform(dcfg, train=False)),
        batch_size=eval_batch_size,
        shuffle=False,
        **common,
    )
    return train_loader, eval_loader, eval_name
