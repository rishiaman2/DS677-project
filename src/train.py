"""Train or pilot one model on CIFAR-10 from a YAML config.

Examples
    python src/train.py --config experiments/configs/vitlite_p4.yaml --seed 0
    python src/train.py --config experiments/configs/resnet18.yaml --seed 0 \
        --set optim.lr=0.05 data.val_size=5000 train.epochs=30 run_name=resnet18_lr0.05
    python src/train.py --config experiments/configs/vitlite_p2.yaml --pilot-steps 60

Outputs go to results/<run_name>/seed<seed>/: config.yaml, log.jsonl (one line per epoch),
metrics.json (final numbers) and last.pt (checkpoint, resumed automatically).
A pilot writes results/pilot/<run_name>.json with measured images/second and peak GPU memory.
"""
import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torchvision.transforms import v2

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import make_loaders  # noqa: E402
from models import NUM_CLASSES, build_model, count_params  # noqa: E402

TRAIN_SIZE_FULL = 50_000


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--set", nargs="*", default=[], help="dotted overrides, e.g. optim.lr=0.05 train.epochs=30")
    p.add_argument("--pilot-steps", type=int, default=0, help="time this many training steps, write a pilot JSON, exit")
    p.add_argument("--results", default="results")
    p.add_argument("--no-resume", action="store_true")
    return p.parse_args()


def apply_overrides(cfg: dict, overrides):
    for item in overrides:
        key, value = item.split("=", 1)
        node = cfg
        *parents, leaf = key.split(".")
        for k in parents:
            node = node.setdefault(k, {})
        node[leaf] = yaml.safe_load(value)
    return cfg


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def param_groups(model, weight_decay: float):
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        skip = p.ndim <= 1 or "cls_token" in name or "position_embeddings" in name
        (no_decay if skip else decay).append(p)
    return [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}]


def build_optimizer(model, ocfg: dict):
    groups = param_groups(model, float(ocfg["weight_decay"]))
    if ocfg["name"] == "adamw":
        return torch.optim.AdamW(groups, lr=float(ocfg["lr"]), betas=tuple(ocfg.get("betas", [0.9, 0.999])))
    if ocfg["name"] == "sgd":
        return torch.optim.SGD(
            groups, lr=float(ocfg["lr"]), momentum=float(ocfg.get("momentum", 0.9)), nesterov=bool(ocfg.get("nesterov", True))
        )
    raise ValueError(f"unknown optimizer {ocfg['name']}")


def lr_at(step: int, total: int, warmup: int, cooldown: int, base: float, min_lr: float) -> float:
    """Linear warmup, cosine decay, then a flat cooldown at min_lr (as in the CCT/timm schedule)."""
    if warmup > 0 and step < warmup:
        return min_lr + (base - min_lr) * (step + 1) / warmup
    decay_steps = max(1, total - warmup - cooldown)
    t = min(1.0, (step - warmup) / decay_steps)
    return min_lr + (base - min_lr) * 0.5 * (1.0 + math.cos(math.pi * t))


def soft_cross_entropy(logits, target):
    return torch.sum(-target * F.log_softmax(logits.float(), dim=-1), dim=-1).mean()


def make_targets(images, labels, tcfg: dict, mixer, mixing: bool):
    if mixing and mixer is not None:
        images, target = mixer(images, labels)  # soft labels, shape (B, C)
    else:
        target = F.one_hot(labels, NUM_CLASSES).float()
    ls = float(tcfg.get("label_smoothing", 0.0))
    if ls > 0:
        target = target * (1.0 - ls) + ls / NUM_CLASSES
    return images, target


@torch.no_grad()
def evaluate(model, loader, device, amp_dtype):
    model.eval()
    conf = torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.long)
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            pred = model(x).argmax(dim=-1)
        conf += torch.bincount(y * NUM_CLASSES + pred, minlength=NUM_CLASSES**2).view(NUM_CLASSES, NUM_CLASSES).cpu()
    acc = conf.diag().sum().item() / conf.sum().item()
    per_class = (conf.diag().float() / conf.sum(dim=1).clamp(min=1).float()).tolist()
    return acc, per_class, conf.tolist()


def main():
    args = parse_args()
    cfg = apply_overrides(yaml.safe_load(open(args.config)), args.set)
    tcfg, ocfg, dcfg = cfg["train"], cfg["optim"], cfg["data"]
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else None
    torch.backends.cudnn.benchmark = True

    model = build_model(cfg["model"]).to(device)
    n_params = count_params(model)
    train_loader, eval_loader, eval_name = make_loaders(dcfg, tcfg["batch_size"], tcfg.get("eval_batch_size", 512), args.seed)
    optimizer = build_optimizer(model, ocfg)

    mixer = None
    if tcfg.get("mixup_alpha") or tcfg.get("cutmix_alpha"):
        choices = []
        if tcfg.get("mixup_alpha"):
            choices.append(v2.MixUp(alpha=float(tcfg["mixup_alpha"]), num_classes=NUM_CLASSES))
        if tcfg.get("cutmix_alpha"):
            choices.append(v2.CutMix(alpha=float(tcfg["cutmix_alpha"]), num_classes=NUM_CLASSES))
        mixer = v2.RandomChoice(choices)

    epochs = int(tcfg["epochs"])
    steps_per_epoch = len(train_loader)
    total_steps = epochs * steps_per_epoch
    warmup = int(float(ocfg.get("warmup_epochs", 0)) * steps_per_epoch)
    cooldown = int(float(ocfg.get("cooldown_epochs", 0)) * steps_per_epoch)
    base_lr, min_lr = float(ocfg["lr"]), float(ocfg.get("min_lr", 0.0))
    mix_off = int(tcfg.get("mix_off_epoch", epochs))
    clip = tcfg.get("clip_grad")

    def train_step(x, y, step, mixing):
        for g in optimizer.param_groups:
            g["lr"] = lr_at(step, total_steps, warmup, cooldown, base_lr, min_lr)
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        x, target = make_targets(x, y, tcfg, mixer, mixing)
        with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            loss = soft_cross_entropy(model(x), target)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(clip))
        optimizer.step()
        return loss.item()

    # ---------------------------------------------------------------- pilot mode
    if args.pilot_steps:
        model.train()
        skip = min(10, args.pilot_steps // 3)  # first steps include compilation and worker start-up
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        it, step, t0, timed_images = iter(train_loader), 0, None, 0
        while step < args.pilot_steps:
            try:
                x, y = next(it)
            except StopIteration:
                it = iter(train_loader)
                x, y = next(it)
            if step == skip:
                if device.type == "cuda":
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
            train_step(x, y, step, mixing=True)
            if step >= skip:
                timed_images += x.shape[0]
            step += 1
        if device.type == "cuda":
            torch.cuda.synchronize()
        seconds = time.perf_counter() - t0
        ips = timed_images / seconds
        full_images = TRAIN_SIZE_FULL * epochs
        out = {
            "run_name": cfg["run_name"],
            "model": cfg["model"],
            "params": n_params,
            "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
            "batch_size": tcfg["batch_size"],
            "timed_steps": args.pilot_steps - skip,
            "images_per_second": round(ips, 1),
            "peak_gpu_memory_GB": round(torch.cuda.max_memory_allocated() / 1e9, 2) if device.type == "cuda" else None,
            "epochs_in_config": epochs,
            "estimated_hours_full_run": round(full_images / ips / 3600, 2),
        }
        pilot_dir = Path(args.results) / "pilot"
        pilot_dir.mkdir(parents=True, exist_ok=True)
        (pilot_dir / f"{cfg['run_name']}.json").write_text(json.dumps(out, indent=1))
        print(json.dumps(out, indent=1))
        return

    # ---------------------------------------------------------------- full training
    out_dir = Path(args.results) / cfg["run_name"] / f"seed{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    ckpt_path = out_dir / "last.pt"
    start_epoch, train_seconds = 0, 0.0
    if ckpt_path.exists() and not args.no_resume:
        ck = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        start_epoch, train_seconds = ck["epoch"], ck.get("train_seconds", 0.0)
        torch.set_rng_state(ck["torch_rng"])
        print(f"resumed from epoch {start_epoch}")

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    log = open(out_dir / "log.jsonl", "a")
    print(f"{cfg['run_name']} seed {args.seed}: {n_params/1e6:.2f}M params, {steps_per_epoch} steps/epoch, eval on {eval_name}")

    for epoch in range(start_epoch, epochs):
        model.train()
        mixing = epoch < mix_off
        t0, loss_sum, n_seen = time.perf_counter(), 0.0, 0
        for i, (x, y) in enumerate(train_loader):
            loss_sum += train_step(x, y, epoch * steps_per_epoch + i, mixing) * x.shape[0]
            n_seen += x.shape[0]
        if device.type == "cuda":
            torch.cuda.synchronize()
        epoch_seconds = time.perf_counter() - t0
        train_seconds += epoch_seconds
        row = {
            "epoch": epoch + 1,
            "lr": optimizer.param_groups[0]["lr"],
            "train_loss": round(loss_sum / n_seen, 4),
            "epoch_seconds": round(epoch_seconds, 1),
            "images_per_second": round(n_seen / epoch_seconds, 1),
        }
        last = epoch + 1 == epochs
        if last or (epoch + 1) % int(tcfg.get("eval_every", 10)) == 0:
            row[f"{eval_name}_acc"] = round(evaluate(model, eval_loader, device, amp_dtype)[0], 4)
        log.write(json.dumps(row) + "\n")
        log.flush()
        print(row, flush=True)
        if last or (epoch + 1) % int(tcfg.get("ckpt_every", 25)) == 0:
            torch.save(
                {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch + 1,
                 "train_seconds": train_seconds, "torch_rng": torch.get_rng_state()},
                ckpt_path,
            )

    acc, per_class, conf = evaluate(model, eval_loader, device, amp_dtype)
    metrics = {
        "run_name": cfg["run_name"],
        "seed": args.seed,
        f"final_{eval_name}_acc": acc,
        "per_class_acc": per_class,
        "confusion_matrix": conf,
        "params": n_params,
        "epochs": epochs,
        "train_hours": round(train_seconds / 3600, 3),
        "peak_gpu_memory_GB": round(torch.cuda.max_memory_allocated() / 1e9, 2) if device.type == "cuda" else None,
        "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=1))
    print(json.dumps({k: v for k, v in metrics.items() if k not in ("confusion_matrix", "per_class_acc")}, indent=1))


if __name__ == "__main__":
    main()
