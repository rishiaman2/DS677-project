"""Summarize every finished run: learning-rate sweeps, seeds, replication anchor and the claim.

    python experiments/summarize_results.py            # prints markdown, writes results/summary.md

Reads results/<run_name>/seed<k>/metrics.json (written by src/train.py when a run finishes)
and the matching config.yaml. Runs still in progress have no metrics.json and are listed as
pending. Needs only the Python standard library and PyYAML.

Rules fixed before any results were seen:
* Learning rate: the sweep value with the highest validation accuracy (single seed, 30 epochs,
  5,000 held-out training images; the test set is never used). The ViT rate is swept on
  ViT-Lite-7/4 and applied to every ViT-Lite-7 config.
* Accuracy: final-epoch test accuracy, mean and sample standard deviation over seeds.
* Claim: holds if ResNet-18's mean test accuracy is at least 1.0 point above the ViT-Lite-7
  mean at every patch size tested (8, 4, 2). It is provisional until every group has 3 seeds.
"""
import json
import statistics
from collections import defaultdict
from pathlib import Path

import yaml

RESULTS = Path("results")
CONFIGS = Path("experiments/configs")

# Published CIFAR-10 test accuracies used as replication anchors (component C).
PUBLISHED = {
    "vitlite_p8": (89.10, "Hassani et al. 2021, ViT-Lite-7/8, 300 epochs"),
    "vitlite_p4": (93.57, "Hassani et al. 2021, ViT-Lite-7/4, 300 epochs"),
    "vitb16_finetune": (98.95, "Dosovitskiy et al. 2021, ViT-B/16 from ImageNet-21k, fine-tuned at 384 px"),
}
LABELS = {
    "resnet18": "ResNet-18 (baseline)",
    "vitlite_p8": "ViT-Lite-7/8",
    "vitlite_p4": "ViT-Lite-7/4",
    "vitlite_p2": "ViT-Lite-7/2",
    "vitlite_convstem": "ViT-Lite-7 + conv stem",
    "vitlite_p2_65ep": "ViT-Lite-7/2, 65 epochs (compute-matched)",
    "vitb16_finetune": "ViT-B/16 pretrained, fine-tuned",
}
CLAIM_PATCHES = ["vitlite_p8", "vitlite_p4", "vitlite_p2"]
CLAIM_MARGIN = 1.0
SEEDS_EXPECTED = 3
VIT_CONFIGS = ["vitlite_p8", "vitlite_p4", "vitlite_p2", "vitlite_convstem"]


def load_runs():
    finished, pending = defaultdict(list), defaultdict(list)
    for seed_dir in sorted(RESULTS.glob("*/seed*")):
        run = seed_dir.parent.name
        cfg_path = seed_dir / "config.yaml"
        cfg = yaml.safe_load(cfg_path.read_text()) if cfg_path.exists() else {}
        m_path = seed_dir / "metrics.json"
        if m_path.exists():
            m = json.loads(m_path.read_text())
            m["_cfg"] = cfg
            finished[run].append(m)
        else:
            log = seed_dir / "log.jsonl"
            epochs_done = sum(1 for _ in log.open()) if log.exists() else 0
            pending[run].append((seed_dir.name, epochs_done, (cfg.get("train") or {}).get("epochs")))
    return finished, pending


def pct(x):
    return 100.0 * x


def fmt_mean_std(vals):
    if len(vals) == 1:
        return f"{vals[0]:.2f}"
    return f"{statistics.mean(vals):.2f} ± {statistics.stdev(vals):.2f}"


def current_lr(config_name):
    path = CONFIGS / f"{config_name}.yaml"
    return float(yaml.safe_load(path.read_text())["optim"]["lr"]) if path.exists() else None


def sweep_section(finished, out):
    sweeps = {r: ms for r, ms in finished.items() if r.startswith("sweep_")}
    if not sweeps:
        out.append("## Learning-rate sweeps\n\nNo finished sweep runs yet.\n")
        return
    groups = defaultdict(list)
    for run, ms in sweeps.items():
        m = ms[0]
        cfg = m["_cfg"]
        model = cfg["model"]["name"] + (f"_p{cfg['model']['patch_size']}" if "patch_size" in cfg["model"] else "")
        groups[model].append((float(cfg["optim"]["lr"]), pct(m["final_val_acc"]), m.get("train_hours")))
    out.append("## Learning-rate sweeps (validation accuracy, seed 0, 30 epochs)\n")
    out.append("| Model | Learning rate | Validation accuracy (%) | Hours |")
    out.append("| --- | --- | --- | --- |")
    choices = {}
    for model, rows in sorted(groups.items()):
        rows.sort()
        best = max(rows, key=lambda r: r[1])
        choices[model] = best[0]
        for lr, acc, hours in rows:
            mark = " (best)" if lr == best[0] else ""
            out.append(f"| {model} | {lr:g}{mark} | {acc:.2f} | {hours} |")
    out.append("")
    out.append("### Apply the chosen rates (run on your laptop, then commit, push, and `git pull` on Wulver)\n")
    out.append("```bash")
    for model, lr in sorted(choices.items()):
        if model.startswith("resnet18"):
            targets = ["resnet18"]
        elif model.startswith("vit_lite"):
            targets = VIT_CONFIGS
        else:
            continue
        now = current_lr(targets[0])
        if now is not None and abs(now - lr) < 1e-12:
            out.append(f"# {model}: best rate {lr:g} is already in the configs, nothing to change")
            continue
        files = " ".join(f"experiments/configs/{t}.yaml" for t in targets)
        out.append(f"sed -i 's/^  lr: .*/  lr: {lr:g}  # chosen by the validation sweep/' {files}")
    out.append("```\n")


def main_section(finished, pending, out):
    main = {r: ms for r, ms in finished.items() if not r.startswith("sweep_")}
    out.append("## Main runs (final-epoch test accuracy)\n")
    if not main:
        out.append("No finished main runs yet.\n")
    else:
        out.append("| Run | Seeds done | Test accuracy (%), mean ± std | Per seed (%) | Train hours per run | Published (%) | Difference (points) |")
        out.append("| --- | --- | --- | --- | --- | --- | --- |")
        order = [r for r in LABELS if r in main] + sorted(r for r in main if r not in LABELS)
        for run in order:
            ms = sorted(main[run], key=lambda m: m["seed"])
            accs = [pct(m["final_test_acc"]) for m in ms if "final_test_acc" in m]
            if not accs:
                continue
            per_seed = ", ".join(f"s{m['seed']}: {pct(m['final_test_acc']):.2f}" for m in ms)
            hours = statistics.mean(m.get("train_hours") or 0 for m in ms)
            pub = PUBLISHED.get(run)
            pub_s, diff_s = ("", "")
            if pub:
                pub_s = f"{pub[0]:.2f}"
                diff_s = f"{statistics.mean(accs) - pub[0]:+.2f}"
            out.append(f"| {LABELS.get(run, run)} | {len(accs)}/{SEEDS_EXPECTED if run != 'vitb16_finetune' else 1} | "
                       f"{fmt_mean_std(accs)} | {per_seed} | {hours:.2f} | {pub_s} | {diff_s} |")
        out.append("")
        anchors = [r for r in PUBLISHED if r in main]
        if anchors:
            out.append("Published numbers: " + "; ".join(f"{LABELS[r]}: {PUBLISHED[r][1]}" for r in anchors) + ".\n")

    if pending:
        out.append("## Still running or unfinished\n")
        for run, items in sorted(pending.items()):
            for seed, done, total in items:
                out.append(f"- {run} {seed}: {done} of {total} epochs logged")
        out.append("")

    out.append("## Claim check\n")
    if "resnet18" not in main:
        out.append("Waiting for ResNet-18 results.\n")
        return
    base = [pct(m["final_test_acc"]) for m in main["resnet18"]]
    base_mean = statistics.mean(base)
    out.append(f"ResNet-18: {fmt_mean_std(base)} % over {len(base)} seed(s). The claim needs a gap of at least {CLAIM_MARGIN:.1f} point at every patch size.\n")
    out.append("| Patch size | ViT-Lite-7 test accuracy (%) | Gap to ResNet-18 (points) | At least 1 point? |")
    out.append("| --- | --- | --- | --- |")
    complete = len(base) >= SEEDS_EXPECTED
    verdicts = []
    for run in CLAIM_PATCHES:
        if run not in main:
            out.append(f"| {LABELS[run]} | not finished | | |")
            complete = False
            continue
        accs = [pct(m["final_test_acc"]) for m in main[run]]
        complete &= len(accs) >= SEEDS_EXPECTED
        gap = base_mean - statistics.mean(accs)
        ok = gap >= CLAIM_MARGIN
        verdicts.append(ok)
        out.append(f"| {LABELS[run]} | {fmt_mean_std(accs)} | {gap:+.2f} | {'yes' if ok else 'no'} |")
    out.append("")
    if verdicts and not all(verdicts):
        status = "does not hold (at least one patch size is within 1 point of ResNet-18 or above it)"
    elif verdicts:
        status = "holds for the patch sizes finished so far"
    else:
        status = "cannot be judged yet"
    out.append(f"**Claim status:** {status}" + ("." if complete else ", provisional until all 3 seeds of every model are in.") + "\n")


def main():
    finished, pending = load_runs()
    out = ["# Results summary\n", "Generated by `python experiments/summarize_results.py`.\n"]
    sweep_section(finished, out)
    main_section(finished, pending, out)
    text = "\n".join(out)
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "summary.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
