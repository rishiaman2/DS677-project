"""Count parameters and forward FLOPs for every model in the plan, using the same builders as training.

    python experiments/measure_cost.py            # prints a markdown table, writes results/model_cost.json

FLOPs come from torch.utils.flop_counter on one image with eager attention, so the attention
matmuls are included. Training cost per image is roughly 3x the forward cost (forward + backward).
Runs on CPU in a few seconds and needs no data or downloads.
"""
import json
import sys
from pathlib import Path

import torch
from torch.utils.flop_counter import FlopCounterMode

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from models import build_model, count_params  # noqa: E402

MODELS = [
    ("ResNet-18 (CIFAR stem)", {"name": "resnet18_cifar"}, 32, None),
    ("ViT-Lite-7/8", {"name": "vit_lite", "patch_size": 8, "attn": "eager"}, 32, 16 + 1),
    ("ViT-Lite-7/4", {"name": "vit_lite", "patch_size": 4, "attn": "eager"}, 32, 64 + 1),
    ("ViT-Lite-7/2", {"name": "vit_lite", "patch_size": 2, "attn": "eager"}, 32, 256 + 1),
    ("ViT-Lite-7/1 (pixels, not planned)", {"name": "vit_lite", "patch_size": 1, "attn": "eager"}, 32, 1024 + 1),
    ("ViT-Lite-7 + conv stem", {"name": "vit_lite_convstem", "attn": "eager"}, 32, 64 + 1),
    ("ViT-B/16 (pretrained architecture), 224 px", {"name": "vit_b16_architecture", "attn": "eager"}, 224, 196 + 1),
]


def main():
    rows = []
    for label, mcfg, res, tokens in MODELS:
        model = build_model(mcfg).eval()
        with FlopCounterMode(display=False) as fc, torch.no_grad():
            model(torch.randn(1, 3, res, res))
        rows.append({"model": label, "input": f"{res}x{res}", "tokens": tokens,
                     "params_M": round(count_params(model) / 1e6, 2),
                     "forward_GFLOPs": round(fc.get_total_flops() / 1e9, 3)})
    out = Path("results")
    out.mkdir(exist_ok=True)
    (out / "model_cost.json").write_text(json.dumps(rows, indent=1))
    print("| Model | Input | Tokens | Params (M) | Forward GFLOPs/image |")
    print("| --- | --- | --- | --- | --- |")
    for r in rows:
        print(f"| {r['model']} | {r['input']} | {r['tokens'] or 'n/a'} | {r['params_M']} | {r['forward_GFLOPs']} |")


if __name__ == "__main__":
    main()
