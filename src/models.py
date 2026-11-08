"""Model builders.

Every transformer in this project is the Hugging Face ViT implementation
(transformers.ViTForImageClassification), used at two scales:

* ViT-Lite-7 (7 layers, 4 heads, width 256, MLP width 512; Hassani et al., 2021),
  trained from scratch on 32x32 CIFAR-10 images with patch size 8, 4 or 2.
* The same ViT-Lite-7 with its linear patch embedding replaced by a small
  convolutional stem (extension, RQ4).
* google/vit-base-patch16-224-in21k, pretrained on ImageNet-21k (reference).

The baseline is the CIFAR-style ResNet-18 (3x3 stem, no max-pool).
"""
import torch
import torch.nn as nn
from transformers import ViTConfig, ViTForImageClassification

NUM_CLASSES = 10
VIT_B16_CHECKPOINT = "google/vit-base-patch16-224-in21k"


def vit_lite_config(patch_size: int, image_size: int = 32, attn: str = "sdpa") -> ViTConfig:
    cfg = ViTConfig(
        image_size=image_size,
        patch_size=patch_size,
        num_channels=3,
        hidden_size=256,
        num_hidden_layers=7,
        num_attention_heads=4,
        intermediate_size=512,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        num_labels=NUM_CLASSES,
    )
    # "eager" is needed when attention maps are read out (analysis); "sdpa" is faster for training.
    cfg._attn_implementation = attn
    return cfg


class HFClassifier(nn.Module):
    """Wraps a Hugging Face image classifier so forward(x) returns plain logits."""

    def __init__(self, hf_model: ViTForImageClassification):
        super().__init__()
        self.hf = hf_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.hf(pixel_values=x).logits


class ConvStemEmbedding(nn.Module):
    """Drop-in replacement for ViTPatchEmbeddings.

    Two stages of (3x3 conv, ReLU, 3x3 max-pool with stride 2) turn a 32x32 image
    into an 8x8 grid of tokens: 64 tokens, the same count as 4x4 patches, so the
    position embeddings and everything after the tokenizer stay unchanged.
    """

    def __init__(self, hidden_size: int = 256, mid_channels: int = 64):
        super().__init__()
        self.num_patches = 64
        self.stage1 = nn.Sequential(
            nn.Conv2d(3, mid_channels, 3, 1, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1),
        )
        # Named "projection" like the original patch embedding: ViTModel reads its weight dtype.
        self.projection = nn.Conv2d(mid_channels, hidden_size, 3, 1, 1, bias=False)
        self.pool2 = nn.MaxPool2d(3, 2, 1)

    def forward(self, pixel_values: torch.Tensor, interpolate_pos_encoding: bool = False) -> torch.Tensor:
        x = self.pool2(torch.relu(self.projection(self.stage1(pixel_values))))
        return x.flatten(2).transpose(1, 2)  # (B, 64, hidden)


class BasicBlock(nn.Module):
    def __init__(self, cin: int, cout: int, stride: int):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
        self.shortcut = nn.Sequential()
        if stride != 1 or cin != cout:
            self.shortcut = nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout))

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        return torch.relu(self.bn2(self.conv2(out)) + self.shortcut(x))


class ResNet18Cifar(nn.Module):
    """CIFAR-style ResNet-18: 3x3 stem with stride 1 and no max-pool (as in kuangliu/pytorch-cifar)."""

    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, 64, 3, 1, 1, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        blocks, cin = [], 64
        for cout, stride in [(64, 1), (128, 2), (256, 2), (512, 2)]:
            blocks += [BasicBlock(cin, cout, stride), BasicBlock(cout, cout, 1)]
            cin = cout
        self.stages = nn.Sequential(*blocks)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        return self.fc(self.stages(self.stem(x)).mean(dim=(2, 3)))


def build_model(mcfg: dict) -> nn.Module:
    name = mcfg["name"]
    attn = mcfg.get("attn", "sdpa")
    if name == "vit_lite":
        return HFClassifier(ViTForImageClassification(vit_lite_config(mcfg["patch_size"], attn=attn)))
    if name == "vit_lite_convstem":
        hf = ViTForImageClassification(vit_lite_config(4, attn=attn))
        hf.vit.embeddings.patch_embeddings = ConvStemEmbedding(hidden_size=256)
        return HFClassifier(hf)
    if name == "resnet18_cifar":
        return ResNet18Cifar()
    if name == "vit_b16_pretrained":
        hf = ViTForImageClassification.from_pretrained(
            mcfg.get("checkpoint", VIT_B16_CHECKPOINT), num_labels=NUM_CLASSES, attn_implementation=attn
        )
        return HFClassifier(hf)
    if name == "vit_b16_architecture":
        # Same architecture as the pretrained checkpoint but randomly initialised; used only to count cost.
        cfg = ViTConfig(num_labels=NUM_CLASSES)
        cfg._attn_implementation = attn
        return HFClassifier(ViTForImageClassification(cfg))
    raise ValueError(f"unknown model name: {name}")


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
