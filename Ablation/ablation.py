"""Self-contained ablation runner for the final CIFAR-10 DenseNet model.

All model, training, logging, plotting, and experiment-selection code lives in
this file so the ``Ablation`` directory can be copied or run independently from
the older experiment directories.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import AdamW, RMSprop, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR, OneCycleLR
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


ROOT = Path(__file__).resolve().parents[1]
ABLATION_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = ROOT / "data"
DEFAULT_OUTPUT_ROOT = ABLATION_ROOT / "results" / "ablation"

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)
CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


BASE_CONFIG: dict[str, Any] = {
    "depth": 190,
    "growth_rate": 40,
    "compression": 0.5,
    "activation": "silu",
    "se_reduction": 16,
    "stochastic_depth_rate": 0.2,
    "drop_rate": 0.0,
    "transition_dropout": 0.0,
    "classifier_hidden": 512,
    "classifier_dropout": 0.2,
    "loss": "ce",
    "label_smoothing": 0.1,
    "focal_gamma": 2.0,
    "optimizer": "sgd",
    "lr": 0.1,
    "min_lr": 1e-5,
    "momentum": 0.9,
    "weight_decay": 1e-4,
    "scheduler": "cosine",
    "milestones": [150, 225],
    "gamma": 0.1,
    "augment": "autoaugment",
    "cutout_length": 16,
    "no_cutout": False,
    "mix_mode": "cutmix",
    "mix_alpha": 1.0,
    "ema_decay": 0.999,
}


# ---------------------------------------------------------------------------
# Model definition
# ---------------------------------------------------------------------------


def make_activation(name: str, inplace: bool = True) -> nn.Module:
    name = name.lower()
    if name == "relu":
        return nn.ReLU(inplace=inplace)
    if name == "leaky_relu":
        return nn.LeakyReLU(negative_slope=0.1, inplace=inplace)
    if name == "elu":
        return nn.ELU(inplace=inplace)
    if name == "silu":
        return nn.SiLU(inplace=inplace)
    if name == "gelu":
        return nn.GELU()
    if name == "mish":
        return nn.Mish(inplace=inplace)
    raise ValueError(f"Unsupported activation: {name}")


class DropPath(nn.Module):
    """Per-sample stochastic depth for the newly produced DenseNet features."""

    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = x.new_empty(shape).bernoulli_(keep_prob)
        return x.div(keep_prob) * mask


class SEBlock(nn.Module):
    """Squeeze-and-Excitation channel attention."""

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.net(x)


class DenseLayer(nn.Module):
    """DenseNet-BC layer with optional SE, feature dropout, and DropPath."""

    def __init__(
        self,
        in_channels: int,
        growth_rate: int,
        bn_size: int,
        drop_rate: float,
        drop_path_rate: float,
        activation: str,
        se_reduction: int,
    ) -> None:
        super().__init__()
        inter_channels = bn_size * growth_rate
        self.net = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            make_activation(activation),
            nn.Conv2d(in_channels, inter_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(inter_channels),
            make_activation(activation),
            nn.Conv2d(inter_channels, growth_rate, kernel_size=3, padding=1, bias=False),
        )
        self.se = SEBlock(growth_rate, reduction=se_reduction) if se_reduction > 0 else nn.Identity()
        self.drop_rate = float(drop_rate)
        self.drop_path = DropPath(drop_path_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        new_features = self.net(x)
        new_features = self.se(new_features)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        new_features = self.drop_path(new_features)
        return torch.cat([x, new_features], dim=1)


class DenseBlock(nn.Module):
    def __init__(
        self,
        num_layers: int,
        in_channels: int,
        growth_rate: int,
        bn_size: int,
        drop_rate: float,
        drop_path_rates: list[float],
        activation: str,
        se_reduction: int,
    ) -> None:
        super().__init__()
        layers = []
        channels = in_channels
        for layer_idx in range(num_layers):
            layers.append(
                DenseLayer(
                    channels,
                    growth_rate,
                    bn_size,
                    drop_rate,
                    drop_path_rates[layer_idx],
                    activation,
                    se_reduction,
                )
            )
            channels += growth_rate
        self.block = nn.Sequential(*layers)
        self.out_channels = channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Transition(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, activation: str, transition_dropout: float) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.BatchNorm2d(in_channels),
            make_activation(activation),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
        ]
        if transition_dropout > 0:
            layers.append(nn.Dropout2d(p=transition_dropout))
        layers.append(nn.AvgPool2d(kernel_size=2, stride=2))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MyDenseNet(nn.Module):
    """SE-DenseNet-BC for CIFAR-10."""

    def __init__(
        self,
        depth: int = 190,
        growth_rate: int = 40,
        compression: float = 0.5,
        num_classes: int = 10,
        bn_size: int = 4,
        drop_rate: float = 0.0,
        transition_dropout: float = 0.0,
        stochastic_depth_rate: float = 0.2,
        activation: str = "silu",
        se_reduction: int = 16,
        classifier_hidden: int = 512,
        classifier_dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if not 0 < compression <= 1:
            raise ValueError("compression must be in (0, 1].")
        if (depth - 4) % 6 != 0:
            raise ValueError("DenseNet-BC depth should satisfy depth = 6n + 4.")

        layers_per_block = (depth - 4) // 6
        total_layers = layers_per_block * 3
        drop_path_rates = torch.linspace(0, stochastic_depth_rate, total_layers).tolist()

        self.depth = depth
        self.growth_rate = growth_rate
        self.compression = compression
        self.layers_per_block = layers_per_block
        self.activation_name = activation
        self.se_reduction = se_reduction
        self.stochastic_depth_rate = stochastic_depth_rate

        channels = 2 * growth_rate
        self.stem = nn.Sequential(
            nn.Conv2d(3, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            make_activation(activation),
        )

        cursor = 0
        self.block1 = DenseBlock(
            layers_per_block,
            channels,
            growth_rate,
            bn_size,
            drop_rate,
            drop_path_rates[cursor : cursor + layers_per_block],
            activation,
            se_reduction,
        )
        cursor += layers_per_block
        channels = self.block1.out_channels
        out_channels = math.floor(channels * compression)
        self.trans1 = Transition(channels, out_channels, activation, transition_dropout)
        channels = out_channels

        self.block2 = DenseBlock(
            layers_per_block,
            channels,
            growth_rate,
            bn_size,
            drop_rate,
            drop_path_rates[cursor : cursor + layers_per_block],
            activation,
            se_reduction,
        )
        cursor += layers_per_block
        channels = self.block2.out_channels
        out_channels = math.floor(channels * compression)
        self.trans2 = Transition(channels, out_channels, activation, transition_dropout)
        channels = out_channels

        self.block3 = DenseBlock(
            layers_per_block,
            channels,
            growth_rate,
            bn_size,
            drop_rate,
            drop_path_rates[cursor : cursor + layers_per_block],
            activation,
            se_reduction,
        )
        channels = self.block3.out_channels

        self.norm = nn.BatchNorm2d(channels)
        self.act = make_activation(activation)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        if classifier_hidden > 0:
            self.classifier = nn.Sequential(
                nn.Linear(channels, classifier_hidden),
                make_activation(activation, inplace=False),
                nn.Dropout(p=classifier_dropout),
                nn.Linear(classifier_hidden, num_classes),
            )
        else:
            self.classifier = nn.Linear(channels, num_classes)
        self._init_weights()

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.trans1(self.block1(x))
        x = self.trans2(self.block2(x))
        x = self.block3(x)
        return self.act(self.norm(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.01)
                nn.init.zeros_(module.bias)


def build_model(
    depth: int = 190,
    growth_rate: int = 40,
    compression: float = 0.5,
    activation: str = "silu",
    se_reduction: int = 16,
    stochastic_depth_rate: float = 0.2,
    classifier_hidden: int = 512,
    classifier_dropout: float = 0.2,
    drop_rate: float = 0.0,
    transition_dropout: float = 0.0,
    num_classes: int = 10,
) -> MyDenseNet:
    return MyDenseNet(
        depth=depth,
        growth_rate=growth_rate,
        compression=compression,
        activation=activation,
        se_reduction=se_reduction,
        stochastic_depth_rate=stochastic_depth_rate,
        classifier_hidden=classifier_hidden,
        classifier_dropout=classifier_dropout,
        drop_rate=drop_rate,
        transition_dropout=transition_dropout,
        num_classes=num_classes,
    )


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def describe_model(model: nn.Module) -> dict[str, int | float | str]:
    return {
        "model": model.__class__.__name__,
        "depth": getattr(model, "depth", "unknown"),
        "growth_rate": getattr(model, "growth_rate", "unknown"),
        "compression": getattr(model, "compression", "unknown"),
        "layers_per_block": getattr(model, "layers_per_block", "unknown"),
        "activation": getattr(model, "activation_name", "unknown"),
        "se_reduction": getattr(model, "se_reduction", "unknown"),
        "stochastic_depth_rate": getattr(model, "stochastic_depth_rate", "unknown"),
        "trainable_parameters": count_parameters(model),
    }


# ---------------------------------------------------------------------------
# Training utilities
# ---------------------------------------------------------------------------


class Cutout:
    def __init__(self, length: int = 16) -> None:
        self.length = length

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        _, height, width = image.shape
        y = np.random.randint(height)
        x = np.random.randint(width)
        y1 = max(0, y - self.length // 2)
        y2 = min(height, y + self.length // 2)
        x1 = max(0, x - self.length // 2)
        x2 = min(width, x + self.length // 2)
        image = image.clone()
        image[:, y1:y2, x1:x2] = 0
        return image


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, label_smoothing: float = 0.0) -> None:
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=1)
        probs = log_probs.exp()
        num_classes = logits.size(1)
        with torch.no_grad():
            true_dist = torch.zeros_like(log_probs)
            true_dist.fill_(self.label_smoothing / max(num_classes - 1, 1))
            true_dist.scatter_(1, targets.unsqueeze(1), 1.0 - self.label_smoothing)
        focal_weight = (1.0 - probs).pow(self.gamma)
        return -(true_dist * focal_weight * log_probs).sum(dim=1).mean()


class ModelEMA:
    def __init__(self, model: nn.Module, decay: float) -> None:
        self.decay = decay
        self.module = deepcopy(model).eval()
        for parameter in self.module.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        source = model.state_dict()
        target = self.module.state_dict()
        for key, value in target.items():
            source_value = source[key].detach()
            if value.dtype.is_floating_point:
                value.mul_(self.decay).add_(source_value, alpha=1.0 - self.decay)
            else:
                value.copy_(source_value)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def maybe_add_autoaugment(steps: list[Any]) -> None:
    if hasattr(transforms, "AutoAugment") and hasattr(transforms, "AutoAugmentPolicy"):
        steps.append(transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10))


def maybe_add_randaugment(steps: list[Any]) -> None:
    if hasattr(transforms, "RandAugment"):
        steps.append(transforms.RandAugment(num_ops=2, magnitude=9))


def build_loaders(args: SimpleNamespace, device: torch.device) -> tuple[DataLoader, DataLoader]:
    if args.augment == "none":
        train_steps: list[Any] = []
    else:
        train_steps = [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()]
        if args.augment == "autoaugment":
            maybe_add_autoaugment(train_steps)
        elif args.augment == "randaugment":
            maybe_add_randaugment(train_steps)

    train_steps.extend([transforms.ToTensor(), transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)])
    if not args.no_cutout and args.cutout_length > 0:
        train_steps.append(Cutout(args.cutout_length))

    test_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)])
    train_set = datasets.CIFAR10(
        root=str(args.data_dir),
        train=True,
        download=args.download,
        transform=transforms.Compose(train_steps),
    )
    test_set = datasets.CIFAR10(root=str(args.data_dir), train=False, download=args.download, transform=test_transform)
    if args.subset > 0:
        train_set = Subset(train_set, list(range(min(args.subset, len(train_set)))))

    loader_kwargs: dict[str, Any] = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
    }
    if args.num_workers > 0:
        loader_kwargs["persistent_workers"] = True
    return (
        DataLoader(train_set, shuffle=True, **loader_kwargs),
        DataLoader(test_set, shuffle=False, **loader_kwargs),
    )


def build_clean_train_loader(args: SimpleNamespace, device: torch.device) -> DataLoader:
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)])
    train_set = datasets.CIFAR10(root=str(args.data_dir), train=True, download=args.download, transform=transform)
    if args.subset > 0:
        train_set = Subset(train_set, list(range(min(args.subset, len(train_set)))))
    loader_kwargs: dict[str, Any] = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
    }
    if args.num_workers > 0:
        loader_kwargs["persistent_workers"] = True
    return DataLoader(train_set, shuffle=False, **loader_kwargs)


def build_criterion(args: SimpleNamespace) -> nn.Module:
    if args.loss == "focal":
        return FocalLoss(gamma=args.focal_gamma, label_smoothing=args.label_smoothing)
    return nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)


def build_optimizer(args: SimpleNamespace, model: nn.Module) -> torch.optim.Optimizer:
    if args.optimizer == "adamw":
        return AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if args.optimizer == "rmsprop":
        return RMSprop(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    return SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)


def build_scheduler(args: SimpleNamespace, optimizer: torch.optim.Optimizer, steps_per_epoch: int):
    if args.scheduler == "cosine":
        return CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.min_lr)
    if args.scheduler == "onecycle":
        return OneCycleLR(optimizer, max_lr=args.lr, epochs=args.epochs, steps_per_epoch=steps_per_epoch, pct_start=0.3)
    if args.scheduler == "multistep":
        return MultiStepLR(optimizer, milestones=args.milestones, gamma=args.gamma)
    return None


def make_grad_scaler(enabled: bool):
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda", enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def autocast_context(device: torch.device, enabled: bool):
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast(device_type=device.type, enabled=enabled)
    return torch.cuda.amp.autocast(enabled=enabled)


def rand_bbox(size: torch.Size, lam: float) -> tuple[int, int, int, int]:
    width = size[-1]
    height = size[-2]
    cut_ratio = math.sqrt(1.0 - lam)
    cut_w = int(width * cut_ratio)
    cut_h = int(height * cut_ratio)
    cx = np.random.randint(width)
    cy = np.random.randint(height)
    x1 = np.clip(cx - cut_w // 2, 0, width)
    y1 = np.clip(cy - cut_h // 2, 0, height)
    x2 = np.clip(cx + cut_w // 2, 0, width)
    y2 = np.clip(cy + cut_h // 2, 0, height)
    return int(x1), int(y1), int(x2), int(y2)


def apply_mix(
    images: torch.Tensor,
    targets: torch.Tensor,
    mode: str,
    alpha: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    if mode == "none" or alpha <= 0:
        return images, targets, targets, 1.0
    if mode == "random":
        mode = "mixup" if random.random() < 0.5 else "cutmix"

    lam = float(np.random.beta(alpha, alpha))
    index = torch.randperm(images.size(0), device=images.device)
    targets_b = targets[index]

    if mode == "mixup":
        return lam * images + (1.0 - lam) * images[index], targets, targets_b, lam
    if mode == "cutmix":
        x1, y1, x2, y2 = rand_bbox(images.size(), lam)
        mixed = images.clone()
        mixed[:, :, y1:y2, x1:x2] = images[index, :, y1:y2, x1:x2]
        lam = 1.0 - ((x2 - x1) * (y2 - y1) / (images.size(-1) * images.size(-2)))
        return mixed, targets, targets_b, lam
    raise ValueError(f"Unsupported mix mode: {mode}")


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
    scaler,
    args: SimpleNamespace,
    ema: ModelEMA | None,
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0.0
    total_samples = 0

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        images, targets_a, targets_b, lam = apply_mix(images, targets, args.mix_mode, args.mix_alpha)

        optimizer.zero_grad(set_to_none=True)
        with autocast_context(device, scaler.is_enabled()):
            logits = model(images)
            loss = lam * criterion(logits, targets_a) + (1.0 - lam) * criterion(logits, targets_b)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if scheduler is not None and args.scheduler == "onecycle":
            scheduler.step()
        if ema is not None:
            ema.update(model)

        batch_size = images.size(0)
        predictions = logits.argmax(dim=1)
        total_loss += loss.item() * batch_size
        total_correct += (
            lam * (predictions == targets_a).sum().item()
            + (1.0 - lam) * (predictions == targets_b).sum().item()
        )
        total_samples += batch_size

    return total_loss / total_samples, total_correct / total_samples


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    max_batches: int = 0,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    for batch_idx, (images, targets) in enumerate(loader):
        if max_batches > 0 and batch_idx >= max_batches:
            break
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, targets)
        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=1) == targets).sum().item()
        total_samples += batch_size
    return total_loss / max(total_samples, 1), total_correct / max(total_samples, 1)


def namespace_to_jsonable(args: SimpleNamespace) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in vars(args).items():
        result[key] = str(value) if isinstance(value, Path) else value
    return result


def save_config(path: Path, args: SimpleNamespace, model: nn.Module) -> None:
    config = namespace_to_jsonable(args)
    config["model_info"] = describe_model(model)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(config, file, indent=2)


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    best_acc: float,
    args: SimpleNamespace,
    ema: ModelEMA | None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "ema_state": ema.module.state_dict() if ema is not None else None,
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
            "best_acc": best_acc,
            "args": namespace_to_jsonable(args),
            "model": "AblationMyDenseNet",
        },
        path,
    )


def append_log(log_path: Path, row: dict[str, float | int]) -> None:
    fieldnames = list(row.keys())
    existing_rows: list[dict[str, str]] = []
    if log_path.exists():
        with log_path.open("r", newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            fieldnames = list(reader.fieldnames or fieldnames)
            existing_rows = list(reader)
        extra_fields = [key for key in row.keys() if key not in fieldnames]
        if extra_fields:
            fieldnames.extend(extra_fields)
            with log_path.open("w", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(file, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(existing_rows)

    with log_path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        if not existing_rows and log_path.stat().st_size == 0:
            writer.writeheader()
        writer.writerow(row)


def read_history(history_path: Path) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    with history_path.open("r", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            rows.append({key: float(value) for key, value in row.items() if key and value not in ("", None)})
    return rows


def summarize_history(history_path: Path) -> dict[str, float | int]:
    rows = read_history(history_path)
    if not rows:
        raise ValueError(f"No rows found in {history_path}")
    best = max(rows, key=lambda row: row["test_acc"])
    final = rows[-1]
    total_time = sum(row.get("time_sec", 0.0) for row in rows)
    summary: dict[str, float | int] = {
        "best_epoch": int(best["epoch"]),
        "best_test_acc": best["test_acc"],
        "best_test_error": 1.0 - best["test_acc"],
        "best_test_loss": best["test_loss"],
        "final_epoch": int(final["epoch"]),
        "final_train_acc": final["train_acc"],
        "final_train_loss": final["train_loss"],
        "final_test_acc": final["test_acc"],
        "final_test_loss": final["test_loss"],
        "total_time_sec": total_time,
        "total_time_min": total_time / 60.0,
    }
    if "clean_train_acc" in final:
        summary["final_clean_train_acc"] = final["clean_train_acc"]
        summary["final_clean_train_loss"] = final["clean_train_loss"]
    return summary


def write_summary(history_path: Path, output_path: Path) -> dict[str, float | int]:
    summary = summarize_history(history_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)
    return summary


def plot_history(history_path: Path, output_path: Path, title: str = "Ablation MyDenseNet on CIFAR-10") -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = read_history(history_path)
    epochs = [int(row["epoch"]) for row in rows]
    train_loss = [row["train_loss"] for row in rows]
    test_loss = [row["test_loss"] for row in rows]
    train_acc = [row["train_acc"] * 100.0 for row in rows]
    test_acc = [row["test_acc"] * 100.0 for row in rows]
    lr = [row["lr"] for row in rows]
    has_clean_train = all("clean_train_acc" in row for row in rows)
    clean_train_acc = [row["clean_train_acc"] * 100.0 for row in rows] if has_clean_train else None

    best_acc = []
    running_best = 0.0
    for acc in test_acc:
        running_best = max(running_best, acc)
        best_acc.append(running_best)

    train_error = [100.0 - acc for acc in train_acc]
    test_error = [100.0 - acc for acc in test_acc]
    if clean_train_acc is not None:
        clean_train_error = [100.0 - acc for acc in clean_train_acc]
        gap = [test - train for train, test in zip(clean_train_error, test_error)]
        gap_title = "Generalization Gap (Clean Train)"
    else:
        gap = [test - train for train, test in zip(train_error, test_error)]
        gap_title = "Logged Gap (Mixed Train)"

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes[0, 0].plot(epochs, train_loss, label="mixed train")
    axes[0, 0].plot(epochs, test_loss, label="test")
    axes[0, 0].set_title("Loss")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.25)

    axes[0, 1].plot(epochs, train_acc, label="mixed train")
    if clean_train_acc is not None:
        axes[0, 1].plot(epochs, clean_train_acc, label="clean train")
    axes[0, 1].plot(epochs, test_acc, label="test")
    axes[0, 1].plot(epochs, best_acc, linestyle="--", label="best test")
    axes[0, 1].set_title("Accuracy")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Accuracy (%)")
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.25)

    axes[0, 2].plot(epochs, lr)
    axes[0, 2].set_title("Learning Rate")
    axes[0, 2].set_xlabel("Epoch")
    axes[0, 2].set_ylabel("LR")
    axes[0, 2].set_yscale("log")
    axes[0, 2].grid(alpha=0.25)

    axes[1, 0].plot(epochs, train_error, label="mixed train")
    if clean_train_acc is not None:
        axes[1, 0].plot(epochs, clean_train_error, label="clean train")
    axes[1, 0].plot(epochs, test_error, label="test")
    axes[1, 0].set_title("Error")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Error (%)")
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.25)

    axes[1, 1].plot(epochs, gap, color="tab:purple")
    axes[1, 1].axhline(0, color="black", linewidth=0.8)
    axes[1, 1].set_title(gap_title)
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("Test error - train error (%)")
    axes[1, 1].grid(alpha=0.25)

    axes[1, 2].plot(epochs, test_error, color="tab:red")
    axes[1, 2].set_title("Test Error Zoom")
    axes[1, 2].set_xlabel("Epoch")
    axes[1, 2].set_ylabel("Test error (%)")
    axes[1, 2].grid(alpha=0.25)

    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def make_train_args(config: dict[str, Any], runtime: argparse.Namespace | SimpleNamespace, output_dir: Path) -> SimpleNamespace:
    merged = dict(BASE_CONFIG)
    merged.update(config)
    runtime_fields = {
        "data_dir": Path(runtime.data_dir),
        "output_dir": output_dir,
        "epochs": int(runtime.epochs),
        "batch_size": int(runtime.batch_size),
        "num_workers": int(runtime.num_workers),
        "seed": int(runtime.seed),
        "download": bool(runtime.download),
        "amp": bool(runtime.amp),
        "subset": int(getattr(runtime, "subset", 0)),
        "eval_max_batches": int(getattr(runtime, "eval_max_batches", 0)),
        "clean_train_eval": bool(getattr(runtime, "clean_train_eval", False)),
        "clean_train_max_batches": int(getattr(runtime, "clean_train_max_batches", 0)),
        "save_every": int(getattr(runtime, "save_every", 0)),
    }
    merged.update(runtime_fields)
    return SimpleNamespace(**merged)


def train_experiment(config: dict[str, Any], output_dir: Path, runtime: argparse.Namespace | SimpleNamespace) -> dict[str, Any]:
    args = make_train_args(config, runtime, output_dir)
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    weights_dir = args.output_dir / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader = build_loaders(args, device)
    clean_train_loader = build_clean_train_loader(args, device) if args.clean_train_eval else None
    model = build_model(
        depth=args.depth,
        growth_rate=args.growth_rate,
        compression=args.compression,
        activation=args.activation,
        se_reduction=args.se_reduction,
        stochastic_depth_rate=args.stochastic_depth_rate,
        classifier_hidden=args.classifier_hidden,
        classifier_dropout=args.classifier_dropout,
        drop_rate=args.drop_rate,
        transition_dropout=args.transition_dropout,
    ).to(device)
    criterion = build_criterion(args)
    optimizer = build_optimizer(args, model)
    scheduler = build_scheduler(args, optimizer, len(train_loader))
    scaler = make_grad_scaler(args.amp and device.type == "cuda")
    ema = ModelEMA(model, args.ema_decay) if args.ema_decay > 0 else None

    save_config(args.output_dir / "config.json", args, model)
    log_path = args.output_dir / "history.csv"
    if log_path.exists():
        log_path.unlink()

    print(f"Device: {device}")
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    print(f"Model: MyDenseNet depth={args.depth} growth={args.growth_rate} activation={args.activation}")
    print(f"Trainable parameters: {count_parameters(model):,}")
    print(f"Output dir: {args.output_dir.resolve()}")

    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        tic = time.time()
        lr = optimizer.param_groups[0]["lr"]
        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scheduler,
            device,
            scaler,
            args,
            ema,
        )
        eval_model = ema.module if ema is not None else model
        clean_train_loss = None
        clean_train_acc = None
        if clean_train_loader is not None:
            clean_train_loss, clean_train_acc = evaluate(
                eval_model,
                clean_train_loader,
                criterion,
                device,
                args.clean_train_max_batches,
            )
        test_loss, test_acc = evaluate(eval_model, test_loader, criterion, device, args.eval_max_batches)
        elapsed = time.time() - tic

        row: dict[str, float | int] = {
            "epoch": epoch,
            "lr": lr,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "test_loss": test_loss,
            "test_acc": test_acc,
            "time_sec": elapsed,
        }
        if clean_train_loss is not None and clean_train_acc is not None:
            row["clean_train_loss"] = clean_train_loss
            row["clean_train_acc"] = clean_train_acc
        append_log(log_path, row)

        if test_acc > best_acc or (epoch == 1 and not (weights_dir / "best.pt").exists()):
            best_acc = test_acc
            save_checkpoint(weights_dir / "best.pt", model, optimizer, scheduler, epoch, best_acc, args, ema)

        if args.save_every > 0 and epoch % args.save_every == 0:
            save_checkpoint(weights_dir / f"epoch_{epoch:03d}.pt", model, optimizer, scheduler, epoch, best_acc, args, ema)

        if scheduler is not None and args.scheduler != "onecycle":
            scheduler.step()

        save_checkpoint(weights_dir / "last.pt", model, optimizer, scheduler, epoch, best_acc, args, ema)
        clean_part = ""
        if clean_train_acc is not None:
            clean_part = f" | clean train acc {clean_train_acc * 100:.2f}%"
        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"lr {lr:.5f} | "
            f"train loss {train_loss:.4f} acc {train_acc * 100:.2f}% | "
            f"test loss {test_loss:.4f} acc {test_acc * 100:.2f}% | "
            f"best {best_acc * 100:.2f}% | "
            f"{elapsed:.1f}s"
            f"{clean_part}"
        )

    title = f"Ablation-MyDenseNet-{args.depth}-{args.growth_rate}-{args.activation}"
    plot_history(log_path, args.output_dir / "curves.png", title)
    summary = write_summary(log_path, args.output_dir / "summary.json")
    print(f"Best test accuracy: {summary['best_test_acc'] * 100:.2f}%")
    print(f"Best test error: {summary['best_test_error'] * 100:.2f}%")
    return summary


# ---------------------------------------------------------------------------
# Ablation orchestration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Experiment:
    name: str
    group: str
    description: str
    overrides: dict[str, Any]
    suites: tuple[str, ...] = ("report", "full")

    @property
    def config(self) -> dict[str, Any]:
        merged = dict(BASE_CONFIG)
        merged.update(self.overrides)
        return merged


EXPERIMENTS: tuple[Experiment, ...] = (
    Experiment(
        name="baseline_d190_k40_silu_ce_sgd",
        group="baseline",
        description="Default SE-DenseNet-BC-190-40 short run.",
        overrides={},
    ),
    Experiment("capacity_growth32", "capacity", "Reduce growth rate from 40 to 32.", {"growth_rate": 32}),
    Experiment("capacity_growth24", "capacity", "Reduce growth rate from 40 to 24.", {"growth_rate": 24}),
    Experiment("activation_relu", "activation", "Use ReLU instead of SiLU.", {"activation": "relu"}),
    Experiment("activation_gelu", "activation", "Use GELU instead of SiLU.", {"activation": "gelu"}),
    Experiment(
        "loss_focal",
        "loss",
        "Use focal loss with light label smoothing.",
        {"loss": "focal", "label_smoothing": 0.05, "focal_gamma": 2.0},
    ),
    Experiment(
        "optimizer_adamw",
        "optimizer",
        "Use AdamW from torch.optim.",
        {"optimizer": "adamw", "lr": 0.001, "weight_decay": 0.05},
    ),
    Experiment(
        "optimizer_rmsprop",
        "optimizer",
        "Use RMSprop from torch.optim.",
        {"optimizer": "rmsprop", "lr": 0.01, "weight_decay": 1e-4},
    ),
    Experiment("component_no_se", "component", "Remove SE channel attention.", {"se_reduction": 0}, ("full",)),
    Experiment(
        "component_no_stochastic_depth",
        "component",
        "Remove stochastic depth.",
        {"stochastic_depth_rate": 0.0},
        ("full",),
    ),
    Experiment(
        "regularization_no_classifier_dropout",
        "regularization",
        "Remove classifier dropout.",
        {"classifier_dropout": 0.0},
        ("full",),
    ),
    Experiment(
        "regularization_no_cutmix_cutout",
        "regularization",
        "Disable CutMix and Cutout.",
        {"mix_mode": "none", "no_cutout": True},
        ("full",),
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run self-contained MyDenseNet ablation experiments.")
    parser.add_argument("--epochs", type=int, default=50, help="Epoch budget for each ablation run.")
    parser.add_argument("--suite", choices=["report", "full"], default="report")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--subset", type=int, default=0, help="Optional train subset for smoke tests.")
    parser.add_argument("--eval-max-batches", type=int, default=0, help="Optional eval limit for smoke tests.")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--amp", dest="amp", action="store_true", default=True)
    parser.add_argument("--no-amp", dest="amp", action="store_false")
    parser.add_argument("--skip-existing", action="store_true", default=True)
    parser.add_argument("--rerun", dest="skip_existing", action="store_false")
    parser.add_argument("--continue-on-error", action="store_true")
    return parser.parse_args()


def selected_experiments(suite: str) -> list[Experiment]:
    return [experiment for experiment in EXPERIMENTS if suite in experiment.suites]


def run_dir(output_root: Path, experiment: Experiment) -> Path:
    return output_root / experiment.group / experiment.name


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def row_from_run(experiment: Experiment, destination: Path, status: str, message: str = "") -> dict[str, Any]:
    summary = load_json(destination / "summary.json")
    config = load_json(destination / "config.json")
    model_info = config.get("model_info", {}) if isinstance(config.get("model_info", {}), dict) else {}
    return {
        "experiment": experiment.name,
        "group": experiment.group,
        "description": experiment.description,
        "status": status,
        "message": message,
        "best_epoch": summary.get("best_epoch", ""),
        "best_test_acc": summary.get("best_test_acc", ""),
        "best_test_error": summary.get("best_test_error", ""),
        "best_test_loss": summary.get("best_test_loss", ""),
        "final_epoch": summary.get("final_epoch", ""),
        "final_test_acc": summary.get("final_test_acc", ""),
        "final_test_loss": summary.get("final_test_loss", ""),
        "total_time_min": summary.get("total_time_min", ""),
        "trainable_parameters": model_info.get("trainable_parameters", ""),
        "output_dir": str(destination),
        "history": str(destination / "history.csv"),
        "summary": str(destination / "summary.json"),
    }


def write_summary_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "experiment",
        "group",
        "description",
        "status",
        "message",
        "best_epoch",
        "best_test_acc",
        "best_test_error",
        "best_test_loss",
        "final_epoch",
        "final_test_acc",
        "final_test_loss",
        "total_time_min",
        "trainable_parameters",
        "output_dir",
        "history",
        "summary",
    ]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def as_float(value: Any, default: float = -1.0) -> float:
    try:
        if value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def choose_recommended_config(experiments: list[Experiment], rows: list[dict[str, Any]]) -> dict[str, Any]:
    ok_rows = [row for row in rows if row["status"] == "ok" and as_float(row["best_test_acc"]) >= 0.0]
    if not ok_rows:
        return {
            "recommended_config": dict(BASE_CONFIG),
            "best_single_experiment": None,
            "best_single_config": dict(BASE_CONFIG),
            "group_choices": [],
        }

    experiment_by_name = {experiment.name: experiment for experiment in experiments}
    row_by_name = {row["experiment"]: row for row in ok_rows}
    baseline = row_by_name.get("baseline_d190_k40_silu_ce_sgd")
    best_single = max(ok_rows, key=lambda row: as_float(row["best_test_acc"]))
    best_single_config = experiment_by_name[best_single["experiment"]].config

    recommended = dict(BASE_CONFIG)
    group_choices: list[dict[str, Any]] = []
    groups = [group for group in sorted({experiment.group for experiment in experiments}) if group != "baseline"]
    for group in groups:
        candidates = [row for row in ok_rows if row["group"] == group]
        if baseline is not None:
            candidates.append(baseline)
        if not candidates:
            continue
        best = max(candidates, key=lambda row: as_float(row["best_test_acc"]))
        source = experiment_by_name.get(best["experiment"])
        applied = {}
        if source is not None and source.group != "baseline":
            recommended.update(source.overrides)
            applied = source.overrides
        group_choices.append(
            {
                "group": group,
                "selected_experiment": best["experiment"],
                "best_test_acc": as_float(best["best_test_acc"]),
                "applied_overrides": applied,
            }
        )

    return {
        "recommended_config": recommended,
        "best_single_experiment": best_single,
        "best_single_config": best_single_config,
        "group_choices": group_choices,
    }


def main() -> None:
    args = parse_args()
    experiments = selected_experiments(args.suite)
    args.output_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for index, experiment in enumerate(experiments, start=1):
        destination = run_dir(args.output_root, experiment)
        summary_path = destination / "summary.json"
        print(f"\n[{index}/{len(experiments)}] {experiment.name} ({experiment.group})")
        print(f"Output: {destination}")
        if args.skip_existing and summary_path.exists():
            print("Skip existing finished run.")
            rows.append(row_from_run(experiment, destination, "ok", "skipped_existing"))
            continue
        try:
            train_experiment(experiment.config, destination, args)
            rows.append(row_from_run(experiment, destination, "ok"))
        except Exception as exc:
            rows.append(row_from_run(experiment, destination, "failed", repr(exc)))
            write_summary_csv(rows, args.output_root / "summary.csv")
            if not args.continue_on_error:
                raise

    summary_csv = args.output_root / "summary.csv"
    write_summary_csv(rows, summary_csv)
    recommendation = choose_recommended_config(experiments, rows)
    best_config_path = args.output_root / "best_config.json"
    with best_config_path.open("w", encoding="utf-8") as file:
        json.dump(
            {
                "suite": args.suite,
                "ablation_epochs": args.epochs,
                "summary_csv": str(summary_csv),
                "base_config": BASE_CONFIG,
                **recommendation,
            },
            file,
            indent=2,
        )

    print(f"\nSaved ablation summary: {summary_csv.resolve()}")
    print(f"Saved recommended config: {best_config_path.resolve()}")
    best = recommendation.get("best_single_experiment")
    if best:
        print(f"Best single short run: {best['experiment']} ({as_float(best['best_test_acc']) * 100:.2f}%)")


if __name__ == "__main__":
    main()
