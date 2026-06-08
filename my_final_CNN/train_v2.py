"""Train FinalCNN v2, a validated SE-DenseNet-BC model on CIFAR-10."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import AdamW, RMSprop, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from model_v2 import build_model, count_parameters, describe_model
from plot_results import plot_history, write_summary


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


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


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    exp_root = Path(__file__).resolve().parent
    default_output = exp_root / "final_result" / "final_densenet_v2_seed42"

    parser = argparse.ArgumentParser(description="FinalCNN v2: high-accuracy SE-DenseNet for CIFAR-10")
    parser.add_argument("--data-dir", type=Path, default=project_root / "data")
    parser.add_argument("--output-dir", type=Path, default=default_output)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--depth", type=int, default=190)
    parser.add_argument("--growth-rate", type=int, default=40)
    parser.add_argument("--compression", type=float, default=0.5)
    parser.add_argument("--activation", choices=["relu", "leaky_relu", "elu", "silu", "gelu", "mish"], default="silu")
    parser.add_argument("--se-reduction", type=int, default=16)
    parser.add_argument("--stochastic-depth-rate", type=float, default=0.2)
    parser.add_argument("--drop-rate", type=float, default=0.0)
    parser.add_argument("--transition-dropout", type=float, default=0.0)
    parser.add_argument("--classifier-hidden", type=int, default=512)
    parser.add_argument("--classifier-dropout", type=float, default=0.2)
    parser.add_argument("--loss", choices=["ce", "focal"], default="ce")
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--optimizer", choices=["sgd", "adamw", "rmsprop"], default="sgd")
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--min-lr", type=float, default=1e-5)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--scheduler", choices=["cosine", "multistep", "none"], default="cosine")
    parser.add_argument("--milestones", type=int, nargs="*", default=[150, 225])
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--augment", choices=["basic", "autoaugment", "randaugment", "none"], default="autoaugment")
    parser.add_argument("--cutout-length", type=int, default=16)
    parser.add_argument("--no-cutout", action="store_true")
    parser.add_argument("--mix-mode", choices=["none", "mixup", "cutmix", "random"], default="cutmix")
    parser.add_argument("--mix-alpha", type=float, default=1.0)
    parser.add_argument("--mix-prob", type=float, default=1.0)
    parser.add_argument("--mix-off-epochs", type=int, default=0, help="Disable mixing only for an optional late fine-tune.")
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--eval-tta", action="store_true", help="Average original and horizontal-flipped logits in evaluation.")
    parser.add_argument("--clean-train-eval", action="store_true", help="Also log accuracy on the unaugmented train set.")
    parser.add_argument("--clean-train-max-batches", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--amp", action="store_true", help="Use CUDA mixed precision.")
    parser.add_argument("--subset", type=int, default=0, help="Use a small train subset for smoke tests.")
    parser.add_argument("--eval-max-batches", type=int, default=0, help="Limit eval batches for smoke tests.")
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--save-every", type=int, default=0, help="Save epoch_NNN.pt every N epochs.")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def build_transform(args: argparse.Namespace, train: bool) -> transforms.Compose:
    if not train:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ])

    if args.augment == "none":
        steps = []
    else:
        steps = [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()]
        if args.augment == "autoaugment" and hasattr(transforms, "AutoAugment"):
            steps.append(transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10))
        elif args.augment == "randaugment" and hasattr(transforms, "RandAugment"):
            steps.append(transforms.RandAugment(num_ops=2, magnitude=9))
    steps.extend([transforms.ToTensor(), transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)])
    if not args.no_cutout and args.cutout_length > 0:
        steps.append(Cutout(args.cutout_length))
    return transforms.Compose(steps)


def make_loader(dataset, args: argparse.Namespace, device: torch.device, shuffle: bool) -> DataLoader:
    kwargs = {
        "batch_size": args.batch_size,
        "shuffle": shuffle,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
    }
    if args.num_workers > 0:
        kwargs["persistent_workers"] = True
    return DataLoader(dataset, **kwargs)


def build_loaders(args: argparse.Namespace, device: torch.device) -> tuple[DataLoader, DataLoader, DataLoader | None]:
    train_set = datasets.CIFAR10(str(args.data_dir), train=True, download=args.download, transform=build_transform(args, True))
    test_set = datasets.CIFAR10(str(args.data_dir), train=False, download=args.download, transform=build_transform(args, False))
    clean_train_set = None
    if args.subset > 0:
        indices = list(range(min(args.subset, len(train_set))))
        train_set = Subset(train_set, indices)
        if args.clean_train_eval:
            full_clean_set = datasets.CIFAR10(str(args.data_dir), train=True, download=args.download, transform=build_transform(args, False))
            clean_train_set = Subset(full_clean_set, indices)
    elif args.clean_train_eval:
        clean_train_set = datasets.CIFAR10(str(args.data_dir), train=True, download=args.download, transform=build_transform(args, False))
    train_loader = make_loader(train_set, args, device, True)
    test_loader = make_loader(test_set, args, device, False)
    clean_loader = make_loader(clean_train_set, args, device, False) if clean_train_set is not None else None
    return train_loader, test_loader, clean_loader


def build_criterion(args: argparse.Namespace) -> nn.Module:
    if args.loss == "focal":
        return FocalLoss(args.focal_gamma, args.label_smoothing)
    return nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)


def build_optimizer(args: argparse.Namespace, model: nn.Module) -> torch.optim.Optimizer:
    if args.optimizer == "adamw":
        return AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if args.optimizer == "rmsprop":
        return RMSprop(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    return SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)


def build_scheduler(args: argparse.Namespace, optimizer: torch.optim.Optimizer):
    if args.scheduler == "cosine":
        return CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.min_lr)
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
    width, height = size[-1], size[-2]
    cut_ratio = math.sqrt(1.0 - lam)
    cut_w, cut_h = int(width * cut_ratio), int(height * cut_ratio)
    cx, cy = np.random.randint(width), np.random.randint(height)
    x1 = np.clip(cx - cut_w // 2, 0, width)
    y1 = np.clip(cy - cut_h // 2, 0, height)
    x2 = np.clip(cx + cut_w // 2, 0, width)
    y2 = np.clip(cy + cut_h // 2, 0, height)
    return int(x1), int(y1), int(x2), int(y2)


def active_mix_mode(args: argparse.Namespace, epoch: int) -> str:
    if args.mix_off_epochs > 0 and epoch > args.epochs - args.mix_off_epochs:
        return "none"
    return args.mix_mode


def apply_mix(images: torch.Tensor, targets: torch.Tensor, mode: str, alpha: float, probability: float):
    if mode == "none" or alpha <= 0 or random.random() > probability:
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


def train_one_epoch(model, loader, criterion, optimizer, device, scaler, args, ema, epoch: int):
    model.train()
    total_loss, total_correct, total_samples = 0.0, 0.0, 0
    mix_mode = active_mix_mode(args, epoch)
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        images, targets_a, targets_b, lam = apply_mix(images, targets, mix_mode, args.mix_alpha, args.mix_prob)
        optimizer.zero_grad(set_to_none=True)
        with autocast_context(device, scaler.is_enabled()):
            logits = model(images)
            loss = lam * criterion(logits, targets_a) + (1.0 - lam) * criterion(logits, targets_b)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        if ema is not None:
            ema.update(model)
        batch_size = images.size(0)
        predictions = logits.argmax(dim=1)
        total_loss += loss.item() * batch_size
        total_correct += lam * (predictions == targets_a).sum().item() + (1.0 - lam) * (predictions == targets_b).sum().item()
        total_samples += batch_size
    return total_loss / total_samples, total_correct / total_samples, mix_mode


@torch.no_grad()
def evaluate(model, loader, criterion, device, max_batches: int = 0, tta: bool = False):
    model.eval()
    total_loss, total_correct, total_samples = 0.0, 0, 0
    for batch_idx, (images, targets) in enumerate(loader):
        if max_batches > 0 and batch_idx >= max_batches:
            break
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model(images)
        if tta:
            logits = 0.5 * (logits + model(torch.flip(images, dims=(-1,))))
        loss = criterion(logits, targets)
        total_loss += loss.item() * images.size(0)
        total_correct += (logits.argmax(dim=1) == targets).sum().item()
        total_samples += images.size(0)
    return total_loss / total_samples, total_correct / total_samples


def serialize_args(args: argparse.Namespace) -> dict[str, object]:
    return {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}


def save_checkpoint(path: Path, model, optimizer, scheduler, epoch: int, best_acc: float, args, ema) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "ema_state": ema.module.state_dict() if ema is not None else None,
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
            "best_acc": best_acc,
            "args": serialize_args(args),
            "model": "FinalDenseNetV2",
        },
        path,
    )


def append_log(path: Path, row: dict[str, float | int | str]) -> None:
    fieldnames = list(row.keys())
    existing: list[dict[str, str]] = []
    if path.exists():
        with path.open("r", newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            fieldnames = list(reader.fieldnames or fieldnames)
            existing = list(reader)
        extras = [key for key in row if key not in fieldnames]
        if extras:
            fieldnames.extend(extras)
            with path.open("w", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(file, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(existing)
    with path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        if not existing and path.stat().st_size == 0:
            writer.writeheader()
        writer.writerow(row)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    weights_dir = args.output_dir / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader, clean_loader = build_loaders(args, device)
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
    scheduler = build_scheduler(args, optimizer)
    scaler = make_grad_scaler(args.amp and device.type == "cuda")
    ema = ModelEMA(model, args.ema_decay) if args.ema_decay > 0 else None

    start_epoch, best_acc = 1, 0.0
    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        if scheduler is not None and checkpoint.get("scheduler_state") is not None:
            scheduler.load_state_dict(checkpoint["scheduler_state"])
        if ema is not None and checkpoint.get("ema_state") is not None:
            ema.module.load_state_dict(checkpoint["ema_state"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_acc = float(checkpoint.get("best_acc", 0.0))

    config = serialize_args(args)
    config["model_info"] = describe_model(model)
    with (args.output_dir / "config.json").open("w", encoding="utf-8") as file:
        json.dump(config, file, indent=2)
    log_path = args.output_dir / "history.csv"
    if start_epoch == 1 and log_path.exists():
        log_path.unlink()

    print(f"Device: {device}")
    print(f"Model: FinalDenseNetV2 depth={args.depth} growth={args.growth_rate} activation={args.activation}")
    print(f"Trainable parameters: {count_parameters(model):,}")
    print(f"Output dir: {args.output_dir.resolve()}")

    for epoch in range(start_epoch, args.epochs + 1):
        tic = time.time()
        lr = optimizer.param_groups[0]["lr"]
        train_loss, train_acc, mix_mode = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler, args, ema, epoch)
        eval_model = ema.module if ema is not None else model
        test_loss, test_acc = evaluate(eval_model, test_loader, criterion, device, args.eval_max_batches, args.eval_tta)
        row: dict[str, float | int | str] = {
            "epoch": epoch,
            "lr": lr,
            "mix_mode": mix_mode,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "test_loss": test_loss,
            "test_acc": test_acc,
            "time_sec": time.time() - tic,
        }
        clean_acc = None
        if clean_loader is not None:
            clean_loss, clean_acc = evaluate(eval_model, clean_loader, criterion, device, args.clean_train_max_batches)
            row["clean_train_loss"] = clean_loss
            row["clean_train_acc"] = clean_acc
        append_log(log_path, row)
        if test_acc > best_acc:
            best_acc = test_acc
            save_checkpoint(weights_dir / "best.pt", model, optimizer, scheduler, epoch, best_acc, args, ema)
        if args.save_every > 0 and epoch % args.save_every == 0:
            save_checkpoint(weights_dir / f"epoch_{epoch:03d}.pt", model, optimizer, scheduler, epoch, best_acc, args, ema)
        if scheduler is not None:
            scheduler.step()
        save_checkpoint(weights_dir / "last.pt", model, optimizer, scheduler, epoch, best_acc, args, ema)
        clean_part = f" | clean train acc {clean_acc * 100:.2f}%" if clean_acc is not None else ""
        print(
            f"Epoch {epoch:03d}/{args.epochs} | lr {lr:.5f} | mix {mix_mode} | "
            f"train loss {train_loss:.4f} acc {train_acc * 100:.2f}% | "
            f"test loss {test_loss:.4f} acc {test_acc * 100:.2f}% | "
            f"best {best_acc * 100:.2f}% | {row['time_sec']:.1f}s{clean_part}"
        )

    title = f"FinalDenseNetV2-{args.depth}-{args.growth_rate}-{args.activation} on CIFAR-10"
    plot_history(log_path, args.output_dir / "curves.png", title)
    summary = write_summary(log_path, args.output_dir / "summary.json")
    print(f"Best test accuracy: {summary['best_test_acc'] * 100:.2f}%")
    print(f"Artifacts saved to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
