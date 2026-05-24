"""Train the final SE-stochastic-depth WideResNet on CIFAR-10."""

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
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from model import build_model, count_parameters, describe_model
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


class EpochLRScheduler:
    """Small epoch-level scheduler with warmup support."""

    def __init__(self, args: argparse.Namespace, optimizer: torch.optim.Optimizer) -> None:
        self.args = args
        self.optimizer = optimizer
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]
        self.last_epoch = 0

    def _scale(self, epoch: int) -> float:
        if self.args.scheduler == "none":
            scale = 1.0
        elif self.args.scheduler == "multistep":
            scale = self.args.gamma ** sum(epoch > milestone for milestone in self.args.milestones)
        elif self.args.scheduler == "cosine":
            warmup = min(self.args.warmup_epochs, self.args.epochs - 1)
            if epoch <= warmup and warmup > 0:
                return max(epoch / warmup, 1e-8)
            progress = (epoch - warmup) / max(1, self.args.epochs - warmup)
            cosine = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
            min_scale = self.args.min_lr / max(self.args.lr, 1e-12)
            scale = min_scale + (1.0 - min_scale) * cosine
        else:
            raise ValueError(f"Unsupported scheduler: {self.args.scheduler}")

        if self.args.warmup_epochs > 0 and epoch <= self.args.warmup_epochs:
            scale *= max(epoch / self.args.warmup_epochs, 1e-8)
        return scale

    def step(self, epoch: int) -> None:
        self.last_epoch = epoch
        scale = self._scale(epoch)
        for group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            group["lr"] = base_lr * scale

    def state_dict(self) -> dict[str, object]:
        return {"last_epoch": self.last_epoch, "base_lrs": self.base_lrs}

    def load_state_dict(self, state: dict[str, object]) -> None:
        self.last_epoch = int(state.get("last_epoch", 0))


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    exp_root = Path(__file__).resolve().parent
    default_output = exp_root / "final_result" / "final_swrn40_10"

    parser = argparse.ArgumentParser(description="Final SE-stochastic-depth WideResNet for CIFAR-10")
    parser.add_argument("--data-dir", type=Path, default=project_root / "data")
    parser.add_argument("--output-dir", type=Path, default=default_output)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--depth", type=int, default=40)
    parser.add_argument("--widen-factor", type=int, default=10)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--head-dropout", type=float, default=0.0)
    parser.add_argument("--activation", choices=["relu", "leaky_relu", "elu", "silu", "gelu", "mish"], default="silu")
    parser.add_argument("--se-reduction", type=int, default=16, help="0 disables SE; 16 is the default final setting.")
    parser.add_argument("--stochastic-depth-rate", type=float, default=0.1)
    parser.add_argument("--loss", choices=["ce", "focal"], default="ce")
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--optimizer", choices=["sgd", "adamw", "rmsprop"], default="sgd")
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--min-lr", type=float, default=1e-5)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--scheduler", choices=["cosine", "multistep", "none"], default="cosine")
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--milestones", type=int, nargs="*", default=[150, 225])
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--augment", choices=["basic", "autoaugment", "randaugment", "none"], default="randaugment")
    parser.add_argument("--randaugment-ops", type=int, default=2)
    parser.add_argument("--randaugment-mag", type=int, default=9)
    parser.add_argument("--cutout-length", type=int, default=16)
    parser.add_argument("--no-cutout", action="store_true")
    parser.add_argument("--mix-mode", choices=["none", "mixup", "cutmix", "random"], default="cutmix")
    parser.add_argument("--mix-alpha", type=float, default=1.0)
    parser.add_argument("--mix-prob", type=float, default=1.0)
    parser.add_argument("--mix-off-epochs", type=int, default=20, help="Disable Mixup/CutMix for the final N epochs.")
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--grad-clip", type=float, default=0.0)
    parser.add_argument("--eval-tta", action="store_true", help="Use horizontal-flip TTA during evaluation.")
    parser.add_argument("--clean-train-eval", action="store_true", help="Also evaluate clean train accuracy each epoch.")
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


def maybe_add_autoaugment(steps: list[nn.Module]) -> None:
    if hasattr(transforms, "AutoAugment") and hasattr(transforms, "AutoAugmentPolicy"):
        steps.append(transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10))


def maybe_add_randaugment(args: argparse.Namespace, steps: list[nn.Module]) -> None:
    if hasattr(transforms, "RandAugment"):
        steps.append(transforms.RandAugment(num_ops=args.randaugment_ops, magnitude=args.randaugment_mag))


def build_transforms(args: argparse.Namespace, train: bool) -> transforms.Compose:
    if not train:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ])

    if args.augment == "none":
        steps: list[nn.Module] = []
    else:
        steps = [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
        ]
        if args.augment == "autoaugment":
            maybe_add_autoaugment(steps)
        elif args.augment == "randaugment":
            maybe_add_randaugment(args, steps)

    steps.extend([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    if not args.no_cutout and args.cutout_length > 0:
        steps.append(Cutout(args.cutout_length))
    return transforms.Compose(steps)


def build_loaders(args: argparse.Namespace, device: torch.device) -> tuple[DataLoader, DataLoader]:
    train_set = datasets.CIFAR10(
        root=str(args.data_dir),
        train=True,
        download=args.download,
        transform=build_transforms(args, train=True),
    )
    test_set = datasets.CIFAR10(
        root=str(args.data_dir),
        train=False,
        download=args.download,
        transform=build_transforms(args, train=False),
    )
    if args.subset > 0:
        train_set = Subset(train_set, list(range(min(args.subset, len(train_set)))))

    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
    }
    if args.num_workers > 0:
        loader_kwargs["persistent_workers"] = True

    train_loader = DataLoader(train_set, shuffle=True, **loader_kwargs)
    test_loader = DataLoader(test_set, shuffle=False, **loader_kwargs)
    return train_loader, test_loader


def build_clean_train_loader(args: argparse.Namespace, device: torch.device) -> DataLoader:
    train_set = datasets.CIFAR10(
        root=str(args.data_dir),
        train=True,
        download=args.download,
        transform=build_transforms(args, train=False),
    )
    if args.subset > 0:
        train_set = Subset(train_set, list(range(min(args.subset, len(train_set)))))
    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
        "shuffle": False,
    }
    if args.num_workers > 0:
        loader_kwargs["persistent_workers"] = True
    return DataLoader(train_set, **loader_kwargs)


def build_criterion(args: argparse.Namespace) -> nn.Module:
    if args.loss == "focal":
        return FocalLoss(gamma=args.focal_gamma, label_smoothing=args.label_smoothing)
    return nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)


def build_optimizer(args: argparse.Namespace, model: nn.Module) -> torch.optim.Optimizer:
    if args.optimizer == "adamw":
        return AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if args.optimizer == "rmsprop":
        return RMSprop(
            model.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
    return SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        nesterov=True,
    )


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
    probability: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    if mode == "none" or alpha <= 0 or random.random() > probability:
        return images, targets, targets, 1.0
    if mode == "random":
        mode = "mixup" if random.random() < 0.5 else "cutmix"

    lam = float(np.random.beta(alpha, alpha))
    index = torch.randperm(images.size(0), device=images.device)
    targets_b = targets[index]

    if mode == "mixup":
        mixed = lam * images + (1.0 - lam) * images[index]
        return mixed, targets, targets_b, lam

    if mode == "cutmix":
        x1, y1, x2, y2 = rand_bbox(images.size(), lam)
        mixed = images.clone()
        mixed[:, :, y1:y2, x1:x2] = images[index, :, y1:y2, x1:x2]
        lam = 1.0 - ((x2 - x1) * (y2 - y1) / (images.size(-1) * images.size(-2)))
        return mixed, targets, targets_b, lam

    raise ValueError(f"Unsupported mix mode: {mode}")


def active_mix_mode(args: argparse.Namespace, epoch: int) -> str:
    if args.mix_off_epochs > 0 and epoch > args.epochs - args.mix_off_epochs:
        return "none"
    return args.mix_mode


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler,
    args: argparse.Namespace,
    ema: ModelEMA | None,
    epoch: int,
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0.0
    total_samples = 0
    mode = active_mix_mode(args, epoch)

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        images, targets_a, targets_b, lam = apply_mix(images, targets, mode, args.mix_alpha, args.mix_prob)

        optimizer.zero_grad(set_to_none=True)
        with autocast_context(device, scaler.is_enabled()):
            logits = model(images)
            loss = lam * criterion(logits, targets_a) + (1.0 - lam) * criterion(logits, targets_b)

        scaler.scale(loss).backward()
        if args.grad_clip > 0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()

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
    tta: bool = False,
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
        if tta:
            logits = 0.5 * (logits + model(torch.flip(images, dims=(-1,))))
        loss = criterion(logits, targets)

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=1) == targets).sum().item()
        total_samples += batch_size

    return total_loss / total_samples, total_correct / total_samples


def serialize_args(args: argparse.Namespace) -> dict[str, object]:
    serialized: dict[str, object] = {}
    for key, value in vars(args).items():
        serialized[key] = str(value) if isinstance(value, Path) else value
    return serialized


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: EpochLRScheduler,
    epoch: int,
    best_acc: float,
    args: argparse.Namespace,
    ema: ModelEMA | None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "ema_state": ema.module.state_dict() if ema is not None else None,
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "best_acc": best_acc,
            "args": serialize_args(args),
            "model": "FinalCNN",
        },
        path,
    )


def append_log(log_path: Path, row: dict[str, float | int | str]) -> None:
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


def save_config(path: Path, args: argparse.Namespace, model: nn.Module) -> None:
    config = serialize_args(args)
    config["model_info"] = describe_model(model)
    with path.open("w", encoding="utf-8") as file:
        json.dump(config, file, indent=2)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    weights_dir = args.output_dir / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader = build_loaders(args, device)
    clean_train_loader = build_clean_train_loader(args, device) if args.clean_train_eval else None
    model = build_model(
        depth=args.depth,
        widen_factor=args.widen_factor,
        dropout=args.dropout,
        activation=args.activation,
        se_reduction=args.se_reduction,
        stochastic_depth_rate=args.stochastic_depth_rate,
        head_dropout=args.head_dropout,
    ).to(device)
    criterion = build_criterion(args)
    optimizer = build_optimizer(args, model)
    scheduler = EpochLRScheduler(args, optimizer)
    scaler = make_grad_scaler(args.amp and device.type == "cuda")
    ema = ModelEMA(model, args.ema_decay) if args.ema_decay > 0 else None

    start_epoch = 1
    best_acc = 0.0
    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        if checkpoint.get("scheduler_state") is not None:
            scheduler.load_state_dict(checkpoint["scheduler_state"])
        if ema is not None and checkpoint.get("ema_state") is not None:
            ema.module.load_state_dict(checkpoint["ema_state"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_acc = float(checkpoint.get("best_acc", 0.0))

    save_config(args.output_dir / "config.json", args, model)
    log_path = args.output_dir / "history.csv"
    if start_epoch == 1 and log_path.exists():
        log_path.unlink()

    print(f"Device: {device}")
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    print(f"Model: FinalCNN WRN-{args.depth}-{args.widen_factor}, SE={args.se_reduction}")
    print(f"Trainable parameters: {count_parameters(model):,}")
    print(f"Output dir: {args.output_dir.resolve()}")

    for epoch in range(start_epoch, args.epochs + 1):
        tic = time.time()
        scheduler.step(epoch)
        lr = optimizer.param_groups[0]["lr"]
        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            scaler,
            args,
            ema,
            epoch,
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
                tta=False,
            )
        test_loss, test_acc = evaluate(eval_model, test_loader, criterion, device, args.eval_max_batches, args.eval_tta)
        elapsed = time.time() - tic

        row: dict[str, float | int | str] = {
            "epoch": epoch,
            "lr": lr,
            "mix_mode": active_mix_mode(args, epoch),
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

        if test_acc > best_acc or (epoch == start_epoch and not (weights_dir / "best.pt").exists()):
            best_acc = test_acc
            save_checkpoint(weights_dir / "best.pt", model, optimizer, scheduler, epoch, best_acc, args, ema)

        if args.save_every > 0 and epoch % args.save_every == 0:
            save_checkpoint(weights_dir / f"epoch_{epoch:03d}.pt", model, optimizer, scheduler, epoch, best_acc, args, ema)

        save_checkpoint(weights_dir / "last.pt", model, optimizer, scheduler, epoch, best_acc, args, ema)
        clean_part = f" | clean train acc {clean_train_acc * 100:.2f}%" if clean_train_acc is not None else ""
        tta_part = " TTA" if args.eval_tta else ""
        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"lr {lr:.5f} | "
            f"mix {active_mix_mode(args, epoch)} | "
            f"train loss {train_loss:.4f} acc {train_acc * 100:.2f}% | "
            f"test{tta_part} loss {test_loss:.4f} acc {test_acc * 100:.2f}% | "
            f"best {best_acc * 100:.2f}% | "
            f"{elapsed:.1f}s"
            f"{clean_part}"
        )

    title = f"FinalCNN WRN-{args.depth}-{args.widen_factor} SE-SD on CIFAR-10"
    plot_history(log_path, args.output_dir / "curves.png", title)
    summary = write_summary(log_path, args.output_dir / "summary.json")
    print(f"Best test accuracy: {summary['best_test_acc'] * 100:.2f}%")
    print(f"Best test error: {summary['best_test_error'] * 100:.2f}%")
    print(f"Artifacts saved to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
