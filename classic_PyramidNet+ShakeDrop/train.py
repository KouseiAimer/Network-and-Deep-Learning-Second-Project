"""Train PyramidNet + ShakeDrop on CIFAR-10."""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from model import count_parameters, describe_model, pyramidnet_shakedrop
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


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    exp_root = Path(__file__).resolve().parent
    default_name = "pyramidnet110_a270_shakedrop"

    parser = argparse.ArgumentParser(description="PyramidNet + ShakeDrop for CIFAR-10")
    parser.add_argument("--data-dir", type=Path, default=project_root / "data")
    parser.add_argument("--weights-dir", type=Path, default=exp_root / "weights" / default_name)
    parser.add_argument("--results-dir", type=Path, default=exp_root / "results" / default_name)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--depth", type=int, default=110)
    parser.add_argument("--alpha", type=int, default=270)
    parser.add_argument("--final-survival-prob", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--scheduler", choices=["cosine", "multistep", "none"], default="cosine")
    parser.add_argument("--milestones", type=int, nargs="*", default=[150, 225])
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--mixup-alpha", type=float, default=0.0)
    parser.add_argument("--cutout-length", type=int, default=16)
    parser.add_argument("--no-cutout", action="store_true")
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--amp", action="store_true", help="Use CUDA mixed precision when available.")
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


def build_loaders(args: argparse.Namespace, device: torch.device) -> tuple[DataLoader, DataLoader]:
    if args.no_augment:
        train_steps = [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    else:
        train_steps = [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
        if not args.no_cutout and args.cutout_length > 0:
            train_steps.append(Cutout(args.cutout_length))

    train_transform = transforms.Compose(train_steps)
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    train_set = datasets.CIFAR10(
        root=str(args.data_dir),
        train=True,
        download=args.download,
        transform=train_transform,
    )
    test_set = datasets.CIFAR10(
        root=str(args.data_dir),
        train=False,
        download=args.download,
        transform=test_transform,
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


def build_scheduler(args: argparse.Namespace, optimizer: torch.optim.Optimizer):
    if args.scheduler == "cosine":
        return CosineAnnealingLR(optimizer, T_max=args.epochs)
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


def mixup_batch(
    images: torch.Tensor,
    targets: torch.Tensor,
    alpha: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    if alpha <= 0:
        return images, targets, targets, 1.0
    lam = np.random.beta(alpha, alpha)
    index = torch.randperm(images.size(0), device=images.device)
    mixed_images = lam * images + (1.0 - lam) * images[index]
    return mixed_images, targets, targets[index], float(lam)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler,
    mixup_alpha: float,
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0.0
    total_samples = 0

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        images, targets_a, targets_b, lam = mixup_batch(images, targets, mixup_alpha)

        optimizer.zero_grad(set_to_none=True)
        with autocast_context(device, scaler.is_enabled()):
            logits = model(images)
            loss = lam * criterion(logits, targets_a) + (1.0 - lam) * criterion(logits, targets_b)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

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
    scheduler,
    epoch: int,
    best_acc: float,
    args: argparse.Namespace,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
            "best_acc": best_acc,
            "args": serialize_args(args),
            "model": "PyramidNet+ShakeDrop",
        },
        path,
    )


def append_log(log_path: Path, row: dict[str, float | int]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not log_path.exists()
    with log_path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(row.keys()))
        if is_new:
            writer.writeheader()
        writer.writerow(row)


def save_config(path: Path, args: argparse.Namespace, model: nn.Module) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    config = serialize_args(args)
    config["model_info"] = describe_model(model)
    with path.open("w", encoding="utf-8") as file:
        json.dump(config, file, indent=2)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    args.weights_dir.mkdir(parents=True, exist_ok=True)
    args.results_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader = build_loaders(args, device)

    model = pyramidnet_shakedrop(
        depth=args.depth,
        alpha=args.alpha,
        final_survival_prob=args.final_survival_prob,
    ).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        nesterov=True,
    )
    scheduler = build_scheduler(args, optimizer)
    scaler = make_grad_scaler(args.amp and device.type == "cuda")

    start_epoch = 1
    best_acc = 0.0
    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        if scheduler is not None and checkpoint.get("scheduler_state") is not None:
            scheduler.load_state_dict(checkpoint["scheduler_state"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_acc = float(checkpoint.get("best_acc", 0.0))

    save_config(args.results_dir / "config.json", args, model)
    log_path = args.results_dir / "history.csv"
    if start_epoch == 1 and log_path.exists():
        log_path.unlink()

    print(f"Device: {device}")
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    print(f"Model: PyramidNet-{args.depth}-a{args.alpha}+ShakeDrop")
    print(f"Trainable parameters: {count_parameters(model):,}")
    print(f"Weights dir: {args.weights_dir.resolve()}")
    print(f"Results dir: {args.results_dir.resolve()}")

    for epoch in range(start_epoch, args.epochs + 1):
        tic = time.time()
        lr = optimizer.param_groups[0]["lr"]
        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            scaler,
            args.mixup_alpha,
        )
        test_loss, test_acc = evaluate(model, test_loader, criterion, device, args.eval_max_batches)
        elapsed = time.time() - tic

        row = {
            "epoch": epoch,
            "lr": lr,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "test_loss": test_loss,
            "test_acc": test_acc,
            "time_sec": elapsed,
        }
        append_log(log_path, row)

        if test_acc > best_acc or (epoch == start_epoch and not (args.weights_dir / "best.pt").exists()):
            best_acc = test_acc
            save_checkpoint(args.weights_dir / "best.pt", model, optimizer, scheduler, epoch, best_acc, args)

        if args.save_every > 0 and epoch % args.save_every == 0:
            save_checkpoint(
                args.weights_dir / f"epoch_{epoch:03d}.pt",
                model,
                optimizer,
                scheduler,
                epoch,
                best_acc,
                args,
            )

        if scheduler is not None:
            scheduler.step()

        save_checkpoint(args.weights_dir / "last.pt", model, optimizer, scheduler, epoch, best_acc, args)
        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"lr {lr:.5f} | "
            f"train loss {train_loss:.4f} acc {train_acc * 100:.2f}% | "
            f"test loss {test_loss:.4f} acc {test_acc * 100:.2f}% | "
            f"best {best_acc * 100:.2f}% | "
            f"{elapsed:.1f}s"
        )

    title = f"PyramidNet-{args.depth}-a{args.alpha} + ShakeDrop on CIFAR-10"
    plot_history(log_path, args.results_dir / "curves.png", title)
    summary = write_summary(log_path, args.results_dir / "summary.json")
    print(f"Best test accuracy: {summary['best_test_acc'] * 100:.2f}%")
    print(f"Best test error: {summary['best_test_error'] * 100:.2f}%")
    print(f"Artifacts saved to: {args.results_dir.resolve()}")


if __name__ == "__main__":
    main()
