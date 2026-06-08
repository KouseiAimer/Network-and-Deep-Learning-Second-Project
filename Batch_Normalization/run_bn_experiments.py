"""
Train VGG-A with and without Batch Normalization and plot loss landscapes.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

from data_utils import get_cifar_loaders
from models import VGG_A, VGG_A_BatchNorm, get_number_of_parameters


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data"
DEFAULT_RESULTS_DIR = Path(__file__).resolve().parent / "results"


@dataclass
class RunResult:
    model_name: str
    lr: float
    run_dir: str
    parameter_count: int
    best_val_accuracy: float
    best_epoch: int
    final_train_loss: float
    final_train_accuracy: float
    final_val_loss: float
    final_val_accuracy: float
    batch_losses_path: str
    grad_norms_path: str
    history_path: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run CIFAR-10 VGG-A BatchNorm comparison experiments."
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rates", type=float, nargs="+", default=[1e-3, 2e-3, 5e-4, 1e-4])
    parser.add_argument("--optimizer", choices=["adam", "sgd"], default="adam")
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2020)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--n-train-items", type=int, default=-1)
    parser.add_argument("--n-val-items", type=int, default=-1)
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--no-save-model", action="store_true")
    return parser.parse_args()


def set_random_seeds(seed_value: int, device: torch.device) -> None:
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if device.type == "cuda":
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def build_model(model_name: str) -> nn.Module:
    if model_name == "no_bn":
        return VGG_A()
    if model_name == "bn":
        return VGG_A_BatchNorm()
    raise ValueError(f"Unknown model name: {model_name}")


def build_optimizer(args: argparse.Namespace, model: nn.Module, lr: float) -> torch.optim.Optimizer:
    if args.optimizer == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=args.weight_decay)
    return torch.optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )


def tensor_grad_norm(parameters: Iterable[torch.nn.Parameter]) -> float:
    squared_norm = 0.0
    for parameter in parameters:
        if parameter.grad is None:
            continue
        squared_norm += float(parameter.grad.detach().norm(2).item() ** 2)
    return math.sqrt(squared_norm)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_items = 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        loss = criterion(logits, y)

        batch_size = y.size(0)
        total_loss += float(loss.item()) * batch_size
        total_correct += int((logits.argmax(dim=1) == y).sum().item())
        total_items += batch_size

    avg_loss = total_loss / max(total_items, 1)
    accuracy = total_correct / max(total_items, 1)
    return avg_loss, accuracy


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_series(path: Path, values: list[float], column_name: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["step", column_name])
        for step, value in enumerate(values):
            writer.writerow([step, value])


def load_series(path: Path, column_name: str) -> np.ndarray:
    with path.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        return np.asarray([float(row[column_name]) for row in reader], dtype=np.float64)


def train_one_run(
    args: argparse.Namespace,
    model_name: str,
    lr: float,
    device: torch.device,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
) -> RunResult:
    run_tag = f"{model_name}_lr_{lr:g}".replace(".", "p")
    run_dir = args.results_dir / run_tag
    run_dir.mkdir(parents=True, exist_ok=True)

    set_random_seeds(args.seed, device)
    model = build_model(model_name).to(device)
    optimizer = build_optimizer(args, model, lr=lr)
    criterion = nn.CrossEntropyLoss()

    batch_losses: list[float] = []
    grad_norms: list[float] = []
    history_rows: list[dict] = []
    best_val_accuracy = 0.0
    best_epoch = 0

    print(f"\n[{model_name}] lr={lr:g}, parameters={get_number_of_parameters(model):,}")
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        running_correct = 0
        running_items = 0

        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()

            grad_norm = tensor_grad_norm(model.parameters())
            optimizer.step()

            batch_size = y.size(0)
            loss_value = float(loss.item())
            batch_losses.append(loss_value)
            grad_norms.append(grad_norm)
            running_loss += loss_value * batch_size
            running_correct += int((logits.argmax(dim=1) == y).sum().item())
            running_items += batch_size

        train_loss = running_loss / max(running_items, 1)
        train_accuracy = running_correct / max(running_items, 1)
        val_loss, val_accuracy = evaluate(model, val_loader, criterion, device)

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_epoch = epoch
            if not args.no_save_model:
                torch.save(model.state_dict(), run_dir / "best.pt")

        history_rows.append(
            {
                "epoch": epoch,
                "lr": lr,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "val_loss": val_loss,
                "val_accuracy": val_accuracy,
                "mean_grad_norm": float(np.mean(grad_norms[-len(train_loader) :])),
                "max_grad_norm": float(np.max(grad_norms[-len(train_loader) :])),
            }
        )
        print(
            f"  epoch {epoch:03d}/{args.epochs}: "
            f"train_loss={train_loss:.4f}, train_acc={train_accuracy:.4f}, "
            f"val_loss={val_loss:.4f}, val_acc={val_accuracy:.4f}"
        )

    history_path = run_dir / "history.csv"
    batch_losses_path = run_dir / "batch_losses.csv"
    grad_norms_path = run_dir / "grad_norms.csv"
    write_rows(history_path, history_rows)
    write_series(batch_losses_path, batch_losses, "loss")
    write_series(grad_norms_path, grad_norms, "grad_norm")
    if not args.no_save_model:
        torch.save(model.state_dict(), run_dir / "last.pt")

    final_row = history_rows[-1]
    return RunResult(
        model_name=model_name,
        lr=lr,
        run_dir=str(run_dir),
        parameter_count=get_number_of_parameters(model),
        best_val_accuracy=best_val_accuracy,
        best_epoch=best_epoch,
        final_train_loss=float(final_row["train_loss"]),
        final_train_accuracy=float(final_row["train_accuracy"]),
        final_val_loss=float(final_row["val_loss"]),
        final_val_accuracy=float(final_row["val_accuracy"]),
        batch_losses_path=str(batch_losses_path),
        grad_norms_path=str(grad_norms_path),
        history_path=str(history_path),
    )


def save_landscape(model_name: str, results: list[RunResult], results_dir: Path) -> Path:
    series = []
    for result in results:
        if result.model_name == model_name:
            series.append(load_series(Path(result.batch_losses_path), "loss"))

    min_len = min(len(values) for values in series)
    aligned = np.vstack([values[:min_len] for values in series])
    min_curve = aligned.min(axis=0)
    max_curve = aligned.max(axis=0)

    output_path = results_dir / f"landscape_{model_name}.csv"
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["step", "min_loss", "max_loss"])
        for step, (min_loss, max_loss) in enumerate(zip(min_curve, max_curve)):
            writer.writerow([step, float(min_loss), float(max_loss)])
    return output_path


def read_landscape(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    steps, min_losses, max_losses = [], [], []
    with path.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            steps.append(int(row["step"]))
            min_losses.append(float(row["min_loss"]))
            max_losses.append(float(row["max_loss"]))
    return np.asarray(steps), np.asarray(min_losses), np.asarray(max_losses)


def plot_loss_landscape(landscape_paths: dict[str, Path], figure_dir: Path) -> Path:
    figure_dir.mkdir(parents=True, exist_ok=True)
    output_path = figure_dir / "loss_landscape_envelope.png"

    plt.figure(figsize=(10, 5))
    styles = {
        "no_bn": ("#cf5c36", "VGG-A without BN"),
        "bn": ("#2f7f6f", "VGG-A with BN"),
    }
    for model_name, path in landscape_paths.items():
        steps, min_curve, max_curve = read_landscape(path)
        color, label = styles[model_name]
        plt.plot(steps, (min_curve + max_curve) / 2, color=color, linewidth=1.5, label=label)
        plt.fill_between(steps, min_curve, max_curve, color=color, alpha=0.18)

    plt.xlabel("Training step")
    plt.ylabel("Cross-entropy loss")
    plt.title("Loss landscape envelope across learning rates")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()
    return output_path


def plot_main_training_curves(results: list[RunResult], figure_dir: Path) -> Path:
    figure_dir.mkdir(parents=True, exist_ok=True)
    output_path = figure_dir / "training_curves_first_lr.png"
    first_lr = results[0].lr
    selected = [result for result in results if result.lr == first_lr]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    styles = {
        "no_bn": ("#cf5c36", "VGG-A without BN"),
        "bn": ("#2f7f6f", "VGG-A with BN"),
    }
    for result in selected:
        with Path(result.history_path).open("r", newline="", encoding="utf-8") as file:
            rows = list(csv.DictReader(file))
        epochs = [int(row["epoch"]) for row in rows]
        train_loss = [float(row["train_loss"]) for row in rows]
        val_accuracy = [float(row["val_accuracy"]) for row in rows]
        color, label = styles[result.model_name]
        axes[0].plot(epochs, train_loss, color=color, marker="o", linewidth=1.5, label=label)
        axes[1].plot(epochs, val_accuracy, color=color, marker="o", linewidth=1.5, label=label)

    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Train loss")
    axes[0].grid(alpha=0.25)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Validation accuracy")
    axes[1].grid(alpha=0.25)
    axes[1].legend()
    fig.suptitle(f"Training comparison at lr={first_lr:g}")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def plot_grad_norms(results: list[RunResult], figure_dir: Path) -> Path:
    figure_dir.mkdir(parents=True, exist_ok=True)
    output_path = figure_dir / "gradient_norms_first_lr.png"
    first_lr = results[0].lr
    selected = [result for result in results if result.lr == first_lr]
    styles = {
        "no_bn": ("#cf5c36", "VGG-A without BN"),
        "bn": ("#2f7f6f", "VGG-A with BN"),
    }

    plt.figure(figsize=(10, 4))
    for result in selected:
        grad_norms = load_series(Path(result.grad_norms_path), "grad_norm")
        color, label = styles[result.model_name]
        plt.plot(np.arange(len(grad_norms)), grad_norms, color=color, linewidth=1.0, label=label, alpha=0.85)

    plt.xlabel("Training step")
    plt.ylabel("Global gradient norm")
    plt.title(f"Gradient norm comparison at lr={first_lr:g}")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()
    return output_path


def write_summary(results: list[RunResult], results_dir: Path, args: argparse.Namespace) -> None:
    rows = [asdict(result) for result in results]
    write_rows(results_dir / "summary.csv", rows)

    serializable_args = vars(args).copy()
    serializable_args["data_root"] = str(serializable_args["data_root"])
    serializable_args["results_dir"] = str(serializable_args["results_dir"])
    with (results_dir / "config.json").open("w", encoding="utf-8") as file:
        json.dump(serializable_args, file, indent=2, ensure_ascii=False)


def main() -> None:
    args = parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)
    figure_dir = args.results_dir / "figures"
    device = resolve_device(args.device)
    set_random_seeds(args.seed, device)

    train_loader, val_loader = get_cifar_loaders(
        root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        n_train_items=args.n_train_items,
        n_val_items=args.n_val_items,
        augment=args.augment,
    )

    all_results: list[RunResult] = []
    for lr in args.learning_rates:
        for model_name in ("no_bn", "bn"):
            all_results.append(
                train_one_run(
                    args=args,
                    model_name=model_name,
                    lr=lr,
                    device=device,
                    train_loader=train_loader,
                    val_loader=val_loader,
                )
            )

    landscape_paths = {
        "no_bn": save_landscape("no_bn", all_results, args.results_dir),
        "bn": save_landscape("bn", all_results, args.results_dir),
    }
    training_figure = plot_main_training_curves(all_results, figure_dir)
    landscape_figure = plot_loss_landscape(landscape_paths, figure_dir)
    grad_figure = plot_grad_norms(all_results, figure_dir)
    write_summary(all_results, args.results_dir, args)

    print("\nSaved experiment outputs:")
    print(f"  summary: {args.results_dir / 'summary.csv'}")
    print(f"  training curves: {training_figure}")
    print(f"  loss landscape: {landscape_figure}")
    print(f"  gradient norms: {grad_figure}")


if __name__ == "__main__":
    main()
