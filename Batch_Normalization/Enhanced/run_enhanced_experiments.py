"""
Run extended experiments for the Batch Normalization section.

The script supports three report-friendly extensions:
1. learning-rate sensitivity;
2. gradient predictiveness;
3. batch-size sensitivity;
plus a normalization-placement ablation.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
BN_ROOT = CURRENT_DIR.parent
PROJECT_ROOT = BN_ROOT.parent
if str(BN_ROOT) not in sys.path:
    sys.path.insert(0, str(BN_ROOT))

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

from data_utils import get_cifar_loaders
from enhanced_models import MODEL_DISPLAY_NAMES, build_enhanced_model
from models import get_number_of_parameters


DEFAULT_DATA_ROOT = PROJECT_ROOT / "data"
DEFAULT_RESULTS_DIR = CURRENT_DIR / "results"
DEFAULT_LR_SWEEP = [5e-5, 1e-4, 5e-4, 1e-3, 2e-3, 3e-3, 5e-3]
DEFAULT_BATCH_SIZES = [32, 64, 128, 256]
DEFAULT_NORM_MODELS = [
    "no_bn",
    "bn",
    "bn_after_relu",
    "bn_first_half",
    "bn_second_half",
    "groupnorm",
]


@dataclass(frozen=True)
class RunSpec:
    suite: str
    model_name: str
    lr: float
    batch_size: int

    @property
    def run_id(self) -> str:
        lr_part = f"{self.lr:g}".replace(".", "p")
        return f"{self.suite}_{self.model_name}_bs{self.batch_size}_lr{lr_part}"


@dataclass
class RunSummary:
    suite: str
    run_id: str
    model_name: str
    display_name: str
    lr: float
    batch_size: int
    parameter_count: int
    best_val_accuracy: float
    best_epoch: int
    final_train_loss: float
    final_train_accuracy: float
    final_val_loss: float
    final_val_accuracy: float
    mean_grad_norm: float
    mean_grad_change_norm: float
    mean_relative_grad_change: float
    mean_grad_cosine: float
    max_grad_diff_over_distance: float
    run_dir: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Enhanced BN experiments on CIFAR-10.")
    parser.add_argument(
        "--suite",
        choices=["lr_sweep", "batch_size", "norm_ablation", "gradient", "all"],
        default="all",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=DEFAULT_BATCH_SIZES)
    parser.add_argument("--learning-rates", type=float, nargs="+", default=DEFAULT_LR_SWEEP)
    parser.add_argument("--gradient-lrs", type=float, nargs="+", default=[1e-3, 2e-3])
    parser.add_argument("--norm-lr", type=float, default=1e-3)
    parser.add_argument("--batch-lr", type=float, default=1e-3)
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
    parser.add_argument("--group-norm-groups", type=int, default=32)
    parser.add_argument("--no-save-model", action="store_true")
    parser.add_argument(
        "--track-grad-every",
        type=int,
        default=1,
        help="Record full-gradient predictiveness every N steps. Use 0 to disable.",
    )
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


def build_optimizer(args: argparse.Namespace, model: nn.Module, lr: float) -> torch.optim.Optimizer:
    if args.optimizer == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=args.weight_decay)
    return torch.optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )


def flatten_parameters(model: nn.Module) -> torch.Tensor:
    return torch.cat([parameter.detach().flatten() for parameter in model.parameters()])


def flatten_gradients(model: nn.Module) -> torch.Tensor:
    grads = []
    for parameter in model.parameters():
        if parameter.grad is None:
            grads.append(torch.zeros_like(parameter, memory_format=torch.preserve_format).flatten())
        else:
            grads.append(parameter.grad.detach().flatten())
    return torch.cat(grads)


def scalar_mean(values: list[float], default: float = float("nan")) -> float:
    finite_values = [value for value in values if math.isfinite(value)]
    if not finite_values:
        return default
    return float(np.mean(finite_values))


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

    return total_loss / max(total_items, 1), total_correct / max(total_items, 1)


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def generate_specs(args: argparse.Namespace) -> list[RunSpec]:
    specs: list[RunSpec] = []
    suites = ["lr_sweep", "batch_size", "norm_ablation", "gradient"]
    selected_suites = suites if args.suite == "all" else [args.suite]

    if "lr_sweep" in selected_suites:
        for lr in args.learning_rates:
            for model_name in ("no_bn", "bn"):
                specs.append(RunSpec("lr_sweep", model_name, lr, args.batch_size))

    if "batch_size" in selected_suites:
        for batch_size in args.batch_sizes:
            for model_name in ("no_bn", "bn"):
                specs.append(RunSpec("batch_size", model_name, args.batch_lr, batch_size))

    if "norm_ablation" in selected_suites:
        for model_name in DEFAULT_NORM_MODELS:
            specs.append(RunSpec("norm_ablation", model_name, args.norm_lr, args.batch_size))

    if "gradient" in selected_suites:
        for lr in args.gradient_lrs:
            for model_name in ("no_bn", "bn"):
                specs.append(RunSpec("gradient", model_name, lr, args.batch_size))

    deduped: dict[str, RunSpec] = {}
    for spec in specs:
        deduped[spec.run_id] = spec
    return list(deduped.values())


def train_one_spec(
    args: argparse.Namespace,
    spec: RunSpec,
    device: torch.device,
) -> RunSummary:
    set_random_seeds(args.seed, device)
    train_loader, val_loader = get_cifar_loaders(
        root=args.data_root,
        batch_size=spec.batch_size,
        num_workers=args.num_workers,
        n_train_items=args.n_train_items,
        n_val_items=args.n_val_items,
        augment=args.augment,
    )

    run_dir = args.results_dir / spec.suite / spec.run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    model = build_enhanced_model(
        spec.model_name,
        group_norm_groups=args.group_norm_groups,
    ).to(device)
    optimizer = build_optimizer(args, model, lr=spec.lr)
    criterion = nn.CrossEntropyLoss()

    history_rows: list[dict] = []
    step_rows: list[dict] = []
    best_val_accuracy = 0.0
    best_epoch = 0
    global_step = 0
    prev_grad: torch.Tensor | None = None
    prev_params: torch.Tensor | None = None

    print(
        f"\n[{spec.suite}] {spec.model_name} "
        f"lr={spec.lr:g}, batch_size={spec.batch_size}, "
        f"parameters={get_number_of_parameters(model):,}"
    )

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_items = 0
        epoch_grad_norms: list[float] = []
        epoch_grad_changes: list[float] = []
        epoch_relative_changes: list[float] = []
        epoch_grad_cosines: list[float] = []
        epoch_grad_diff_over_distance: list[float] = []

        for x, y in train_loader:
            global_step += 1
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()

            should_track = args.track_grad_every > 0 and global_step % args.track_grad_every == 0
            grad_norm = float("nan")
            grad_change_norm = float("nan")
            relative_grad_change = float("nan")
            grad_cosine = float("nan")
            param_distance = float("nan")
            grad_diff_over_distance = float("nan")

            if should_track:
                current_grad = flatten_gradients(model)
                current_params = flatten_parameters(model)
                grad_norm = float(torch.linalg.vector_norm(current_grad).item())
                if prev_grad is not None and prev_params is not None:
                    grad_diff = current_grad - prev_grad
                    grad_change_norm = float(torch.linalg.vector_norm(grad_diff).item())
                    prev_grad_norm = float(torch.linalg.vector_norm(prev_grad).item())
                    relative_grad_change = grad_change_norm / (prev_grad_norm + 1e-12)
                    grad_cosine = float(
                        torch.nn.functional.cosine_similarity(
                            current_grad.unsqueeze(0),
                            prev_grad.unsqueeze(0),
                            dim=1,
                            eps=1e-12,
                        ).item()
                    )
                    param_distance = float(
                        torch.linalg.vector_norm(current_params - prev_params).item()
                    )
                    grad_diff_over_distance = grad_change_norm / (param_distance + 1e-12)

                prev_grad = current_grad.detach().clone()
                prev_params = current_params.detach().clone()

                epoch_grad_norms.append(grad_norm)
                epoch_grad_changes.append(grad_change_norm)
                epoch_relative_changes.append(relative_grad_change)
                epoch_grad_cosines.append(grad_cosine)
                epoch_grad_diff_over_distance.append(grad_diff_over_distance)

            optimizer.step()

            batch_size = y.size(0)
            loss_value = float(loss.item())
            epoch_loss += loss_value * batch_size
            epoch_correct += int((logits.argmax(dim=1) == y).sum().item())
            epoch_items += batch_size

            step_rows.append(
                {
                    "step": global_step,
                    "epoch": epoch,
                    "loss": loss_value,
                    "grad_norm": grad_norm,
                    "grad_change_norm": grad_change_norm,
                    "relative_grad_change": relative_grad_change,
                    "grad_cosine": grad_cosine,
                    "param_distance": param_distance,
                    "grad_diff_over_distance": grad_diff_over_distance,
                }
            )

        train_loss = epoch_loss / max(epoch_items, 1)
        train_accuracy = epoch_correct / max(epoch_items, 1)
        val_loss, val_accuracy = evaluate(model, val_loader, criterion, device)

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_epoch = epoch
            if not args.no_save_model:
                torch.save(model.state_dict(), run_dir / "best.pt")

        history_rows.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "val_loss": val_loss,
                "val_accuracy": val_accuracy,
                "mean_grad_norm": scalar_mean(epoch_grad_norms),
                "mean_grad_change_norm": scalar_mean(epoch_grad_changes),
                "mean_relative_grad_change": scalar_mean(epoch_relative_changes),
                "mean_grad_cosine": scalar_mean(epoch_grad_cosines),
                "mean_grad_diff_over_distance": scalar_mean(epoch_grad_diff_over_distance),
            }
        )
        print(
            f"  epoch {epoch:03d}/{args.epochs}: "
            f"train_loss={train_loss:.4f}, train_acc={train_accuracy:.4f}, "
            f"val_loss={val_loss:.4f}, val_acc={val_accuracy:.4f}"
        )

    if not args.no_save_model:
        torch.save(model.state_dict(), run_dir / "last.pt")

    write_rows(run_dir / "history.csv", history_rows)
    write_rows(run_dir / "step_metrics.csv", step_rows)

    final_row = history_rows[-1]
    finite_grad_diff_over_distance = [
        float(row["grad_diff_over_distance"])
        for row in step_rows
        if math.isfinite(float(row["grad_diff_over_distance"]))
    ]

    return RunSummary(
        suite=spec.suite,
        run_id=spec.run_id,
        model_name=spec.model_name,
        display_name=MODEL_DISPLAY_NAMES[spec.model_name],
        lr=spec.lr,
        batch_size=spec.batch_size,
        parameter_count=get_number_of_parameters(model),
        best_val_accuracy=best_val_accuracy,
        best_epoch=best_epoch,
        final_train_loss=float(final_row["train_loss"]),
        final_train_accuracy=float(final_row["train_accuracy"]),
        final_val_loss=float(final_row["val_loss"]),
        final_val_accuracy=float(final_row["val_accuracy"]),
        mean_grad_norm=scalar_mean([float(row["grad_norm"]) for row in step_rows]),
        mean_grad_change_norm=scalar_mean(
            [float(row["grad_change_norm"]) for row in step_rows]
        ),
        mean_relative_grad_change=scalar_mean(
            [float(row["relative_grad_change"]) for row in step_rows]
        ),
        mean_grad_cosine=scalar_mean([float(row["grad_cosine"]) for row in step_rows]),
        max_grad_diff_over_distance=max(finite_grad_diff_over_distance, default=float("nan")),
        run_dir=str(run_dir),
    )


def plot_lr_sensitivity(rows: list[RunSummary], figure_dir: Path) -> None:
    selected = [row for row in rows if row.suite == "lr_sweep"]
    if not selected:
        return
    figure_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5))
    for model_name, color in (("no_bn", "#cf5c36"), ("bn", "#2f7f6f")):
        model_rows = sorted([row for row in selected if row.model_name == model_name], key=lambda r: r.lr)
        plt.plot(
            [row.lr for row in model_rows],
            [row.best_val_accuracy for row in model_rows],
            marker="o",
            linewidth=1.8,
            color=color,
            label=MODEL_DISPLAY_NAMES[model_name],
        )
    plt.xscale("log")
    plt.xlabel("Learning rate")
    plt.ylabel("Best validation accuracy")
    plt.title("Learning-rate sensitivity")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(figure_dir / "lr_sensitivity.png", dpi=180)
    plt.close()


def plot_batch_size_sensitivity(rows: list[RunSummary], figure_dir: Path) -> None:
    selected = [row for row in rows if row.suite == "batch_size"]
    if not selected:
        return
    figure_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5))
    for model_name, color in (("no_bn", "#cf5c36"), ("bn", "#2f7f6f")):
        model_rows = sorted(
            [row for row in selected if row.model_name == model_name],
            key=lambda r: r.batch_size,
        )
        plt.plot(
            [row.batch_size for row in model_rows],
            [row.best_val_accuracy for row in model_rows],
            marker="o",
            linewidth=1.8,
            color=color,
            label=MODEL_DISPLAY_NAMES[model_name],
        )
    plt.xlabel("Batch size")
    plt.ylabel("Best validation accuracy")
    plt.title("Batch-size sensitivity")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(figure_dir / "batch_size_sensitivity.png", dpi=180)
    plt.close()


def plot_norm_ablation(rows: list[RunSummary], figure_dir: Path) -> None:
    selected = [row for row in rows if row.suite == "norm_ablation"]
    if not selected:
        return
    figure_dir.mkdir(parents=True, exist_ok=True)
    selected = sorted(selected, key=lambda row: row.best_val_accuracy, reverse=True)
    labels = [MODEL_DISPLAY_NAMES[row.model_name] for row in selected]
    values = [row.best_val_accuracy for row in selected]
    colors = ["#2f7f6f" if row.model_name == "bn" else "#7d8a99" for row in selected]
    plt.figure(figsize=(10, 5))
    plt.bar(labels, values, color=colors)
    plt.ylabel("Best validation accuracy")
    plt.title("Normalization placement ablation")
    plt.xticks(rotation=25, ha="right")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(figure_dir / "norm_ablation.png", dpi=180)
    plt.close()


def _load_step_metric(run_dir: Path, column: str) -> tuple[np.ndarray, np.ndarray]:
    steps, values = [], []
    with (run_dir / "step_metrics.csv").open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            value = float(row[column])
            if math.isfinite(value):
                steps.append(int(row["step"]))
                values.append(value)
    return np.asarray(steps), np.asarray(values)


def _rolling_mean(values: np.ndarray, window: int = 80) -> np.ndarray:
    if len(values) == 0:
        return values
    window = min(window, len(values))
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="same")


def plot_gradient_predictiveness(rows: list[RunSummary], figure_dir: Path) -> None:
    selected = [row for row in rows if row.suite == "gradient"]
    if not selected:
        return
    figure_dir.mkdir(parents=True, exist_ok=True)
    lrs = sorted({row.lr for row in selected})
    target_lr = lrs[0]
    selected = [row for row in selected if row.lr == target_lr]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for row, color in zip(
        sorted(selected, key=lambda item: item.model_name),
        ["#2f7f6f", "#cf5c36"],
    ):
        steps, rel_change = _load_step_metric(Path(row.run_dir), "relative_grad_change")
        _, cosine = _load_step_metric(Path(row.run_dir), "grad_cosine")
        label = MODEL_DISPLAY_NAMES[row.model_name]
        axes[0].plot(steps, _rolling_mean(rel_change), color=color, linewidth=1.4, label=label)
        axes[1].plot(steps, _rolling_mean(cosine), color=color, linewidth=1.4, label=label)

    axes[0].set_title(f"Relative gradient change, lr={target_lr:g}")
    axes[0].set_xlabel("Training step")
    axes[0].set_ylabel("Rolling mean")
    axes[0].grid(alpha=0.25)
    axes[1].set_title(f"Gradient cosine similarity, lr={target_lr:g}")
    axes[1].set_xlabel("Training step")
    axes[1].set_ylabel("Rolling mean")
    axes[1].grid(alpha=0.25)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(figure_dir / "gradient_predictiveness.png", dpi=180)
    plt.close(fig)


def save_config(args: argparse.Namespace, specs: list[RunSpec]) -> None:
    args.results_dir.mkdir(parents=True, exist_ok=True)
    config = vars(args).copy()
    config["data_root"] = str(config["data_root"])
    config["results_dir"] = str(config["results_dir"])
    config["planned_runs"] = [spec.__dict__ | {"run_id": spec.run_id} for spec in specs]
    with (args.results_dir / "config.json").open("w", encoding="utf-8") as file:
        json.dump(config, file, indent=2, ensure_ascii=False)


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    figure_dir = args.results_dir / "figures"
    specs = generate_specs(args)
    save_config(args, specs)

    print(f"Device: {device}")
    print(f"Planned runs: {len(specs)}")
    summaries: list[RunSummary] = []
    for spec in specs:
        summaries.append(train_one_spec(args, spec, device))

    summary_rows = [asdict(summary) for summary in summaries]
    write_rows(args.results_dir / "summary.csv", summary_rows)
    plot_lr_sensitivity(summaries, figure_dir)
    plot_batch_size_sensitivity(summaries, figure_dir)
    plot_norm_ablation(summaries, figure_dir)
    plot_gradient_predictiveness(summaries, figure_dir)

    print("\nSaved enhanced experiment outputs:")
    print(f"  summary: {args.results_dir / 'summary.csv'}")
    print(f"  figures: {figure_dir}")


if __name__ == "__main__":
    main()
