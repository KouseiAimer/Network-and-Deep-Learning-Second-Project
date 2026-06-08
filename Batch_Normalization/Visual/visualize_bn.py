"""
Paper-style visualizations for the baseline Batch Normalization experiment.

This script uses the existing outputs in Batch_Normalization/results:
- history.csv and summary.csv for accuracy curves;
- best.pt / last.pt checkpoints for 3D loss surfaces and activation distributions.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from collections import defaultdict
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
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from torch import nn

from data_utils import get_cifar_loader
from models import VGG_A, VGG_A_BatchNorm


DEFAULT_RESULTS_DIR = BN_ROOT / "results"
DEFAULT_OUTPUT_DIR = CURRENT_DIR / "figures"
DEFAULT_CACHE_DIR = CURRENT_DIR / "cache"
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data"
MODEL_LABELS = {
    "no_bn": "Standard",
    "bn": "Standard + BatchNorm",
}
MODEL_COLORS = {
    "no_bn": "#f04b23",
    "bn": "#0b72b9",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize baseline VGG-A BN results.")
    parser.add_argument(
        "--mode",
        choices=["all", "accuracy", "surface", "activation"],
        default="all",
    )
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=2020)
    parser.add_argument("--checkpoint", choices=["best", "last"], default="best")
    parser.add_argument("--accuracy-lrs", type=float, nargs="+", default=[1e-3, 2e-3])
    parser.add_argument("--surface-lr", type=float, default=1e-3)
    parser.add_argument("--surface-grid", type=int, default=25)
    parser.add_argument("--surface-radius", type=float, default=0.45)
    parser.add_argument("--surface-samples", type=int, default=1024)
    parser.add_argument("--surface-batch-size", type=int, default=128)
    parser.add_argument("--surface-z-percentile", type=float, default=98.0)
    parser.add_argument("--activation-lr", type=float, default=1e-3)
    parser.add_argument("--activation-samples", type=int, default=1024)
    parser.add_argument("--activation-batch-size", type=int, default=128)
    parser.add_argument("--activation-layers", type=int, nargs="+", default=[1, 4, 8])
    parser.add_argument("--activation-max-channels", type=int, default=48)
    parser.add_argument("--activation-samples-per-channel", type=int, default=48)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--force", action="store_true", help="Recompute cached surface grids.")
    return parser.parse_args()


def set_random_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def lr_to_run_token(lr: float) -> str:
    return f"{lr:g}".replace(".", "p")


def run_dir(results_dir: Path, model_name: str, lr: float) -> Path:
    return results_dir / f"{model_name}_lr_{lr_to_run_token(lr)}"


def load_history(path: Path) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    with path.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            rows.append(
                {
                    "epoch": float(row["epoch"]),
                    "train_loss": float(row["train_loss"]),
                    "train_accuracy": float(row["train_accuracy"]),
                    "val_loss": float(row["val_loss"]),
                    "val_accuracy": float(row["val_accuracy"]),
                }
            )
    return rows


def count_steps_per_epoch(batch_loss_path: Path, epochs: int) -> int:
    with batch_loss_path.open("r", newline="", encoding="utf-8") as file:
        row_count = sum(1 for _ in file) - 1
    return max(row_count // max(epochs, 1), 1)


def build_model(model_name: str) -> nn.Module:
    if model_name == "no_bn":
        return VGG_A()
    if model_name == "bn":
        return VGG_A_BatchNorm()
    raise ValueError(f"Unknown model: {model_name}")


def load_checkpoint_model(
    model_name: str,
    lr: float,
    checkpoint: str,
    results_dir: Path,
    device: torch.device,
) -> nn.Module:
    model = build_model(model_name)
    ckpt_path = run_dir(results_dir, model_name, lr) / f"{checkpoint}.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def plot_accuracy_panels(args: argparse.Namespace) -> Path:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "paper_style_accuracy_panels.png"

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.7), sharex=True)
    line_styles = {
        args.accuracy_lrs[0]: "-",
        args.accuracy_lrs[1] if len(args.accuracy_lrs) > 1 else args.accuracy_lrs[0]: "--",
    }

    for lr in args.accuracy_lrs:
        for model_name in ("no_bn", "bn"):
            directory = run_dir(args.results_dir, model_name, lr)
            history_path = directory / "history.csv"
            batch_loss_path = directory / "batch_losses.csv"
            if not history_path.exists():
                print(f"Skip missing history: {history_path}")
                continue
            history = load_history(history_path)
            steps_per_epoch = count_steps_per_epoch(batch_loss_path, len(history))
            steps = [row["epoch"] * steps_per_epoch for row in history]
            linestyle = line_styles.get(lr, ":")
            color = MODEL_COLORS[model_name]
            label = f"{MODEL_LABELS[model_name]}, LR={lr:g}"
            axes[0].plot(
                steps,
                [row["train_accuracy"] * 100 for row in history],
                linestyle=linestyle,
                color=color,
                linewidth=2.0,
                label=label,
            )
            axes[1].plot(
                steps,
                [row["val_accuracy"] * 100 for row in history],
                linestyle=linestyle,
                color=color,
                linewidth=2.0,
                label=label,
            )

    for ax, ylabel in zip(axes, ("Training Accuracy (%)", "Test Accuracy (%)")):
        ax.set_facecolor("#e9e9f2")
        ax.set_xlabel("Steps")
        ax.set_ylabel(ylabel)
        ax.set_ylim(0, 100)
        ax.grid(color="white", linewidth=1.2)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].legend(frameon=False, loc="lower right")
    axes[1].legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    return output_path


def direction_like_parameter(parameter: torch.Tensor) -> torch.Tensor:
    direction = torch.randn_like(parameter)
    if parameter.ndim <= 1:
        return torch.zeros_like(parameter)

    direction_view = direction.view(direction.shape[0], -1)
    parameter_view = parameter.detach().view(parameter.shape[0], -1)
    direction_norm = direction_view.norm(dim=1, keepdim=True).clamp_min(1e-12)
    parameter_norm = parameter_view.norm(dim=1, keepdim=True)
    direction_view.mul_(parameter_norm / direction_norm)
    return direction


def make_filter_normalized_directions(model: nn.Module) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    direction_a: list[torch.Tensor] = []
    direction_b: list[torch.Tensor] = []
    for parameter in model.parameters():
        direction_a.append(direction_like_parameter(parameter.data))
        direction_b.append(direction_like_parameter(parameter.data))
    return direction_a, direction_b


@torch.no_grad()
def assign_perturbed_parameters(
    model: nn.Module,
    base_parameters: list[torch.Tensor],
    direction_a: list[torch.Tensor],
    direction_b: list[torch.Tensor],
    alpha: float,
    beta: float,
) -> None:
    for parameter, base, dir_a, dir_b in zip(
        model.parameters(), base_parameters, direction_a, direction_b
    ):
        parameter.copy_(base + alpha * dir_a + beta * dir_b)


@torch.no_grad()
def compute_average_loss(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.eval()
    total_loss = 0.0
    total_items = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        loss = criterion(logits, y)
        batch_size = y.size(0)
        total_loss += float(loss.item()) * batch_size
        total_items += batch_size
    return total_loss / max(total_items, 1)


def cache_name(model_name: str, args: argparse.Namespace) -> str:
    return (
        f"surface_{model_name}_lr{lr_to_run_token(args.surface_lr)}_"
        f"{args.checkpoint}_g{args.surface_grid}_r{args.surface_radius:g}_"
        f"n{args.surface_samples}_seed{args.seed}.npz"
    )


def compute_or_load_surface(
    model_name: str,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = args.cache_dir / cache_name(model_name, args)
    if cache_path.exists() and not args.force:
        cached = np.load(cache_path)
        return cached["alpha"], cached["beta"], cached["loss"]

    set_random_seeds(args.seed)
    model = load_checkpoint_model(
        model_name,
        lr=args.surface_lr,
        checkpoint=args.checkpoint,
        results_dir=args.results_dir,
        device=device,
    )
    base_parameters = [parameter.detach().clone() for parameter in model.parameters()]
    direction_a, direction_b = make_filter_normalized_directions(model)
    criterion = nn.CrossEntropyLoss()
    loader = get_cifar_loader(
        root=args.data_root,
        batch_size=args.surface_batch_size,
        train=False,
        shuffle=False,
        num_workers=args.num_workers,
        n_items=args.surface_samples,
        augment=False,
    )

    coordinates = np.linspace(-args.surface_radius, args.surface_radius, args.surface_grid)
    alpha_grid, beta_grid = np.meshgrid(coordinates, coordinates)
    loss_grid = np.zeros_like(alpha_grid, dtype=np.float64)

    total_points = args.surface_grid * args.surface_grid
    point_id = 0
    for i, alpha in enumerate(coordinates):
        for j, beta in enumerate(coordinates):
            point_id += 1
            assign_perturbed_parameters(
                model,
                base_parameters,
                direction_a,
                direction_b,
                float(alpha),
                float(beta),
            )
            loss_grid[j, i] = compute_average_loss(model, loader, criterion, device)
            if point_id % max(args.surface_grid, 1) == 0:
                print(f"  {model_name} surface: {point_id}/{total_points} points")

    assign_perturbed_parameters(model, base_parameters, direction_a, direction_b, 0.0, 0.0)
    np.savez_compressed(cache_path, alpha=alpha_grid, beta=beta_grid, loss=loss_grid)
    return alpha_grid, beta_grid, loss_grid


def plot_3d_surface(args: argparse.Namespace, device: torch.device) -> Path:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "loss_surface_3d_no_bn_vs_bn.png"
    surfaces = {
        model_name: compute_or_load_surface(model_name, args, device)
        for model_name in ("no_bn", "bn")
    }

    z_values = np.concatenate([surface[2].reshape(-1) for surface in surfaces.values()])
    z_min = float(np.nanmin(z_values))
    z_max = float(np.nanpercentile(z_values, args.surface_z_percentile))

    fig = plt.figure(figsize=(14, 5.8))
    for index, model_name in enumerate(("no_bn", "bn"), start=1):
        alpha_grid, beta_grid, loss_grid = surfaces[model_name]
        ax = fig.add_subplot(1, 2, index, projection="3d")
        ax.plot_surface(
            alpha_grid,
            beta_grid,
            loss_grid,
            cmap="coolwarm",
            linewidth=0,
            antialiased=True,
            shade=True,
            alpha=0.97,
        )
        ax.set_title(MODEL_LABELS[model_name], pad=10)
        ax.set_xlabel("Direction 1")
        ax.set_ylabel("Direction 2")
        ax.set_zlabel("Loss")
        ax.set_zlim(z_min, z_max)
        ax.view_init(elev=28, azim=-58)
        ax.grid(False)
        ax.xaxis.pane.set_alpha(0.0)
        ax.yaxis.pane.set_alpha(0.0)
        ax.zaxis.pane.set_alpha(0.0)

    fig.suptitle(
        f"2D loss surface around {args.checkpoint}.pt, lr={args.surface_lr:g}",
        y=0.98,
        fontsize=14,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output_path


def conv_or_bn_modules_by_ordinal(model: nn.Module, model_name: str) -> dict[int, nn.Module]:
    modules: dict[int, nn.Module] = {}
    conv_count = 0
    last_conv_ordinal = 0
    for module in model.features:
        if isinstance(module, nn.Conv2d):
            conv_count += 1
            last_conv_ordinal = conv_count
            if model_name == "no_bn":
                modules[conv_count] = module
        elif isinstance(module, nn.BatchNorm2d) and model_name == "bn":
            modules[last_conv_ordinal] = module
    return modules


def collect_activation_samples(
    model: nn.Module,
    model_name: str,
    layer_ordinals: list[int],
    args: argparse.Namespace,
    device: torch.device,
) -> dict[int, np.ndarray]:
    selected_modules = conv_or_bn_modules_by_ordinal(model, model_name)
    missing_layers = [layer for layer in layer_ordinals if layer not in selected_modules]
    if missing_layers:
        raise ValueError(f"Missing layer ordinals for {model_name}: {missing_layers}")

    samples: dict[int, list[np.ndarray]] = {layer: [] for layer in layer_ordinals}
    chosen_channels: dict[int, np.ndarray] = {}

    def make_hook(layer: int):
        def hook(_module: nn.Module, _inputs, output: torch.Tensor) -> None:
            activation = output.detach().float().cpu()
            channels = activation.shape[1]
            if layer not in chosen_channels:
                max_channels = min(channels, args.activation_max_channels)
                chosen_channels[layer] = np.linspace(
                    0, channels - 1, num=max_channels, dtype=np.int64
                )

            selected = activation[:, chosen_channels[layer], :, :]
            channel_first = selected.permute(1, 0, 2, 3).reshape(len(chosen_channels[layer]), -1)
            per_channel_values: list[np.ndarray] = []
            for channel_values in channel_first:
                sample_n = min(args.activation_samples_per_channel, channel_values.numel())
                index = torch.randperm(channel_values.numel())[:sample_n]
                per_channel_values.append(channel_values[index].numpy())
            samples[layer].append(np.stack(per_channel_values, axis=0))

        return hook

    handles = []
    for layer in layer_ordinals:
        handles.append(selected_modules[layer].register_forward_hook(make_hook(layer)))

    loader = get_cifar_loader(
        root=args.data_root,
        batch_size=args.activation_batch_size,
        train=False,
        shuffle=False,
        num_workers=args.num_workers,
        n_items=args.activation_samples,
        augment=False,
    )

    try:
        with torch.no_grad():
            for x, _y in loader:
                model(x.to(device, non_blocking=True))
    finally:
        for handle in handles:
            handle.remove()

    merged: dict[int, np.ndarray] = {}
    for layer, layer_samples in samples.items():
        merged[layer] = np.concatenate(layer_samples, axis=1)
    return merged


def write_activation_stats(
    all_samples: dict[str, dict[int, np.ndarray]],
    output_path: Path,
) -> None:
    rows = []
    for model_name, layer_samples in all_samples.items():
        for layer, values in layer_samples.items():
            flat = values.reshape(-1)
            rows.append(
                {
                    "model_name": model_name,
                    "layer": layer,
                    "mean": float(np.mean(flat)),
                    "std": float(np.std(flat)),
                    "p01": float(np.percentile(flat, 1)),
                    "p50": float(np.percentile(flat, 50)),
                    "p99": float(np.percentile(flat, 99)),
                }
            )
    write_csv(output_path, rows)


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def smooth_histogram(values: np.ndarray, bins: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    hist, edges = np.histogram(values, bins=bins, density=True)
    if len(hist) >= 5:
        kernel = np.array([1, 2, 3, 2, 1], dtype=np.float64)
        kernel /= kernel.sum()
        hist = np.convolve(hist, kernel, mode="same")
    centers = (edges[:-1] + edges[1:]) / 2
    return centers, hist


def plot_activation_ridgeline(args: argparse.Namespace, device: torch.device) -> Path:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "activation_distribution_ridgeline.png"
    stats_path = args.output_dir / "activation_distribution_stats.csv"

    set_random_seeds(args.seed)
    all_samples: dict[str, dict[int, np.ndarray]] = {}
    for model_name in ("no_bn", "bn"):
        model = load_checkpoint_model(
            model_name,
            lr=args.activation_lr,
            checkpoint=args.checkpoint,
            results_dir=args.results_dir,
            device=device,
        )
        all_samples[model_name] = collect_activation_samples(
            model,
            model_name,
            args.activation_layers,
            args,
            device,
        )

    write_activation_stats(all_samples, stats_path)

    fig, axes = plt.subplots(
        len(args.activation_layers),
        2,
        figsize=(10.8, 3.0 * len(args.activation_layers)),
        sharex=False,
    )
    if len(args.activation_layers) == 1:
        axes = np.asarray([axes])

    for row_idx, layer in enumerate(args.activation_layers):
        combined = np.concatenate(
            [
                all_samples["no_bn"][layer].reshape(-1),
                all_samples["bn"][layer].reshape(-1),
            ]
        )
        lower, upper = np.percentile(combined, [0.5, 99.5])
        if math.isclose(lower, upper):
            lower, upper = lower - 1.0, upper + 1.0
        bins = np.linspace(lower, upper, 90)

        for col_idx, model_name in enumerate(("no_bn", "bn")):
            ax = axes[row_idx, col_idx]
            layer_values = all_samples[model_name][layer]
            color = MODEL_COLORS[model_name]
            channel_count = layer_values.shape[0]
            for channel_idx in range(channel_count):
                values = np.clip(layer_values[channel_idx], lower, upper)
                centers, density = smooth_histogram(values, bins)
                if np.nanmax(density) > 0:
                    density = density / np.nanmax(density)
                offset = channel_idx * 0.45
                ax.plot(centers, density * 0.8 + offset, color=color, alpha=0.35, linewidth=0.8)

            ax.set_title(MODEL_LABELS[model_name] if row_idx == 0 else "")
            ax.set_yticks([])
            ax.set_xlim(lower, upper)
            ax.set_facecolor("#fbfbfb")
            ax.grid(axis="x", alpha=0.18)
            if col_idx == 0:
                ax.set_ylabel(f"Layer #{layer}")
            if row_idx == len(args.activation_layers) - 1:
                ax.set_xlabel("Activation value")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

    fig.suptitle(
        f"Activation distributions around {args.checkpoint}.pt, lr={args.activation_lr:g}",
        fontsize=14,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output_path


def save_visual_config(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = vars(args).copy()
    for key in ("results_dir", "output_dir", "cache_dir", "data_root"):
        config[key] = str(config[key])
    with (args.output_dir / "visual_config.json").open("w", encoding="utf-8") as file:
        json.dump(config, file, indent=2, ensure_ascii=False)


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    set_random_seeds(args.seed)
    save_visual_config(args)

    print(f"Device: {device}")
    print(f"Results dir: {args.results_dir}")
    print(f"Output dir: {args.output_dir}")

    outputs: list[Path] = []
    if args.mode in {"all", "accuracy"}:
        outputs.append(plot_accuracy_panels(args))
    if args.mode in {"all", "surface"}:
        outputs.append(plot_3d_surface(args, device))
    if args.mode in {"all", "activation"}:
        outputs.append(plot_activation_ridgeline(args, device))

    print("\nSaved visualizations:")
    for output in outputs:
        print(f"  {output}")


if __name__ == "__main__":
    main()
