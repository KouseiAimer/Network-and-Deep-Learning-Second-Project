"""Visualize the final Ablation MyDenseNet run.

This script is self-contained inside ``Ablation`` except for using torchvision
to load CIFAR-10. It imports the local model and plotting utilities from
``Ablation/ablation.py``.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from ablation import (
    ABLATION_ROOT,
    CIFAR10_MEAN,
    CIFAR10_STD,
    CLASSES,
    DEFAULT_DATA_DIR,
    build_model,
    plot_history,
    write_summary,
)


DEFAULT_RUN_DIR = ABLATION_ROOT / "results" / "final" / "best_from_ablation"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize the final Ablation MyDenseNet run.")
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-batches", type=int, default=0)
    parser.add_argument("--max-misclassified", type=int, default=25)
    parser.add_argument("--gradcam-samples", type=int, default=8)
    parser.add_argument("--raw-filters", type=int, default=32)
    parser.add_argument("--landscape-max-batches", type=int, default=4)
    parser.add_argument("--landscape-points", type=int, default=21)
    parser.add_argument("--landscape-radius", type=float, default=0.5)
    parser.add_argument("--landscape-seed", type=int, default=123)
    parser.add_argument("--skip-interpretation", action="store_true")
    parser.add_argument("--skip-landscape", action="store_true")
    parser.add_argument("--use-ema", action="store_true", default=True)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def cfg(config: dict[str, Any], key: str, default):
    return config.get(key, default)


def build_from_config(config: dict[str, Any]) -> nn.Module:
    return build_model(
        depth=int(cfg(config, "depth", 190)),
        growth_rate=int(cfg(config, "growth_rate", 40)),
        compression=float(cfg(config, "compression", 0.5)),
        activation=str(cfg(config, "activation", "silu")),
        se_reduction=int(cfg(config, "se_reduction", 16)),
        stochastic_depth_rate=float(cfg(config, "stochastic_depth_rate", 0.2)),
        classifier_hidden=int(cfg(config, "classifier_hidden", 512)),
        classifier_dropout=float(cfg(config, "classifier_dropout", 0.2)),
        drop_rate=float(cfg(config, "drop_rate", 0.0)),
        transition_dropout=float(cfg(config, "transition_dropout", 0.0)),
    )


def load_checkpoint_model(run_dir: Path, use_ema: bool, device: torch.device) -> tuple[nn.Module, dict[str, Any]]:
    config = load_json(run_dir / "config.json")
    checkpoint = torch.load(run_dir / "weights" / "best.pt", map_location=device)
    if not config and isinstance(checkpoint.get("args"), dict):
        config = checkpoint["args"]
    model = build_from_config(config).to(device)
    state = checkpoint.get("ema_state") if use_ema and checkpoint.get("ema_state") is not None else checkpoint["model_state"]
    model.load_state_dict(state)
    return model, config


def build_loader(data_dir: Path, batch_size: int, num_workers: int, device: torch.device) -> DataLoader:
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)])
    dataset = datasets.CIFAR10(root=str(data_dir), train=False, download=False, transform=transform)
    loader_kwargs: dict[str, Any] = {
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
    return DataLoader(dataset, **loader_kwargs)


def unnormalize(image: torch.Tensor) -> np.ndarray:
    mean = torch.tensor(CIFAR10_MEAN).view(3, 1, 1)
    std = torch.tensor(CIFAR10_STD).view(3, 1, 1)
    image = image.detach().cpu() * std + mean
    return image.clamp(0, 1).permute(1, 2, 0).numpy()


@torch.no_grad()
def collect_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    max_batches: int,
    max_misclassified: int,
):
    confusion = np.zeros((10, 10), dtype=np.int64)
    rows: list[dict[str, object]] = []
    correct_confidence: list[float] = []
    wrong_confidence: list[float] = []
    misclassified: list[tuple[np.ndarray, int, int, float]] = []
    sample_index = 0

    model.eval()
    for batch_idx, (images, targets) in enumerate(loader):
        if max_batches > 0 and batch_idx >= max_batches:
            break
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model(images)
        probs = logits.softmax(dim=1)
        confidence, predictions = probs.max(dim=1)

        for idx in range(images.size(0)):
            label = int(targets[idx].item())
            pred = int(predictions[idx].item())
            conf = float(confidence[idx].item())
            confusion[label, pred] += 1
            correct = label == pred
            rows.append(
                {
                    "index": sample_index,
                    "label": CLASSES[label],
                    "prediction": CLASSES[pred],
                    "confidence": conf,
                    "correct": int(correct),
                }
            )
            if correct:
                correct_confidence.append(conf)
            else:
                wrong_confidence.append(conf)
                if len(misclassified) < max_misclassified:
                    misclassified.append((unnormalize(images[idx]), label, pred, conf))
            sample_index += 1

    return confusion, rows, correct_confidence, wrong_confidence, misclassified


def write_predictions(rows: list[dict[str, object]], output_path: Path) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["index", "label", "prediction", "confidence", "correct"])
        writer.writeheader()
        writer.writerows(rows)


def plot_confusion_matrix(confusion: np.ndarray, output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    normalized = confusion / np.maximum(confusion.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(7.4, 6.4))
    image = ax.imshow(normalized, cmap="Blues", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    ax.set_xticklabels(CLASSES, rotation=45, ha="right")
    ax.set_yticklabels(CLASSES)
    ax.set_xlabel("Prediction")
    ax.set_ylabel("Label")
    ax.set_title("Normalized Confusion Matrix")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    for i in range(10):
        for j in range(10):
            value = normalized[i, j]
            if value >= 0.01:
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=7)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_per_class_accuracy(confusion: np.ndarray, output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    acc = np.diag(confusion) / np.maximum(confusion.sum(axis=1), 1)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(CLASSES, acc * 100.0, color="#4c78a8")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Per-Class Accuracy")
    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=0.25)
    ax.tick_params(axis="x", rotation=35)
    for idx, value in enumerate(acc):
        ax.text(idx, value * 100.0 + 1.0, f"{value * 100.0:.1f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_confidence_histogram(correct: list[float], wrong: list[float], output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bins = np.linspace(0, 1, 21)
    ax.hist(correct, bins=bins, alpha=0.7, label="correct", color="#4c78a8")
    ax.hist(wrong, bins=bins, alpha=0.7, label="wrong", color="#e45756")
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Count")
    ax.set_title("Prediction Confidence")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_misclassified_examples(
    examples: list[tuple[np.ndarray, int, int, float]],
    output_path: Path,
    columns: int = 5,
) -> None:
    if not examples:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = math.ceil(len(examples) / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(columns * 2.1, rows * 2.35))
    axes = np.atleast_1d(axes).ravel()
    for ax, (image, label, pred, conf) in zip(axes, examples):
        ax.imshow(image)
        ax.set_title(f"{CLASSES[label]} -> {CLASSES[pred]}\n{conf:.2f}", fontsize=8)
        ax.axis("off")
    for ax in axes[len(examples) :]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_first_conv_filters(model: nn.Module, output_path: Path, max_filters: int) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    first_conv = next(module for module in model.modules() if isinstance(module, nn.Conv2d))
    weights = first_conv.weight.detach().cpu()
    count = min(max_filters, weights.size(0))
    columns = 8
    rows = math.ceil(count / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(columns * 1.4, rows * 1.4))
    axes = np.atleast_1d(axes).ravel()
    for idx in range(count):
        filt = weights[idx]
        filt = (filt - filt.min()) / (filt.max() - filt.min() + 1e-8)
        axes[idx].imshow(filt.permute(1, 2, 0).numpy())
        axes[idx].axis("off")
    for ax in axes[count:]:
        ax.axis("off")
    fig.suptitle("First Convolution Filters", fontsize=12)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


class GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module | None = None) -> None:
        self.model = model
        self.activations = None
        self.gradients = None
        if target_layer is None:
            target_layer = self._find_last_conv()
        self.forward_handle = target_layer.register_forward_hook(self._forward_hook)
        self.backward_handle = target_layer.register_full_backward_hook(self._backward_hook)

    def _find_last_conv(self) -> nn.Conv2d:
        last_conv = None
        for module in self.model.modules():
            if isinstance(module, nn.Conv2d):
                last_conv = module
        if last_conv is None:
            raise ValueError("No Conv2d layer found for Grad-CAM.")
        return last_conv

    def _forward_hook(self, _module, _inputs, output) -> None:
        self.activations = output.detach()

    def _backward_hook(self, _module, _grad_input, grad_output) -> None:
        self.gradients = grad_output[0].detach()

    def remove(self) -> None:
        self.forward_handle.remove()
        self.backward_handle.remove()

    def __call__(self, image: torch.Tensor, class_idx: int) -> np.ndarray:
        self.model.zero_grad(set_to_none=True)
        logits = self.model(image)
        score = logits[:, class_idx].sum()
        score.backward()
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=image.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam[0, 0]
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam.detach().cpu().numpy()


def plot_gradcam_examples(model: nn.Module, loader: DataLoader, device: torch.device, output_path: Path, samples: int) -> None:
    if samples <= 0:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    model.eval()
    images, targets = next(iter(loader))
    images = images[:samples].to(device)
    targets = targets[:samples].to(device)
    gradcam = GradCAM(model)
    rows = images.size(0)
    fig, axes = plt.subplots(rows, 2, figsize=(5, rows * 2.3))
    axes = np.atleast_2d(axes)
    for idx in range(rows):
        image = images[idx : idx + 1]
        logits = model(image)
        pred = int(logits.argmax(dim=1).item())
        heatmap = gradcam(image, pred)
        raw = unnormalize(images[idx])
        axes[idx, 0].imshow(raw)
        axes[idx, 0].set_title(f"Input / true: {CLASSES[int(targets[idx].item())]}")
        axes[idx, 0].axis("off")
        axes[idx, 1].imshow(raw)
        axes[idx, 1].imshow(heatmap, cmap="jet", alpha=0.45)
        axes[idx, 1].set_title(f"Grad-CAM / pred: {CLASSES[pred]}")
        axes[idx, 1].axis("off")
    gradcam.remove()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def random_direction(state: dict[str, torch.Tensor], seed: int) -> dict[str, torch.Tensor]:
    generator = torch.Generator()
    generator.manual_seed(seed)
    direction: dict[str, torch.Tensor] = {}
    for key, value in state.items():
        if not value.dtype.is_floating_point:
            direction[key] = torch.zeros_like(value)
            continue
        noise = torch.randn(value.shape, generator=generator, dtype=value.dtype)
        if "weight" in key and value.ndim >= 2:
            noise_norm = noise.norm()
            value_norm = value.cpu().norm()
            noise = noise / (noise_norm + 1e-12) * (value_norm + 1e-12)
        else:
            noise.zero_()
        direction[key] = noise
    return direction


def apply_direction(
    model: nn.Module,
    base_state: dict[str, torch.Tensor],
    direction: dict[str, torch.Tensor],
    alpha: float,
    device: torch.device,
) -> None:
    new_state = {}
    for key, value in base_state.items():
        if value.dtype.is_floating_point:
            new_state[key] = (value.cpu() + alpha * direction[key]).to(device)
        else:
            new_state[key] = value.to(device)
    model.load_state_dict(new_state, strict=True)


@torch.no_grad()
def evaluate_loss(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device, max_batches: int) -> float:
    model.eval()
    total_loss = 0.0
    total_samples = 0
    for batch_idx, (images, targets) in enumerate(loader):
        if max_batches > 0 and batch_idx >= max_batches:
            break
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, targets)
        total_loss += loss.item() * images.size(0)
        total_samples += images.size(0)
    return total_loss / max(total_samples, 1)


def plot_loss_landscape(
    model: nn.Module,
    config: dict[str, Any],
    loader: DataLoader,
    device: torch.device,
    output_dir: Path,
    max_batches: int,
    points: int,
    radius: float,
    seed: int,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    criterion = nn.CrossEntropyLoss(label_smoothing=float(cfg(config, "label_smoothing", 0.0)))
    base_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    direction = random_direction(base_state, seed)
    alphas = np.linspace(-radius, radius, points)
    losses = []
    for alpha in alphas:
        apply_direction(model, base_state, direction, float(alpha), device)
        losses.append(evaluate_loss(model, loader, criterion, device, max_batches))
    apply_direction(model, base_state, direction, 0.0, device)

    with (output_dir / "loss_landscape.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["alpha", "loss"])
        writer.writeheader()
        for alpha, loss in zip(alphas, losses):
            writer.writerow({"alpha": float(alpha), "loss": float(loss)})

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(alphas, losses, marker="o", color="#4c78a8")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Normalized random direction scale")
    ax.set_ylabel("Evaluation loss")
    ax.set_title("Local 1D Loss Landscape")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "loss_landscape.png", dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir
    history = run_dir / "history.csv"
    config_path = run_dir / "config.json"
    checkpoint = run_dir / "weights" / "best.pt"
    for path in (history, config_path, checkpoint):
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")

    plot_history(history, run_dir / "curves.png", "Final Ablation MyDenseNet on CIFAR-10")
    write_summary(history, run_dir / "summary.json")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, config = load_checkpoint_model(run_dir, args.use_ema, device)
    loader = build_loader(args.data_dir, args.batch_size, args.num_workers, device)

    if not args.skip_interpretation:
        output_dir = run_dir / "visualizations"
        output_dir.mkdir(parents=True, exist_ok=True)
        confusion, rows, correct, wrong, misclassified = collect_predictions(
            model,
            loader,
            device,
            args.max_batches,
            args.max_misclassified,
        )
        write_predictions(rows, output_dir / "predictions.csv")
        plot_confusion_matrix(confusion, output_dir / "confusion_matrix.png")
        plot_per_class_accuracy(confusion, output_dir / "per_class_accuracy.png")
        plot_confidence_histogram(correct, wrong, output_dir / "confidence_histogram.png")
        plot_misclassified_examples(misclassified, output_dir / "misclassified_examples.png")
        plot_first_conv_filters(model, output_dir / "first_conv_filters.png", args.raw_filters)
        plot_gradcam_examples(model, loader, device, output_dir / "gradcam_examples.png", args.gradcam_samples)
        accuracy = np.diag(confusion).sum() / max(confusion.sum(), 1)
        print(f"Visualization accuracy: {accuracy * 100:.2f}%")

    if not args.skip_landscape:
        plot_loss_landscape(
            model,
            config,
            loader,
            device,
            run_dir / "loss_landscape",
            args.landscape_max_batches,
            args.landscape_points,
            args.landscape_radius,
            args.landscape_seed,
        )

    overview = {
        "run_dir": str(run_dir),
        "curves": str(run_dir / "curves.png"),
        "summary": str(run_dir / "summary.json"),
        "visualizations": str(run_dir / "visualizations"),
        "loss_landscape": str(run_dir / "loss_landscape"),
    }
    with (run_dir / "visual_overview.json").open("w", encoding="utf-8") as file:
        json.dump(overview, file, indent=2)
    print(f"Saved final visual overview to: {(run_dir / 'visual_overview.json').resolve()}")


if __name__ == "__main__":
    main()
