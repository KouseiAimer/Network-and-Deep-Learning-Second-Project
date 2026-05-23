"""Generate evaluation visualizations for a trained PyramidNet + ShakeDrop."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import pyramidnet_shakedrop


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


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    project_root = root.parents[0]
    default_results = root / "results" / "pyramidnet110_a270_shakedrop"
    parser = argparse.ArgumentParser(description="Visualize PyramidNet + ShakeDrop predictions")
    parser.add_argument("--data-dir", type=Path, default=project_root / "data")
    parser.add_argument("--checkpoint", type=Path, default=root / "weights" / "pyramidnet110_a270_shakedrop" / "best.pt")
    parser.add_argument("--config", type=Path, default=default_results / "config.json")
    parser.add_argument("--output-dir", type=Path, default=default_results / "visualizations")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-batches", type=int, default=0)
    parser.add_argument("--max-misclassified", type=int, default=25)
    return parser.parse_args()


def load_config(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def config_value(config: dict[str, object], key: str, default):
    return config.get(key, default)


def build_model(config: dict[str, object]) -> torch.nn.Module:
    return pyramidnet_shakedrop(
        depth=int(config_value(config, "depth", 110)),
        alpha=int(config_value(config, "alpha", 270)),
        final_survival_prob=float(config_value(config, "final_survival_prob", 0.5)),
    )


def build_loader(args: argparse.Namespace, device: torch.device) -> DataLoader:
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    dataset = datasets.CIFAR10(
        root=str(args.data_dir),
        train=False,
        download=False,
        transform=transform,
    )
    loader_kwargs = {
        "batch_size": args.batch_size,
        "shuffle": False,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
    }
    if args.num_workers > 0:
        loader_kwargs["persistent_workers"] = True
    return DataLoader(dataset, **loader_kwargs)


def unnormalize(image: torch.Tensor) -> np.ndarray:
    mean = torch.tensor(CIFAR10_MEAN).view(3, 1, 1)
    std = torch.tensor(CIFAR10_STD).view(3, 1, 1)
    image = image.cpu() * std + mean
    image = image.clamp(0, 1).permute(1, 2, 0).numpy()
    return image


@torch.no_grad()
def collect_predictions(
    model: torch.nn.Module,
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
        probabilities = torch.softmax(logits, dim=1)
        confidence, predictions = probabilities.max(dim=1)

        for i in range(images.size(0)):
            target = int(targets[i].item())
            pred = int(predictions[i].item())
            conf = float(confidence[i].item())
            confusion[target, pred] += 1
            rows.append(
                {
                    "index": sample_index,
                    "label": CLASSES[target],
                    "prediction": CLASSES[pred],
                    "confidence": conf,
                    "correct": target == pred,
                }
            )
            if target == pred:
                correct_confidence.append(conf)
            else:
                wrong_confidence.append(conf)
                if len(misclassified) < max_misclassified:
                    misclassified.append((unnormalize(images[i]), target, pred, conf))
            sample_index += 1

    return confusion, rows, correct_confidence, wrong_confidence, misclassified


def plot_confusion_matrix(confusion: np.ndarray, output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    normalized = confusion / np.maximum(confusion.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(8, 7))
    image = ax.imshow(normalized, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(10), CLASSES, rotation=45, ha="right")
    ax.set_yticks(range(10), CLASSES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Normalized Confusion Matrix")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    for y in range(10):
        for x in range(10):
            value = normalized[y, x]
            ax.text(x, y, f"{value:.2f}", ha="center", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_per_class_accuracy(confusion: np.ndarray, output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    counts = np.maximum(confusion.sum(axis=1), 1)
    acc = np.diag(confusion) / counts * 100.0
    fig, ax = plt.subplots(figsize=(9, 4.8))
    bars = ax.bar(CLASSES, acc, color="tab:green")
    ax.set_ylim(0, 100)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Per-Class Accuracy")
    ax.tick_params(axis="x", rotation=35)
    for bar, value in zip(bars, acc):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 1, f"{value:.1f}", ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_confidence_histogram(
    correct_confidence: list[float],
    wrong_confidence: list[float],
    output_path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4.8))
    bins = np.linspace(0, 1, 21)
    ax.hist(correct_confidence, bins=bins, alpha=0.65, label="correct", color="tab:blue")
    ax.hist(wrong_confidence, bins=bins, alpha=0.65, label="wrong", color="tab:red")
    ax.set_xlabel("Prediction confidence")
    ax.set_ylabel("Count")
    ax.set_title("Confidence Distribution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_misclassified_examples(
    examples: list[tuple[np.ndarray, int, int, float]],
    output_path: Path,
) -> None:
    if not examples:
        return

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cols = 5
    rows = int(np.ceil(len(examples) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.2, rows * 2.4))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes:
        ax.axis("off")

    for ax, (image, target, pred, conf) in zip(axes, examples):
        ax.imshow(image)
        ax.set_title(f"T:{CLASSES[target]}\nP:{CLASSES[pred]} ({conf:.2f})", fontsize=8)
        ax.axis("off")

    fig.suptitle("Misclassified Examples", fontsize=13)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def write_predictions(rows: list[dict[str, object]], output_path: Path) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["index", "label", "prediction", "confidence", "correct"])
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = load_config(args.config)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    if not config and isinstance(checkpoint.get("args"), dict):
        config = checkpoint["args"]

    model = build_model(config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    loader = build_loader(args, device)

    confusion, rows, correct_confidence, wrong_confidence, misclassified = collect_predictions(
        model,
        loader,
        device,
        args.max_batches,
        args.max_misclassified,
    )
    write_predictions(rows, args.output_dir / "predictions.csv")
    plot_confusion_matrix(confusion, args.output_dir / "confusion_matrix.png")
    plot_per_class_accuracy(confusion, args.output_dir / "per_class_accuracy.png")
    plot_confidence_histogram(correct_confidence, wrong_confidence, args.output_dir / "confidence_histogram.png")
    plot_misclassified_examples(misclassified, args.output_dir / "misclassified_examples.png")

    accuracy = np.diag(confusion).sum() / max(confusion.sum(), 1)
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"Saved visualizations to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
