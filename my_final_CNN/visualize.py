"""Generate interpretation figures for the final CNN checkpoint."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import build_model as build_wrn_model
from model_v2 import build_model as build_densenet_model


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
    default_dir = root / "final_result" / "final_swrn40_10"
    parser = argparse.ArgumentParser(description="Visualize FinalCNN predictions and filters")
    parser.add_argument("--data-dir", type=Path, default=root.parents[0] / "data")
    parser.add_argument("--checkpoint", type=Path, default=default_dir / "weights" / "best.pt")
    parser.add_argument("--config", type=Path, default=default_dir / "config.json")
    parser.add_argument("--output-dir", type=Path, default=default_dir / "visualizations")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-batches", type=int, default=0)
    parser.add_argument("--max-misclassified", type=int, default=25)
    parser.add_argument("--gradcam-samples", type=int, default=8)
    parser.add_argument("--raw-filters", type=int, default=32)
    parser.add_argument("--use-ema", action="store_true", default=True)
    parser.add_argument("--tta", action="store_true", help="Use horizontal-flip TTA for predictions.")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def cfg(config: dict[str, object], key: str, default):
    return config.get(key, default)


def build_from_config(config: dict[str, object]) -> torch.nn.Module:
    if "growth_rate" in config or str(config.get("model", "")).lower().find("dense") >= 0:
        return build_densenet_model(
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
    return build_wrn_model(
        depth=int(cfg(config, "depth", 40)),
        widen_factor=int(cfg(config, "widen_factor", 10)),
        dropout=float(cfg(config, "dropout", 0.3)),
        activation=str(cfg(config, "activation", "silu")),
        se_reduction=int(cfg(config, "se_reduction", 16)),
        stochastic_depth_rate=float(cfg(config, "stochastic_depth_rate", 0.1)),
        head_dropout=float(cfg(config, "head_dropout", 0.0)),
    )


def build_loader(args: argparse.Namespace, device: torch.device) -> DataLoader:
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    dataset = datasets.CIFAR10(root=str(args.data_dir), train=False, download=False, transform=transform)
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
    image = image.detach().cpu() * std + mean
    return image.clamp(0, 1).permute(1, 2, 0).numpy()


@torch.no_grad()
def collect_predictions(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    max_batches: int,
    max_misclassified: int,
    tta: bool,
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
        if tta:
            logits = 0.5 * (logits + model(torch.flip(images, dims=(-1,))))
        probs = torch.softmax(logits, dim=1)
        confidence, predictions = probs.max(dim=1)

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
            ax.text(x, y, f"{normalized[y, x]:.2f}", ha="center", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_per_class_accuracy(confusion: np.ndarray, output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    acc = np.diag(confusion) / np.maximum(confusion.sum(axis=1), 1) * 100.0
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


def plot_confidence_histogram(correct: list[float], wrong: list[float], output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4.8))
    bins = np.linspace(0, 1, 21)
    ax.hist(correct, bins=bins, alpha=0.65, label="correct", color="tab:blue")
    ax.hist(wrong, bins=bins, alpha=0.65, label="wrong", color="tab:red")
    ax.set_xlabel("Prediction confidence")
    ax.set_ylabel("Count")
    ax.set_title("Confidence Distribution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_misclassified_examples(examples: list[tuple[np.ndarray, int, int, float]], output_path: Path) -> None:
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


def plot_first_conv_filters(model: torch.nn.Module, output_path: Path, max_filters: int) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    first_conv = next(module for module in model.modules() if isinstance(module, torch.nn.Conv2d))
    weights = first_conv.weight.detach().cpu()
    num_filters = min(max_filters, weights.size(0))
    weights = weights[:num_filters]
    weights = (weights - weights.amin(dim=(1, 2, 3), keepdim=True)) / (
        weights.amax(dim=(1, 2, 3), keepdim=True) - weights.amin(dim=(1, 2, 3), keepdim=True) + 1e-8
    )

    cols = 8
    rows = int(np.ceil(num_filters / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.5, rows * 1.5))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes:
        ax.axis("off")
    for ax, weight in zip(axes, weights):
        ax.imshow(weight.permute(1, 2, 0).numpy())
    fig.suptitle("First Convolution Filters", fontsize=13)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


class GradCAM:
    def __init__(self, model: torch.nn.Module) -> None:
        self.model = model
        self.activations = None
        self.gradients = None
        target_layer = self._find_last_conv()
        self.forward_handle = target_layer.register_forward_hook(self._forward_hook)
        self.backward_handle = target_layer.register_full_backward_hook(self._backward_hook)

    def _find_last_conv(self) -> torch.nn.Conv2d:
        last_conv = None
        for module in self.model.modules():
            if isinstance(module, torch.nn.Conv2d):
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
        gradients = self.gradients
        activations = self.activations
        weights = gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=image.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam[0, 0]
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam.detach().cpu().numpy()


def plot_gradcam_examples(model: torch.nn.Module, loader: DataLoader, device: torch.device, output_path: Path, samples: int) -> None:
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


def write_predictions(rows: list[dict[str, object]], output_path: Path) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["index", "label", "prediction", "confidence", "correct"])
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = load_json(args.config)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    if not config and isinstance(checkpoint.get("args"), dict):
        config = checkpoint["args"]

    model = build_from_config(config).to(device)
    state = checkpoint.get("ema_state") if args.use_ema and checkpoint.get("ema_state") is not None else checkpoint["model_state"]
    model.load_state_dict(state)
    loader = build_loader(args, device)

    confusion, rows, correct, wrong, misclassified = collect_predictions(
        model,
        loader,
        device,
        args.max_batches,
        args.max_misclassified,
        args.tta,
    )

    write_predictions(rows, args.output_dir / "predictions.csv")
    plot_confusion_matrix(confusion, args.output_dir / "confusion_matrix.png")
    plot_per_class_accuracy(confusion, args.output_dir / "per_class_accuracy.png")
    plot_confidence_histogram(correct, wrong, args.output_dir / "confidence_histogram.png")
    plot_misclassified_examples(misclassified, args.output_dir / "misclassified_examples.png")
    plot_first_conv_filters(model, args.output_dir / "first_conv_filters.png", args.raw_filters)
    plot_gradcam_examples(model, loader, device, args.output_dir / "gradcam_examples.png", args.gradcam_samples)

    accuracy = np.diag(confusion).sum() / max(confusion.sum(), 1)
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"Saved visualizations to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
