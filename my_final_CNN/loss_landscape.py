"""Approximate a local 1D loss landscape around a trained FinalCNN checkpoint."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import build_model


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    default_dir = root / "final_result" / "final_swrn40_10"
    parser = argparse.ArgumentParser(description="Plot a small loss landscape around FinalCNN")
    parser.add_argument("--data-dir", type=Path, default=root.parents[0] / "data")
    parser.add_argument("--checkpoint", type=Path, default=default_dir / "weights" / "best.pt")
    parser.add_argument("--config", type=Path, default=default_dir / "config.json")
    parser.add_argument("--output-dir", type=Path, default=default_dir / "loss_landscape")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-batches", type=int, default=4)
    parser.add_argument("--radius", type=float, default=0.5)
    parser.add_argument("--points", type=int, default=21)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--use-ema", action="store_true", default=True)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def cfg(config: dict[str, object], key: str, default):
    return config.get(key, default)


def build_from_config(config: dict[str, object]) -> torch.nn.Module:
    return build_model(
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
    model: torch.nn.Module,
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
def evaluate_loss(model: torch.nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device, max_batches: int) -> float:
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
    return total_loss / total_samples


def plot_landscape(rows: list[dict[str, float]], output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    alphas = [row["alpha"] for row in rows]
    losses = [row["loss"] for row in rows]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(alphas, losses, marker="o")
    ax.set_xlabel("Parameter displacement alpha")
    ax.set_ylabel("Loss")
    ax.set_title("1D Loss Landscape Around Best Checkpoint")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = load_json(args.config)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    if not config and isinstance(checkpoint.get("args"), dict):
        config = checkpoint["args"]

    model = build_from_config(config).to(device)
    state = checkpoint.get("ema_state") if args.use_ema and checkpoint.get("ema_state") is not None else checkpoint["model_state"]
    model.load_state_dict(state)
    base_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    direction = random_direction(base_state, args.seed)
    loader = build_loader(args, device)
    criterion = nn.CrossEntropyLoss()

    alphas = torch.linspace(-args.radius, args.radius, args.points).tolist()
    rows = []
    for alpha in alphas:
        apply_direction(model, base_state, direction, float(alpha), device)
        loss = evaluate_loss(model, loader, criterion, device, args.max_batches)
        rows.append({"alpha": float(alpha), "loss": float(loss)})
        print(f"alpha={alpha:.3f}, loss={loss:.4f}")

    model.load_state_dict({key: value.to(device) for key, value in base_state.items()}, strict=True)

    csv_path = args.output_dir / "loss_landscape_1d.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["alpha", "loss"])
        writer.writeheader()
        writer.writerows(rows)
    plot_landscape(rows, args.output_dir / "loss_landscape_1d.png")
    print(f"Saved loss landscape to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
