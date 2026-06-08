"""Evaluate one or more FinalDenseNetV2-compatible checkpoints with TTA/ensembling."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model_v2 import build_model


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Evaluate or ensemble FinalDenseNetV2 checkpoints")
    parser.add_argument("--data-dir", type=Path, default=root.parents[0] / "data")
    parser.add_argument("--checkpoints", type=Path, nargs="+", required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-batches", type=int, default=0, help="Limit batches for quick local checks.")
    parser.add_argument("--tta", action="store_true", help="Average original and horizontally flipped predictions.")
    parser.add_argument("--raw-model", action="store_true", help="Use raw weights instead of EMA weights.")
    parser.add_argument("--output", type=Path, default=root / "final_result" / "ensemble_summary.json")
    return parser.parse_args()


def load_config(checkpoint_path: Path) -> dict[str, object]:
    config_path = checkpoint_path.parent.parent / "config.json"
    with config_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def build_from_config(config: dict[str, object]) -> torch.nn.Module:
    return build_model(
        depth=int(config.get("depth", 190)),
        growth_rate=int(config.get("growth_rate", 40)),
        compression=float(config.get("compression", 0.5)),
        activation=str(config.get("activation", "silu")),
        se_reduction=int(config.get("se_reduction", 16)),
        stochastic_depth_rate=float(config.get("stochastic_depth_rate", 0.2)),
        classifier_hidden=int(config.get("classifier_hidden", 512)),
        classifier_dropout=float(config.get("classifier_dropout", 0.2)),
        drop_rate=float(config.get("drop_rate", 0.0)),
        transition_dropout=float(config.get("transition_dropout", 0.0)),
    )


def build_loader(args: argparse.Namespace, device: torch.device) -> DataLoader:
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    dataset = datasets.CIFAR10(str(args.data_dir), train=False, download=False, transform=transform)
    kwargs = {
        "batch_size": args.batch_size,
        "shuffle": False,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
    }
    if args.num_workers > 0:
        kwargs["persistent_workers"] = True
    return DataLoader(dataset, **kwargs)


def load_models(args: argparse.Namespace, device: torch.device) -> list[torch.nn.Module]:
    models = []
    for checkpoint_path in args.checkpoints:
        config = load_config(checkpoint_path)
        model = build_from_config(config).to(device)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        state = checkpoint["model_state"]
        if not args.raw_model and checkpoint.get("ema_state") is not None:
            state = checkpoint["ema_state"]
        model.load_state_dict(state, strict=True)
        model.eval()
        models.append(model)
    return models


@torch.no_grad()
def evaluate(models: list[torch.nn.Module], loader: DataLoader, device: torch.device, tta: bool, max_batches: int) -> dict[str, object]:
    correct_each = [0 for _ in models]
    ensemble_correct = 0
    total = 0
    for batch_idx, (images, targets) in enumerate(loader):
        if max_batches > 0 and batch_idx >= max_batches:
            break
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits_list = []
        for index, model in enumerate(models):
            logits = model(images)
            if tta:
                logits = 0.5 * (logits + model(torch.flip(images, dims=(-1,))))
            logits_list.append(logits)
            correct_each[index] += (logits.argmax(dim=1) == targets).sum().item()
        ensemble_logits = torch.stack(logits_list, dim=0).mean(dim=0)
        ensemble_correct += (ensemble_logits.argmax(dim=1) == targets).sum().item()
        total += images.size(0)
    return {
        "tta": tta,
        "num_models": len(models),
        "per_model_accuracy": [correct / total for correct in correct_each],
        "ensemble_accuracy": ensemble_correct / total,
        "test_samples": total,
    }


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = load_models(args, device)
    summary = evaluate(models, build_loader(args, device), device, args.tta, args.max_batches)
    summary["checkpoints"] = [str(path) for path in args.checkpoints]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)
    for index, accuracy in enumerate(summary["per_model_accuracy"], start=1):
        print(f"Model {index} accuracy: {accuracy * 100:.2f}%")
    print(f"Ensemble accuracy: {summary['ensemble_accuracy'] * 100:.2f}%")
    print(f"Saved evaluation summary to: {args.output.resolve()}")


if __name__ == "__main__":
    main()
