from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


ROOT = Path(__file__).resolve().parent
ABLATION_DIR = ROOT / "Ablation"
if str(ABLATION_DIR) not in sys.path:
    sys.path.insert(0, str(ABLATION_DIR))

from ablation import CIFAR10_MEAN, CIFAR10_STD, build_model, count_parameters  # noqa: E402


CLASS_NAMES = (
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
)


def resolve_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return ROOT / path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Quickly verify the final Ablation model on CIFAR-10 test set."
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("Ablation/results/final/best_from_ablation"),
        help="Directory containing config.json, summary.json and weights/.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to model config JSON. Default: <run-dir>/config.json.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Path to checkpoint. Default: <run-dir>/weights/best.pt.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="CIFAR-10 data directory.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Evaluation batch size.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=2,
        help="DataLoader workers. Use 0 if Windows multiprocessing has issues.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu"),
        default="auto",
        help="Device used for evaluation.",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download CIFAR-10 if it is not already present.",
    )
    parser.add_argument(
        "--no-ema",
        action="store_true",
        help="Use raw model_state instead of ema_state when ema_state exists.",
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=0,
        help="Debug option: stop after N batches. 0 means evaluate the full test set.",
    )
    return parser.parse_args()


def choose_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")
    return torch.device(device_arg)


def load_json(path: Path) -> Dict:
    if not path.exists():
        raise FileNotFoundError(f"Cannot find JSON file: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_test_loader(data_dir: Path, batch_size: int, num_workers: int, download: bool) -> DataLoader:
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )
    dataset = datasets.CIFAR10(
        root=str(data_dir),
        train=False,
        transform=transform,
        download=download,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def load_model(config: Dict, checkpoint_path: Path, device: torch.device, use_ema: bool) -> Tuple[nn.Module, str]:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Cannot find checkpoint: {checkpoint_path}")

    model_kwargs = {
        "depth": int(config.get("depth", 190)),
        "growth_rate": int(config.get("growth_rate", 40)),
        "compression": float(config.get("compression", 0.5)),
        "activation": str(config.get("activation", "silu")),
        "se_reduction": int(config.get("se_reduction", 16)),
        "stochastic_depth_rate": float(config.get("stochastic_depth_rate", 0.2)),
        "classifier_hidden": int(config.get("classifier_hidden", 512)),
        "classifier_dropout": float(config.get("classifier_dropout", 0.2)),
        "drop_rate": float(config.get("drop_rate", 0.0)),
        "transition_dropout": float(config.get("transition_dropout", 0.0)),
        "num_classes": int(config.get("num_classes", 10)),
    }
    model = build_model(**model_kwargs).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if use_ema and checkpoint.get("ema_state") is not None:
        state_dict = checkpoint["ema_state"]
        state_name = "ema_state"
    elif checkpoint.get("model_state") is not None:
        state_dict = checkpoint["model_state"]
        state_name = "model_state"
    else:
        state_dict = checkpoint
        state_name = "raw_state_dict"

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            "Checkpoint does not match the model. "
            f"Missing keys: {missing[:8]} "
            f"Unexpected keys: {unexpected[:8]}"
        )

    model.eval()
    return model, state_name


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    label_smoothing: float,
    max_batches: int,
) -> Dict:
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    class_correct = torch.zeros(10, dtype=torch.long)
    class_total = torch.zeros(10, dtype=torch.long)
    total_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (images, targets) in enumerate(loader, start=1):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        outputs = model(images)
        loss = criterion(outputs, targets)
        predictions = outputs.argmax(dim=1)

        batch_size = targets.size(0)
        total_loss += loss.item() * batch_size
        correct += predictions.eq(targets).sum().item()
        total += batch_size

        for cls in range(10):
            mask = targets == cls
            class_total[cls] += mask.sum().cpu()
            class_correct[cls] += predictions[mask].eq(targets[mask]).sum().cpu()

        if max_batches > 0 and batch_idx >= max_batches:
            break

    accuracy = correct / total
    error_rate = 1.0 - accuracy
    return {
        "loss": total_loss / total,
        "accuracy": accuracy,
        "error_rate": error_rate,
        "correct": correct,
        "total": total,
        "class_correct": class_correct,
        "class_total": class_total,
    }


def print_reference(summary_path: Path, accuracy: float, error_rate: float, is_partial: bool) -> None:
    if not summary_path.exists():
        return

    summary = load_json(summary_path)
    best_acc = summary.get("best_test_acc")
    best_error = summary.get("best_test_error")
    best_epoch = summary.get("best_epoch")

    if best_acc is None or best_error is None:
        return

    print("\nReference from final summary:")
    print(f"  best_epoch      : {best_epoch}")
    print(f"  best_test_acc   : {best_acc * 100:.2f}%")
    print(f"  best_test_error : {best_error * 100:.2f}%")
    if is_partial:
        print("  note            : current run used --max-batches, so deltas are not comparable.")
    else:
        print(f"  acc_delta       : {(accuracy - best_acc) * 100:+.3f}%")
        print(f"  error_delta     : {(error_rate - best_error) * 100:+.3f}%")


def main() -> int:
    args = parse_args()
    run_dir = resolve_path(args.run_dir)
    config_path = resolve_path(args.config) if args.config else run_dir / "config.json"
    checkpoint_path = resolve_path(args.checkpoint) if args.checkpoint else run_dir / "weights" / "best.pt"
    data_dir = resolve_path(args.data_dir)
    summary_path = run_dir / "summary.json"

    config = load_json(config_path)
    device = choose_device(args.device)
    model, state_name = load_model(
        config=config,
        checkpoint_path=checkpoint_path,
        device=device,
        use_ema=not args.no_ema,
    )
    loader = build_test_loader(
        data_dir=data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        download=args.download,
    )

    label_smoothing = float(config.get("label_smoothing", 0.0))
    metrics = evaluate(
        model=model,
        loader=loader,
        device=device,
        label_smoothing=label_smoothing,
        max_batches=args.max_batches,
    )

    print("Quick CIFAR-10 check")
    print(f"  config          : {config_path}")
    print(f"  checkpoint      : {checkpoint_path}")
    print(f"  loaded weights  : {state_name}")
    print(f"  device          : {device}")
    print(
        "  model           : "
        f"depth={config.get('depth')}, "
        f"growth_rate={config.get('growth_rate')}, "
        f"activation={config.get('activation')}, "
        f"params={count_parameters(model):,}"
    )
    print(f"  samples         : {metrics['total']}")
    if args.max_batches > 0:
        print(f"  partial_eval    : first {args.max_batches} batch(es)")
    print(f"  test_loss       : {metrics['loss']:.4f}")
    print(f"  test_accuracy   : {metrics['accuracy'] * 100:.2f}%")
    print(f"  test_error      : {metrics['error_rate'] * 100:.2f}%")
    print(f"  correct/total   : {metrics['correct']}/{metrics['total']}")

    print("\nPer-class accuracy:")
    class_correct = metrics["class_correct"]
    class_total = metrics["class_total"]
    for idx, name in enumerate(CLASS_NAMES):
        total = int(class_total[idx].item())
        correct = int(class_correct[idx].item())
        acc = correct / total if total else 0.0
        print(f"  {name:10s}: {acc * 100:6.2f}% ({correct:4d}/{total:4d})")

    print_reference(
        summary_path=summary_path,
        accuracy=metrics["accuracy"],
        error_rate=metrics["error_rate"],
        is_partial=args.max_batches > 0,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
