"""Plot and summarize DenseNet training logs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def read_history(history_path: Path) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    with history_path.open("r", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            rows.append({key: float(value) for key, value in row.items()})
    return rows


def summarize_history(history_path: Path) -> dict[str, float | int]:
    rows = read_history(history_path)
    if not rows:
        raise ValueError(f"No rows found in {history_path}")

    best = max(rows, key=lambda row: row["test_acc"])
    final = rows[-1]
    total_time = sum(row.get("time_sec", 0.0) for row in rows)
    return {
        "best_epoch": int(best["epoch"]),
        "best_test_acc": best["test_acc"],
        "best_test_error": 1.0 - best["test_acc"],
        "best_test_loss": best["test_loss"],
        "final_epoch": int(final["epoch"]),
        "final_train_acc": final["train_acc"],
        "final_train_loss": final["train_loss"],
        "final_test_acc": final["test_acc"],
        "final_test_loss": final["test_loss"],
        "total_time_sec": total_time,
        "total_time_min": total_time / 60.0,
    }


def write_summary(history_path: Path, output_path: Path) -> dict[str, float | int]:
    summary = summarize_history(history_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)
    return summary


def plot_history(history_path: Path, output_path: Path, title: str = "DenseNet-BC on CIFAR-10") -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = read_history(history_path)
    epochs = [int(row["epoch"]) for row in rows]
    train_loss = [row["train_loss"] for row in rows]
    test_loss = [row["test_loss"] for row in rows]
    train_acc = [row["train_acc"] * 100.0 for row in rows]
    test_acc = [row["test_acc"] * 100.0 for row in rows]
    lr = [row["lr"] for row in rows]
    best_acc = []
    current_best = 0.0
    for acc in test_acc:
        current_best = max(current_best, acc)
        best_acc.append(current_best)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    axes[0, 0].plot(epochs, train_loss, label="train")
    axes[0, 0].plot(epochs, test_loss, label="test")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].set_title("Loss")
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.25)

    axes[0, 1].plot(epochs, train_acc, label="train")
    axes[0, 1].plot(epochs, test_acc, label="test")
    axes[0, 1].plot(epochs, best_acc, linestyle="--", label="best test")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Accuracy (%)")
    axes[0, 1].set_title("Accuracy")
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.25)

    axes[1, 0].plot(epochs, lr)
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Learning rate")
    axes[1, 0].set_title("Learning Rate")
    axes[1, 0].set_yscale("log")
    axes[1, 0].grid(alpha=0.25)

    test_error = [100.0 - acc for acc in test_acc]
    axes[1, 1].plot(epochs, test_error, color="tab:red")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("Test error (%)")
    axes[1, 1].set_title("Test Error")
    axes[1, 1].grid(alpha=0.25)

    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Plot DenseNet training curves")
    parser.add_argument("--history", type=Path, default=root / "results" / "densenet_bc_100_24" / "history.csv")
    parser.add_argument("--output", type=Path, default=root / "results" / "densenet_bc_100_24" / "curves.png")
    parser.add_argument("--summary", type=Path, default=root / "results" / "densenet_bc_100_24" / "summary.json")
    parser.add_argument("--title", type=str, default="DenseNet-BC-100-24 on CIFAR-10")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plot_history(args.history, args.output, args.title)
    summary = write_summary(args.history, args.summary)
    print(f"Saved curves to: {args.output.resolve()}")
    print(f"Saved summary to: {args.summary.resolve()}")
    print(f"Best test accuracy: {summary['best_test_acc'] * 100:.2f}%")


if __name__ == "__main__":
    main()
