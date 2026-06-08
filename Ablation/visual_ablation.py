"""Create plots and tables from Ablation/ ablation runs."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


ABLATION_ROOT = Path(__file__).resolve().parent
DEFAULT_RESULTS = ABLATION_ROOT / "results" / "ablation"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize ablation summaries.")
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--summary", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--top-k-curves", type=int, default=8)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def as_float(value: Any, default: float = 0.0) -> float:
    try:
        if value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def read_history(path: Path) -> tuple[list[int], list[float]]:
    epochs: list[int] = []
    acc: list[float] = []
    if not path.exists():
        return epochs, acc
    with path.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            epochs.append(int(float(row["epoch"])))
            acc.append(float(row["test_acc"]) * 100.0)
    return epochs, acc


def ok_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return [row for row in rows if row.get("status") == "ok" and row.get("best_test_acc") not in ("", None)]


def plot_best_accuracy(rows: list[dict[str, str]], output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = sorted(rows, key=lambda row: as_float(row["best_test_acc"]), reverse=True)
    labels = [row["experiment"].replace("_", "\n") for row in rows]
    values = [as_float(row["best_test_acc"]) * 100.0 for row in rows]
    colors_by_group = {
        "baseline": "#4c78a8",
        "capacity": "#f58518",
        "activation": "#54a24b",
        "loss": "#e45756",
        "optimizer": "#b279a2",
        "component": "#72b7b2",
        "regularization": "#ff9da6",
    }
    colors = [colors_by_group.get(row["group"], "#9d9d9d") for row in rows]

    fig, ax = plt.subplots(figsize=(max(9, len(rows) * 1.2), 5.5))
    ax.bar(range(len(rows)), values, color=colors)
    ax.set_ylabel("Best test accuracy (%)")
    ax.set_title("Ablation Best Accuracy")
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels(labels, rotation=0, fontsize=8)
    low = max(0.0, min(values) - 2.0) if values else 0.0
    high = min(100.0, max(values) + 0.5) if values else 100.0
    ax.set_ylim(low, high)
    ax.grid(axis="y", alpha=0.25)
    for index, value in enumerate(values):
        ax.text(index, value + 0.05, f"{value:.2f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_learning_curves(rows: list[dict[str, str]], output_path: Path, top_k: int) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = sorted(rows, key=lambda row: as_float(row["best_test_acc"]), reverse=True)[:top_k]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for row in rows:
        epochs, acc = read_history(Path(row["history"]))
        if epochs:
            ax.plot(epochs, acc, label=row["experiment"])
    ax.set_title(f"Top {len(rows)} Ablation Test Accuracy Curves")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Test accuracy (%)")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def best_by_group(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["group"]].append(row)
    result = []
    for group, group_rows in sorted(grouped.items()):
        best = max(group_rows, key=lambda row: as_float(row["best_test_acc"]))
        result.append(
            {
                "group": group,
                "best_experiment": best["experiment"],
                "best_test_acc": f"{as_float(best['best_test_acc']) * 100.0:.2f}",
                "best_test_error": f"{as_float(best['best_test_error']) * 100.0:.2f}",
                "best_epoch": best["best_epoch"],
                "trainable_parameters": best["trainable_parameters"],
            }
        )
    return result


def write_markdown(rows: list[dict[str, str]], group_rows: list[dict[str, Any]], output_path: Path) -> None:
    best = max(rows, key=lambda row: as_float(row["best_test_acc"])) if rows else None
    lines = [
        "# Ablation Summary",
        "",
        "This file is generated by `visual_ablation.py`.",
        "",
    ]
    if best is not None:
        lines.append(
            f"Best short ablation run: `{best['experiment']}` with "
            f"{as_float(best['best_test_acc']) * 100.0:.2f}% test accuracy."
        )
        lines.append("")
    lines.extend(["## Best By Group", "", "| Group | Best experiment | Acc (%) | Error (%) | Best epoch | Params |", "|---|---|---:|---:|---:|---:|"])
    for row in group_rows:
        lines.append(
            f"| {row['group']} | {row['best_experiment']} | {row['best_test_acc']} | "
            f"{row['best_test_error']} | {row['best_epoch']} | {row['trainable_parameters']} |"
        )
    lines.extend(["", "## All Runs", "", "| Experiment | Group | Acc (%) | Error (%) | Epoch | Status |", "|---|---|---:|---:|---:|---|"])
    for row in sorted(rows, key=lambda item: as_float(item["best_test_acc"]), reverse=True):
        lines.append(
            f"| {row['experiment']} | {row['group']} | {as_float(row['best_test_acc']) * 100.0:.2f} | "
            f"{as_float(row['best_test_error']) * 100.0:.2f} | {row['best_epoch']} | {row['status']} |"
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    summary_path = args.summary or (args.results_dir / "summary.csv")
    output_dir = args.output_dir or (args.results_dir / "figures")
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = ok_rows(read_csv(summary_path))
    if not rows:
        raise ValueError(f"No completed ablation rows found in {summary_path}")

    plot_best_accuracy(rows, output_dir / "ablation_best_accuracy.png")
    plot_learning_curves(rows, output_dir / "ablation_learning_curves.png", args.top_k_curves)
    group_rows = best_by_group(rows)
    write_csv(output_dir / "best_by_group.csv", group_rows)
    write_markdown(rows, group_rows, output_dir / "ablation_summary.md")

    best_config = args.results_dir / "best_config.json"
    if best_config.exists():
        payload = json.loads(best_config.read_text(encoding="utf-8"))
        (output_dir / "recommended_config.json").write_text(
            json.dumps(payload.get("recommended_config", {}), indent=2),
            encoding="utf-8",
        )

    print(f"Saved ablation figures and tables to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
