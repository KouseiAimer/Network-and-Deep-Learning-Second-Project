"""Run the final experiment using settings selected by Ablation/ablation.py."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ablation import ABLATION_ROOT, DEFAULT_DATA_DIR, train_experiment


DEFAULT_BEST_CONFIG = ABLATION_ROOT / "results" / "ablation" / "best_config.json"
DEFAULT_OUTPUT_DIR = ABLATION_ROOT / "results" / "final" / "best_from_ablation"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run final MyDenseNet training from local ablation results.")
    parser.add_argument("--best-config", type=Path, default=DEFAULT_BEST_CONFIG)
    parser.add_argument("--mode", choices=["recommended", "best-single"], default="recommended")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--amp", dest="amp", action="store_true", default=True)
    parser.add_argument("--no-amp", dest="amp", action="store_false")
    parser.add_argument("--subset", type=int, default=0)
    parser.add_argument("--eval-max-batches", type=int, default=0)
    parser.add_argument("--clean-train-eval", action="store_true")
    parser.add_argument("--clean-train-max-batches", type=int, default=0)
    parser.add_argument("--save-every", type=int, default=0)
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def main() -> None:
    args = parse_args()
    best_config = load_json(args.best_config)
    if args.mode == "recommended":
        selected_config = dict(best_config["recommended_config"])
    else:
        selected_config = dict(best_config["best_single_config"])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "selected_final_config.json").open("w", encoding="utf-8") as file:
        json.dump(
            {
                "mode": args.mode,
                "source_best_config": str(args.best_config),
                "selected_config": selected_config,
            },
            file,
            indent=2,
        )

    print("Running final experiment with selected config:")
    print(json.dumps(selected_config, indent=2))
    train_experiment(selected_config, args.output_dir, args)
    print(f"Final artifacts saved to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
