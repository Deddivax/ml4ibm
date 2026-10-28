"""Print a concise, machine-independent summary of a training run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "run_path",
        type=Path,
        help="History JSON file or directory containing stage_01_history.json or history.json.",
    )
    return parser.parse_args()


def find_history(path: Path) -> Path:
    if path.is_file():
        return path
    for filename in ("stage_01_history.json", "history.json"):
        candidate = path / filename
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"No history JSON found in {path}")


def main() -> None:
    args = parse_args()
    history_path = find_history(args.run_path.expanduser())
    payload = json.loads(history_path.read_text(encoding="utf-8"))
    epochs = payload.get("epochs", [])

    print(f"History: {history_path}")
    print(f"Model: {payload.get('model_type', 'unknown')}")
    print(f"Completed epochs: {len(epochs)}")
    if not epochs:
        return

    latest = epochs[-1]
    best = min(epochs, key=lambda row: row["val_loss"])
    print(f"Latest: epoch {latest['epoch']}, train={latest['train_loss']:.6%}, val={latest['val_loss']:.6%}")
    print(f"Best: epoch {best['epoch']}, val={best['val_loss']:.6%}")
    if "learning_rate" in latest.get("train_metrics", {}):
        print(f"Latest learning rate: {latest['train_metrics']['learning_rate']:.3e}")