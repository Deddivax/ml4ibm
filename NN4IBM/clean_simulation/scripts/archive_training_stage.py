from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Archive a training stage and save its convergence plot.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--stage", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    history_path = run_dir / f"{args.stage}_history.json"
    checkpoint_path = run_dir / f"{args.stage}_best.pt"
    log_path = run_dir / f"{args.stage}.log"

    with history_path.open(encoding="utf-8") as handle:
        history = json.load(handle)
    epochs = history["epochs"]
    if not epochs:
        raise RuntimeError(f"No epochs found in {history_path}")

    last_epoch = int(epochs[-1]["epoch"])
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    archive_dir = run_dir / f"{args.stage}_interrupted_epoch{last_epoch}_{timestamp}"
    archive_dir.mkdir()

    for path in (history_path, checkpoint_path, log_path):
        if path.exists():
            shutil.copy2(path, archive_dir / path.name)

    x_values = [entry["epoch"] for entry in epochs]
    train_loss = [100.0 * entry["train_loss"] for entry in epochs]
    val_loss = [100.0 * entry["val_loss"] for entry in epochs]
    best_entry = min(epochs, key=lambda entry: entry["val_loss"])

    figure, axis = plt.subplots(figsize=(10, 6), constrained_layout=True)
    axis.plot(x_values, train_loss, label="Training", linewidth=1.5)
    axis.plot(x_values, val_loss, label="Validation", linewidth=1.8)
    axis.scatter(
        [best_entry["epoch"]], [100.0 * best_entry["val_loss"]], color="tab:green", zorder=3,
        label=f"Best validation: {100.0 * best_entry['val_loss']:.4f}%",
    )
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Relative L2 loss (%)")
    axis.set_title(f"{args.stage.replace('_', ' ').title()} convergence")
    axis.grid(alpha=0.3)
    axis.legend()
    figure.savefig(archive_dir / f"{args.stage}_convergence.png", dpi=180)
    plt.close(figure)

    summary = {
        "archived_at_utc": datetime.now(timezone.utc).isoformat(),
        "last_epoch": last_epoch,
        "latest_train_loss": float(epochs[-1]["train_loss"]),
        "latest_val_loss": float(epochs[-1]["val_loss"]),
        "best_epoch": int(best_entry["epoch"]),
        "best_val_loss": float(best_entry["val_loss"]),
        "source_history": history_path.name,
        "source_checkpoint": checkpoint_path.name,
    }
    with (archive_dir / f"{args.stage}_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(archive_dir)


if __name__ == "__main__":
    main()