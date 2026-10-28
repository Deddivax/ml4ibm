"""Run a reproducible, four-stage adaptive learning-rate training chain."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = ("-m", "clean_simulation.scripts.training.run")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--model-type", choices=("cutcellnet", "transformer"), required=True)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--blocks", type=int, required=True)
    parser.add_argument("--residual-scaling", default="fixed")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--stage-epochs", type=int, required=True)
    parser.add_argument("--stage-wall-seconds", type=int, default=1750)
    parser.add_argument("--stages", type=int, default=4)
    parser.add_argument("--initial-lr", type=float, default=2.0e-4)
    parser.add_argument("--resume-from-stage", type=int, default=0)
    return parser.parse_args()


def next_learning_rate(current_lr: float, previous_best: float | None, stage_best: float) -> tuple[float, str]:
    if previous_best is None:
        return current_lr * 0.7, "initial exploration complete; anneal by 0.70"
    improvement = (previous_best - stage_best) / previous_best
    if improvement >= 0.03:
        return current_lr * 0.75, f"{improvement:.2%} improvement; anneal by 0.75"
    if improvement >= 0.005:
        return current_lr * 0.60, f"{improvement:.2%} improvement; anneal by 0.60"
    if improvement >= -0.005:
        return current_lr * 0.45, f"plateau ({improvement:.2%}); anneal by 0.45"
    return current_lr * 0.30, f"regression ({improvement:.2%}); reduce by 0.30"


def write_combined_artifacts(run_dir: Path, records: list[dict]) -> None:
    (run_dir / "chain_summary.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    rows = []
    global_epoch = 0
    for record in records:
        for item in record["history"]["epochs"]:
            global_epoch += 1
            rows.append(
                {
                    "global_epoch": global_epoch,
                    "stage": record["stage"],
                    "stage_epoch": item["epoch"],
                    "learning_rate": record["learning_rate"],
                    "train_loss": item["train_loss"],
                    "val_loss": item["val_loss"],
                    "train_rel_l2": item["train_metrics"].get("rel_l2"),
                    "val_rel_l2": item["val_metrics"].get("rel_l2"),
                }
            )
    with (run_dir / "combined_history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys() if rows else ["global_epoch"])
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    run_dir = ROOT / "clean_simulation" / "outputs" / "experiments" / "training" / args.name
    run_dir.mkdir(parents=True, exist_ok=True)
    summary_path = run_dir / "chain_summary.json"
    records: list[dict] = []
    current_lr = args.initial_lr
    previous_best: float | None = None
    previous_checkpoint: Path | None = None

    if args.resume_from_stage:
        if not summary_path.is_file():
            raise FileNotFoundError(f"Cannot resume stage {args.resume_from_stage}: missing {summary_path}")
        records = json.loads(summary_path.read_text(encoding="utf-8"))
        if len(records) != args.resume_from_stage:
            raise ValueError(
                f"Expected {args.resume_from_stage} completed stages, found {len(records)} in {summary_path}"
            )
        previous_checkpoint = run_dir / f"stage_{args.resume_from_stage:02d}_best.pt"
        if not previous_checkpoint.is_file():
            raise FileNotFoundError(f"Missing resume checkpoint: {previous_checkpoint}")
        previous_best = float(records[-1]["best_val_loss"])
        saved_next_lr = records[-1]["next_learning_rate"]
        if saved_next_lr is None:
            preceding_best = float(records[-2]["best_val_loss"]) if len(records) > 1 else None
            current_lr, _ = next_learning_rate(
                float(records[-1]["learning_rate"]),
                preceding_best,
                previous_best,
            )
        else:
            current_lr = float(saved_next_lr)

    for stage in range(args.resume_from_stage + 1, args.stages + 1):
        checkpoint = run_dir / f"stage_{stage:02d}_best.pt"
        history_path = run_dir / f"stage_{stage:02d}_history.json"
        stage_log = run_dir / f"stage_{stage:02d}.log"
        environment = os.environ.copy()
        environment.update(
            {
                "CUT_CELL_MODEL_TYPE": args.model_type,
                "CUT_CELL_FUSION_DIM": str(args.width),
                "CUT_CELL_RESIDUAL_BLOCKS": str(args.blocks),
                "CUT_CELL_RESIDUAL_SCALING": args.residual_scaling,
                "CUT_CELL_SEED": str(args.seed),
                "CUT_CELL_EPOCHS": str(args.stage_epochs),
                "CUT_CELL_BATCH_SIZE": "2048",
                "CUT_CELL_INITIAL_LR": f"{current_lr:.12g}",
                "CUT_CELL_SCHEDULER_TYPE": "cosine",
                "CUT_CELL_COSINE_ETA_MIN_FACTOR": "0.1",
                "CUT_CELL_MAX_WALL_SECONDS": str(args.stage_wall_seconds),
                "CUT_CELL_SPIKE_THRESHOLD": "0.15",
                "CUT_CELL_OUTPUT_CHECKPOINT": str(checkpoint),
                "CUT_CELL_HISTORY_PATH": str(history_path),
                "CUT_CELL_INIT_CHECKPOINT": str(previous_checkpoint or run_dir / "no_checkpoint.pt"),
            }
        )
        started_at = time.time()
        print(f"Stage {stage}/{args.stages}: lr={current_lr:.3e}, init={previous_checkpoint}", flush=True)
        with stage_log.open("w", encoding="utf-8") as handle:
            completed = subprocess.run([sys.executable, *LAUNCHER], cwd=ROOT, env=environment, stdout=handle, stderr=subprocess.STDOUT)
        if completed.returncode != 0 or not history_path.is_file() or not checkpoint.is_file():
            raise RuntimeError(f"Stage {stage} failed; inspect {stage_log}")
        history = json.loads(history_path.read_text(encoding="utf-8"))
        stage_best = float(history["best_val_loss"])
        next_lr, adaptation = next_learning_rate(current_lr, previous_best, stage_best)
        records.append(
            {
                "stage": stage,
                "learning_rate": current_lr,
                "best_val_loss": stage_best,
                "elapsed_seconds": time.time() - started_at,
                "checkpoint": str(checkpoint),
                "history": history,
                "next_learning_rate": next_lr if stage < args.stages else None,
                "adaptation": adaptation if stage < args.stages else "final stage",
            }
        )
        write_combined_artifacts(run_dir, records)
        print(f"Stage {stage} best={stage_best:.6%}; {adaptation}", flush=True)
        previous_checkpoint = checkpoint
        previous_best = stage_best
        current_lr = next_lr


if __name__ == "__main__":
    main()