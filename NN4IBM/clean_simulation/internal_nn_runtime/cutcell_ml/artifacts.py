"""Checkpoint and history artifacts produced by training runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from .training import TrainingHistory


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    """Return the underlying model when DDP or torch.compile wraps it."""
    model_to_save = model.module if hasattr(model, "module") else model
    while hasattr(model_to_save, "_orig_mod"):
        model_to_save = model_to_save._orig_mod
    return model_to_save


def save_checkpoint(
    checkpoint_path: Path,
    model: torch.nn.Module,
    history: TrainingHistory,
    epoch: int,
    validation_loss: float,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    scaler: torch.amp.GradScaler | None = None,
) -> None:
    """Write the existing checkpoint payload format used by inference tools."""
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "val_loss": validation_loss,
            "model_state_dict": unwrap_model(model).state_dict(),
            "history": history,
            "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        },
        checkpoint_path,
    )


def history_epochs(history: TrainingHistory) -> list[dict[str, Any]]:
    """Convert in-memory epoch history to the JSON run-history schema."""
    return [
        {
            "epoch": epoch,
            "train_loss": float(train_loss),
            "val_loss": float(validation_loss),
            "train_metrics": {name: float(value) for name, value in train_metrics.items()},
            "val_metrics": {name: float(value) for name, value in validation_metrics.items()},
        }
        for epoch, (train_loss, validation_loss, train_metrics, validation_metrics) in enumerate(
            zip(history.train_loss, history.val_loss, history.train_metrics, history.val_metrics),
            start=1,
        )
    ]


def save_history(
    history_path: Path | None,
    history: TrainingHistory,
    run_config: dict[str, Any],
    best_validation_loss: float,
) -> None:
    """Write an inspectable JSON run manifest and epoch history."""
    if history_path is None:
        return
    history_path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(run_config)
    payload["best_val_loss"] = float(best_validation_loss)
    payload["epochs"] = history_epochs(history)
    with history_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)