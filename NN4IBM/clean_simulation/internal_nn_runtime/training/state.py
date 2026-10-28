"""Rollback and learning-rate state helpers for resilient training epochs."""

from __future__ import annotations

from copy import deepcopy


def capture_epoch_state(model, optimizer, scheduler, scaler) -> dict:
    """Capture state needed to repeat an epoch without retaining its updates."""
    return {
        "model": deepcopy(model.state_dict()),
        "opt": deepcopy(optimizer.state_dict()),
        "scheduler": deepcopy(scheduler.state_dict()) if scheduler is not None else None,
        "scaler": deepcopy(scaler.state_dict()) if scaler is not None else None,
    }


def restore_epoch_state(model, optimizer, scheduler, scaler, snapshot: dict) -> None:
    """Restore a snapshot captured before a rejected training epoch."""
    model.load_state_dict(snapshot["model"])
    optimizer.load_state_dict(snapshot["opt"])
    if scheduler is not None and snapshot["scheduler"] is not None:
        scheduler.load_state_dict(snapshot["scheduler"])
    if scaler is not None and snapshot["scaler"] is not None:
        scaler.load_state_dict(snapshot["scaler"])


def reduce_learning_rate(optimizer, scheduler, factor: float) -> None:
    """Apply a rollback reduction that later scheduler steps retain."""
    for parameter_group in optimizer.param_groups:
        parameter_group["lr"] *= factor
    if scheduler is not None:
        scheduler.base_lrs = [learning_rate * factor for learning_rate in scheduler.base_lrs]
        scheduler._last_lr = [learning_rate * factor for learning_rate in scheduler.get_last_lr()]