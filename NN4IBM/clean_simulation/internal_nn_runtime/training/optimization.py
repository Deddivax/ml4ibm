"""Optimizer and scheduler construction for the active training workflow."""

from __future__ import annotations

import torch

from clean_simulation.internal_nn_runtime.cutcell_ml.config import TrainingConfig


def build_optimizer_and_scheduler(
    model: torch.nn.Module,
    config: TrainingConfig,
    *,
    distributed: bool,
    steps_per_epoch: int,
) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler | None, bool]:
    """Build configured optimization state and identify per-batch scheduling."""
    optimizer_type = config.optimizer_type.lower()
    if optimizer_type == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    elif optimizer_type == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=config.lr,
            momentum=config.sgd_momentum,
            nesterov=config.sgd_momentum > 0.0,
            weight_decay=config.weight_decay,
        )
    elif optimizer_type == "lbfgs":
        if distributed:
            raise ValueError("L-BFGS is supported only for single-process late-stage refinement, not DDP.")
        if config.use_amp:
            raise ValueError("L-BFGS requires use_amp=False so each closure has deterministic full precision.")
        if config.weight_decay != 0.0:
            raise ValueError("L-BFGS has no decoupled weight decay; set weight_decay=0.0 for refinement.")
        optimizer = torch.optim.LBFGS(
            model.parameters(),
            lr=config.lr,
            max_iter=config.lbfgs_max_iter,
            history_size=config.lbfgs_history_size,
            line_search_fn=config.lbfgs_line_search_fn,
        )
    else:
        raise ValueError(
            f"Unsupported optimizer_type={config.optimizer_type!r}; expected 'adamw', 'sgd', or 'lbfgs'."
        )

    scheduler_is_onecycle = config.scheduler_type == "onecycle"
    if optimizer_type == "lbfgs" and scheduler_is_onecycle:
        raise ValueError("L-BFGS does not support the per-batch OneCycle scheduler; use cosine or none.")
    if scheduler_is_onecycle:
        print(
            "Problem: OneCycleLR scheduler is enabled. This scheduler will adjust the learning rate every batch, "
            "and it is not compatible with the spike rollback mechanism. If a spike occurs, the learning rate "
            "will be reset to its original value, which may not be what you want. Consider using a different "
            "scheduler or disabling spike rollback.",
            flush=True,
        )
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=config.lr * config.onecycle_max_lr_factor,
            epochs=config.epochs,
            steps_per_epoch=steps_per_epoch,
            pct_start=config.onecycle_pct_start,
            final_div_factor=100.0,
        )
    elif config.scheduler_type == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.epochs,
            eta_min=config.lr * config.cosine_eta_min_factor,
        )
    else:
        scheduler = None
    return optimizer, scheduler, scheduler_is_onecycle