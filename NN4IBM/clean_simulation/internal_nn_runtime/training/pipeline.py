from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
import os
import re
import time
from typing import Any, Callable
import h5py

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from clean_simulation.internal_nn_runtime.cutcell_ml.config import RunConfig
from clean_simulation.internal_nn_runtime.cutcell_ml.data import CutCellDataset, load_dataset_from_h5
from clean_simulation.internal_nn_runtime.cutcell_ml.normalizers import (
    estimate_input_normalizers as _estimate_input_normalizers,
)
from clean_simulation.internal_nn_runtime.cutcell_ml.training import (
    DEFAULT_LOSS_COMPONENTS,
    LossComponent,
    TrainingHistory,
    composite_loss,
    evaluate,
    FastTensorDataLoader,
    make_loaders,
)
from clean_simulation.internal_nn_runtime.training.diagnostics import model_structural_diagnostics
from clean_simulation.internal_nn_runtime.training.distributed import (
    assert_model_parity as _assert_distributed_model_parity,
    average_metrics as _distributed_average_metrics,
)
from clean_simulation.internal_nn_runtime.training.model_factory import build_training_model
from clean_simulation.internal_nn_runtime.training.optimization import build_optimizer_and_scheduler
from clean_simulation.internal_nn_runtime.training.state import (
    capture_epoch_state as _capture_epoch_state,
    reduce_learning_rate as _reduce_learning_rate,
    restore_epoch_state as _restore_epoch_state,
)

@torch.no_grad()
def _drop_hardest_training_examples(
    model: torch.nn.Module,
    train_loader: FastTensorDataLoader,
    device: str,
    drop_fraction: float,
    *,
    rank: int,
    distributed: bool,
) -> None:
    """Remove the highest-error train examples using the initialized model.

    The selection is performed once on rank zero before DDP wrapping and the
    resulting mask is broadcast, so every rank trains on the same filtered
    corpus. Validation data and its membership are intentionally unchanged.
    """
    if not 0.0 <= drop_fraction < 1.0:
        raise ValueError("hard_example_drop_fraction must be in [0, 1).")
    if drop_fraction == 0.0:
        return
    if not isinstance(train_loader, FastTensorDataLoader):
        raise TypeError("hard-example filtering requires FastTensorDataLoader.")

    keep_mask = torch.empty(train_loader.num_samples, dtype=torch.bool, device=device)
    removed_count = 0
    if rank == 0:
        was_training = model.training
        model.eval()
        reference_norm = torch.linalg.matrix_norm(train_loader.k_mat, ord="fro", dim=(-2, -1))
        floor = torch.quantile(reference_norm, 0.05).clamp(min=1.0e-4).to(device)
        errors = []
        for start in range(0, train_loader.num_samples, train_loader.batch_size):
            stop = min(start + train_loader.batch_size, train_loader.num_samples)
            distances = train_loader.d[start:stop].to(device, non_blocking=True)
            targets = train_loader.k_mat[start:stop].to(device, non_blocking=True)
            predicted_unit, log_norm = model(distances)
            predicted = predicted_unit.float() * torch.exp(log_norm.float())[:, None, None]
            error_norm = torch.linalg.matrix_norm(predicted - targets, ord="fro", dim=(-2, -1))
            target_norm = torch.linalg.matrix_norm(targets, ord="fro", dim=(-2, -1))
            errors.append((error_norm / torch.clamp(target_norm, min=floor)).cpu())
        errors = torch.cat(errors)
        removed_count = max(1, int(drop_fraction * train_loader.num_samples))
        drop_indices = torch.topk(errors, k=removed_count, largest=True, sorted=False).indices
        keep_mask.fill_(True)
        keep_mask[drop_indices.to(device)] = False
        if was_training:
            model.train()

    if distributed:
        dist.broadcast(keep_mask, src=0)

    keep_mask_cpu = keep_mask.cpu()
    kept_count = int(keep_mask_cpu.sum())
    train_loader.d = train_loader.d[keep_mask_cpu].contiguous()
    train_loader.nu = train_loader.nu[keep_mask_cpu].contiguous()
    train_loader.k_mat = train_loader.k_mat[keep_mask_cpu].contiguous()
    train_loader.classes = train_loader.classes[keep_mask_cpu].contiguous()
    train_loader.num_samples = kept_count
    if rank == 0:
        print(
            f"Filtered {removed_count:,} hard training examples ({drop_fraction:.2%}); "
            f"retained {kept_count:,}. Validation is unchanged.",
            flush=True,
        )


@dataclass(slots=True)
class PipelineResult:
    model: torch.nn.Module
    history: TrainingHistory
    group: str
    train_size: int
    val_size: int
    total_size: int
    device: str
    symmetry_error: float
    rigid_body_residual: float
    min_eigenvalue: float
    min_physical_eigenvalue: float
    max_eigenvalue: float


def _train_one_epoch(
    model, opt, scheduler, scaler, train_loader, device: str,
    loss_components: list[LossComponent], grad_clip_norm: float,
    scheduler_is_onecycle: bool = False, lbfgs_reset_history_per_batch: bool = True,
) -> dict:
    """One pass over train_loader."""
    model.train()
    dev = torch.device(device)

    if isinstance(opt, torch.optim.LBFGS):
        total_samples = getattr(train_loader, "num_samples", None)
        if total_samples is None:
            raise ValueError("L-BFGS requires a loader that exposes num_samples for full-objective weighting.")
        closure_metrics: dict[str, torch.Tensor] = {}

        def closure() -> torch.Tensor:
            opt.zero_grad(set_to_none=True)
            metric_totals: dict[str, torch.Tensor] = {}
            objective = torch.zeros((), device=dev)

            for d, nu, k_ref, cls in train_loader:
                d = torch.nan_to_num(d.to(dev, non_blocking=True), nan=0.0, posinf=1e6, neginf=-1e6)
                nu = torch.nan_to_num(nu.to(dev, non_blocking=True), nan=0.3, posinf=1.0, neginf=-1.0)
                k_ref = torch.nan_to_num(k_ref.to(dev, non_blocking=True), nan=0.0, posinf=1e9, neginf=-1e9)
                cls = cls.to(dev, non_blocking=True).long()
                if cls.numel() == 0:
                    continue

                loss, metrics = composite_loss(model, d, nu, k_ref, cut_class=cls, loss_components=loss_components)
                if not torch.isfinite(loss):
                    raise FloatingPointError("L-BFGS closure produced a non-finite loss")
                batch_weight = d.shape[0] / total_samples
                (loss * batch_weight).backward()
                objective = objective + loss.detach() * batch_weight
                for name, value in metrics.items():
                    weighted_value = value.detach() * d.shape[0]
                    metric_totals[name] = metric_totals.get(name, torch.zeros_like(weighted_value)) + weighted_value

            if grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            closure_metrics.clear()
            closure_metrics.update({name: value / total_samples for name, value in metric_totals.items()})
            return objective

        opt.step(closure)
        metrics = closure_metrics
        metrics["n_samples"] = total_samples
        return {name: float(value.item()) if isinstance(value, torch.Tensor) else value for name, value in metrics.items()}

    totals: dict[str, torch.Tensor] = {}
    n_samples = 0

    for d, nu, k_ref, cls in train_loader:
        d = torch.nan_to_num(d.to(dev, non_blocking=True), nan=0.0, posinf=1e6, neginf=-1e6)
        nu = torch.nan_to_num(nu.to(dev, non_blocking=True), nan=0.3, posinf=1.0, neginf=-1.0)
        k_ref = torch.nan_to_num(k_ref.to(dev, non_blocking=True), nan=0.0, posinf=1e9, neginf=-1e9)
        cls = cls.to(dev, non_blocking=True).long()
        if cls.numel() == 0:
            continue

        if scaler is not None:
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast():
                loss, metrics = composite_loss(model, d, nu, k_ref, cut_class=cls, loss_components=loss_components)
            if not torch.isfinite(loss):
                continue
            scaler.scale(loss).backward()
            if grad_clip_norm > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            scaler.step(opt)
            scaler.update()
        else:
            opt.zero_grad(set_to_none=True)
            loss, metrics = composite_loss(model, d, nu, k_ref, cut_class=cls, loss_components=loss_components)
            if not torch.isfinite(loss):
                continue
            loss.backward()
            if grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            opt.step()


        bsz = d.shape[0]
        for k, v in metrics.items():
            weighted_value = v.detach() * bsz
            totals[k] = totals.get(k, torch.zeros_like(weighted_value)) + weighted_value
        n_samples += bsz

    if n_samples == 0:
        return {"total": float("nan"), "n_samples": 0}
    avg = {k: float((v / n_samples).item()) for k, v in totals.items()}
    avg["n_samples"] = n_samples
    return avg


def run_training_pipeline(
    config: RunConfig,
    base_dir: Path | None = None,
    init_checkpoint_path: Path | None = None,
    preserve_estimated_input_normalizers: bool = False,
    checkpoint_callback: Callable[[torch.nn.Module, TrainingHistory, int, float, torch.optim.Optimizer, Any, Any], None] | None = None,
    loss_components: list[LossComponent] | None = None,
    spike_threshold: float = 0.05,
    spike_lr_reduction_factor: float = 0.5,
    max_spike_retries: int = 3,
    canonicalize_inputs: bool = False,
    resume_learning_rate: float | None = None,
) -> PipelineResult:
    """
    loss_components         : list of LossComponent to optimize (defaults
                               to just relative-Frobenius reconstruction
                               error). See training.py's module docstring
                               for how to add a new component.
    spike_threshold          : an epoch's train loss is a "spike" if it
                               exceeds the previous ACCEPTED epoch's train
                               loss by more than this fraction. The first
                               epoch is never a spike (no prior baseline).
    spike_lr_reduction_factor : LR multiplier for a retried epoch,
                               compounding across consecutive retries of
                               the SAME epoch (factor ** retry_count).
    max_spike_retries        : cap on retries per epoch; if still spiking
                               after this many, the epoch is accepted
                               anyway (with a warning) so training can't
                               stall indefinitely.
    canonicalize_inputs      : passed to CutCellNet's constructor
                               (model_type="cutcellnet" only). Raw
                               physical-space learning is the default;
                               enable this only after validating the
                               full-cell complement convention for the
                               specific dataset.
    """
    loss_components = loss_components if loss_components is not None else DEFAULT_LOSS_COMPONENTS
    distributed = bool(config.training.use_distributed)
    if distributed:
        if not torch.cuda.is_available():
            raise RuntimeError("Distributed training requires CUDA.")
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        device = f"cuda:{local_rank}"
    else:
        rank, world_size, device = 0, 1, "cuda"

    base_dir = Path.cwd() if base_dir is None else base_dir
    config.resolve_paths(base_dir)
    torch.manual_seed(config.training.seed)

    h5_paths = [Path(p) for p in config.data.h5_path] if isinstance(config.data.h5_path, (list, tuple)) \
        else [Path(config.data.h5_path)]

    requested_group = f"integration_{config.data.integration_order}_poly_{config.data.poly_order}"
    expected_n_distance_nodes = (config.data.refined_subcells + 1) ** 3

    def _resolve_group(h5_file: h5py.File) -> str | None:
        if requested_group in h5_file:
            return requested_group
        legacy = f"depth_{config.data.depth}_{requested_group}"
        if legacy in h5_file:
            return legacy
        for key in h5_file.keys():
            if re.sub(r"^depth_\d+_", "", key) == requested_group:
                return key
        return None

    all_d, all_nu, all_k = [], [], []
    excluded_geometry_prefixes = tuple(config.training.excluded_geometry_prefixes)
    for h5_p in h5_paths:
        if not h5_p.exists():
            print(f"Warning: training h5 does not exist: {h5_p}", flush=True)
            continue
        with h5py.File(h5_p, "r") as h5_f:
            file_grp = _resolve_group(h5_f)
            if file_grp is None:
                print(f"Warning: no group matching '{requested_group}' found in {h5_p}", flush=True)
                continue
            d_file, nu_file, k_file = load_dataset_from_h5(
                h5_p, file_grp, config.data.poisson_ratio, max_samples=config.data.max_load_samples,
            )
            if excluded_geometry_prefixes:
                if "geometry_name" not in h5_f[file_grp]:
                    raise ValueError("excluded_geometry_prefixes requires a geometry_name dataset.")
                raw_names = h5_f[file_grp]["geometry_name"][...]
                names = [name.decode("utf-8") if isinstance(name, bytes) else str(name) for name in raw_names]
                keep_mask = torch.tensor(
                    [not name.startswith(excluded_geometry_prefixes) for name in names], dtype=torch.bool,
                )
                removed = int((~keep_mask).sum())
                if removed:
                    d_file, nu_file, k_file = d_file[keep_mask], nu_file[keep_mask], k_file[keep_mask]
                if rank == 0:
                    print(
                        f"Excluded {removed:,} row(s) for geometry prefixes {excluded_geometry_prefixes}; "
                        f"retained {int(keep_mask.sum()):,} from {h5_p.name} before splitting.",
                        flush=True,
                    )
            if d_file.shape[1] != expected_n_distance_nodes:
                raise ValueError(
                    f"{h5_p}/{file_grp} has {d_file.shape[1]} distance inputs, but "
                    f"refined_subcells={config.data.refined_subcells} expects {expected_n_distance_nodes}."
                )
            all_d.append(d_file)
            all_nu.append(nu_file)
            all_k.append(k_file)

    if not all_d:
        raise RuntimeError(f"No valid data loaded from any of: {h5_paths}")

    dataset = CutCellDataset(torch.cat(all_d), torch.cat(all_nu), torch.cat(all_k))
    total_size = len(dataset)
    train_size = int(config.training.train_split * total_size)
    val_size = total_size - train_size
    if train_size == 0 or val_size == 0:
        raise ValueError(f"Dataset too small: total={total_size}")

    train_loader, val_loader, n_train, n_val = make_loaders(
        dataset=dataset,
        batch_size=config.training.batch_size,
        seed=config.training.seed,
        train_split=config.training.train_split,
        split_seed=config.training.split_seed,
        shuffle_seed=config.training.shuffle_seed,
        num_workers=config.training.num_workers,
        pin_memory=config.training.pin_memory,
        persistent_workers=config.training.persistent_workers,
        prefetch_factor=config.data.prefetch_factor if config.training.num_workers > 0 else None,
        rank=rank,
        world_size=world_size,
    )

    if rank == 0:
        print(f"Executing training on {world_size} GPU(s); device={device}.", flush=True)

    in_mean, in_std = _estimate_input_normalizers(
        train_loader, include_nu=config.training.include_nu, n_distance_nodes=expected_n_distance_nodes,
    )
    if distributed:
        in_mean = in_mean.to(device)
        in_std = in_std.to(device)
        dist.broadcast(in_mean, src=0)
        dist.broadcast(in_std, src=0)

    model = build_training_model(
        config,
        device=device,
        n_distance_nodes=expected_n_distance_nodes,
    )
    model.to(device=device, dtype=torch.float32)
    model.set_normalizers(in_mean, in_std)

    checkpoint_payload = None
    if init_checkpoint_path is not None and init_checkpoint_path.exists():
        if rank == 0:
            print(f"Resuming/initializing from: {init_checkpoint_path}", flush=True)
        checkpoint_payload = torch.load(init_checkpoint_path, map_location=device, weights_only=False)
        state_dict = checkpoint_payload["model_state_dict"]
        if "k_scale" in state_dict and state_dict["k_scale"].shape == torch.Size([]):
            state_dict["k_scale"] = state_dict["k_scale"].unsqueeze(0)
        if preserve_estimated_input_normalizers:
            state_dict = {
                name: value
                for name, value in state_dict.items()
                if name not in {"in_mean", "in_std"}
            }
            model.load_state_dict(state_dict, strict=False)
        else:
            model.load_state_dict(state_dict)

    _drop_hardest_training_examples(
        model,
        train_loader,
        device,
        config.training.hard_example_drop_fraction,
        rank=rank,
        distributed=distributed,
    )

    compile_mode = os.environ.get("CUT_CELL_COMPILE_MODE", "none").lower()
    if compile_mode != "none":
        model = torch.compile(model, mode=compile_mode, dynamic=False)

    if distributed:
        model = DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)

    opt, scheduler, scheduler_is_onecycle = build_optimizer_and_scheduler(
        model,
        config.training,
        distributed=distributed,
        steps_per_epoch=len(train_loader),
    )

    scaler = torch.cuda.amp.GradScaler() if (device == "cuda" and config.training.use_amp) else None
    if checkpoint_payload is not None:
        optimizer_state = checkpoint_payload.get("optimizer_state_dict")
        scheduler_state = checkpoint_payload.get("scheduler_state_dict")
        scaler_state = checkpoint_payload.get("scaler_state_dict")
        if optimizer_state is not None:
            opt.load_state_dict(optimizer_state)
            if resume_learning_rate is None and scheduler is not None and scheduler_state is not None:
                scheduler.load_state_dict(scheduler_state)
            if scaler is not None and scaler_state is not None:
                scaler.load_state_dict(scaler_state)
            if resume_learning_rate is not None:
                if resume_learning_rate <= 0.0:
                    raise ValueError("resume_learning_rate must be positive when provided.")
                for parameter_group in opt.param_groups:
                    parameter_group["lr"] = resume_learning_rate
            if rank == 0:
                state_message = "Restored optimizer, scheduler, and AMP scaler state from checkpoint."
                if resume_learning_rate is not None:
                    state_message = (
                        f"Restored optimizer and AMP scaler state; started a new schedule at "
                        f"lr={resume_learning_rate:.3e}."
                    )
                print(state_message, flush=True)

    history = TrainingHistory()
    best_loss = float("inf")
    best_state = None
    patience_counter = 0
    prev_accepted_train_loss: float | None = None
    epoch_snapshot = _capture_epoch_state(model, opt, scheduler, scaler)
    retry_count = 0

    loss_component_names = ", ".join(c.name for c in loss_components)
    if rank == 0:
        print(f"Beginning training over {config.training.epochs} epochs (loss: {loss_component_names})", flush=True)

    started_at = time.perf_counter()
    epoch = 1
    while epoch <= config.training.epochs:
        t0 = time.perf_counter()
        if hasattr(train_loader, "set_epoch"):
            train_loader.set_epoch(epoch)

        train_metrics = _train_one_epoch(
            model,
            opt,
            scheduler,
            scaler,
            train_loader,
            device,
            loss_components,
            config.training.grad_clip_norm,
            scheduler_is_onecycle,
            config.training.lbfgs_reset_history_per_batch,
        )
        if distributed:
            train_metrics = _distributed_average_metrics(train_metrics, device)

        train_loss = train_metrics["total"]

        if distributed:
            _assert_distributed_model_parity(model, device)

        spiked = prev_accepted_train_loss is not None and train_loss > prev_accepted_train_loss * (1.0 + spike_threshold)

        if spiked and retry_count < max_spike_retries:
            pct = (train_loss / prev_accepted_train_loss - 1.0) * 100.0
            retry_count += 1
            next_lr = spike_lr_reduction_factor ** retry_count
            print(f"Epoch {epoch:4d}: loss spiked +{pct:.1f}% (train={train_loss:.6f} vs "
                  f"prev={prev_accepted_train_loss:.6f}) -- rolling back, retry {retry_count}/{max_spike_retries} "
                  f"with LR x{next_lr:.3g}", flush=True)
            _restore_epoch_state(model, opt, scheduler, scaler, epoch_snapshot)
            _reduce_learning_rate(opt, scheduler, spike_lr_reduction_factor ** retry_count)
            continue

        if spiked:
            print(f"Epoch {epoch:4d}: still spiking after {max_spike_retries} retries -- accepting anyway", flush=True)

        if scheduler is not None and not scheduler_is_onecycle:
            scheduler.step()

        val_loss, val_metrics = evaluate(model, val_loader, device, loss_components=loss_components)
        if distributed:
            val_metrics = _distributed_average_metrics(val_metrics, device)
            val_loss = val_metrics["total"]

        history.train_loss.append(train_loss)
        history.train_metrics.append(train_metrics)
        history.val_loss.append(val_loss)
        history.val_metrics.append(val_metrics)

        curr_lr = opt.param_groups[0]["lr"]
        epoch_seconds = time.perf_counter() - t0
        if rank == 0:
            print(
                f"Epoch {epoch:4d}/{config.training.epochs:4d} | "
                f"Train: {train_loss:.6f} | Val: {val_loss:.6f} | "
                f"Val rel_l2: {val_metrics.get('rel_l2', 0.0)*100:.2f}% | "
                f"Val ref_norm_p50: {val_metrics.get('ref_norm_p50', 0.0):.4g} | "
                f"Lr: {curr_lr:.6e} | Time: {epoch_seconds:.2f}s",
                flush=True,
            )

        max_epoch_seconds = config.training.max_epoch_seconds
        if max_epoch_seconds is not None and epoch_seconds > max_epoch_seconds:
            raise RuntimeError(
                f"Epoch {epoch} exceeded the {max_epoch_seconds:.1f}s budget: "
                f"{epoch_seconds:.2f}s."
            )

        if val_loss < best_loss - config.training.early_stopping_min_delta:
            best_loss = val_loss
            best_state = deepcopy(model.state_dict())
            patience_counter = 0
            if checkpoint_callback is not None and rank == 0:
                checkpoint_callback(model.module if distributed else model, history, epoch, val_loss, opt, scheduler, scaler)
        else:
            patience_counter += 1

        prev_accepted_train_loss = train_loss
        epoch_snapshot = _capture_epoch_state(model, opt, scheduler, scaler)
        retry_count = 0
        epoch += 1

        if patience_counter >= config.training.early_stopping_patience:
            print(f"Early stopping at epoch {epoch - 1} (patience={config.training.early_stopping_patience})", flush=True)
            break

        max_wall_seconds = config.training.max_wall_seconds
        if max_wall_seconds is not None and time.perf_counter() - started_at >= max_wall_seconds:
            print(
                f"Stopping after epoch {epoch - 1}: reached the {max_wall_seconds:.1f}s wall-clock budget.",
                flush=True,
            )
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    val_loss, val_metrics = evaluate(model, val_loader, device, loss_components=loss_components, return_K_phys_stats=True)
    if distributed:
        val_metrics = _distributed_average_metrics(val_metrics, device)
        val_loss = val_metrics["total"]
    diagnostics = model_structural_diagnostics(model, val_loader, device)

    print("\n--- Final Evaluation ---")
    print(f"  Val loss                 : {val_loss:.6e}")
    print(f"  Val rel_l2                : {val_metrics.get('rel_l2', 0.0)*100:.6f}%")
    print(f"  Symmetry error (K-K^T)   : {diagnostics['symmetry_error']:.3e}  (expect ~0, sanity check)")
    print(f"  Rigid-body residual (KR) : {diagnostics['rigid_body_residual']:.3e}  (expect ~0)")
    print(f"  Min physical eigenvalue  : {diagnostics['min_physical_eigenvalue']:.6e}  (should be > 0)")

    if distributed and dist.is_initialized():
        dist.destroy_process_group()

    return PipelineResult(
        model=model, history=history, group=requested_group,
        train_size=n_train, val_size=n_val, total_size=total_size, device=device,
        symmetry_error=diagnostics["symmetry_error"],
        rigid_body_residual=diagnostics["rigid_body_residual"],
        min_eigenvalue=diagnostics["min_eigenvalue"],
        min_physical_eigenvalue=diagnostics["min_physical_eigenvalue"],
        max_eigenvalue=diagnostics["max_eigenvalue"],
    )