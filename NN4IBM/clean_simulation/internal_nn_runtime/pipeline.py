from dataclasses import dataclass
from pathlib import Path
import time
from copy import deepcopy
from typing import Callable
import h5py

import torch

from clean_simulation.internal_nn_runtime.cutcell_ml.model import (
    CutCellNet,
    CutCellNetGNN,
    CutCellNetTransformer,
    build_rigid_body_modes,
    HEX_CORNER_COORDS,
)

from .config import RunConfig
from .data import (
    CutCellDataset,
    H5CutCellDataset,
    load_dataset_from_h5,
)
from .training import (
    composite_loss,
    DEFAULT_LOSS_COMPONENTS,
    evaluate,
    TrainingHistory,
    make_loaders,
)

@torch.no_grad()
def _estimate_input_normalizers(loader, *, include_nu, n_distance_nodes=8, max_batches=128):
    sum_x, sum_x2, count = None, None, 0

    for i, (d, nu, _, _) in enumerate(loader):
        d = d.float()
        x = torch.cat([d, nu.unsqueeze(-1)], dim=-1) if include_nu else d

        bsum = x.sum(0)
        bsum2 = (x * x).sum(0)

        sum_x = bsum if sum_x is None else sum_x + bsum
        sum_x2 = bsum2 if sum_x2 is None else sum_x2 + bsum2
        count += x.shape[0]

        if i + 1 >= max_batches:
            break

    if count == 0:
        dim = n_distance_nodes + (1 if include_nu else 0)
        return torch.zeros(dim), torch.ones(dim)

    mean = sum_x / count
    var = sum_x2 / count - mean**2
    std = torch.sqrt(torch.clamp(var, min=1e-8)).clamp(min=1e-4)

    return mean, std


@torch.no_grad()
def _compute_output_diagnostics(
    model: CutCellNet,
    loader,
    device: str,
    *,
    max_batches: int = 16,
) -> dict[str, float]:
    """
    Real, meaningful diagnostics on the model's predicted K, now that
    symmetry and rigid-body nullity are guaranteed by construction rather
    than learned:

      - symmetry_error   : max ||K - K^T|| over sampled batches
                           (sanity check only -- should be ~0 to float
                           precision; a nonzero value would indicate a
                           bug in the projection, not a training issue)
      - rigid_body_residual : max ||K @ R|| over sampled batches
                           (should be ~0 to float precision; nonzero
                           means HEX_CORNER_COORDS doesn't match your
                           FEM dof ordering -- see check_rigid_body_nullity)
      - min_eigenvalue   : smallest eigenvalue overall (should be ~0,
                           corresponding to the 6 constructed rigid-body
                           modes -- sanity check)
      - min_physical_eigenvalue : smallest eigenvalue EXCLUDING the 6
                           known-null ones (7th smallest) -- this is the
                           one that actually reflects how well the model
                           has learned the physical stiffness; should be
                           > 0 and not collapsing toward 0 for well-posed
                           (non-degenerate) trimmed cells
      - max_eigenvalue   : largest eigenvalue overall
    """
    model.eval()
    R = build_rigid_body_modes(HEX_CORNER_COORDS).to(device)

    sym_err = 0.0
    rb_resid = 0.0
    min_ev = float("inf")
    min_phys_ev = float("inf")
    max_ev = float("-inf")

    for i, (d, nu, k_ref, cls) in enumerate(loader):
        d = d.to(device)
        nu = nu.to(device)
        cls = cls.to(device).long()

        K_unit, log_norm = model(d, nu=nu, cut_class=cls)
        K = K_unit * torch.exp(log_norm)[:, None, None]

        sym_err = max(sym_err, (K - K.transpose(-1, -2)).abs().amax().item())
        rb_resid = max(rb_resid, (K @ R).abs().amax().item())

        evals = torch.linalg.eigvalsh(K.double())  # (B, 24), ascending
        min_ev = min(min_ev, evals[:, 0].min().item())
        min_phys_ev = min(min_phys_ev, evals[:, 6].min().item())
        max_ev = max(max_ev, evals[:, -1].max().item())

        if i + 1 >= max_batches:
            break

    return {
        "symmetry_error": sym_err,
        "rigid_body_residual": rb_resid,
        "min_eigenvalue": 0.0 if min_ev == float("inf") else min_ev,
        "min_physical_eigenvalue": 0.0 if min_phys_ev == float("inf") else min_phys_ev,
        "max_eigenvalue": 0.0 if max_ev == float("-inf") else max_ev,
    }


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


def run_training_pipeline(
    config: RunConfig,
    base_dir: Path | None = None,
    init_checkpoint_path: Path | None = None,
    checkpoint_callback: Callable[[torch.nn.Module, TrainingHistory, int, float], None] | None = None,
) -> PipelineResult:

    base_dir = Path.cwd() if base_dir is None else base_dir

    config.resolve_paths(base_dir)
    torch.manual_seed(config.training.seed)

    # Ensure config.data.h5_path is treated as a list internally for uniform multi-file support
    if isinstance(config.data.h5_path, (list, tuple)):
        h5_paths = [Path(p) for p in config.data.h5_path]
    else:
        h5_paths = [Path(config.data.h5_path)]

    grp = f"integration_{config.data.integration_order}_poly_{config.data.poly_order}"

    all_d, all_nu, all_k = [], [], []
    expected_n_distance_nodes = (config.data.refined_subcells + 1) ** 3
    if expected_n_distance_nodes > 8 and config.training.model_type != "cutcellnet":
        raise ValueError("Refined distance inputs are currently supported by model_type='cutcellnet' only.")
    for h5_p in h5_paths:
        if not h5_p.exists():
            print(f"Warning: Specified training h5 file does not exist: {h5_p}", flush=True)
            continue
        # Resolve group name for this file
        with h5py.File(h5_p, "r") as h5_f:
            file_grp = f"integration_{config.data.integration_order}_poly_{config.data.poly_order}"
            if file_grp not in h5_f:
                # Fallback to legacy group template if not found
                grp_legacy = f"depth_{config.data.depth}_integration_{config.data.integration_order}_poly_{config.data.poly_order}"
                if grp_legacy in h5_f:
                    file_grp = grp_legacy
                else:
                    # Let's check with regex fallback
                    import re

                    found = False
                    for gk in h5_f.keys():
                        trimmed_gk = re.sub(r"^depth_\d+_", "", gk)
                        if trimmed_gk == f"integration_{config.data.integration_order}_poly_{config.data.poly_order}":
                            file_grp = gk
                            found = True
                            break
                    if not found:

                        print(f"Warning: No valid group f'integration_{config.data.integration_order}_poly_{config.data.poly_order}' found in {h5_p}", flush=True)
                        continue

            d_file, nu_file, k_file = load_dataset_from_h5(
                h5_p,
                file_grp,
                config.data.poisson_ratio,
                max_samples=config.data.max_load_samples,
            )
            if d_file.shape[1] != expected_n_distance_nodes:
                raise ValueError(
                    f"{h5_p}/{file_grp} has {d_file.shape[1]} distance inputs, but "
                    f"refined_subcells={config.data.refined_subcells} expects "
                    f"{expected_n_distance_nodes}."
                )
            all_d.append(d_file)
            all_nu.append(nu_file)
            all_k.append(k_file)

    if not all_d:
        raise RuntimeError(f"No valid data loaded from any of the specified H5 files: {h5_paths}")

    combined_d = torch.cat(all_d)
    combined_nu = torch.cat(all_nu)
    combined_k = torch.cat(all_k)

    dataset = CutCellDataset(combined_d, combined_nu, combined_k)
    collate_fn = None

    total_size = len(dataset)
    train_size = int(config.training.train_split * total_size)
    val_size = total_size - train_size

    if train_size == 0 or val_size == 0:
        raise ValueError(
            f"Dataset too small! Total samples: {total_size}. Check your configuration. "
            "Ensure data has been generated for this geometry first."
        )

    generator = torch.Generator().manual_seed(config.training.seed)
    train_ds, val_ds = torch.utils.data.random_split(
        dataset, [train_size, val_size], generator=generator
    )

    train_loader, val_loader, n_train, n_val = make_loaders(
        dataset=dataset,
        batch_size=config.training.batch_size,
        seed=config.training.seed,
        train_split=config.training.train_split,
        num_workers=config.training.num_workers,
        pin_memory=config.training.pin_memory,
        persistent_workers=config.training.persistent_workers,
        prefetch_factor=config.data.prefetch_factor if config.training.num_workers > 0 else None,
        collate_fn=collate_fn,
    )

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )
    print(f"Executing training on device: {device}", flush=True)

    # Normalizer calculation
    in_mean, in_std = _estimate_input_normalizers(
        train_loader,
        include_nu=config.training.include_nu,
        n_distance_nodes=expected_n_distance_nodes,
    )

    if config.training.model_type == "cutcellnet":

        model = CutCellNet(
            fusion_dim=config.training.fusion_dim,
            n_residual_blocks=config.training.n_residual_blocks,
            include_nu=config.training.include_nu,
            trunk_type=config.training.trunk_type,
            use_class_conditioning=config.training.use_class_conditioning,
            include_class_inputs=config.training.include_class_inputs,
            class_embed_dim=config.training.class_embed_dim,
            class_encoder_type=config.training.class_encoder_type,
            residual_scaling=config.training.residual_scaling,
            spd_mode=config.training.spd_mode,
            dropout_p=config.training.dropout_p,
            use_reduced_null_space=config.training.use_reduced_null_space,
            matrix_size=config.training.matrix_size,
            n_distance_nodes=expected_n_distance_nodes,
        )

    elif config.training.model_type == "gnn":

        model = CutCellNetGNN(
            hidden_dim=64,
            latent_dim=128,
            n_layers=4,
            matrix_size=config.training.matrix_size,
        )

    elif config.training.model_type == "transformer":

        model = CutCellNetTransformer(
            embed_dim=getattr(config.training, "fusion_dim", 128),
            n_heads=getattr(config.training, "n_heads", 4),
            n_layers=getattr(config.training, "n_residual_blocks", 5),
            ffn_dim=getattr(config.training, "fusion_dim", 128) * 2,
            dropout=getattr(config.training, "dropout_p", 0.0),
        )

    else:
        raise ValueError(config.training.model_type)

    model.to(device=device, dtype=torch.float32)
    # Use the model's own setter (clamps std, keeps buffer dtype/device
    # consistent) rather than calling register_buffer directly again.
    model.set_normalizers(in_mean, in_std)

    if init_checkpoint_path is not None and init_checkpoint_path.exists():
        print(f"Resuming/Initializing with weights from: {init_checkpoint_path}", flush=True)
        ckpt = torch.load(init_checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])

    # Register target stiffness matrix scale
    avg_fro = 0.0
    k_batches = 0
    for _, _, k_mat, _ in train_loader:
        avg_fro += float(torch.linalg.matrix_norm(k_mat, ord="fro").mean().item())
        k_batches += 1
        if k_batches >= 32:
            break
    k_scale = max(avg_fro / max(k_batches, 1), 1e-6)
    model.k_scale.fill_(k_scale)

    # ------------------------------------------------------
    # Optimization
    # ------------------------------------------------------
    optimizer_type = config.training.optimizer_type.lower()
    if optimizer_type == "adamw":
        opt = torch.optim.AdamW(
            model.parameters(),
            lr=config.training.lr,
            weight_decay=config.training.weight_decay,
        )
    elif optimizer_type == "lbfgs":
        if config.training.use_distributed:
            raise ValueError("L-BFGS is supported only for single-process late-stage refinement, not DDP.")
        if config.training.use_amp:
            raise ValueError("L-BFGS requires use_amp=False so each closure has deterministic full precision.")
        if config.training.weight_decay != 0.0:
            raise ValueError("L-BFGS has no decoupled weight decay; set weight_decay=0.0 for refinement.")
        opt = torch.optim.LBFGS(
            model.parameters(),
            lr=config.training.lr,
            max_iter=config.training.lbfgs_max_iter,
            history_size=config.training.lbfgs_history_size,
            line_search_fn=config.training.lbfgs_line_search_fn,
        )
    else:
        raise ValueError(f"Unsupported optimizer_type={config.training.optimizer_type!r}; expected 'adamw' or 'lbfgs'.")

    steps_per_epoch = len(train_loader)
    scheduler_is_onecycle = config.training.scheduler_type == "onecycle"
    if optimizer_type == "lbfgs" and scheduler_is_onecycle:
        raise ValueError("L-BFGS does not support the per-batch OneCycle scheduler; use cosine or none.")

    if scheduler_is_onecycle:
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            opt,
            max_lr=config.training.lr * config.training.onecycle_max_lr_factor,
            epochs=config.training.epochs,
            steps_per_epoch=steps_per_epoch,
            pct_start=config.training.onecycle_pct_start,
            final_div_factor=100.0,
        )
    elif config.training.scheduler_type == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt,
            T_max=config.training.epochs,
            eta_min=config.training.lr * config.training.cosine_eta_min_factor,
        )
    else:
        scheduler = None

    scaler = (
        torch.cuda.amp.GradScaler()
        if (device == "cuda" and config.training.use_amp)
        else None
    )

    history = TrainingHistory()
    best_loss = float("inf")
    best_state = None
    patience_counter = 0

    # NOTE on nullspace_loss_weight: with the hard-constrained head in
    # CutCellNet (K = T @ K_reduced @ T.T), K @ R == 0 and K == K.T hold
    # exactly for every sample regardless of this weight -- it's now inert
    # and only kept here so existing configs don't break. Consider it a
    # no-op; real verification is via _compute_output_diagnostics below,
    # not via a loss term.
    print(f"Beginning training over {config.training.epochs} epochs", flush=True)
    for epoch in range(1, config.training.epochs + 1):
        t0 = time.perf_counter()
        model.train()

        tot_loss = 0.0
        tot_rel_l2 = 0.0
        n_batches = 0

        for d, nu, k_mat, cl in train_loader:
            d = d.to(device=device, non_blocking=True)
            nu = nu.to(device=device, non_blocking=True)
            k_mat = k_mat.to(device=device, non_blocking=True)
            cl = cl.to(device=device, non_blocking=True)

            if isinstance(opt, torch.optim.LBFGS):
                closure_metrics: dict[str, torch.Tensor] = {}

                def closure() -> torch.Tensor:
                    opt.zero_grad(set_to_none=True)
                    closure_loss, closure_metric_values = composite_loss(
                        model,
                        d,
                        nu,
                        k_mat,
                        cut_class=cl,
                        loss_components=DEFAULT_LOSS_COMPONENTS,
                    )
                    if not torch.isfinite(closure_loss):
                        raise FloatingPointError("L-BFGS closure produced a non-finite loss")
                    closure_loss.backward()
                    if config.training.grad_clip_norm > 0:
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), config.training.grad_clip_norm
                        )
                    closure_metrics.clear()
                    closure_metrics.update({name: value.detach() for name, value in closure_metric_values.items()})
                    return closure_loss

                loss = opt.step(closure)
                metrics = closure_metrics
            elif scaler is not None:
                opt.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast():
                    loss, metrics = composite_loss(
                        model,
                        d,
                        nu,
                        k_mat,
                        cut_class=cl,
                        loss_components=DEFAULT_LOSS_COMPONENTS,
                    )
                scaler.scale(loss).backward()
                if config.training.grad_clip_norm > 0:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), config.training.grad_clip_norm
                    )
                scaler.step(opt)
                scaler.update()
            else:
                opt.zero_grad(set_to_none=True)
                loss, metrics = composite_loss(
                    model,
                    d,
                    nu,
                    k_mat,
                    cut_class=cl,
                    loss_components=DEFAULT_LOSS_COMPONENTS
                )
                loss.backward()
                if config.training.grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), config.training.grad_clip_norm
                    )
                opt.step()

            # OneCycleLR must step once per optimizer step (not per epoch) --
            # previously this scheduler never stepped at all.
            if scheduler_is_onecycle:
                scheduler.step()

            tot_loss += float(loss.item())
            batch_rel_l2 = float(metrics.get("rel_l2", 0.0))
            tot_rel_l2 += batch_rel_l2
            n_batches += 1

        if scheduler is not None and not scheduler_is_onecycle:
            scheduler.step()

        val_loss, val_metrics = evaluate(
            model=model,
            loader=val_loader,
            device=device,
            physical_loss_weight=config.training.physical_loss_weight,
            classifier_weight=config.training.classifier_weight,
            energy_loss_weight=config.training.energy_loss_weight,
            nullspace_loss_weight=config.training.nullspace_loss_weight,
            energy_probe_count=config.training.energy_probe_count,
            fourier_loss_weight=config.training.fourier_loss_weight,
            fourier_loss_order=config.training.fourier_loss_order,
            use_decoupled_loss=config.training.use_decoupled_loss,
            return_metrics=True,
        )

        history.train_loss.append(tot_loss / max(n_batches, 1))
        history.val_loss.append(val_loss)

        history.train_metrics.append({
            "rel_l2": avg_train_rel_l2,
        })

        history.val_metrics.append(val_metrics)

        epoch_time = time.perf_counter() - t0

        # Print every single epoch
        curr_lr = opt.param_groups[0]["lr"]
        avg_train_rel_l2 = tot_rel_l2 / max(n_batches, 1)
        val_rel_l2 = val_metrics.get("rel_l2", 0.0)
        print(
            f"Epoch {epoch:4d}/{config.training.epochs:4d} | "
            f"Train Loss: {history.train_loss[-1]:.6f} | "
            f"Train Rel L2: {avg_train_rel_l2 * 100:.2f}% | "
            f"Val Loss: {val_loss:.6f} | "
            f"Val Rel L2: {val_metrics['rel_l2'] * 100:.2f}% | "
            f"Lr: {curr_lr:.2e} | "
            f"Time: {epoch_time:.2f}s",
            flush=True,
        )

        # Early Stopping
        if val_loss < best_loss - config.training.early_stopping_min_delta:
            best_loss = val_loss
            best_state = deepcopy(model.state_dict())
            patience_counter = 0
            if checkpoint_callback is not None:
                checkpoint_callback(model, history, epoch, val_loss)
        else:
            patience_counter += 1

        if patience_counter >= config.training.early_stopping_patience:
            print(f"Early stopping triggered at epoch {epoch} (patience={config.training.early_stopping_patience})", flush=True)
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    # Validate output properties
    model.eval()
    val_loss, val_metrics = evaluate(
        model=model,
        loader=val_loader,
        device=device,
    )

    diagnostics = _compute_output_diagnostics(model, val_loader, device)
    print(
        f"Epoch {epoch:4d}/{config.training.epochs:4d} | "
        f"Train Loss: {history.train_loss[-1]:.6f} | "
        f"Train Rel L2: {avg_train_rel_l2 * 100:.2f}% | "
        f"Val Loss: {val_loss:.6f} | "
        f"Val Rel L2: {val_metrics['rel_l2'] * 100:.2f}% | "
        f"Lr: {curr_lr:.2e} | "
        f"Time: {epoch_time:.2f}s",
        flush=True,
    )

    return PipelineResult(
        model=model,
        history=history,
        group=grp,
        train_size=train_size,
        val_size=val_size,
        total_size=total_size,
        device=device,
        symmetry_error=diagnostics["symmetry_error"],
        rigid_body_residual=diagnostics["rigid_body_residual"],
        min_eigenvalue=diagnostics["min_eigenvalue"],
        min_physical_eigenvalue=diagnostics["min_physical_eigenvalue"],
        max_eigenvalue=diagnostics["max_eigenvalue"],
    )