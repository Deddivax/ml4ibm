"""
Training utilities for CutCellNet.

Loss design
-----------
A loss is a weighted sum of independent LossComponents. Each component is a
plain function of (K_phys, k_ref, ref_norm, ref_floor) -> scalar tensor:

    K_phys    : (B, 24, 24) predicted physical stiffness
    k_ref     : (B, 24, 24) ground-truth physical stiffness
    ref_norm  : (B,)        Frobenius norm of k_ref, per sample
    ref_floor : scalar      small quantile of ref_norm, used to avoid
                             dividing by ~0 for near-singular reference
                             matrices (e.g. tiny cut-fraction cells)

To add a new loss term, write a function with that signature and wrap it
in a LossComponent -- no changes needed anywhere else. See
`fourier_loss` below for a ready-to-use example of an optional extra
component.

`relative_frobenius_loss` is the only component included by default.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import torch
from torch.utils.data import DataLoader, Dataset

# ── Numerical safety constants ─────────────────────────────────────────────
LOSS_EPS = 1e-8
REL_FLOOR_QUANTILE = 0.05      # ref-norm quantile used as the division floor
LOG_NORM_CLAMP = 20.0          # exp() overflows well before this in fp16/AMP
K_UNIT_CLAMP = 1.0e4
NORM_CLAMP = 1.0e9


# ==========================================================
# Loss components
# ==========================================================

LossFn = Callable[[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]


@dataclass(slots=True)
class LossComponent:
    name: str
    fn: LossFn
    weight: float = 1.0


def relative_frobenius_loss(
    K_phys: torch.Tensor,
    k_ref: torch.Tensor,
    ref_norm: torch.Tensor,
    ref_floor: torch.Tensor,
) -> torch.Tensor:
    """Mean relative Frobenius reconstruction error: ||K_phys - k_ref||_F,
    normalized per-sample by ||k_ref||_F (floored at ref_floor so
    near-singular reference matrices, e.g. tiny cut-fraction cells,
    don't blow up the ratio)."""
    err = torch.linalg.matrix_norm(K_phys - k_ref, ord="fro", dim=(-2, -1))
    err = torch.clamp(err, min=0.0)
    return (err / torch.clamp(ref_norm, min=ref_floor)).mean()


def fourier_loss(
    K_phys: torch.Tensor,
    k_ref: torch.Tensor,
    ref_norm: torch.Tensor,
    ref_floor: torch.Tensor,
    order: int = 4,
) -> torch.Tensor:
    """Optional extra component: relative error restricted to the
    low-frequency part of the 2D Fourier transform of the matrix
    difference. Not included by default -- add via
    `make_fourier_loss_component()` if wanted."""
    diff = K_phys - k_ref
    H, W = diff.shape[-2], diff.shape[-1]

    diff_fft = torch.fft.fftshift(torch.fft.fft2(diff, norm="ortho"), dim=(-2, -1))
    ref_fft = torch.fft.fftshift(torch.fft.fft2(k_ref, norm="ortho"), dim=(-2, -1))

    ch, cw = H // 2, W // 2
    h0, h1 = max(0, ch - order), min(H, ch + order + 1)
    w0, w1 = max(0, cw - order), min(W, cw + order + 1)

    diff_low = torch.linalg.matrix_norm(diff_fft[..., h0:h1, w0:w1], ord="fro", dim=(-2, -1))
    ref_low = torch.linalg.matrix_norm(ref_fft[..., h0:h1, w0:w1], ord="fro", dim=(-2, -1))
    ref_low = torch.clamp(ref_low, min=1e-5)

    return (diff_low / ref_low).mean()


def make_fourier_loss_component(weight: float = 0.5, order: int = 4) -> LossComponent:
    return LossComponent("fourier", lambda Kp, kr, rn, rf: fourier_loss(Kp, kr, rn, rf, order=order), weight=weight)


DEFAULT_LOSS_COMPONENTS: list[LossComponent] = [LossComponent("relative_frobenius", relative_frobenius_loss, weight=1.0)]


# ==========================================================
# Composite loss
# ==========================================================

def composite_loss(
    model: torch.nn.Module,
    d: torch.Tensor,
    nu: torch.Tensor,
    k_ref: torch.Tensor,
    *,
    cut_class: torch.Tensor | None = None,
    loss_components: list[LossComponent] | None = None,
    return_K_phys: bool = False,
):
    """Forward pass + weighted sum of loss_components (defaults to just
    relative_frobenius_loss). Returns (total_loss, metrics) where
    metrics contains each component's un-weighted value plus "rel_l2"
    (mean relative Frobenius error, always reported regardless of which
    components are active -- useful as a comparable headline number
    across different loss configurations) and "total"."""
    loss_components = loss_components if loss_components is not None else DEFAULT_LOSS_COMPONENTS

    K_unit_pred, log_norm_pred = model(d, nu=nu, cut_class=cut_class)

    K_unit_pred = torch.nan_to_num(K_unit_pred.float(), nan=0.0, posinf=K_UNIT_CLAMP, neginf=-K_UNIT_CLAMP)
    K_unit_pred = torch.clamp(K_unit_pred, min=-K_UNIT_CLAMP, max=K_UNIT_CLAMP)
    log_norm_pred = torch.nan_to_num(log_norm_pred.float(), nan=0.0, posinf=LOG_NORM_CLAMP, neginf=-LOG_NORM_CLAMP)
    log_norm_pred = torch.clamp(log_norm_pred, min=-LOG_NORM_CLAMP, max=LOG_NORM_CLAMP)
    k_ref = torch.nan_to_num(k_ref.float(), nan=0.0, posinf=NORM_CLAMP, neginf=-NORM_CLAMP)
    k_ref = torch.clamp(k_ref, min=-NORM_CLAMP, max=NORM_CLAMP)

    K_phys = K_unit_pred * torch.exp(log_norm_pred)[:, None, None]

    ref_norm = torch.linalg.matrix_norm(k_ref, ord="fro", dim=(-2, -1))
    ref_norm = torch.clamp(torch.nan_to_num(ref_norm, nan=LOSS_EPS, posinf=NORM_CLAMP), min=LOSS_EPS)
    ref_floor = torch.quantile(ref_norm.detach(), REL_FLOOR_QUANTILE).clamp(min=1e-4)

    total = k_ref.new_zeros(())
    metrics: dict[str, torch.Tensor] = {}
    for component in loss_components:
        value = torch.nan_to_num(component.fn(K_phys, k_ref, ref_norm, ref_floor), nan=1e6, posinf=1e6)
        metrics[component.name] = value.detach()
        total = total + component.weight * value

    err_fro = torch.linalg.matrix_norm(K_phys - k_ref, ord="fro", dim=(-2, -1))
    metrics["rel_l2"] = (err_fro / torch.clamp(ref_norm, min=ref_floor)).mean().detach()
    metrics["ref_norm_p50"] = torch.quantile(ref_norm.detach(), 0.5)
    total = torch.nan_to_num(total, nan=1e6, posinf=1e6)
    metrics["total"] = total.detach()

    if return_K_phys:
        return total, metrics, K_phys
    return total, metrics


# ==========================================================
# Training history
# ==========================================================

@dataclass(slots=True)
class TrainingHistory:
    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    train_metrics: list[dict[str, float]] = field(default_factory=list)
    val_metrics: list[dict[str, float]] = field(default_factory=list)


# ==========================================================
# Evaluation
# ==========================================================

@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    *,
    loss_components: list[LossComponent] | None = None,
    return_K_phys_stats: bool = False,
) -> tuple[float, dict[str, float]]:
    """Sample-weighted average of composite_loss's metrics over the whole
    loader (weighted by batch size, not by number of batches -- matters
    when the last batch is a different size from the rest)."""
    model.eval()
    dev = torch.device(device)

    totals: dict[str, float] = {}
    n_samples = 0
    n_skipped_batches = 0
    eig_min_phys: float | None = None

    for batch in loader:
        if batch is None:
            continue
        d, nu, k_ref, cls = batch
        d = torch.nan_to_num(d.to(dev), nan=0.0, posinf=1e6, neginf=-1e6)
        nu = torch.nan_to_num(nu.to(dev), nan=0.3, posinf=1.0, neginf=-1.0)
        k_ref = torch.nan_to_num(k_ref.to(dev), nan=0.0, posinf=1e9, neginf=-1e9)
        cls = cls.to(dev).long()
        if cls.numel() == 0:
            continue

        loss, metrics, K_phys = composite_loss(
            model, d, nu, k_ref, cut_class=cls, loss_components=loss_components, return_K_phys=True,
        )
        if not torch.isfinite(loss):
            n_skipped_batches += 1
            continue

        bsz = d.shape[0]
        for k, v in metrics.items():
            totals[k] = totals.get(k, 0.0) + float(v.item()) * bsz
        n_samples += bsz

        if return_K_phys_stats:
            evals = torch.linalg.eigvalsh(K_phys.double())  # (B, 24), ascending
            batch_min_phys = evals[:, 6].min().item()  # 7th smallest: first non-rigid-body mode
            eig_min_phys = batch_min_phys if eig_min_phys is None else min(eig_min_phys, batch_min_phys)

    if n_samples == 0:
        return float("nan"), {"n_skipped_batches": float(n_skipped_batches)}

    avg_metrics = {k: v / n_samples for k, v in totals.items()}
    avg_metrics["n_samples"] = float(n_samples)
    avg_metrics["n_skipped_batches"] = float(n_skipped_batches)
    if return_K_phys_stats and eig_min_phys is not None:
        avg_metrics["min_physical_eigenvalue"] = eig_min_phys

    return avg_metrics["total"], avg_metrics


# ==========================================================
# Data loading
# ==========================================================

class FastTensorDataLoader:
    """Vectorized in-memory loader.

    The complete dataset intentionally remains in host memory. Keeping a
    large stiffness corpus on a small GPU leaves too little workspace for
    the matrix multiplies in the model and severely reduces throughput.
    """

    def __init__(
        self,
        d: torch.Tensor,
        nu: torch.Tensor,
        k_mat: torch.Tensor,
        classes: torch.Tensor,
        batch_size: int,
        shuffle: bool,
        generator: torch.Generator | None = None,
        device: torch.device | None = None,
        pin_memory: bool = True,
        rank: int = 0,
        world_size: int = 1,
    ):
        self.device = device if device is not None else torch.device("cpu")
        self.d = d.cpu().float().contiguous()
        self.nu = nu.cpu().float().contiguous()
        self.k_mat = k_mat.cpu().float().contiguous()
        self.classes = classes.cpu().long().contiguous()
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.generator = generator
        self.pin_memory = bool(pin_memory and torch.cuda.is_available())
        self.num_samples = d.shape[0]
        self.rank = int(rank)
        self.world_size = int(world_size)
        if not 0 <= self.rank < self.world_size:
            raise ValueError(f"rank must be in [0, {self.world_size}); got {self.rank}.")
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        samples_per_rank = -(-self.num_samples // self.world_size)
        return -(-samples_per_rank // self.batch_size)  # ceil div

    def __iter__(self):
        if self.shuffle:
            generator = torch.Generator()
            generator.manual_seed(self.epoch if self.generator is None else self.generator.initial_seed() + self.epoch)
            idx = torch.randperm(self.num_samples, generator=generator)
        else:
            idx = torch.arange(self.num_samples)

        samples_per_rank = -(-self.num_samples // self.world_size)
        total_samples = samples_per_rank * self.world_size
        if total_samples > self.num_samples:
            idx = torch.cat((idx, idx[:total_samples - self.num_samples]))
        idx = idx[self.rank:total_samples:self.world_size]

        for start in range(0, len(idx), self.batch_size):
            batch = idx[start:start+self.batch_size]
            tensors = (
                self.d[batch],
                self.nu[batch],
                self.k_mat[batch],
                self.classes[batch],
            )
            if self.pin_memory:
                tensors = tuple(tensor.pin_memory() for tensor in tensors)
            yield tensors


def make_loaders(
    dataset: Dataset,
    batch_size: int,
    seed: int,
    train_split: float,
    split_seed: int | None = None,
    shuffle_seed: int | None = None,
    num_workers: int = 0,
    pin_memory: bool = True,
    persistent_workers: bool = True,
    prefetch_factor: int = 2,
    collate_fn: Callable | None = None,
    rank: int = 0,
    world_size: int = 1,
) -> tuple[Any, Any, int, int]:
    n_total = len(dataset)
    if n_total < 2:
        raise RuntimeError(f"Not enough samples to split train/val: n_total={n_total}")

    n_train = max(1, int(train_split * n_total))
    n_val = n_total - n_train
    if n_val == 0:
        n_train -= 1
        n_val = 1

    split_rng = torch.Generator()
    split_rng.manual_seed(int(seed if split_seed is None else split_seed))
    perm = torch.randperm(n_total, generator=split_rng)
    train_idx, val_idx = perm[:n_train], perm[n_train:]

    shuffle_rng = torch.Generator()
    shuffle_rng.manual_seed(int(seed if shuffle_seed is None else shuffle_seed))

    from clean_simulation.internal_nn_runtime.cutcell_ml.data.datasets import CutCellDataset
    if isinstance(dataset, CutCellDataset):
        train_loader = FastTensorDataLoader(
            dataset.d[train_idx], dataset.nu[train_idx], dataset.k_mat[train_idx], dataset.classes[train_idx],
            batch_size=batch_size, shuffle=True, generator=shuffle_rng, pin_memory=pin_memory, rank=rank, world_size=world_size,
        )
        val_loader = FastTensorDataLoader(
            dataset.d[val_idx], dataset.nu[val_idx], dataset.k_mat[val_idx], dataset.classes[val_idx],
            batch_size=batch_size, shuffle=False, pin_memory=pin_memory,
            rank=rank, world_size=world_size,
        )
        return train_loader, val_loader, n_train, n_val

    # Fallback for H5-streamed datasets that can't be materialized as
    # in-memory tensors up front.
    train_set = torch.utils.data.Subset(dataset, train_idx.tolist())
    val_set = torch.utils.data.Subset(dataset, val_idx.tolist())

    loader_kwargs: dict[str, object] = {"num_workers": max(int(num_workers), 0), "pin_memory": bool(pin_memory)}
    if int(num_workers) > 0:
        loader_kwargs["persistent_workers"] = bool(persistent_workers)
        loader_kwargs["prefetch_factor"] = max(int(prefetch_factor), 1)

    loader_rng = torch.Generator()
    loader_rng.manual_seed(int(seed if shuffle_seed is None else shuffle_seed))

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, generator=loader_rng,
                               collate_fn=collate_fn, **loader_kwargs)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, **loader_kwargs)
    return train_loader, val_loader, n_train, n_val