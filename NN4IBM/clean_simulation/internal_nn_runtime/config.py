from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True)
class DataConfig:
    h5_path: Path = Path("outputs/data/local_stiffness_matrices_3k_stats.h5")
    cut_mask_csv: Path = Path("data_ibm/cut_mask_r6.csv")

    depth: int = 4
    integration_order: int = 2
    poly_order: int = 1

    poisson_ratio: float = 0.3

    min_samples: int = 3000
    value_resolution: int = 6

    sampling_strategy: str = "class_first_hybrid"
    sampling_seed: int = 20260608
    replace_existing: bool = False

    # None = streaming HDF5
    max_load_samples: int | None = None

    # Number of regular subcells per axis used for distance inputs. A value of
    # one preserves the legacy eight corner distances.
    refined_subcells: int = 3
    prefetch_factor: int = 11

@dataclass(slots=True)
class TrainingConfig:

    # ──────────────────────────────────────────
    # Model
    # ──────────────────────────────────────────

    topology_dim: int = 64
    geometry_dim: int = 128
    fusion_dim: int = 256
    n_residual_blocks: int = 4
    class_embed_dim: int = 32
    class_encoder_type: str = "one_hot"
    dropout_p: float = 0.0
    include_nu: bool = False
    model_type: str = "cutcell_net"
    include_class_inputs: bool = False
    trunk_type: str = "resnet"
    residual_scaling: str = "rezero"
    spd_mode: str = "cholesky"   # "cholesky" (K=U^T@U, softplus diagonal) or "triangular" (symmetric completion)
    use_reduced_null_space: bool = True
    matrix_size: int = 24

    # ──────────────────────────────────────────
    # Optimization
    # ──────────────────────────────────────────
    batch_size: int = 512

    epochs: int = 1000

    lr: float = 1e-4

    weight_decay: float = 1e-5

    # Optimizer: "adamw" for regular training or "lbfgs" for a late,
    # single-process late-stage refinement.
    optimizer_type: str = "adamw"
    lbfgs_max_iter: int = 4
    lbfgs_history_size: int = 20
    lbfgs_line_search_fn: str | None = "strong_wolfe"

    grad_clip_norm: float = 5.0

    use_amp: bool = True

    # Scheduler: "onecycle", "cosine", or "none" (constant learning rate)
    scheduler_type: str = "cosine"

    # OneCycleLR / CosineAnnealingLR limits
    onecycle_max_lr_factor: float = 3.0
    onecycle_pct_start: float = 0.10
    cosine_eta_min_factor: float = 0.01  # Value to determine what the lowest learning rate settles to (determines eta_min = lr * cosine_eta_min_factor)

    # ──────────────────────────────────────────
    # Data augmentation
    # ──────────────────────────────────────────

    input_noise_std: float = 0.0

    standardize_inputs: bool = True

    # ──────────────────────────────────────────
    # Train/validation
    # ──────────────────────────────────────────

    train_split: float = 0.8

    seed: int = 0

    split_seed: int | None = None
    shuffle_seed: int | None = None

    # ──────────────────────────────────────────
    # Loss weights
    # ──────────────────────────────────────────
    auto_balance_scale_loss: bool = True
    use_decoupled_loss: bool = True

    physical_loss_weight: float = 1.0
    classifier_weight: float = 1.0
    energy_loss_weight: float = 0.0
    nullspace_loss_weight: float = 0.0
    soft_mode_loss_weight: float = 0.0
    energy_probe_count: int = 8
    fourier_loss_weight: float = 1.0
    orbit_loss_weight: float = 0.0
    fourier_loss_order: int = 20
    use_symmetry_canonicalization: bool = False
    use_class_conditioning: bool = False



    # ──────────────────────────────────────────
    # Curriculum
    # ──────────────────────────────────────────

    warmup_phys_only_epochs: int = 100

    aux_ramp_epochs: int = 200

    # ──────────────────────────────────────────
    # Early stopping
    # ──────────────────────────────────────────

    early_stopping_patience: int = 100

    early_stopping_min_delta: float = 1e-5

    max_epoch_seconds: float | None = None

    max_wall_seconds: float | None = None

    use_distributed: bool = False

    # ──────────────────────────────────────────
    # DataLoader
    # ──────────────────────────────────────────

    num_workers: int = 0

    pin_memory: bool = True

    persistent_workers: bool = False

    prefetch_factor: int = 2


@dataclass(slots=True)
class RunConfig:

    data: DataConfig = field(default_factory=DataConfig)

    training: TrainingConfig = field(default_factory=TrainingConfig)

    def resolve_paths(self, base_dir: Path) -> "RunConfig":

        if isinstance(self.data.h5_path, (list, tuple)):
            resolved = []
            for p in self.data.h5_path:
                p_path = Path(p)
                if not p_path.is_absolute():
                    p_path = (base_dir / p_path).resolve()
                resolved.append(p_path)
            self.data.h5_path = resolved
        else:
            self.data.h5_path = Path(self.data.h5_path)
            if not self.data.h5_path.is_absolute():
                self.data.h5_path = (
                    base_dir / self.data.h5_path
                ).resolve()

        if not self.data.cut_mask_csv.is_absolute():
            self.data.cut_mask_csv = (
                base_dir / self.data.cut_mask_csv
            ).resolve()

        return self
