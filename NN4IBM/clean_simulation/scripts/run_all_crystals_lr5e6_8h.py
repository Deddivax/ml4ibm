from __future__ import annotations

from datetime import datetime
import os
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clean_simulation.internal_nn_runtime.cutcell_ml.config import DataConfig, RunConfig, TrainingConfig
from clean_simulation.internal_nn_runtime.workflows import run_training_pipeline


EPOCHS = int(os.environ.get("CUT_CELL_EPOCHS", "26"))
BATCH_SIZE = int(os.environ.get("CUT_CELL_BATCH_SIZE", "1024"))
LR = float(os.environ.get("CUT_CELL_LR", "5.0e-6"))
MAX_EPOCH_SECONDS = float(os.environ.get("CUT_CELL_MAX_EPOCH_SECONDS", "0")) or None
CHECKPOINT_DIR = ROOT / "clean_simulation" / "outputs" / "checkpoints"
DATA_PATH = ROOT / "clean_simulation" / "outputs" / "data" / (
    "grid_locked_source_sdf_rebuilt_queso_i2_p1_simple_harsh_crystals30_filtered.h5"
)
INITIAL_CHECKPOINT = Path(
    os.environ.get(
        "CUT_CELL_INIT_CHECKPOINT",
        CHECKPOINT_DIR / "source_sdf_rebuilt_queso_d10_f1024_all_crystals_lr5e6_calibration.pt",
    )
)
OUTPUT_CHECKPOINT = CHECKPOINT_DIR / "source_sdf_rebuilt_queso_d10_f1024_all_crystals_lr5e6_8h.pt"


def checkpoint_saver(model, history, epoch, val_loss) -> None:
    if OUTPUT_CHECKPOINT.is_file():
        existing = torch.load(OUTPUT_CHECKPOINT, map_location="cpu", weights_only=False)
        existing_loss = float(existing.get("val_loss", float("inf")))
        if val_loss >= existing_loss:
            return
    torch.save(
        {
            "epoch": epoch,
            "val_loss": val_loss,
            "model_state_dict": model.state_dict(),
            "history": history,
        },
        OUTPUT_CHECKPOINT,
    )


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this continuation.")
    if not DATA_PATH.is_file():
        raise FileNotFoundError(f"Missing expanded training data: {DATA_PATH}")
    if not INITIAL_CHECKPOINT.is_file():
        raise FileNotFoundError(f"Missing restart checkpoint: {INITIAL_CHECKPOINT}")

    config = RunConfig(
        data=DataConfig(
            h5_path=DATA_PATH,
            depth=2,
            integration_order=2,
            poly_order=1,
            poisson_ratio=0.3,
            min_samples=0,
            max_load_samples=None,
            refined_subcells=1,
            value_resolution=1,
        ),
        training=TrainingConfig(
            model_type="cutcellnet",
            batch_size=BATCH_SIZE,
            fusion_dim=1024,
            n_residual_blocks=10,
            trunk_type="resnet",
            residual_scaling="rezero",
            spd_mode="cholesky",
            use_reduced_null_space=True,
            matrix_size=24,
            dropout_p=0.0,
            weight_decay=1.0e-5,
            grad_clip_norm=1.0,
            epochs=EPOCHS,
            lr=LR,
            scheduler_type="cosine",
            cosine_eta_min_factor=0.5,
            train_split=0.9,
            seed=15,
            standardize_inputs=True,
            use_decoupled_loss=False,
            physical_loss_weight=1.0,
            classifier_weight=0.0,
            fourier_loss_weight=0.0,
            energy_loss_weight=0.0,
            nullspace_loss_weight=0.0,
            soft_mode_loss_weight=0.0,
            warmup_phys_only_epochs=0,
            aux_ramp_epochs=0,
            early_stopping_patience=EPOCHS + 1,
            early_stopping_min_delta=0.0,
            max_epoch_seconds=MAX_EPOCH_SECONDS,
            num_workers=0,
            pin_memory=True,
            persistent_workers=False,
            use_amp=True,
        ),
    )

    started = datetime.now()
    print(f"[{started:%H:%M:%S}] Starting {EPOCHS}-epoch all-crystal continuation.", flush=True)
    result = run_training_pipeline(
        config=config,
        base_dir=ROOT,
        init_checkpoint_path=INITIAL_CHECKPOINT,
        checkpoint_callback=checkpoint_saver,
    )
    best = min(result.history.val_loss)
    print(
        f"[{datetime.now():%H:%M:%S}] Complete: best validation relative L2={best:.6%}; "
        f"checkpoint={OUTPUT_CHECKPOINT.name}; wall time={datetime.now() - started}",
        flush=True,
    )


if __name__ == "__main__":
    main()