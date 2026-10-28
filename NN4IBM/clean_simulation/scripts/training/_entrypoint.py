"""Portable CUT_CELL_* training application used by local and remote launchers."""

from __future__ import annotations

from datetime import datetime
import os
from pathlib import Path

import torch

from clean_simulation.internal_nn_runtime.cutcell_ml.artifacts import save_checkpoint, save_history as save_run_history
from clean_simulation.internal_nn_runtime.training.pipeline import run_training_pipeline
from ._configuration import build_run_config


ROOT = Path(__file__).resolve().parents[3]
EPOCHS = int(os.environ.get("CUT_CELL_EPOCHS", "50"))
BATCH_SIZE = int(os.environ.get("CUT_CELL_BATCH_SIZE", "512"))
INITIAL_LR = float(os.environ.get("CUT_CELL_INITIAL_LR", "8.0e-6"))
RESUME_LEARNING_RATE = float(os.environ.get("CUT_CELL_RESUME_LEARNING_RATE", "0")) or None
COSINE_ETA_MIN_FACTOR = float(os.environ.get("CUT_CELL_COSINE_ETA_MIN_FACTOR", "0.01"))
OPTIMIZER_TYPE = os.environ.get("CUT_CELL_OPTIMIZER", "adamw").lower()
WEIGHT_DECAY = float(os.environ.get("CUT_CELL_WEIGHT_DECAY", "1.0e-5"))
SGD_MOMENTUM = float(os.environ.get("CUT_CELL_SGD_MOMENTUM", "0.9"))
LBFGS_MAX_ITER = int(os.environ.get("CUT_CELL_LBFGS_MAX_ITER", "4"))
LBFGS_HISTORY_SIZE = int(os.environ.get("CUT_CELL_LBFGS_HISTORY_SIZE", "20"))
LBFGS_LINE_SEARCH_FN = os.environ.get("CUT_CELL_LBFGS_LINE_SEARCH_FN", "strong_wolfe") or None
LBFGS_RESET_HISTORY_PER_BATCH = os.environ.get("CUT_CELL_LBFGS_RESET_HISTORY_PER_BATCH", "1") == "1"
USE_AMP = os.environ.get("CUT_CELL_USE_AMP", "1") == "1"
NUM_WORKERS = int(os.environ.get("CUT_CELL_NUM_WORKERS", "16"))
PREFETCH_FACTOR = int(os.environ.get("CUT_CELL_PREFETCH_FACTOR", "4"))
CPU_THREADS = int(os.environ.get("CUT_CELL_CPU_THREADS", "0"))
ALLOW_TF32 = os.environ.get("CUT_CELL_ALLOW_TF32", "1") == "1"
COMPILE_MODE = os.environ.get("CUT_CELL_COMPILE_MODE", "none").lower()
MAX_WALL_SECONDS = float(os.environ.get("CUT_CELL_MAX_WALL_SECONDS", "0")) or None
SPIKE_THRESHOLD = float(os.environ.get("CUT_CELL_SPIKE_THRESHOLD", "0.05"))
SPIKE_LR_REDUCTION_FACTOR = float(os.environ.get("CUT_CELL_SPIKE_LR_REDUCTION_FACTOR", "0.5"))
MAX_SPIKE_RETRIES = int(os.environ.get("CUT_CELL_MAX_SPIKE_RETRIES", "3"))
DISTRIBUTED = os.environ.get("CUT_CELL_DISTRIBUTED", "0") == "1"
MODEL_TYPE = os.environ.get("CUT_CELL_MODEL_TYPE", "cutcellnet")
FUSION_DIM = int(os.environ.get("CUT_CELL_FUSION_DIM", "4096"))
RESIDUAL_BLOCKS = int(os.environ.get("CUT_CELL_RESIDUAL_BLOCKS", "20"))
N_ATTENTION_HEADS = int(os.environ.get("CUT_CELL_N_ATTENTION_HEADS", "0")) or None
RESIDUAL_SCALING = os.environ.get("CUT_CELL_RESIDUAL_SCALING", "rezero")
TRANSFORMER_ORDERED_SDF_BRANCH = os.environ.get("CUT_CELL_TRANSFORMER_ORDERED_SDF_BRANCH", "0") == "1"
REFINED_SUBCELLS = int(os.environ.get("CUT_CELL_REFINED_SUBCELLS", "1"))
SCHEDULER_TYPE = os.environ.get("CUT_CELL_SCHEDULER_TYPE", "cosine")
TRAINING_SEED = int(os.environ.get("CUT_CELL_SEED", "15"))
SPLIT_SEED = int(os.environ.get("CUT_CELL_SPLIT_SEED", str(TRAINING_SEED)))
SHUFFLE_SEED = int(os.environ.get("CUT_CELL_SHUFFLE_SEED", str(TRAINING_SEED)))
PRESERVE_ESTIMATED_INPUT_NORMALIZERS = os.environ.get("CUT_CELL_PRESERVE_ESTIMATED_INPUT_NORMALIZERS", "0") == "1"
HARD_EXAMPLE_DROP_FRACTION = float(os.environ.get("CUT_CELL_HARD_EXAMPLE_DROP_FRACTION", "0.0"))
EXCLUDED_GEOMETRY_PREFIXES = tuple(
    prefix.strip() for prefix in os.environ.get("CUT_CELL_EXCLUDED_GEOMETRY_PREFIXES", "").split(",") if prefix.strip()
)
CHECKPOINT_DIR = ROOT / "clean_simulation" / "outputs" / "checkpoints"
DATA_PATH = Path(os.environ.get(
    "CUT_CELL_DATA_PATH",
    ROOT / "clean_simulation" / "outputs" / "data" / "grid_locked_source_sdf_rebuilt_queso_i2_p1_simple_harsh_crystals30_filtered.h5",
)).expanduser()
DEFAULT_CHECKPOINT = CHECKPOINT_DIR / "source_sdf_rebuilt_queso_d20_f4096_aws.pt"
OUTPUT_CHECKPOINT = Path(os.environ.get("CUT_CELL_OUTPUT_CHECKPOINT", DEFAULT_CHECKPOINT)).expanduser()
INIT_CHECKPOINT = Path(os.environ.get("CUT_CELL_INIT_CHECKPOINT", OUTPUT_CHECKPOINT)).expanduser()
HISTORY_PATH_VALUE = os.environ.get("CUT_CELL_HISTORY_PATH")
HISTORY_PATH = Path(HISTORY_PATH_VALUE).expanduser() if HISTORY_PATH_VALUE else None


def checkpoint_saver(model, history, epoch, val_loss, optimizer, scheduler, scaler) -> None:
    save_checkpoint(OUTPUT_CHECKPOINT, model, history, epoch, val_loss, optimizer, scheduler, scaler)
    print(
        f"[{datetime.now():%H:%M:%S}] Epoch {epoch}: validation relative L2={val_loss:.6%}; "
        f"checkpoint={OUTPUT_CHECKPOINT.name}",
        flush=True,
    )
    save_history(history, val_loss)


def save_history(history, best_loss: float) -> None:
    save_run_history(
        HISTORY_PATH,
        history,
        {
            "model_type": MODEL_TYPE,
            "fusion_dim": FUSION_DIM,
            "n_residual_blocks": RESIDUAL_BLOCKS,
            "n_attention_heads": N_ATTENTION_HEADS or (8 if FUSION_DIM % 8 == 0 else 4),
            "residual_scaling": RESIDUAL_SCALING,
            "transformer_ordered_sdf_branch": TRANSFORMER_ORDERED_SDF_BRANCH,
            "refined_subcells": REFINED_SUBCELLS,
            "n_distance_nodes": (REFINED_SUBCELLS + 1) ** 3,
            "seed": TRAINING_SEED,
            "split_seed": SPLIT_SEED,
            "shuffle_seed": SHUFFLE_SEED,
            "initial_lr": INITIAL_LR,
            "optimizer_type": OPTIMIZER_TYPE,
            "weight_decay": WEIGHT_DECAY,
            "sgd_momentum": SGD_MOMENTUM,
            "lbfgs_max_iter": LBFGS_MAX_ITER,
            "lbfgs_history_size": LBFGS_HISTORY_SIZE,
            "lbfgs_line_search_fn": LBFGS_LINE_SEARCH_FN,
            "lbfgs_reset_history_per_batch": LBFGS_RESET_HISTORY_PER_BATCH,
            "scheduler_type": SCHEDULER_TYPE,
            "hard_example_drop_fraction": HARD_EXAMPLE_DROP_FRACTION,
        },
        best_loss,
    )


def main() -> None:
    if CPU_THREADS > 0:
        torch.set_num_threads(CPU_THREADS)
    if ALLOW_TF32:
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    if COMPILE_MODE not in {"none", "default", "reduce-overhead", "max-autotune"}:
        raise ValueError("CUT_CELL_COMPILE_MODE must be none, default, reduce-overhead, or max-autotune.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this training run.")
    if DISTRIBUTED and "LOCAL_RANK" not in os.environ:
        raise RuntimeError("CUT_CELL_DISTRIBUTED=1 requires launch via torchrun --nproc_per_node=2.")
    if not DATA_PATH.is_file():
        raise FileNotFoundError(f"Missing expanded training data: {DATA_PATH}")

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    init_checkpoint_path = INIT_CHECKPOINT if INIT_CHECKPOINT.is_file() else None
    config = build_run_config(
        data_path=DATA_PATH,
        model_type=MODEL_TYPE,
        batch_size=BATCH_SIZE,
        fusion_dim=FUSION_DIM,
        residual_blocks=RESIDUAL_BLOCKS,
        n_attention_heads=N_ATTENTION_HEADS,
        residual_scaling=RESIDUAL_SCALING,
        transformer_ordered_sdf_branch=TRANSFORMER_ORDERED_SDF_BRANCH,
        weight_decay=WEIGHT_DECAY,
        optimizer_type=OPTIMIZER_TYPE,
        sgd_momentum=SGD_MOMENTUM,
        lbfgs_max_iter=LBFGS_MAX_ITER,
        lbfgs_history_size=LBFGS_HISTORY_SIZE,
        lbfgs_line_search_fn=LBFGS_LINE_SEARCH_FN,
        lbfgs_reset_history_per_batch=LBFGS_RESET_HISTORY_PER_BATCH,
        epochs=EPOCHS,
        initial_lr=INITIAL_LR,
        scheduler_type=SCHEDULER_TYPE,
        cosine_eta_min_factor=COSINE_ETA_MIN_FACTOR,
        training_seed=TRAINING_SEED,
        split_seed=SPLIT_SEED,
        shuffle_seed=SHUFFLE_SEED,
        hard_example_drop_fraction=HARD_EXAMPLE_DROP_FRACTION,
        excluded_geometry_prefixes=EXCLUDED_GEOMETRY_PREFIXES,
        max_wall_seconds=MAX_WALL_SECONDS,
        distributed=DISTRIBUTED,
        num_workers=NUM_WORKERS,
        prefetch_factor=PREFETCH_FACTOR,
        use_amp=USE_AMP,
        refined_subcells=REFINED_SUBCELLS,
    )

    started = datetime.now()
    if not DISTRIBUTED or int(os.environ["RANK"]) == 0:
        print(
            f"[{started:%H:%M:%S}] Starting {MODEL_TYPE}: width={FUSION_DIM}, blocks={RESIDUAL_BLOCKS}, "
            f"residual_scaling={RESIDUAL_SCALING}, attention_heads={N_ATTENTION_HEADS or ('8' if FUSION_DIM % 8 == 0 else '4')}, "
            f"ordered_sdf_branch={TRANSFORMER_ORDERED_SDF_BRANCH}, batch={BATCH_SIZE}, "
            f"lr={INITIAL_LR:.1e} -> {INITIAL_LR * COSINE_ETA_MIN_FACTOR:.1e}, epochs={EPOCHS}, wall_seconds={MAX_WALL_SECONDS}, "
            f"resume_learning_rate={RESUME_LEARNING_RATE}, "
            f"optimizer={OPTIMIZER_TYPE}, scheduler={SCHEDULER_TYPE}, amp={USE_AMP}, seed={TRAINING_SEED}, split_seed={SPLIT_SEED}, shuffle_seed={SHUFFLE_SEED}, "
            f"spike_threshold={SPIKE_THRESHOLD:.1%}, spike_factor={SPIKE_LR_REDUCTION_FACTOR:g}, "
            f"max_spike_retries={MAX_SPIKE_RETRIES}, resume={init_checkpoint_path is not None}, "
            f"cpu_threads={CPU_THREADS or torch.get_num_threads()}, workers={NUM_WORKERS}, prefetch={PREFETCH_FACTOR}, tf32={ALLOW_TF32}, "
            f"compile_mode={COMPILE_MODE}, "
            f"preserve_estimated_input_normalizers={PRESERVE_ESTIMATED_INPUT_NORMALIZERS}, "
            f"hard_example_drop_fraction={HARD_EXAMPLE_DROP_FRACTION:.2%}, "
            f"excluded_geometry_prefixes={EXCLUDED_GEOMETRY_PREFIXES}, "
            f"distributed={DISTRIBUTED}, refined_subcells={REFINED_SUBCELLS}, "
            f"inputs={(REFINED_SUBCELLS + 1) ** 3} signed distances, gpu={torch.cuda.get_device_name(0)}.",
            flush=True,
        )
    result = run_training_pipeline(
        config=config,
        base_dir=ROOT,
        init_checkpoint_path=init_checkpoint_path,
        preserve_estimated_input_normalizers=PRESERVE_ESTIMATED_INPUT_NORMALIZERS,
        checkpoint_callback=checkpoint_saver,
        spike_threshold=SPIKE_THRESHOLD,
        spike_lr_reduction_factor=SPIKE_LR_REDUCTION_FACTOR,
        max_spike_retries=MAX_SPIKE_RETRIES,
        resume_learning_rate=RESUME_LEARNING_RATE,
    )
    if not DISTRIBUTED or int(os.environ["RANK"]) == 0:
        best = min(result.history.val_loss)
        save_history(result.history, best)
        print(
            f"[{datetime.now():%H:%M:%S}] Complete: best validation relative L2={best:.6%}; "
            f"checkpoint={OUTPUT_CHECKPOINT.name}; wall time={datetime.now() - started}",
            flush=True,
        )