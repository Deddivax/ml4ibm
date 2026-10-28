"""Export exact held-out relative-Frobenius loss distributions for a checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import h5py
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clean_simulation.internal_nn_runtime.cutcell_ml.classification_utils import get_class_label_batch
from clean_simulation.internal_nn_runtime.cutcell_ml.data.datasets import reconstruct_matrices_from_ut
from clean_simulation.internal_nn_runtime.cutcell_ml.model import CutCellNet, CutCellNetTransformer


GROUP = "integration_2_poly_1"
TRAIN_SPLIT = 0.9
BATCH_SIZE = 2048
NORMALIZER_BATCHES = 128


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model-type", choices=("cutcellnet", "transformer"), required=True)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--blocks", type=int, required=True)
    parser.add_argument("--residual-scaling", default="fixed")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def validation_indices(size: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(size, generator=generator)
    train_size = int(TRAIN_SPLIT * size)
    return permutation[:train_size], permutation[train_size:]


def estimate_normalizer(distances: np.ndarray, train_indices: torch.Tensor, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    shuffled = torch.randperm(train_indices.numel(), generator=generator)
    sampled = train_indices[shuffled[: NORMALIZER_BATCHES * BATCH_SIZE]].numpy()
    values = torch.from_numpy(distances[sampled].astype(np.float32, copy=False))
    mean = values.mean(dim=0)
    std = values.std(dim=0, correction=0).clamp(min=1.0e-4)
    return mean, std


def make_model(args: argparse.Namespace) -> torch.nn.Module:
    if args.model_type == "cutcellnet":
        return CutCellNet(
            fusion_dim=args.width,
            n_residual_blocks=args.blocks,
            include_nu=False,
            trunk_type="resnet",
            use_class_conditioning=False,
            include_class_inputs=False,
            residual_scaling=args.residual_scaling,
            spd_mode="cholesky",
            dropout_p=0.0,
            use_reduced_null_space=True,
            matrix_size=24,
            n_distance_nodes=8,
        )
    return CutCellNetTransformer(
        embed_dim=args.width,
        n_heads=8,
        n_layers=args.blocks,
        ffn_dim=4 * args.width,
        dropout=0.0,
        reduced_matrix_size=18,
    )


def main() -> None:
    args = parse_args()
    if not args.checkpoint.is_file() or not args.data.is_file():
        raise FileNotFoundError("Checkpoint or HDF5 data file is missing.")

    with h5py.File(args.data, "r") as handle:
        group = handle[GROUP]
        distances = np.asarray(group["d_values"], dtype=np.float32)
        packed_stiffness = np.asarray(group["stiffness_ut"], dtype=np.float64)
        volume_fraction = np.asarray(group["cut_volume_fraction"], dtype=np.float64)
        geometry_name = np.char.decode(np.asarray(group["geometry_name"]), "utf-8")

    train_indices, val_indices = validation_indices(len(distances), args.seed)
    input_mean, input_std = estimate_normalizer(distances, train_indices, args.seed)
    model = make_model(args).cuda().eval()
    model.set_normalizers(input_mean.cuda(), input_std.cuda())
    checkpoint = torch.load(args.checkpoint, map_location="cuda", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    val_indices_np = val_indices.numpy()
    target_norms = []
    error_norms = []
    raw_relative_errors = []
    with torch.no_grad():
        for start in range(0, val_indices_np.size, BATCH_SIZE):
            indices = val_indices_np[start:start + BATCH_SIZE]
            batch_distances = torch.from_numpy(distances[indices]).cuda()
            targets = torch.from_numpy(reconstruct_matrices_from_ut(packed_stiffness[indices]).astype(np.float32)).cuda()
            classes = get_class_label_batch(batch_distances)
            predicted_unit, log_norm = model(batch_distances, nu=None, cut_class=classes)
            predicted = predicted_unit.float() * torch.exp(log_norm.float())[:, None, None]
            error_norm = torch.linalg.matrix_norm(predicted - targets, ord="fro", dim=(-2, -1)).cpu().numpy()
            target_norm = torch.linalg.matrix_norm(targets, ord="fro", dim=(-2, -1)).cpu().numpy()
            error_norms.append(error_norm)
            target_norms.append(target_norm)

    target_norm = np.concatenate(target_norms)
    error_norm = np.concatenate(error_norms)
    floor = max(float(np.quantile(target_norm, 0.05)), 1.0e-4)
    raw_relative = error_norm / np.maximum(target_norm, 1.0e-8)
    floored_relative = error_norm / np.maximum(target_norm, floor)
    output = pd.DataFrame(
        {
            "row_index": val_indices_np,
            "geometry_name": geometry_name[val_indices_np],
            "cut_volume_fraction": volume_fraction[val_indices_np],
            "sdf_min": distances[val_indices_np].min(axis=1),
            "sdf_max": distances[val_indices_np].max(axis=1),
            "sdf_span": np.ptp(distances[val_indices_np], axis=1),
            "target_frobenius_norm": target_norm,
            "error_frobenius_norm": error_norm,
            "raw_relative_frobenius_error": raw_relative,
            "floored_relative_frobenius_error": floored_relative,
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    summary = {
        "checkpoint": str(args.checkpoint),
        "model_type": args.model_type,
        "seed": args.seed,
        "validation_samples": int(len(output)),
        "reference_norm_floor": floor,
        "mean_floored_relative_error": float(floored_relative.mean()),
        "quantiles": {str(level): float(np.quantile(floored_relative, level)) for level in (0.0, 0.5, 0.9, 0.95, 0.99, 0.999, 1.0)},
    }
    args.output.with_suffix(".json").write_text(pd.Series(summary).to_json(indent=2), encoding="utf-8")
    print(pd.Series(summary).to_json(indent=2))


if __name__ == "__main__":
    main()
