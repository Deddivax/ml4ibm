"""Construct supported NN training models from a run configuration."""

from __future__ import annotations

import torch

from clean_simulation.internal_nn_runtime.cutcell_ml.config import RunConfig
from clean_simulation.internal_nn_runtime.cutcell_ml.model import (
    CutCellNet,
    CutCellNetTransformer,
)


def build_training_model(
    config: RunConfig,
    *,
    device: str,
    n_distance_nodes: int,
) -> torch.nn.Module:
    """Build the configured model before normalizer and checkpoint setup."""
    if config.training.model_type == "cutcellnet":
        torch.manual_seed(config.training.seed)
        if device == "cuda":
            torch.cuda.manual_seed_all(config.training.seed)
        return CutCellNet(
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
            n_distance_nodes=n_distance_nodes,
        )
    if config.training.model_type == "transformer":
        if config.training.include_nu or config.training.use_class_conditioning or config.training.include_class_inputs:
            raise ValueError("CutCellNetTransformer accepts signed-distance inputs only.")
        torch.manual_seed(config.training.seed)
        if device == "cuda":
            torch.cuda.manual_seed_all(config.training.seed)
        embed_dim = config.training.fusion_dim
        n_heads = config.training.n_attention_heads or (8 if embed_dim % 8 == 0 else 4)
        if embed_dim % n_heads:
            raise ValueError(f"embed_dim={embed_dim} must be divisible by n_attention_heads={n_heads}.")
        return CutCellNetTransformer(
            embed_dim=embed_dim,
            n_heads=n_heads,
            n_layers=config.training.n_residual_blocks,
            ffn_dim=4 * embed_dim,
            dropout=config.training.dropout_p,
            reduced_matrix_size=18,
            ordered_sdf_branch=config.training.transformer_ordered_sdf_branch,
            n_distance_nodes=n_distance_nodes,
        )
    raise ValueError(f"Unsupported model_type={config.training.model_type!r}")