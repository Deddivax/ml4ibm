"""Structural stiffness diagnostics for trained cut-cell models."""

from __future__ import annotations

import torch

from clean_simulation.internal_nn_runtime.cutcell_ml.model import (
    HEX_CORNER_COORDS,
    build_rigid_body_modes,
)


@torch.no_grad()
def model_structural_diagnostics(
    model: torch.nn.Module,
    loader,
    device: str,
    max_batches: int = 16,
) -> dict[str, float]:
    """Measure stiffness symmetry, rigid-body nullity, and eigenvalue bounds."""
    model.eval()
    rigid_body_modes = build_rigid_body_modes(HEX_CORNER_COORDS).to(device)

    symmetry_error = rigid_body_residual = 0.0
    minimum_eigenvalue = minimum_physical_eigenvalue = float("inf")
    maximum_eigenvalue = float("-inf")

    for batch_index, (distances, poisson_ratio, _, cut_class) in enumerate(loader):
        distances = distances.to(device)
        poisson_ratio = poisson_ratio.to(device)
        cut_class = cut_class.to(device).long()
        stiffness_unit, log_norm = model(distances, nu=poisson_ratio, cut_class=cut_class)
        stiffness = stiffness_unit * torch.exp(log_norm)[:, None, None]

        symmetry_error = max(symmetry_error, (stiffness - stiffness.transpose(-1, -2)).abs().amax().item())
        rigid_body_residual = max(rigid_body_residual, (stiffness @ rigid_body_modes).abs().amax().item())

        eigenvalues = torch.linalg.eigvalsh(stiffness.double())
        minimum_eigenvalue = min(minimum_eigenvalue, eigenvalues[:, 0].min().item())
        minimum_physical_eigenvalue = min(minimum_physical_eigenvalue, eigenvalues[:, 6].min().item())
        maximum_eigenvalue = max(maximum_eigenvalue, eigenvalues[:, -1].max().item())

        if batch_index + 1 >= max_batches:
            break

    return {
        "symmetry_error": symmetry_error,
        "rigid_body_residual": rigid_body_residual,
        "min_eigenvalue": 0.0 if minimum_eigenvalue == float("inf") else minimum_eigenvalue,
        "min_physical_eigenvalue": 0.0 if minimum_physical_eigenvalue == float("inf") else minimum_physical_eigenvalue,
        "max_eigenvalue": 0.0 if maximum_eigenvalue == float("-inf") else maximum_eigenvalue,
    }