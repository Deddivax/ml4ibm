"""Symmetry canonicalization and stiffness representation utilities."""

from __future__ import annotations

import importlib
from functools import lru_cache
from typing import Any

import torch

from clean_simulation.internal_fem_kernel.runtime.common import as_tensor


@lru_cache(maxsize=1)
def _cached_symmetry_table() -> dict[str, Any]:
    symmetries = importlib.import_module("cutcell_ml.symmetries")
    table = getattr(symmetries, "build_symmetry_table")()
    return {
        "node_perms": as_tensor(table["node_perms"], dtype=torch.int64, device="cpu"),
        "dof_perms": as_tensor(table["dof_perms"], dtype=torch.int64, device="cpu"),
        "dof_signs": as_tensor(table["dof_signs"], dtype=torch.float64, device="cpu"),
    }


def get_symmetry_cache() -> dict[str, Any]:
    return _cached_symmetry_table()


def canonicalize_d_batch(d_batch: torch.Tensor, node_perms: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    d_orbits = d_batch[:, node_perms]
    key = torch.round(d_orbits * 1.0e12) / 1.0e12
    active = torch.ones((d_batch.shape[0], node_perms.shape[0]), dtype=torch.bool, device=d_batch.device)
    for dimension in range(8):
        values = key[:, :, dimension]
        masked = torch.where(active, values, torch.full_like(values, float("inf")))
        minimum = masked.min(dim=1, keepdim=True).values
        active &= values <= (minimum + 1e-15)
        if bool(torch.all(active.sum(dim=1) == 1).item()):
            break
    group_index = torch.argmax(active.to(dtype=torch.int64), dim=1)
    canonical = d_orbits[torch.arange(d_batch.shape[0], device=d_batch.device), group_index]
    return canonical.to(dtype=torch.float64), group_index.to(dtype=torch.int64)


def output_to_k24_batch(output: torch.Tensor) -> torch.Tensor:
    if output.ndim == 3 and output.shape[-2:] == (24, 24):
        return 0.5 * (output + output.transpose(-1, -2))
    triangle = torch.triu_indices(24, 24, device=output.device)
    if output.ndim == 2 and output.shape[1] == 300:
        batch_size = output.shape[0]
        stiffness = torch.zeros((batch_size, 24, 24), dtype=output.dtype, device=output.device)
        stiffness[:, triangle[0], triangle[1]] = output
        stiffness[:, triangle[1], triangle[0]] = output
        return stiffness
    if output.ndim == 1 and output.shape[0] == 300:
        stiffness = torch.zeros((1, 24, 24), dtype=output.dtype, device=output.device)
        stiffness[:, triangle[0], triangle[1]] = output[None, :]
        stiffness[:, triangle[1], triangle[0]] = output[None, :]
        return stiffness
    raise RuntimeError(f"Unexpected NN output shape: {tuple(output.shape)}")


def inverse_transform_k_batch(
    canonical_stiffness: torch.Tensor,
    group_index: torch.Tensor,
    dof_permutations: torch.Tensor,
    dof_signs: torch.Tensor,
) -> torch.Tensor:
    permutation = dof_permutations[group_index]
    signs = dof_signs[group_index]
    if canonical_stiffness.shape[0] == 0:
        return canonical_stiffness

    indices = torch.arange(permutation.shape[1], dtype=torch.int64, device=permutation.device).unsqueeze(0).expand_as(permutation)
    inverse_permutation = torch.empty_like(permutation)
    inverse_permutation.scatter_(1, permutation, indices)
    inverse_signs = torch.gather(signs, 1, inverse_permutation)

    row_indices = inverse_permutation[:, :, None].expand(-1, -1, 24)
    column_indices = inverse_permutation[:, None, :].expand(-1, 24, -1)
    permuted = torch.gather(canonical_stiffness, 1, row_indices)
    permuted = torch.gather(permuted, 2, column_indices)
    return permuted * inverse_signs[:, :, None] * inverse_signs[:, None, :]