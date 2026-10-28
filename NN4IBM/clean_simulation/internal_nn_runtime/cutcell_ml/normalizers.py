"""Input normalization statistics shared by supported training workflows."""

from __future__ import annotations

import torch


@torch.no_grad()
def estimate_input_normalizers(
    loader,
    *,
    include_nu: bool,
    n_distance_nodes: int = 8,
    max_batches: int = 128,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Estimate feature mean and standard deviation over a bounded loader prefix."""
    sum_x, sum_x2, count = None, None, 0

    for index, (distances, poisson_ratio, _, _) in enumerate(loader):
        distances = distances.float()
        values = (
            torch.cat([distances, poisson_ratio.unsqueeze(-1)], dim=-1)
            if include_nu
            else distances
        )
        batch_sum, batch_sum_squared = values.sum(0), (values * values).sum(0)
        sum_x = batch_sum if sum_x is None else sum_x + batch_sum
        sum_x2 = batch_sum_squared if sum_x2 is None else sum_x2 + batch_sum_squared
        count += values.shape[0]
        if index + 1 >= max_batches:
            break

    if count == 0:
        dimension = n_distance_nodes + (1 if include_nu else 0)
        return torch.zeros(dimension), torch.ones(dimension)

    mean = sum_x / count
    variance = sum_x2 / count - mean**2
    standard_deviation = torch.sqrt(torch.clamp(variance, min=1e-8)).clamp(min=1e-4)
    return mean, standard_deviation