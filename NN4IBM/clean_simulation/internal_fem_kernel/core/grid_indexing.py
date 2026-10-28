import torch


def global_to_local(x, cell_min, dx):
    """Map physical coordinates to reference element coordinates in [-1, 1]."""
    return 2.0 * (x - cell_min) / dx - 1.0


def cell_index(ijk, resolution):
    """Flatten structured cell indices (i, j, k) to a global element index."""
    nx, ny, nz = resolution
    i, j, k = ijk[..., 0], ijk[..., 1], ijk[..., 2]
    return i * ny * nz + j * nz + k


def structured_node_shape(resolution, poly_order=1):
    nx, ny, nz = resolution
    p = int(poly_order)
    if p < 1:
        raise ValueError("poly_order must be >= 1")
    return nx * p + 1, ny * p + 1, nz * p + 1


def structured_total_nodes(resolution, poly_order=1):
    sx, sy, sz = structured_node_shape(resolution, poly_order=poly_order)
    return sx * sy * sz


def active_element_nodes(active_indices, resolution, poly_order=1, device="cuda"):
    """Return node indices for active hex elements on a structured grid.

    Parameters
    ----------
    active_indices : array-like (M,)  global element indices
    resolution     : (nx, ny, nz)
    poly_order     : int >= 1
    device         : torch device string or torch.device

    Returns
    -------
    (M, K) int64 tensor on `device`, where K = (poly_order+1)^3
    """
    nx, ny, nz = resolution
    p = int(poly_order)
    sx, sy, sz = structured_node_shape(resolution, poly_order=p)

    node_id = torch.arange(sx * sy * sz, dtype=torch.int64, device=device).reshape(sx, sy, sz)

    e = torch.as_tensor(active_indices, dtype=torch.int64, device=device)
    i = e // (ny * nz)
    j = (e %  (ny * nz)) // nz
    k = e %  nz

    if p == 1:
        # torch.stack has no `device` kwarg — tensors already live on `device`
        return torch.stack([
            node_id[i,     j,     k    ],
            node_id[i + 1, j,     k    ],
            node_id[i + 1, j + 1, k    ],
            node_id[i,     j + 1, k    ],
            node_id[i,     j,     k + 1],
            node_id[i + 1, j,     k + 1],
            node_id[i + 1, j + 1, k + 1],
            node_id[i,     j + 1, k + 1],
        ], dim=1)  # (M, 8)

    # Higher-order: build offsets on the correct device from the start
    r = torch.arange(p + 1, dtype=torch.int64, device=device)
    ga, gb, gc = torch.meshgrid(r, r, r, indexing="ij")   # each (p+1, p+1, p+1)
    local_offsets = torch.stack([ga, gb, gc], dim=-1).reshape(-1, 3)  # (K, 3)

    return node_id[
        i[:, None] * p + local_offsets[None, :, 0],
        j[:, None] * p + local_offsets[None, :, 1],
        k[:, None] * p + local_offsets[None, :, 2],
    ]  # (M, K)