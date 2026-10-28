from __future__ import annotations

import itertools
import numpy as np


# ==========================================================
# Cube rotation setup
# ==========================================================

def _build_cube_rotation_permutations() -> list[tuple[int, ...]]:
    verts = np.asarray([(i & 1, (i >> 1) & 1, (i >> 2) & 1) for i in range(8)], dtype=np.int64)
    coord_to_idx = {tuple(v.tolist()): int(i) for i, v in enumerate(verts)}

    perms = []

    for perm in itertools.permutations((0, 1, 2)):
        p = np.eye(3, dtype=np.int64)[:, perm]
        for signs in itertools.product((-1, 1), repeat=3):
            s = np.diag(np.asarray(signs, dtype=np.int64))
            m = s @ p

            if int(round(np.linalg.det(m))) != 1:
                continue

            mapped = []
            for v in verts:
                x = 2 * v - 1
                y = m @ x
                vb = ((y + 1) // 2).astype(np.int64)
                mapped.append(coord_to_idx[tuple(vb.tolist())])

            perms.append(tuple(mapped))

    return list(dict.fromkeys(perms))


_CUBE_ROT_PERMS = _build_cube_rotation_permutations()


def _permute_case_mask(mask: int, perm: tuple[int, ...]) -> int:
    out = 0
    for i in range(8):
        if (mask >> i) & 1:
            out |= 1 << perm[i]
    return out


def _canonicalize_case_mask(mask: int) -> int:
    best = 255

    for perm in _CUBE_ROT_PERMS:
        m = _permute_case_mask(mask, perm)
        mc = m ^ 0xFF  # complement

        if m < best:
            best = m
        if mc < best:
            best = mc

    return int(best)


# ==========================================================
# Build canonical class system
# ==========================================================

_CANONICAL_CASE_MASKS = sorted(
    {_canonicalize_case_mask(m) for m in range(256)}
)

if len(_CANONICAL_CASE_MASKS) != 15:
    raise RuntimeError("Expected 15 canonical masks")

MASK_TO_CLASS = {m: i for i, m in enumerate(_CANONICAL_CASE_MASKS)}


# ==========================================================
# Classification
# ==========================================================

def classify_cube_configuration(
    nodal_sdf_basic_order: np.ndarray,
    threshold: float = 0.5,
) -> tuple[int, int]:

    sdf = np.asarray(nodal_sdf_basic_order, dtype=np.float64).reshape(8)

    raw_mask = 0
    for i, val in enumerate(sdf):
        if val > threshold:
            raw_mask |= (1 << i)

    canonical_mask = _canonicalize_case_mask(raw_mask)

    return int(raw_mask), int(canonical_mask)


RAW_MASK_TO_CLASS = [
    MASK_TO_CLASS[_canonicalize_case_mask(m)] for m in range(256)
]


def get_class_label_batch(d_batch: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """Vectorized corner-topology labels for (..., 8) or refined (..., N) inputs."""
    import torch
    d_batch = d_batch[..., :8]
    # Powers of two to convert binary mask to base 10 integer index
    powers_of_two = torch.tensor([1, 2, 4, 8, 16, 32, 64, 128], dtype=torch.long, device=d_batch.device)
    
    # Thresholding
    binary = (d_batch > threshold).long()
    raw_masks = (binary * powers_of_two).sum(dim=-1)
    
    # Lookup in RAW_MASK_TO_CLASS using a tensor
    lookup = torch.tensor(RAW_MASK_TO_CLASS, dtype=torch.long, device=d_batch.device)
    return lookup[raw_masks]


def compute_class_label_from_d(d) -> int:
    import torch
    if torch.is_tensor(d):
        binary = (d[..., :8] > 0.5).long()
        powers_of_two = torch.tensor([1, 2, 4, 8, 16, 32, 64, 128], dtype=torch.long, device=d.device)
        raw_mask = int((binary * powers_of_two).sum().item())
    else:
        sdf = np.asarray(d, dtype=np.float64).reshape(-1)[:8]
        raw_mask = 0
        for i, val in enumerate(sdf):
            if val > 0.5:
                raw_mask |= (1 << i)

    return RAW_MASK_TO_CLASS[raw_mask]
