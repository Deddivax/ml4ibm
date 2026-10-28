"""
Computes all 48 symmetries of the unit cube and uses them for data
augmentation of cut cell (d, K) pairs.

Node ordering convention (standard hex, matches FEniCS / VTK):

        6───────7
       /|      /|
      / |     / |        z
     4───────5  |        │  y
     |  2────|──3        │ /
     | /     | /         └──── x
     |/      |/
     0───────1

  node i has coordinates (i&1, (i>>1)&1, (i>>2)&1)
  i.e.:
    0 = (0,0,0)    1 = (1,0,0)
    2 = (0,1,0)    3 = (1,1,0)
    4 = (0,0,1)    5 = (1,0,1)
    6 = (0,1,1)    7 = (1,1,1)
"""

import numpy as np
import torch
from itertools import permutations


# ─────────────────────────────────────────────
#  1.  Build node coordinates
# ─────────────────────────────────────────────

def node_coords() -> np.ndarray:
    """Returns (8, 3) array of node coordinates for the unit hex."""
    return np.array([[i & 1, (i >> 1) & 1, (i >> 2) & 1]
                     for i in range(8)], dtype=float)


# ─────────────────────────────────────────────
#  2.  Generate all 48 symmetries of the cube
# ─────────────────────────────────────────────

def _rotation_matrices() -> list:
    """
    Generate all 24 rotation matrices of the cube (proper rotations, det=+1).
    Uses the fact that cube rotations are all 3x3 signed permutation matrices
    with det = +1.
    """
    rotations = []
    # A signed permutation matrix has exactly one ±1 per row and column.
    for perm in permutations(range(3)):
        for signs in range(8):  # 2^3 sign combinations
            s = np.array([1 - 2 * ((signs >> i) & 1) for i in range(3)])
            R = np.zeros((3, 3))
            for row, col in enumerate(perm):
                R[row, col] = s[row]
            if np.linalg.det(R) > 0:  # keep only proper rotations
                rotations.append(R)
    return rotations


def _all_symmetry_matrices() -> list:
    """
    All 48 symmetries = 24 rotations + 24 improper rotations (rot * inversion).
    """
    rotations = _rotation_matrices()
    inversion = -np.eye(3)
    improper  = [inversion @ R for R in rotations]
    return rotations + improper


def compute_node_permutations() -> np.ndarray:
    """
    For each of the 48 cube symmetries, compute the permutation of the 8
    nodes of the unit hex.

    The unit hex lives in [0,1]^3. A symmetry R (3x3 orthogonal matrix)
    maps node x to R*(x - c) + c where c = (0.5, 0.5, 0.5) is the cube center.
    The result is another node of the hex — this gives the node permutation.

    Returns
    -------
    perms : (48, 8) int array
        perms[s, i] = j means symmetry s maps node i to node j.
    """
    coords   = node_coords()           # (8, 3)
    center   = np.array([0.5, 0.5, 0.5])
    symmetries = _all_symmetry_matrices()

    perms = np.zeros((48, 8), dtype=int)
    for s_idx, R in enumerate(symmetries):
        for i, xi in enumerate(coords):
            # Apply symmetry: reflect/rotate around cube center
            xj = R @ (xi - center) + center
            # Find which node xj corresponds to (round to handle float errors)
            xj_rounded = np.round(xj).astype(int)
            j = xj_rounded[0] | (xj_rounded[1] << 1) | (xj_rounded[2] << 2)
            perms[s_idx, i] = j

    return perms


def compute_dof_permutations(node_perms: np.ndarray,
                              sym_matrices: list) -> tuple:
    """
    For each symmetry, compute the permutation of the 24 DOFs (3 per node).

    A symmetry R permutes nodes AND rotates the displacement vectors at each
    node. If node i maps to node j under symmetry s, then DOF (i, k) maps to
    DOF (j, R*e_k) — i.e. the displacement component is also rotated by R.

    For a permutation matrix R (only ±1 entries), R*e_k = ±e_{perm(k)},
    so each DOF maps to another DOF with a possible sign flip.

    Returns
    -------
    dof_perms : (48, 24) int array
        dof_perms[s, dof_i] = dof_j means symmetry s maps DOF i to DOF j.
    dof_signs : (48, 24) float array
        Sign (+1 or -1) applied to the DOF value under the symmetry.
    """
    n_sym = len(sym_matrices)
    dof_perms = np.zeros((n_sym, 24), dtype=int)
    dof_signs = np.ones((n_sym, 24), dtype=float)

    for s_idx, R in enumerate(sym_matrices):
        for node_i in range(8):
            node_j = node_perms[s_idx, node_i]
            for k in range(3):  # x, y, z component
                # R maps e_k to R[:, k] — find which component and sign
                col = R[:, k]
                # For a signed permutation matrix, col has one nonzero entry
                comp_j = int(np.argmax(np.abs(col)))
                sign_j = col[comp_j]

                dof_i = 3 * node_i + k
                dof_j = 3 * node_j + comp_j

                dof_perms[s_idx, dof_i] = dof_j
                dof_signs[s_idx, dof_i] = sign_j

    return dof_perms, dof_signs


# ─────────────────────────────────────────────
#  3.  Precompute and cache all symmetry data
# ─────────────────────────────────────────────

def build_symmetry_table() -> dict:
    """
    Build and return a dict containing all symmetry permutations and signs,
    ready for use in data augmentation.

    Returns
    -------
    table : dict with keys:
        'node_perms' : (48, 8)  int    — node permutations
        'dof_perms'  : (48, 24) int    — DOF permutations
        'dof_signs'  : (48, 24) float  — DOF signs (±1)
        'n_sym'      : int             — 48
    """
    sym_matrices = _all_symmetry_matrices()
    node_perms   = compute_node_permutations()
    dof_perms, dof_signs = compute_dof_permutations(node_perms, sym_matrices)

    return {
        "node_perms" : node_perms,
        "dof_perms"  : dof_perms,
        "dof_signs"  : dof_signs,
        "n_sym"      : 48,
    }


# ─────────────────────────────────────────────
#  4.  Data augmentation
# ─────────────────────────────────────────────

def augment(
    d:     torch.Tensor,   # (N, 8)
    K:     torch.Tensor,   # (N, 24, 24)
    table: dict,
    sym_indices: list = None,
) -> tuple:
    """
    Apply cube symmetries to (d, K) pairs to produce augmented data.

    Parameters
    ----------
    d           : (N, 8)      signed distances at nodes
    K           : (N, 24, 24) stiffness matrices
    table       : symmetry table from build_symmetry_table()
    sym_indices : which symmetries to apply (default: all 48)
                  pass [0] to skip augmentation (identity is symmetry 0)

    Returns
    -------
    d_aug : (N * len(sym_indices), 8)
    K_aug : (N * len(sym_indices), 24, 24)
    """
    if sym_indices is None:
        sym_indices = list(range(table["n_sym"]))

    node_perms = table["node_perms"]   # (48, 8)
    dof_perms  = table["dof_perms"]    # (48, 24)
    dof_signs  = table["dof_signs"]    # (48, 24)

    d_list, K_list = [], []

    for s in sym_indices:
        np_ = node_perms[s]                               # (8,)
        dp  = dof_perms[s]                                # (24,)
        ds  = torch.tensor(dof_signs[s], dtype=K.dtype)  # (24,)

        # Permute signed distances
        d_s = d[:, np_]                                   # (N, 8)

        # Permute and sign-correct stiffness matrix
        # K_aug[i,j] = sign[i] * sign[j] * K[perm[i], perm[j]]
        # First permute rows and columns, then apply sign correction
        K_s = K[:, dp, :][:, :, dp]                      # (N, 24, 24)
        K_s = K_s * ds[None, :, None] * ds[None, None, :]

        d_list.append(d_s)
        K_list.append(K_s)

    return torch.cat(d_list, dim=0), torch.cat(K_list, dim=0)


# ─────────────────────────────────────────────
#  5.  Validation
# ─────────────────────────────────────────────

def validate_symmetry_table(table: dict, tol: float = 1e-10) -> None:
    """
    Sanity checks on the symmetry table:
    1. Every permutation is a valid bijection on {0..7} / {0..23}
    2. Symmetries form a group: composition of any two is also in the table
    3. Every symmetry has an inverse in the table
    """
    node_perms = table["node_perms"]   # (48, 8)
    dof_perms  = table["dof_perms"]    # (48, 24)
    n = table["n_sym"]

    # 1. Each row is a permutation
    for s in range(n):
        assert set(node_perms[s]) == set(range(8)),  f"node perm {s} not bijective"
        assert set(dof_perms[s])  == set(range(24)), f"dof perm {s} not bijective"

    # 2. Group closure: composition of symmetries s1 ∘ s2 must be in the table
    perm_set = set(tuple(p) for p in node_perms)
    for s1 in range(n):
        for s2 in range(n):
            composed = node_perms[s1][node_perms[s2]]  # s1 ∘ s2
            assert tuple(composed) in perm_set, \
                f"symmetry group not closed for s1={s1}, s2={s2}"

    # 3. Every symmetry has an inverse
    for s in range(n):
        inv_perm = np.argsort(node_perms[s])
        assert tuple(inv_perm) in perm_set, f"no inverse for symmetry {s}"

    print(f"✓ All {n} symmetries validated: bijective, group closure, inverses exist")


def validate_augmentation(table: dict, tol: float = 1e-6) -> None:
    """
    Check that augmented K matrices are still SPD and have the same
    Frobenius norm as the original.
    """
    torch.manual_seed(42)
    N = 16

    # Build a valid SPD K via random Cholesky
    L   = torch.tril(torch.randn(N, 24, 24)) * 0.1
    L[:, range(24), range(24)] = torch.exp(L[:, range(24), range(24)])
    K   = torch.bmm(L, L.transpose(-1, -2))
    d   = torch.randn(N, 8)

    d_aug, K_aug = augment(d, K, table)

    # Check Frobenius norm preserved
    K_norm     = K.norm(dim=(-1, -2))                      # (N,)
    K_aug_norm = K_aug.norm(dim=(-1, -2)).reshape(48, N)   # (48, N)
    for s in range(48):
        err = (K_aug_norm[s] - K_norm).abs().max().item()
        assert err < tol, f"Frobenius norm changed under symmetry {s}: err={err:.2e}"

    # Check SPD preserved
    eigvals = torch.linalg.eigvalsh(K_aug)
    assert eigvals.min().item() > -tol, \
        f"SPD violated after augmentation: min eigenvalue = {eigvals.min():.2e}"

    # Check symmetry preserved
    sym_err = (K_aug - K_aug.transpose(-1, -2)).abs().max().item()
    assert sym_err < tol, f"Symmetry of K violated: err={sym_err:.2e}"

    print(f"✓ Augmentation validated: Frobenius norm, SPD, symmetry all preserved")
    print(f"  Input : {d.shape}, {K.shape}")
    print(f"  Output: {d_aug.shape}, {K_aug.shape}  ({48}× augmentation)")
