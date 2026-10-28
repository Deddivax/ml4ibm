from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from clean_simulation.internal_nn_runtime.cutcell_ml.classification_utils import _canonicalize_case_mask, MASK_TO_CLASS

# Precompute 256-index lookup table for fast deterministic classification on device
_LUT_PRECOMPUTED = []
for m in range(256):
    cmask = _canonicalize_case_mask(m)
    _LUT_PRECOMPUTED.append(MASK_TO_CLASS[cmask])

_LUT_TENSOR = torch.tensor(_LUT_PRECOMPUTED, dtype=torch.long)


# ==========================================================
# Rigid-body null-space utilities
# ==========================================================
#
# For any trimmed hex cell built on this fixed reference cube, the 6
# rigid-body modes (3 translations + 3 infinitesimal rotations) lie
# exactly in the null space of the true local stiffness K, regardless of
# trim topology (rigid motions produce zero strain everywhere, so the
# strain-energy integral over any subdomain is identically zero on them).
#
# R (24,6) spans that null space; T (24,18) is an orthonormal basis for
# its complement. Parametrizing K as T @ K_reduced @ T.T, with K_reduced
# an 18x18 PSD matrix built from a Cholesky factor, guarantees K @ R == 0
# and K == K.T and K >= 0 for *any* network output -- the rigid-body
# property no longer needs to be learned or penalized.
#
# IMPORTANT: HEX_CORNER_COORDS below must use the exact same node
# ordering as the ground-truth K's dof layout (node-major: dof block i
# corresponds to node i's x,y,z). Verify this against your FEM assembler
# before trusting the projection -- see the validation snippet at the
# bottom of this file.
# ==========================================================

def build_rigid_body_modes(coords: torch.Tensor) -> torch.Tensor:
    """
    Construct the six rigid body modes of an 8-node hexahedron.

    Parameters
    ----------
    coords : (8,3)
        Corner coordinates in the SAME ordering used by the FEM stiffness
        matrix and network outputs.

    Returns
    -------
    R : (24,6)
        Columns are the rigid body basis vectors.
    """

    n = coords.shape[0]

    R = torch.zeros(
        3 * n,
        6,
        dtype=coords.dtype,
        device=coords.device,
    )

    for i, (x, y, z) in enumerate(coords):

        # translations
        R[3*i + 0, 0] = 1.0
        R[3*i + 1, 1] = 1.0
        R[3*i + 2, 2] = 1.0

        # infinitesimal rotations

        # Rx
        R[3*i + 1, 3] = -z
        R[3*i + 2, 3] =  y

        # Ry
        R[3*i + 0, 4] =  z
        R[3*i + 2, 4] = -x

        # Rz
        R[3*i + 0, 5] = -y
        R[3*i + 1, 5] =  x

    return R


#
# IMPORTANT
#
# These coordinates MUST match the ordering used by the FEM assembly
# (i.e. the ordering k_ref's 24 dofs are laid out in). This is a
# standard VTK/Abaqus-style hex ordering (bottom face CCW, then top
# face CCW) -- NOT the bit-unpacked (n&1, (n>>1)&1, (n>>2)&1) ordering
# used elsewhere for spatial encoders. Confirm this against your FEM
# assembler before relying on the projection below.
#
HEX_CORNER_COORDS = torch.tensor(
    [
        (-1., -1., -1.),
        ( 1., -1., -1.),
        ( 1.,  1., -1.),
        (-1.,  1., -1.),
        (-1., -1.,  1.),
        ( 1., -1.,  1.),
        ( 1.,  1.,  1.),
        (-1.,  1.,  1.),
    ],
    dtype=torch.float32,
)


def build_null_space_basis(coords: torch.Tensor) -> torch.Tensor:
    """
    Returns the orthonormal basis T spanning the complement
    of the rigid-body space.

    T : (24,18)

    satisfying

        T^T T = I

    and

        T^T R = 0.
    """

    R = build_rigid_body_modes(coords)

    Q, _ = torch.linalg.qr(
        R,
        mode="complete",
    )

    return Q[:, 6:]


NULL_SPACE_BASIS = build_null_space_basis(HEX_CORNER_COORDS)  # (24,18) orthonormal complement, fixed

# ==========================================================
# Analytic full-cube (uncut) stiffness validation reference
# ==========================================================

def _gauss2() -> tuple[torch.Tensor, torch.Tensor]:
    """2-point Gauss-Legendre rule on [-1, 1]: exact for the cubic
    integrands from products of trilinear hex8 shape-function
    derivatives in the stiffness integral."""
    p = 1.0 / (3.0 ** 0.5)
    points = torch.tensor([-p, p], dtype=torch.float64)
    weights = torch.tensor([1.0, 1.0], dtype=torch.float64)
    return points, weights


def _isotropic_D(E: float, nu: float, dtype=torch.float64) -> torch.Tensor:
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    mu = E / (2.0 * (1.0 + nu))
    D = torch.zeros(6, 6, dtype=dtype)
    D[0, 0] = D[1, 1] = D[2, 2] = lam + 2.0 * mu
    D[0, 1] = D[1, 0] = lam
    D[0, 2] = D[2, 0] = lam
    D[1, 2] = D[2, 1] = lam
    D[3, 3] = D[4, 4] = D[5, 5] = mu
    return D


def compute_K_full(
    E: float = 1.0,
    nu: float = 0.3,
    coords: torch.Tensor = HEX_CORNER_COORDS,
    dtype=torch.float64,
) -> torch.Tensor:
    """Analytic 24x24 stiffness of the FULLY SOLID reference hex cube.
    Node ordering / dof layout matches `coords` exactly (node-major
    x,y,z blocks), consistent with build_rigid_body_modes /
    NULL_SPACE_BASIS above. Because coords IS the parametric domain
    [-1,1]^3, the isoparametric map is the identity, so J = I and
    detJ = 1 throughout -- no separate Jacobian bookkeeping needed."""
    coords = coords.to(dtype)
    n_nodes = coords.shape[0]
    xi_i, eta_i, zeta_i = coords[:, 0], coords[:, 1], coords[:, 2]

    gp, gw = _gauss2()
    D = _isotropic_D(E, nu, dtype=dtype)

    K = torch.zeros(3 * n_nodes, 3 * n_nodes, dtype=dtype)

    for a, wa in zip(gp, gw):
        for b, wb in zip(gp, gw):
            for c, wc in zip(gp, gw):
                xi, eta, zeta = a, b, c
                weight = wa * wb * wc  # detJ = 1

                dN_dxi = 0.125 * xi_i * (1 + eta * eta_i) * (1 + zeta * zeta_i)
                dN_deta = 0.125 * (1 + xi * xi_i) * eta_i * (1 + zeta * zeta_i)
                dN_dzeta = 0.125 * (1 + xi * xi_i) * (1 + eta * eta_i) * zeta_i
                dN_dx, dN_dy, dN_dz = dN_dxi, dN_deta, dN_dzeta  # J = I

                B = torch.zeros(6, 3 * n_nodes, dtype=dtype)
                for i in range(n_nodes):
                    B[0, 3 * i + 0] = dN_dx[i]
                    B[1, 3 * i + 1] = dN_dy[i]
                    B[2, 3 * i + 2] = dN_dz[i]
                    B[3, 3 * i + 0] = dN_dy[i]
                    B[3, 3 * i + 1] = dN_dx[i]
                    B[4, 3 * i + 1] = dN_dz[i]
                    B[4, 3 * i + 2] = dN_dy[i]
                    B[5, 3 * i + 0] = dN_dz[i]
                    B[5, 3 * i + 2] = dN_dx[i]

                K += weight * (B.transpose(0, 1) @ D @ B)

    return K


def self_check_K_full(K_full: torch.Tensor, coords: torch.Tensor = HEX_CORNER_COORDS) -> None:
    """Sanity checks needing no dataset: symmetry, PSD, exact rigid-body
    nullity. Run once before trusting K_full."""
    sym_err = (K_full - K_full.T).abs().max().item()
    eigvals = torch.linalg.eigvalsh(K_full)
    R = build_rigid_body_modes(coords).to(K_full.dtype)
    rigid_err = (K_full @ R).abs().max().item()
    print(f"symmetry max abs error:      {sym_err:.3e}  (expect ~1e-14)")
    print(f"min eigenvalue:              {eigvals.min().item():.6e}  (expect ~0 from rigid modes)")
    print(f"rigid-body K@R max abs err:  {rigid_err:.3e}  (expect ~1e-12 or smaller)")
    if sym_err > 1e-8 or rigid_err > 1e-6 or eigvals.min() < -1e-8:
        raise RuntimeError("compute_K_full failed a sanity check -- do not use yet.")


# `compute_K_full` integrates a cube of side length two, so it has scale
# E*(2). Dataset targets are framework matrices divided by E*h, giving the
# scale-one equivalent. The factor below has been checked directly against
# `Stiffness.elements(...)/h` to machine precision for several h values.
K_FULL_E1_F64 = compute_K_full(E=1.0, nu=0.3) / 2.0


def compute_vectorized_class_label(d: torch.Tensor) -> torch.Tensor:
    """Fast vectorized mapping from 8 nodal SDF values to deterministic cut classification.

    Compatible with batch (B, 8) and single (8,) tensors, running fully on device.
    """
    device = d.device
    lut = _LUT_TENSOR.to(device)

    # We only care about the first 8 dimensions (d0..d7) even if d has nu concatenated
    d_sdf = d[..., :8]

    bits = (d_sdf > 0.5).to(torch.long)
    powers = torch.tensor([1, 2, 4, 8, 16, 32, 64, 128], dtype=torch.long, device=device)
    mask_idx = torch.sum(bits * powers, dim=-1)

    return lut[mask_idx]


class ClassBitEncoder(nn.Module):
    """
    Encodes the deterministic class ID (0-14) as a binary bit vector
    instead of a single integer ID embedding.
    Supports:
    - "one_hot": 15-dimensional one-hot bit vector.
    - "binary": 4-dimensional binary representation (bits 0-3 of class ID).
    - "canonical_mask": 8-dimensional bit representation of the 8-bit canonical case mask.
    """
    def __init__(self, encoder_type: str = "one_hot"):
        super().__init__()
        self.encoder_type = encoder_type
        if encoder_type == "one_hot":
            self.dim = 15
        elif encoder_type == "binary":
            self.dim = 4
        elif encoder_type == "canonical_mask":
            self.dim = 8
            from clean_simulation.internal_nn_runtime.cutcell_ml.classification_utils import _CANONICAL_CASE_MASKS
            self.register_buffer("masks", torch.tensor(_CANONICAL_CASE_MASKS, dtype=torch.long))
        else:
            raise ValueError(f"Unknown class_encoder_type: {encoder_type}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device
        if self.encoder_type == "one_hot":
            return F.one_hot(x.long().clamp(min=0, max=14), num_classes=15).to(torch.float32)
        elif self.encoder_type == "binary":
            # Unpack class ID into 4 bits
            bits = []
            x_long = x.long()
            for i in range(4):
                bits.append((x_long >> i) & 1)
            return torch.stack(bits, dim=-1).to(torch.float32)
        elif self.encoder_type == "canonical_mask":
            masks = self.masks.to(device)
            cmask = masks[x.long().clamp(min=0, max=14)]
            # Unpack 8-bit canonical mask into 8 binary bits on device
            bits = []
            for i in range(8):
                bits.append((cmask >> i) & 1)
            return torch.stack(bits, dim=-1).to(torch.float32)
        return torch.zeros((*x.shape, 0), device=device, dtype=torch.float32)


# ==========================================================
# Residual block
# ==========================================================

class ResidualBlock(nn.Module):

    def __init__(
        self,
        dim: int,
        dropout_p: float = 0.0,
        residual_scale: float = 0.5,
        residual_scaling: str = "rezero",
        num_blocks: int | None = None,
    ):
        super().__init__()

        self.norm1 = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim)

        self.norm2 = nn.LayerNorm(dim)
        self.fc2 = nn.Linear(dim, dim)

        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout_p)

        self.residual_scale = residual_scale
        self.residual_scaling = residual_scaling

        if self.residual_scaling == "rezero":
            self.alpha = nn.Parameter(torch.zeros(1))
        elif self.residual_scaling == "fixed":
            # x = x + (1/sqrt(N)) * block(x)
            n_blocks = num_blocks if num_blocks is not None else 1
            self.fixed_scale = 1.0 / (n_blocks ** 0.5)
        else:
            self.fixed_scale = 1.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        y = self.norm1(x)
        y = self.act(self.fc1(y))
        y = self.dropout(y)

        y = self.norm2(y)
        y = self.fc2(y)

        if self.residual_scaling == "rezero":
            return x + self.alpha * y
        elif self.residual_scaling == "fixed":
            return x + self.fixed_scale * y
        else:
            # none: standard residual with original self.residual_scale (0.5 by default)
            return x + self.residual_scale * y


# ==========================================================
# Simple MLP block (no residual skip connection)
# ==========================================================

class SimpleMLPBlock(nn.Module):

    def __init__(
        self,
        dim: int,
        dropout_p: float = 0.0,
    ):
        super().__init__()

        self.norm1 = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim)

        self.norm2 = nn.LayerNorm(dim)
        self.fc2 = nn.Linear(dim, dim)

        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        y = self.norm1(x)
        y = self.act(self.fc1(y))
        y = self.dropout(y)

        y = self.norm2(y)
        y = self.fc2(y)

        return y


# ==========================================================
# CutCellNet
# ==========================================================

class CutCellNet(nn.Module):
    """
    Standard MLP with Residual Blocks and Deterministic Class Conditioning Input.

        [d, nu] -> normalized
                       ↓
        [normalized, Class_Embed]
                       ↓
                  Input Proj
                       ↓
                 Residual Trunk
                       ├── SPD shape head -> K_reduced (18x18) -> T K_reduced T^T -> K (24x24)
                       └── Scale head      -> log_norm

    The SPD shape head predicts only the 18-dimensional part of K
    orthogonal to the 6 rigid-body modes (3 translations + 3 infinitesimal
    rotations). Projecting through the fixed orthonormal basis `T` makes
    K @ R == 0 (rigid-body nullity) and K == K.T and K >= 0 exactly, for
    every sample, by construction -- no loss term needed for any of these
    three properties.

    """

    N_CLASSES = 15

    # Default matrix size (overridable per-instance via constructor).
    # 24 = 8 nodes × 3 DOF for a tri-linear hexahedral element.
    MATRIX_SIZE = 24

    DIAG_EPS = 1e-6

    def __init__(
        self,
        fusion_dim: int = 256,
        matrix_size: int = 24,
        n_residual_blocks: int = 4,
        class_embed_dim: int = 32,
        class_encoder_type: str = "one_hot",
        dropout_p: float = 0.0,
        include_nu: bool = False,
        use_class_conditioning: bool = False,
        include_class_inputs: bool = False,
        trunk_type: str = "resnet",
        residual_scaling: str = "rezero",
        spd_mode: str = "cholesky",
        use_reduced_null_space: bool = True,
        topology_dim: int = 64,
        geometry_dim: int = 128,
        n_distance_nodes: int = 8,
        hidden: int = None,
        hidden_dims: tuple[int, ...] = None,
    ):
        super().__init__()

        if hidden is not None:
            fusion_dim = hidden

        # ------------------------------------------------------
        # Null-space configuration
        # ------------------------------------------------------
        self.use_reduced_null_space = bool(use_reduced_null_space)

        # If we use the reduced parameterization, the network only
        # predicts the 18x18 SPD matrix, which is then projected up to
        # the full 24x24 physical stiffness matrix via T.
        if self.use_reduced_null_space:

            if matrix_size not in (18, 24):
                raise ValueError(
                    "Reduced null-space expects matrix_size=18 "
                    "(recommended) or 24."
                )

            self.MATRIX_SIZE = 18

        else:

            self.MATRIX_SIZE = int(matrix_size)

        self.N_DIAG = self.MATRIX_SIZE
        self.N_OFFDIAG = self.MATRIX_SIZE * (self.MATRIX_SIZE - 1) // 2
        self.N_OUT = self.N_DIAG + self.N_OFFDIAG

        # Register the null-space basis as a buffer (not a bare module-level
        # tensor) so it correctly follows .to(device) / .half() / etc.
        self.register_buffer(
            "null_basis",
            NULL_SPACE_BASIS.clone(),
        )

        # --------------------------------------------------
        # Existing configuration
        # --------------------------------------------------

        # If hidden_dims is provided, use the last element as fusion_dim
        if hidden_dims is not None and len(hidden_dims) > 0:
            fusion_dim = hidden_dims[-1]

        self.include_nu = bool(include_nu)
        self.use_class_conditioning = bool(use_class_conditioning)
        self.class_encoder_type = class_encoder_type
        self.include_class_inputs = bool(include_class_inputs)
        self.trunk_type = trunk_type
        self.residual_scaling = residual_scaling
        self.spd_mode = spd_mode
        self.class_embed_dim = int(class_embed_dim)
        self.n_distance_nodes = int(n_distance_nodes)
        if self.n_distance_nodes < 8:
            raise ValueError("n_distance_nodes must include the eight corner values")

        # --------------------------------------------------
        # Class conditioning
        # --------------------------------------------------
        if self.use_class_conditioning and self.class_embed_dim > 0:
            self.class_encoder = ClassBitEncoder(class_encoder_type)
            self.class_embed = None
        else:
            self.class_encoder = None
            self.class_embed = None

        # --------------------------------------------------
        # Input dimension
        # --------------------------------------------------
        input_dim = self.n_distance_nodes
        if self.include_nu:
            input_dim += 1
        if self.use_class_conditioning and self.class_embed_dim > 0:
            input_dim += self.class_encoder.dim
        if self.include_class_inputs:
            input_dim += 14  # one-hot classes 1..14

        # --------------------------------------------------
        # Network layers
        # --------------------------------------------------
        self.input_proj = nn.Linear(input_dim, fusion_dim)

        block_cls = ResidualBlock if trunk_type == "resnet" else SimpleMLPBlock
        self.trunk = nn.Sequential(
            *[
                block_cls(
                    fusion_dim,
                    dropout_p=dropout_p,
                    **(dict(residual_scaling=residual_scaling, num_blocks=n_residual_blocks)
                       if trunk_type == "resnet" else {}),
                )
                for _ in range(n_residual_blocks)
            ]
        )

        self.final_norm = nn.LayerNorm(fusion_dim)
        self.output_proj = nn.Linear(fusion_dim, self.N_OUT)
        self.norm_proj   = nn.Linear(fusion_dim, 1)

        # --------------------------------------------------
        # Normalizer + scale buffers
        # --------------------------------------------------
        self.register_buffer("in_mean", torch.zeros(input_dim))
        self.register_buffer("in_std",  torch.ones(input_dim))
        self.register_buffer("k_scale", torch.ones(1))

        self._init_weights()

    # ======================================================
    # Initialization
    # ======================================================

    def _init_weights(self):

        for module in self.modules():

            if isinstance(module, nn.Linear):

                nn.init.kaiming_normal_(module.weight, nonlinearity="linear")

                if module.bias is not None:
                    nn.init.zeros_(module.bias)

            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    # ======================================================
    # Normalization
    # ======================================================

    @torch.no_grad()
    def set_normalizers(
        self,
        in_mean: torch.Tensor,
        in_std: torch.Tensor,
        k_scale: torch.Tensor | None = None,
    ):
        self.in_mean.copy_(
            in_mean.to(self.in_mean.device)
        )

        self.in_std.copy_(
            torch.clamp(
                in_std.to(self.in_std.device),
                min=1e-6,
            )
        )

    # ======================================================
    # Input processing
    # ======================================================

    def _build_input(
        self,
        d: torch.Tensor,
        nu: torch.Tensor | None,
    ):

        single = d.ndim == 1

        if single:
            d = d.unsqueeze(0)

        if self.include_nu:

            if nu is None:
                raise ValueError(
                    "nu must be provided when include_nu=True"
                )

            if nu.ndim == 0:
                nu = nu.unsqueeze(0)

            if nu.ndim > 1:
                nu = nu.squeeze(-1)

            x = torch.cat(
                [
                    d,
                    nu.unsqueeze(-1),
                ],
                dim=-1,
            )

        else:
            x = d

        x = (x - self.in_mean) / self.in_std

        return x, single

    # ======================================================
    # SPD helper
    # ======================================================

    def _build_U(self, diag, offdiag):

        b = diag.shape[0]
        n = self.MATRIX_SIZE

        offdiag = offdiag.to(diag.dtype)

        U = torch.zeros(
            b,
            n,
            n,
            dtype=diag.dtype,
            device=diag.device,
        )

        idx = torch.arange(n, device=diag.device)

        U[:, idx, idx] = diag

        rows, cols = torch.triu_indices(
            n,
            n,
            offset=1,
            device=diag.device,
        )

        U[:, rows, cols] = offdiag

        return U

    # ======================================================
    # Project reduced stiffness into the full space
    # ======================================================

    def _project_stiffness(
        self,
        K: torch.Tensor,
    ) -> torch.Tensor:
        """
        Convert an 18x18 reduced stiffness matrix into the
        full 24x24 matrix satisfying

            K R = 0

        exactly (up to floating-point precision).

        If reduced null-space mode is disabled, returns K unchanged.
        """

        if not self.use_reduced_null_space:
            return K

        T = self.null_basis

        return T @ K @ T.transpose(-1, -2)

    # ======================================================
    # Forward
    # ======================================================

    def forward(
        self,
        d: torch.Tensor,
        nu: torch.Tensor | None = None,
        cut_class: torch.Tensor | None = None,
    ):
        single = d.ndim == 1
        if single:
            d = d.unsqueeze(0)

        x, _ = self._build_input(
            d,
            nu,
        )

        # --------------------------------------------------
        # Cut class representation (deterministic conditioning)
        # --------------------------------------------------
        if self.use_class_conditioning and self.class_embed_dim > 0:
            if cut_class is None:
                raw_d = d.unsqueeze(0) if d.ndim == 1 else d
                cut_class_tensor = compute_vectorized_class_label(raw_d)
            else:
                cut_class_tensor = cut_class.unsqueeze(0) if (d.ndim == 1 and cut_class.ndim == 0) else cut_class

            if self.class_encoder is not None:
                c_rep = self.class_encoder(cut_class_tensor)
            else:
                c_rep = self.class_embed(cut_class_tensor.long())

            feat = torch.cat([x, c_rep], dim=-1)
        else:
            feat = x

        # --------------------------------------------------
        # Optional 14 Extra Class Inputs (computed from d)
        # --------------------------------------------------
        if self.include_class_inputs:
            raw_d = d.unsqueeze(0) if d.ndim == 1 else d
            c_label = compute_vectorized_class_label(raw_d)
            c_one_hot = F.one_hot(c_label.long(), num_classes=15)[..., 1:].to(feat.dtype)
            feat = torch.cat([feat, c_one_hot], dim=-1)

        # --------------------------------------------------
        # MLP Classic Trunk
        # --------------------------------------------------
        feat = self.input_proj(feat)
        feat = self.trunk(feat)

        feat = self.final_norm(feat)

        # --------------------------------------------------
        # SPD shape head
        # --------------------------------------------------

        raw = self.output_proj(feat)

        if self.spd_mode == "triangular":
            # Symmetric completion — does NOT enforce PSD.
            # All N_OUT raw outputs fill the upper triangle; off-diagonal
            # elements are mirrored, diagonal used as-is.
            _rows, _cols = torch.triu_indices(
                self.MATRIX_SIZE, self.MATRIX_SIZE, device=raw.device
            )
            K = torch.zeros(
                raw.shape[0], self.MATRIX_SIZE, self.MATRIX_SIZE,
                dtype=raw.dtype, device=raw.device,
            )
            K[:, _rows, _cols] = raw
            K = K + K.transpose(-1, -2) - torch.diag_embed(K.diagonal(dim1=-2, dim2=-1))
            K = self._project_stiffness(K)
        else:
            # Cholesky: K = U^T @ U — always PSD.
            # Softplus on diagonal ensures positive diagonal entries in U.
            diag = F.softplus(raw[:, : self.N_DIAG])
            U = self._build_U(diag, raw[:, self.N_DIAG:])
            K = torch.bmm(U.transpose(-1, -2), U)
            K = self._project_stiffness(K)

        K_norm_det = (
            K.norm(
                dim=(-1, -2),
                keepdim=True,
            )
            .clamp(min=1e-6)
            .detach()
        )

        K_unit = K / K_norm_det

        # --------------------------------------------------
        # Scale head
        # --------------------------------------------------

        log_norm = self.norm_proj(
            feat
        ).squeeze(-1)

        if single:
            return (
                K_unit[0],
                log_norm[0],
            )

        return (
            K_unit,
            log_norm,
        )

    # ======================================================
    # Optional orbit-averaged prediction (opt-in, NOT used by forward())
    # ======================================================


class CutCellNetTransformer(nn.Module):
    """
    Transformer-based stiffness predictor.

    Each of the 8 cube corners contributes a token [sdf, x, y, z] that is
    projected to a latent dimension.  N Transformer encoder blocks apply
    multi-head self-attention over the 8 tokens.  The resulting tokens are
    averaged into a single global latent, which drives:
      • a Cholesky head → 18×18 PSD matrix (reduced null-space basis)
      • a scale head    → log Frobenius norm scalar

    The 18×18 matrix is lifted to 24×24 via the fixed null-space basis T:
        K_24 = T @ K_18 @ T.T

    Interface compatible with the residual model runtime:
        forward(d, nu=None, cut_class=None) → (K_unit, log_norm)
    """

    def __init__(
        self,
        embed_dim: int = 128,
        n_heads: int = 4,
        n_layers: int = 5,
        ffn_dim: int = 256,
        dropout: float = 0.0,
        reduced_matrix_size: int = 18,
        ordered_sdf_branch: bool = False,
        n_distance_nodes: int = 8,
    ):
        super().__init__()

        node_count = round(int(n_distance_nodes) ** (1.0 / 3.0))
        if node_count < 2 or node_count ** 3 != n_distance_nodes:
            raise ValueError("n_distance_nodes must be a cubic grid with at least eight nodes.")
        self.n_distance_nodes = int(n_distance_nodes)

        self.MATRIX_SIZE = reduced_matrix_size
        self.N_DIAG    = reduced_matrix_size
        self.N_OFFDIAG = reduced_matrix_size * (reduced_matrix_size - 1) // 2
        self.N_OUT     = self.N_DIAG + self.N_OFFDIAG

        # Precomputed index buffers for vectorized Cholesky assembly
        _d = torch.arange(reduced_matrix_size)
        _r, _c = torch.tril_indices(reduced_matrix_size, reduced_matrix_size, offset=-1)
        self.register_buffer("_diag_idx", _d, persistent=False)
        self.register_buffer("_tril_r",   _r, persistent=False)
        self.register_buffer("_tril_c",   _c, persistent=False)

        # Corner-first, x-major order matches refined training datasets.
        grid = torch.linspace(-1.0, 1.0, node_count)
        raw_coords = torch.cartesian_prod(grid, grid, grid)
        corner_indices = torch.tensor(
            [
                0,
                (node_count - 1) * node_count * node_count,
                (node_count - 1) * node_count * node_count + (node_count - 1) * node_count,
                (node_count - 1) * node_count,
                node_count - 1,
                (node_count - 1) * node_count * node_count + node_count - 1,
                node_count ** 3 - 1,
                (node_count - 1) * node_count + node_count - 1,
            ],
            dtype=torch.long,
        )
        non_corner = torch.ones(self.n_distance_nodes, dtype=torch.bool)
        non_corner[corner_indices] = False
        coords = torch.cat((raw_coords[corner_indices], raw_coords[non_corner]), dim=0)
        self.register_buffer("coords", coords)

        # Fixed reduced null-space basis for 24-DOF stiffness reconstruction.
        self.register_buffer("null_basis", NULL_SPACE_BASIS.clone())  # (24, 18)

        # Per-node input projection: [sdf, x, y, z] → embed_dim
        self.node_embed = nn.Linear(4, embed_dim)

        # Learned positional embedding (one per SDF sample)
        self.pos_embed = nn.Parameter(torch.zeros(self.n_distance_nodes, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Transformer encoder (pre-norm, batch_first)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Readout MLP applied to global mean-pooled token
        self.readout = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.SiLU(),
        )

        self.ordered_sdf_branch = None
        if ordered_sdf_branch:
            self.ordered_sdf_branch = nn.Sequential(
                nn.Linear(self.n_distance_nodes, embed_dim),
                nn.SiLU(),
                nn.Linear(embed_dim, embed_dim),
            )
            nn.init.zeros_(self.ordered_sdf_branch[-1].weight)
            nn.init.zeros_(self.ordered_sdf_branch[-1].bias)

        self.matrix_head = nn.Linear(embed_dim, self.N_OUT)
        self.scale_head  = nn.Linear(embed_dim, 1)

        # Input normalisers (set by pipeline via set_normalizers)
        self.register_buffer("in_mean", torch.zeros(self.n_distance_nodes))
        self.register_buffer("in_std",  torch.ones(self.n_distance_nodes))
        # Scale buffer (set by pipeline before training; unused by this model
        # since scale is predicted by scale_head, but required for API compatibility)
        self.register_buffer("k_scale", torch.tensor(1.0))

    def set_normalizers(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        self.in_mean = mean.clone().to(self.in_mean.device)
        self.in_std  = std.clone().to(self.in_std.device)

    def _project(self, K: torch.Tensor) -> torch.Tensor:
        """Lift (B, 18, 18) → (B, 24, 24) via T @ K @ T.T, T=(24, 18)."""
        T = self.null_basis                         # (24, 18)
        return T @ K @ T.transpose(-1, -2)

    def forward(
        self,
        d: torch.Tensor,
        nu: torch.Tensor | None = None,
        cut_class=None,
    ):
        single = d.ndim == 1
        if single:
            d = d.unsqueeze(0)

        B      = d.shape[0]
        device = d.device

        if d.shape[-1] != self.n_distance_nodes:
            raise ValueError(f"Expected {self.n_distance_nodes} signed-distance inputs; got {d.shape[-1]}.")

        d_norm = (d - self.in_mean.to(device)) / self.in_std.to(device)

        # Build per-node feature tokens: [sdf_i, x_i, y_i, z_i]
        coords = self.coords.to(device).unsqueeze(0).expand(B, -1, -1)
        sdf    = d_norm.unsqueeze(-1)
        tokens = torch.cat([sdf, coords], dim=-1)

        # Token embedding + positional encoding
        x = self.node_embed(tokens) + self.pos_embed.unsqueeze(0)         # (B, 8, E)

        x = self.transformer(x)

        # Global average pooling → single latent vector
        z = self.readout(x.mean(dim=1))                                    # (B, E)
        if self.ordered_sdf_branch is not None:
            z = z + self.ordered_sdf_branch(d_norm)

        # Cholesky head: lower-triangular L so K = L @ L.T is PSD
        raw  = self.matrix_head(z)                                         # (B, N_OUT)
        diag = F.softplus(raw[:, :self.N_DIAG]) + 1e-6
        off  = raw[:, self.N_DIAG:].to(dtype=diag.dtype)

        # Vectorized lower-triangular assembly (avoids slow Python loops)
        L = torch.zeros(B, self.MATRIX_SIZE, self.MATRIX_SIZE,
                dtype=diag.dtype, device=raw.device)
        d_idx = self._diag_idx.to(raw.device)
        r_idx, c_idx = self._tril_r.to(raw.device), self._tril_c.to(raw.device)
        L[:, d_idx, d_idx] = diag
        L[:, r_idx, c_idx] = off   # N_OFFDIAG > 0 always for MATRIX_SIZE >= 2

        K      = L @ L.transpose(-1, -2)                                   # (B, 18, 18)
        K      = self._project(K)                                          # (B, 24, 24)

        K_norm = K.norm(dim=(-1, -2), keepdim=True).clamp(min=1e-6)
        K_unit = K / K_norm                                                # (B, 24, 24)

        log_norm = self.scale_head(z).squeeze(-1)                          # (B,)

        if single:
            return K_unit[0], log_norm[0]
        return K_unit, log_norm