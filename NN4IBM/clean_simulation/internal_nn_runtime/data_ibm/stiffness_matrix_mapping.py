from __future__ import annotations

"""Generate and append local hexahedral stiffness matrices from cut-mask inputs.

Overview
--------
This script builds a compact HDF5 dataset of local stiffness matrices K_e for
single hexahedral cells. Each input row contains 8 nodal SDF-like values
(d0..d7). Those values define the cell topology and integration domain.

Key workflow
------------
1) Reorder d-values to match the FEM kernel node order.
2) Convert nodal values into a raw 8-bit cube occupancy mask.
3) Canonicalize that mask under cube rotations and inside/outside complement.
	This collapses 256 raw masks to 15 canonical cube classes.
4) Use trilinear nodal SDF directly for octree-based cut-cell integration.
	No marching-cubes mesh reconstruction is used.
5) Assemble K_e from B-matrices and material tensor C(nu), then store only the
	upper-triangular part for compact append-only persistence.

Storage contract
----------------
- HDF5 group naming: depth_{d}_integration_{g}_poly_{p}
- Datasets per group: flat_index, d_values, stiffness_ut
- The script only appends missing indices; existing rows are never rewritten.
"""

import argparse
import csv
import itertools
import shutil
import sys
from functools import lru_cache
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = REPO_ROOT.parent
if str(WORKSPACE_ROOT) not in sys.path:
	sys.path.insert(0, str(WORKSPACE_ROOT))

from clean_simulation.internal_fem_kernel.constants import Material
from clean_simulation.internal_fem_kernel.geometry import HexGrid
from clean_simulation.internal_fem_kernel.integration.gauss_integration import Gauss
from clean_simulation.internal_fem_kernel.integration.octreesampler import OctreeSampler
from clean_simulation.internal_fem_kernel.kinematics import Kinematics
from clean_simulation.internal_fem_kernel.shape import Shape
try:
	from .signed_distance_generator import generate_mask_csv, nodes as CUT_MASK_NODES
except ImportError:
	from signed_distance_generator import generate_mask_csv, nodes as CUT_MASK_NODES


N_HEX_NODES = 8
LOCAL_DOF = 24
UT_SIZE = 300


_REF_HEX_CENTER = np.asarray([0.5, 0.5, 0.5], dtype=np.float64)
_REF_HEX_COORDS = np.asarray(
	HexGrid.cell_nodes_from_centers(_REF_HEX_CENTER, 1.0, 1.0, 1.0, poly_order=1),
	dtype=np.float64,
)
_REF_HEX_COORDS_T = torch.as_tensor(_REF_HEX_COORDS, dtype=torch.float64).unsqueeze(0)


@lru_cache(maxsize=8)
def _gauss_rule_3d(order: int) -> tuple[torch.Tensor, torch.Tensor]:
	pts_1d, w_1d = Gauss.gauss_1d(int(order), 'cpu')
	xi_g = torch.stack(
		torch.meshgrid(pts_1d, pts_1d, pts_1d, indexing='ij'), dim=-1
	).reshape(-1, 3)
	w_g = (w_1d[:, None, None] * w_1d[None, :, None] * w_1d[None, None, :]).reshape(-1)
	w_g = w_g * (8.0 / w_g.sum())
	return xi_g, w_g


@lru_cache(maxsize=16)
def _material_c(nu: float) -> np.ndarray:
	return Material(E=1.0, nu=float(nu)).C


@lru_cache(maxsize=16)
def _material_c_torch(nu: float) -> torch.Tensor:
	return torch.as_tensor(_material_c(float(nu)), dtype=torch.float64)


# Import canonical case and classification utilities from split-out modular classifier file
from clean_simulation.internal_nn_runtime.cutcell_ml.classification_utils import (
    _canonicalize_case_mask,
    _CANONICAL_CASE_MASKS,
    MASK_TO_CLASS,
    classify_cube_configuration,
    compute_class_label_from_d,
)


def _compute_cut_mask_to_basic_order() -> np.ndarray:
	basic_nodes_local = HexGrid._local_node_offsets(2.0, 2.0, 2.0, poly_order=1)
	basic_nodes_01 = tuple(tuple(int(v) for v in row) for row in ((basic_nodes_local + 1.0) * 0.5))
	cut_nodes_01 = tuple(tuple(int(v) for v in row) for row in np.asarray(CUT_MASK_NODES, dtype=np.int64))
	cut_index_by_node = {node: i for i, node in enumerate(cut_nodes_01)}
	return np.asarray([cut_index_by_node[node] for node in basic_nodes_01], dtype=np.int64)


CUT_MASK_TO_BASIC_ORDER = _compute_cut_mask_to_basic_order()


@dataclass(frozen=True)
class PipelineParams:
	depth: int
	integration_order: int
	poly_order: int
	poisson_ratio: float
	refined_subcells: int = 1

	@property
	def group_name(self) -> str:
		return f"depth_{self.depth}_integration_{self.integration_order}_poly_{self.poly_order}"

	@property
	def n_distance_nodes(self) -> int:
		return (self.refined_subcells + 1) ** 3


class TrilinearNodalSDFVolume:
	"""SDF proxy from nodal values on one hexahedral cell."""

	def __init__(self, node_coords: np.ndarray, nodal_sdf_basic_order: np.ndarray):
		self.node_coords = np.asarray(node_coords, dtype=np.float64).reshape(N_HEX_NODES, 3)
		self.nodal_sdf = np.asarray(nodal_sdf_basic_order, dtype=np.float64).reshape(N_HEX_NODES)

		xyz_min = self.node_coords.min(axis=0)
		xyz_max = self.node_coords.max(axis=0)
		self.xyz_min = xyz_min
		self.scale = 2.0 / np.maximum(xyz_max - xyz_min, 1e-15)

	def signed_distance_torch(self, points: torch.Tensor) -> torch.Tensor:
		xi_eta_zeta = (points.to(dtype=torch.float64) - torch.as_tensor(self.xyz_min, dtype=torch.float64, device=points.device))
		xi_eta_zeta = xi_eta_zeta * torch.as_tensor(self.scale, dtype=torch.float64, device=points.device) - 1.0
		N = Shape.N3(xi_eta_zeta[:, 0], xi_eta_zeta[:, 1], xi_eta_zeta[:, 2], poly_order=1)
		phi = N @ torch.as_tensor(self.nodal_sdf, dtype=torch.float64, device=points.device)
		return phi


def iter_cut_mask_rows(cut_mask_csv: Path, valid_only: bool = True) -> Iterable[tuple[int, np.ndarray]]:
	with cut_mask_csv.open("r", newline="") as handle:
		reader = csv.DictReader(handle)
		for row in reader:
			if valid_only and int(row.get("valid", "1")) != 1:
				continue

			flat_index = int(row["index"])
			d_vals = np.array([float(row[f"d{i}"]) for i in range(8)], dtype=np.float64)
			yield flat_index, d_vals


def refine_nodal_distances(d_values_csv_order: np.ndarray, refined_subcells: int) -> np.ndarray:
	"""Sample the trilinear corner field on a regular subcell-node grid.

	The original eight CSV-order corners are retained first so topology labels
	remain independent of the selected refinement.
	"""
	if refined_subcells < 1:
		raise ValueError("refined_subcells must be at least 1")
	d_values = np.asarray(d_values_csv_order, dtype=np.float64).reshape(N_HEX_NODES)
	if refined_subcells == 1:
		return d_values.copy()

	corner_coords = np.asarray(CUT_MASK_NODES, dtype=np.float64)
	grid = np.linspace(0.0, 1.0, refined_subcells + 1)
	refined_values = [d_values]
	for point in itertools.product(grid, repeat=3):
		if all(value in (0.0, 1.0) for value in point):
			continue
		weights = np.prod(np.where(corner_coords == 1.0, point, 1.0 - np.asarray(point)), axis=1)
		refined_values.append(np.asarray([weights @ d_values], dtype=np.float64))
	return np.concatenate(refined_values)


def get_processed_index_set(h5: h5py.File, group_name: str) -> set[int]:
	if group_name not in h5 or "flat_index" not in h5[group_name]:
		return set()
	indices = np.asarray(h5[group_name]["flat_index"], dtype=np.int64)
	return set(indices.tolist())


def ensure_group_layout(h5: h5py.File, params: PipelineParams) -> h5py.Group:
	grp = h5.require_group(params.group_name)

	grp.attrs["depth"] = int(params.depth)
	grp.attrs["integration_order"] = int(params.integration_order)
	grp.attrs["poly_order"] = int(params.poly_order)
	grp.attrs["young_modulus"] = 1.0
	grp.attrs["poisson_ratio"] = float(params.poisson_ratio)
	grp.attrs["storage"] = "upper_triangular"
	grp.attrs["matrix_size"] = 24
	grp.attrs["refined_subcells"] = int(params.refined_subcells)
	grp.attrs["n_distance_nodes"] = int(params.n_distance_nodes)

	if "flat_index" not in grp:
		grp.create_dataset(
			"flat_index",
			shape=(0,),
			maxshape=(None,),
			dtype=np.int64,
			chunks=True,
			compression="gzip",
			shuffle=True,
		)

	if "d_values" not in grp:
		grp.create_dataset(
			"d_values",
			shape=(0, params.n_distance_nodes),
			maxshape=(None, params.n_distance_nodes),
			dtype=np.float64,
			chunks=(1024, params.n_distance_nodes),
			compression="gzip",
			shuffle=True,
		)

	if "stiffness_ut" not in grp:
		grp.create_dataset(
			"stiffness_ut",
			shape=(0, UT_SIZE),
			maxshape=(None, UT_SIZE),
			dtype=np.float64,
			chunks=(256, UT_SIZE),
			compression="gzip",
			shuffle=True,
		)

	return grp


def append_batch(
	grp: h5py.Group,
	indices: list[int],
	d_values: list[np.ndarray],
	stiffness_ut: list[np.ndarray],
) -> None:
	if not indices:
		return

	n_new = len(indices)
	new_slice = slice(grp["flat_index"].shape[0], grp["flat_index"].shape[0] + n_new)

	grp["flat_index"].resize((new_slice.stop,))
	grp["flat_index"][new_slice] = np.asarray(indices, dtype=np.int64)

	grp["d_values"].resize((new_slice.stop, grp["d_values"].shape[1]))
	grp["d_values"][new_slice, :] = np.asarray(d_values, dtype=np.float64)

	grp["stiffness_ut"].resize((new_slice.stop, UT_SIZE))
	grp["stiffness_ut"][new_slice, :] = np.asarray(stiffness_ut, dtype=np.float64)


def compute_single_cell_stiffness(
	d_values_csv_order: np.ndarray,
	params: PipelineParams,
) -> np.ndarray:
	B_all, w_all = compute_integration_structure(
		d_values_csv_order=d_values_csv_order,
		depth=int(params.depth),
		integration_order=int(params.integration_order),
		poly_order=int(params.poly_order),
	)
	if B_all is None:
		return np.zeros((LOCAL_DOF, LOCAL_DOF), dtype=np.float64)
	ke = assemble_k_from_structure(B_all, w_all, float(params.poisson_ratio))
	return ke.cpu().numpy().astype(np.float64, copy=False)


def compute_integration_structure(
	d_values_csv_order: np.ndarray,
	depth: int,
	integration_order: int,
	poly_order: int = 1,
) -> tuple:
	"""Precompute geometry-only integration data (B matrices + weights) for one cut cell.

	The result is independent of Poisson ratio and can be reused across multiple nu
	values via ``assemble_k_from_structure``.  This avoids redundant octree traversals
	when computing K for many nu values on the same geometry.

	Returns
	-------
	(B_all, w_eff) where B_all has shape (G, 6, 24) and w_eff has shape (G,), or
	(None, None) if the cell contributes no stiffness (fully outside the domain).
	"""

	d_basic_order = d_values_csv_order[CUT_MASK_TO_BASIC_ORDER]
	nodal_sdf = d_basic_order - 0.5
	raw_mask, _ = classify_cube_configuration(nodal_sdf, threshold=0.0)

	# Fully outside cell: no stiffness contribution.
	if raw_mask == 0:
		return None, None

	coords_t = _REF_HEX_COORDS_T
	coords_ref = coords_t.squeeze(0)

	# Use direct trilinear nodal SDF for cut classification/integration.
	# Cube configuration is identified by raw/canonical masks above (15 canonical classes).
	volume = TrilinearNodalSDFVolume(node_coords=_REF_HEX_COORDS, nodal_sdf_basic_order=nodal_sdf)

	cut_n_pts = (2 * max(1, integration_order)) ** 3

	(pts_ref_list, w_list, h_ref_list, _, _) = OctreeSampler.sample_batch(
		coords_t, volume, cut_n_pts, 'cpu',
		max_depth=depth, poly_order=poly_order, compute_physical=False,
	)

	if pts_ref_list[0] is None:
		return None, None

	subcell_ctr = pts_ref_list[0]   # (S, 3)
	w_ref       = w_list[0]         # (S,)
	h_ref       = h_ref_list[0]     # (S,)
	order_eff   = int(integration_order)

	if order_eff > 1:
		xi_g, w_g = _gauss_rule_3d(order_eff)

		S, P = subcell_ctr.shape[0], xi_g.shape[0]
		xi_pts = (subcell_ctr[:, None, :] + h_ref[:, None, None] * xi_g[None]).reshape(S * P, 3)
		w_pts  = (w_ref[:, None] * w_g[None] / 8.0).reshape(S * P)
		w_pts  = 8.0 * w_pts
	else:
		xi_pts = subcell_ctr
		w_pts  = 8.0 * w_ref

	dN3_ref  = Shape.dN3_dxi(
		xi_pts[:, 0], xi_pts[:, 1], xi_pts[:, 2], poly_order=poly_order
	)
	dN3_phys, J = Shape.dN3_dx(dN3_ref, coords_ref)
	detJ = torch.linalg.det(J)

	B     = Kinematics.B_matrix(dN3_phys)              # (G, 6, 24)
	w_eff = w_pts * detJ                                # (G,)

	return (B, w_eff)


def assemble_k_from_structure(
	B_all: torch.Tensor,
	w_all: torch.Tensor,
	nu: float,
) -> torch.Tensor:
	"""Assemble K(nu) from precomputed B matrices and effective weights."""
	C = _material_c_torch(float(nu))
	CB = torch.matmul(C.unsqueeze(0), B_all)
	K_each = torch.matmul(B_all.transpose(1, 2), CB)
	return torch.tensordot(w_all, K_each, dims=([0], [0]))


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description=(
			"Incrementally compute and store single-cell local stiffness matrices from cut-mask inputs."
		)
	)
	parser.add_argument(
		"--cut-mask-csv",
		type=Path,
		default=REPO_ROOT / "cut_mask_r6.csv",
		help="Path to cut_mask CSV (e.g. cut_mask_r6.csv).",
	)
	parser.add_argument(
		"--output-h5",
		type=Path,
		default=REPO_ROOT / "data_ibm" / "local_stiffness_matrices.h5",
		help="Output HDF5 file for compact appendable storage.",
	)
	parser.add_argument(
		"--amount",
		type=int,
		default=10000,
		help="Maximum number of new inputs to process this run. Use <=0 for all remaining.",
	)
	parser.add_argument("--depth", type=int, default=4, help="Octree refinement depth.")
	parser.add_argument(
		"--integration-order",
		type=int,
		default=3,
		help="Gauss order used in the single-cell cut integration.",
	)
	parser.add_argument("--poly-order", type=int, default=1, help="Hex element polynomial order.")
	parser.add_argument("--poisson-ratio", type=float, default=0.3, help="Poisson ratio nu.")
	parser.add_argument(
		"--refined-subcells",
		type=int,
		default=1,
		help="Subcells per axis for distance inputs; 1 keeps the legacy eight corners.",
	)
	parser.add_argument(
		"--include-invalid",
		action="store_true",
		help="If set, process all rows instead of filtering by valid=1.",
	)
	parser.add_argument(
		"--value-resolution",
		type=int,
		default=6,
		help="Value resolution used if cut-mask needs generation.",
	)
	return parser.parse_args()


def _infer_value_resolution_from_name(path: Path, fallback: int) -> int:
	name = path.name
	if "_r" in name and name.endswith(".csv"):
		try:
			start = name.rfind("_r") + 2
			end = name.rfind(".csv")
			return int(name[start:end])
		except ValueError:
			pass
	return fallback


def main() -> None:
	args = parse_args()

	cut_mask_csv = args.cut_mask_csv.resolve()
	output_h5 = args.output_h5.resolve()
	if args.refined_subcells < 1:
		raise ValueError("--refined-subcells must be at least 1")
	if args.refined_subcells > 1 and not output_h5.name.startswith("refined_"):
		output_h5 = output_h5.with_name(f"refined_{output_h5.name}")

	if not cut_mask_csv.exists():
		value_resolution = _infer_value_resolution_from_name(cut_mask_csv, int(args.value_resolution))
		fallback_source = REPO_ROOT / f"cut_mask_r{value_resolution}.csv"
		if fallback_source.exists():
			cut_mask_csv.parent.mkdir(parents=True, exist_ok=True)
			shutil.copyfile(fallback_source, cut_mask_csv)
			print(f"cut-mask CSV copied from: {fallback_source}")
		else:
			cut_mask_csv.parent.mkdir(parents=True, exist_ok=True)
			print(f"cut-mask CSV not found. Generating: {cut_mask_csv}")
			generate_mask_csv(
				resolution=1,
				value_resolution=value_resolution,
				output_csv=str(cut_mask_csv),
				batch_size=20000,
			)

	output_h5.parent.mkdir(parents=True, exist_ok=True)

	params = PipelineParams(
		depth=int(args.depth),
		integration_order=int(args.integration_order),
		poly_order=int(args.poly_order),
		poisson_ratio=float(args.poisson_ratio),
		refined_subcells=int(args.refined_subcells),
	)

	amount = int(args.amount)
	valid_only = not bool(args.include_invalid)

	with h5py.File(output_h5, "a") as h5:
		grp = ensure_group_layout(h5, params)
		processed = get_processed_index_set(h5, params.group_name)

		selected: list[tuple[int, np.ndarray]] = []
		for flat_index, d_vals in iter_cut_mask_rows(cut_mask_csv, valid_only=valid_only):
			if flat_index in processed:
				continue

			selected.append((flat_index, d_vals))
			if amount > 0 and len(selected) >= amount:
				break

		if not selected:
			print("No new configurations to process for this parameter set.")
			print(f"output_h5={output_h5}")
			print(f"group={params.group_name}")
			print(f"depth={params.depth}, integration_order={params.integration_order}")
			return

		tri = np.triu_indices(LOCAL_DOF)
		out_idx: list[int] = []
		out_d: list[np.ndarray] = []
		out_ke_ut: list[np.ndarray] = []

		for i, (flat_index, d_vals) in enumerate(selected, start=1):
			ke = compute_single_cell_stiffness(d_vals, params)
			out_idx.append(flat_index)
			out_d.append(refine_nodal_distances(d_vals, params.refined_subcells))
			out_ke_ut.append(ke[tri])

			if i % 25 == 0:
				print(f"Computed {i}/{len(selected)}", flush=True)

		append_batch(grp, out_idx, out_d, out_ke_ut)

		print("Saved local stiffness matrices.")
		print(f"output_h5={output_h5}")
		print(f"group={params.group_name}")
		print(f"added={len(out_idx)}")
		print(f"total_in_group={grp['flat_index'].shape[0]}")
		print(f"depth={params.depth}, integration_order={params.integration_order}")


if __name__ == "__main__":
	main()
