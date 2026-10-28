from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
import sys

root = Path.cwd().resolve()
while root != root.parent and not (root / "clean_simulation").is_dir():
    root = root.parent
if not (root / "clean_simulation").is_dir():
    raise RuntimeError("Could not find workspace root containing clean_simulation/")
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from clean_simulation.internal_nn_runtime.cutcell_ml.classification_utils import compute_class_label_from_d, get_class_label_batch

class CutCellDataset(Dataset):
    def __init__(self, d: torch.Tensor, nu: torch.Tensor, k_mat: torch.Tensor):
        if not (d.shape[0] == nu.shape[0] == k_mat.shape[0]):
            raise ValueError("d, nu and k_mat must have the same leading dimension")
        self.d = d.float()
        self.nu = nu.float()
        self.k_mat = k_mat.float()
        self.classes = get_class_label_batch(self.d)

    def __len__(self) -> int:
        return self.d.shape[0]

    def __getitem__(self, idx):
        return self.d[idx], self.nu[idx], self.k_mat[idx], self.classes[idx]



class H5CutCellDataset(Dataset):
    """Memory-light dataset that reads samples directly from HDF5 on demand."""

    def __init__(
        self,
        h5_path: Path,
        grp_name: str,
        fallback_nu: float,
        *,
        max_samples: int | None = None,
        dof: int = 24,
    ):
        self.h5_path = Path(h5_path)
        self.dof = int(dof)
        self._tri = np.triu_indices(self.dof)
        self._h5: h5py.File | None = None
        self._d_ds: Any = None
        self._k_ds: Any = None
        self._nu_val = float(fallback_nu)

        # Handle fallback for non-depth group name
        with h5py.File(self.h5_path, "r") as h5_file:
            if grp_name not in h5_file:
                # Check for group without 'depth_X_' prefix
                import re
                trimmed_grp = re.sub(r"^depth_\d+_", "", grp_name)
                if trimmed_grp in h5_file:
                    grp_name = trimmed_grp
                else:
                    raise KeyError(f"Group not found in HDF5: {grp_name} (tried fallback {trimmed_grp} too)")
            
            self.grp_name = grp_name
            group = h5_file[self.grp_name]
            n_total = int(group["d_values"].shape[0])
            self._nu_val = float(group.attrs.get("poisson_ratio", fallback_nu))

        if max_samples is not None and int(max_samples) > 0:
            self.n_rows = min(n_total, int(max_samples))
            print(
                f"Dataset group has {n_total:,} rows; streaming first {self.n_rows:,} rows from HDF5.",
                flush=True,
            )
        else:
            self.n_rows = n_total
            print(f"Dataset group has {n_total:,} rows; streaming all rows from HDF5.", flush=True)

    def _ensure_open(self) -> None:
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, "r")
            group = self._h5[self.grp_name]
            self._d_ds = group["d_values"]
            self._k_ds = group["stiffness_ut"]

    def __len__(self) -> int:
        return self.n_rows

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self._ensure_open()
        d_values = np.asarray(self._d_ds[idx], dtype=np.float32)
        stiffness_ut = np.asarray(self._k_ds[idx], dtype=np.float32)

        return (
            torch.from_numpy(d_values),
            torch.tensor(self._nu_val, dtype=torch.float32),
            torch.from_numpy(stiffness_ut),
        )

    def __del__(self) -> None:
        h5_obj = getattr(self, "_h5", None)
        if h5_obj is not None:
            try:
                h5_obj.close()
            except Exception:
                pass


def reconstruct_matrices_from_ut(stiffness_ut: np.ndarray, dof: int = 24) -> np.ndarray:
    n_rows = int(stiffness_ut.shape[0])
    tri = np.triu_indices(dof)

    k_mat = np.zeros((n_rows, dof, dof), dtype=np.float64)
    k_mat[:, tri[0], tri[1]] = stiffness_ut
    k_mat = k_mat + np.transpose(k_mat, (0, 2, 1))

    diag_idx = np.arange(dof)
    k_mat[:, diag_idx, diag_idx] *= 0.5
    return k_mat


def load_dataset_from_h5(
    h5_path: Path,
    grp_name: str,
    fallback_nu: float,
    *,
    max_samples: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    chunk_size = 200_000

    with h5py.File(h5_path, "r") as h5_file:
        if grp_name not in h5_file:
            import re
            trimmed_grp = re.sub(r"^depth_\d+_", "", grp_name)
            if trimmed_grp in h5_file:
                grp_name = trimmed_grp
            else:
                raise KeyError(f"Group not found in HDF5: {grp_name} (tried fallback {trimmed_grp} too)")

        group = h5_file[grp_name]
        d_ds = group["d_values"]
        k_ds = group["stiffness_ut"]
        n_total = int(d_ds.shape[0])

        if max_samples is not None and int(max_samples) > 0 and n_total > int(max_samples):
            n_keep = int(max_samples)
            print(f"Dataset group has {n_total:,} rows; loading first {n_keep:,} rows for this run.", flush=True)
            d_values = np.asarray(d_ds[:n_keep], dtype=np.float32)
            stiffness_ut = np.asarray(k_ds[:n_keep], dtype=np.float64)
        else:
            print(f"Dataset group has {n_total:,} rows; loading all rows.", flush=True)
            d_values = np.empty((n_total, d_ds.shape[1]), dtype=np.float32)
            stiffness_ut = np.empty((n_total, k_ds.shape[1]), dtype=np.float64)
            for start in range(0, n_total, chunk_size):
                end = min(start + chunk_size, n_total)
                d_values[start:end] = np.asarray(d_ds[start:end], dtype=np.float32)
                stiffness_ut[start:end] = np.asarray(k_ds[start:end], dtype=np.float64)
                if (start == 0) or (end == n_total) or ((start // chunk_size + 1) % 5 == 0):
                    pct = 100.0 * float(end) / float(n_total)
                    print(f"Loaded HDF5 rows: {end:,}/{n_total:,} ({pct:.1f}%)", flush=True)
        nu_val = float(group.attrs.get("poisson_ratio", fallback_nu))

    k_values = reconstruct_matrices_from_ut(stiffness_ut).astype(np.float32)
    nu_values = np.full((d_values.shape[0],), nu_val, dtype=np.float32)

    # Remove pathological rows that destabilize relative-loss training:
    # non-finite values or (near-)zero target norm.
    d_finite = np.isfinite(d_values).all(axis=1)
    k_finite = np.isfinite(k_values).all(axis=(1, 2))
    k_fro = np.linalg.norm(k_values, axis=(1, 2))
    nonzero_target = k_fro > 1.0e-8
    keep = d_finite & k_finite & nonzero_target
    dropped = int((~keep).sum())
    if dropped > 0:
        print(
            f"Filtered {dropped:,} unstable row(s) from HDF5 load "
            f"(non-finite or near-zero target norm).",
            flush=True,
        )
        d_values = d_values[keep]
        nu_values = nu_values[keep]
        k_values = k_values[keep]

    return (
        torch.from_numpy(d_values),
        torch.from_numpy(nu_values),
        torch.from_numpy(k_values),
    )


def load_and_concat_h5_files(
    h5_specs: list[tuple[Path, str, float]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    all_d, all_nu, all_k = [], [], []
    for h5_path, grp_name, fallback_nu in h5_specs:
        d, nu, k = load_dataset_from_h5(h5_path, grp_name, fallback_nu)
        all_d.append(d)
        all_nu.append(nu)
        all_k.append(k)
        print(f"  Loaded {d.shape[0]:,} samples  nu={float(nu[0]):.4f}  from {h5_path.name}/{grp_name}")
    return torch.cat(all_d), torch.cat(all_nu), torch.cat(all_k)
