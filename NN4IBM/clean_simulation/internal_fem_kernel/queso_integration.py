import ctypes
import os
import sys
import time
import threading
from pathlib import Path
import numpy as np
from typing import Tuple

try:
    from QuESoPythonModule.PyQuESo import PyQuESo
except ImportError:
    PyQuESo = None


def _load_queso_dynamically(queso_python_path):
    global PyQuESo
    if PyQuESo is not None:
        return
    if queso_python_path is not None:
        p = str(Path(queso_python_path).resolve())
        if p not in sys.path:
            sys.path.insert(0, p)
    try:
        from QuESoPythonModule.PyQuESo import PyQuESo as loaded_pyqueso
        PyQuESo = loaded_pyqueso
    except ImportError:
        pass


_dll_dir_handles = []
_dll_dirs_registered = set()


class _ProgressHeartbeat:
    """Emit elapsed-time status while a native QuESo phase does not return."""

    def __init__(self, label, interval_seconds=30.0):
        self.label = label
        self.interval_seconds = interval_seconds
        self.started_at = time.perf_counter()
        self._stopped = threading.Event()
        self._thread = threading.Thread(target=self._report, daemon=True)

    def __enter__(self):
        print(f"[QuESo] {self.label} started.", flush=True)
        self._thread.start()
        return self

    def _report(self):
        while not self._stopped.wait(self.interval_seconds):
            print(
                f"[QuESo] {self.label} still running; elapsed={time.perf_counter() - self.started_at:.1f}s.",
                flush=True,
            )

    def __exit__(self, exc_type, exc_value, traceback):
        self._stopped.set()
        self._thread.join(timeout=0.1)
        state = "interrupted" if exc_type is not None else "complete"
        print(
            f"[QuESo] {self.label} {state}; elapsed={time.perf_counter() - self.started_at:.1f}s.",
            flush=True,
        )


def run_queso_get_integration_points(
    settings_json_path,
    work_dir,
    queso_python_path=None,
    keep_negative_weights=True,
    weight_zero_tol=0.0,
    return_metadata=False,
    cut_cells_only=False,
):
    _load_queso_dynamically(queso_python_path)
    if PyQuESo is None:
        raise RuntimeError("PyQuESo is None. Make sure QuESo path is configured or available in sys.path")
    tol = float(weight_zero_tol)
    if tol < 0.0:
        raise ValueError('weight_zero_tol must be >= 0.0')

    metadata = {
        'inside_element_count': 0,
        'trimmed_element_count': 0,
        'inside_collect_time': 0.0,
        'trimmed_collect_time': 0.0,
        'cut_element_count': 0,
        'cut_collect_time': 0.0,
        'inside_ip_count_before_weight_filter': 0,
        'trimmed_ip_count_before_weight_filter': 0,
        'cut_ip_count_before_weight_filter': 0,
        'inside_ip_count_kept': 0,
        'trimmed_ip_count_kept': 0,
        'cut_ip_count_kept': 0,
        'total_ip_count_before_weight_filter': 0,
        'total_ip_count_kept': 0,
        'weight_filter_time': 0.0,
        'queso_run_time': 0.0,
        'point_extraction_time': 0.0,
    }

    prev_dir = os.getcwd()
    try:
        os.chdir(work_dir)
        pq = PyQuESo(str(settings_json_path))
        t_run = time.perf_counter()
        with _ProgressHeartbeat('native grid classification and quadrature generation'):
            pq.Run()
        metadata['queso_run_time'] = time.perf_counter() - t_run
        t_extract = time.perf_counter()
        rows = []
        trimmed_flags = []
        with _ProgressHeartbeat('Python integration-point collection'):
            for element in pq.GetElements():
                is_trimmed = bool(element.IsTrimmed())
                if cut_cells_only and not is_trimmed:
                    metadata['inside_element_count'] += 1
                    continue
                t_element = time.perf_counter()
                ip_count_before_filter = 0
                for point in element.GetIntegrationPoints():
                    ip_count_before_filter += 1
                    rows.append([point.X(), point.Y(), point.Z(), point.Weight()])
                    trimmed_flags.append(is_trimmed)
                elapsed = time.perf_counter() - t_element
                if is_trimmed:
                    metadata['trimmed_element_count'] += 1
                    metadata['trimmed_collect_time'] += elapsed
                    metadata['trimmed_ip_count_before_weight_filter'] += ip_count_before_filter
                else:
                    metadata['inside_element_count'] += 1
                    metadata['inside_collect_time'] += elapsed
                    metadata['inside_ip_count_before_weight_filter'] += ip_count_before_filter

            metadata['cut_element_count'] = metadata['trimmed_element_count']
            metadata['cut_collect_time'] = metadata['trimmed_collect_time']
            metadata['cut_ip_count_before_weight_filter'] = metadata['trimmed_ip_count_before_weight_filter']
        metadata['point_extraction_time'] = time.perf_counter() - t_extract
    finally:
        os.chdir(prev_dir)

    metadata['total_ip_count_before_weight_filter'] = (
        metadata['inside_ip_count_before_weight_filter']
        + metadata['trimmed_ip_count_before_weight_filter']
    )

    arr_all = np.array(rows, dtype=np.float64) if rows else np.zeros((0, 4), dtype=np.float64)
    flags_all = np.array(trimmed_flags, dtype=bool) if trimmed_flags else np.zeros((0,), dtype=bool)
    if arr_all.size == 0:
        arr = arr_all.reshape(0, 4)
        return (arr, metadata) if return_metadata else arr

    t_filter = time.perf_counter()
    if keep_negative_weights:
        mask = np.abs(arr_all[:, 3]) > tol
    else:
        mask = arr_all[:, 3] > tol
    metadata['weight_filter_time'] = time.perf_counter() - t_filter

    kept_flags = flags_all[mask]
    metadata['trimmed_ip_mask'] = kept_flags
    metadata['trimmed_ip_count_kept'] = int(np.count_nonzero(kept_flags))
    metadata['cut_ip_count_kept'] = metadata['trimmed_ip_count_kept']
    metadata['inside_ip_count_kept'] = int(kept_flags.size - metadata['trimmed_ip_count_kept'])
    metadata['total_ip_count_kept'] = int(kept_flags.size)

    arr = arr_all[mask]
    return (arr, metadata) if return_metadata else arr


def run_queso_write_integration_points_hdf5(
    settings_json_path,
    work_dir,
    output_path,
    queso_python_path=None,
    keep_negative_weights=True,
    weight_zero_tol=0.0,
    cut_cells_only=False,
    point_batch_size=100_000,
    physical_bounds=None,
):
    """Run QuESo and append retained integration points to a resumable HDF5 file."""
    try:
        import h5py
    except ImportError as exc:
        raise RuntimeError('HDF5-backed QuESo extraction requires h5py.') from exc

    _load_queso_dynamically(queso_python_path)
    if PyQuESo is None:
        raise RuntimeError('PyQuESo is None. Make sure QuESo path is configured or available in sys.path')
    if weight_zero_tol < 0.0:
        raise ValueError('weight_zero_tol must be >= 0.0')
    if point_batch_size < 1:
        raise ValueError('point_batch_size must be positive')

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    bounds = None if physical_bounds is None else np.asarray(physical_bounds, dtype=np.float64)
    if bounds is not None and bounds.shape != (2, 3):
        raise ValueError('physical_bounds must have shape (2, 3)')
    metadata = {'queso_run_time': 0.0, 'point_extraction_time': 0.0}
    with h5py.File(output_path, 'a') as output:
        points = output.require_dataset(
            'points', shape=(0, 4), maxshape=(None, 4), dtype=np.float64,
            chunks=(point_batch_size, 4),
        )
        completed_elements = int(output.attrs.get('completed_elements', 0))
        if bounds is not None:
            output.attrs['physical_bounds'] = bounds
            output.attrs['coordinates_are_physical'] = True
        if output.attrs.get('complete', False):
            return dict(output.attrs)

        prev_dir = os.getcwd()
        try:
            os.chdir(work_dir)
            pq = PyQuESo(str(settings_json_path))
            t_run = time.perf_counter()
            with _ProgressHeartbeat('native grid classification and quadrature generation'):
                pq.Run()
            metadata['queso_run_time'] = time.perf_counter() - t_run
            t_extract = time.perf_counter()
            pending = []
            with _ProgressHeartbeat('HDF5 integration-point collection'):
                for element_index, element in enumerate(pq.GetElements()):
                    if element_index < completed_elements:
                        continue
                    is_trimmed = bool(element.IsTrimmed())
                    if cut_cells_only and not is_trimmed:
                        continue
                    for point in element.GetIntegrationPoints():
                        weight = point.Weight()
                        if (keep_negative_weights and abs(weight) > weight_zero_tol) or (not keep_negative_weights and weight > weight_zero_tol):
                            pending.append((point.X(), point.Y(), point.Z(), weight))
                    if len(pending) >= point_batch_size:
                        batch = np.asarray(pending, dtype=np.float64)
                        if bounds is not None and np.all((batch[:, :3] >= -1.0e-12) & (batch[:, :3] <= 1.0 + 1.0e-12)):
                            batch[:, :3] = bounds[0] + batch[:, :3] * (bounds[1] - bounds[0])
                            batch[:, 3] *= np.prod(bounds[1] - bounds[0])
                        old_size = len(points)
                        points.resize(old_size + len(batch), axis=0)
                        points[old_size:] = batch
                        pending.clear()
                        output.attrs['completed_elements'] = element_index + 1
                        output.flush()
            if pending:
                batch = np.asarray(pending, dtype=np.float64)
                if bounds is not None and np.all((batch[:, :3] >= -1.0e-12) & (batch[:, :3] <= 1.0 + 1.0e-12)):
                    batch[:, :3] = bounds[0] + batch[:, :3] * (bounds[1] - bounds[0])
                    batch[:, 3] *= np.prod(bounds[1] - bounds[0])
                old_size = len(points)
                points.resize(old_size + len(batch), axis=0)
                points[old_size:] = batch
            output.attrs['completed_elements'] = element_index + 1 if 'element_index' in locals() else completed_elements
            output.attrs['point_count'] = len(points)
            output.attrs['complete'] = True
            output.flush()
            metadata['point_extraction_time'] = time.perf_counter() - t_extract
        finally:
            os.chdir(prev_dir)
        output.attrs.update(metadata)
        return dict(output.attrs)


def map_integration_points_to_elements(ip_xyz, grid_bounds, grid_resolution):
    nx, ny, nz = grid_resolution
    bounds_min = np.array(grid_bounds[0], dtype=np.float64)
    bounds_max = np.array(grid_bounds[1], dtype=np.float64)
    dx = (bounds_max - bounds_min) / np.array([nx, ny, nz], dtype=np.float64)
    ijk = np.floor((ip_xyz - bounds_min) / dx).astype(np.int64)
    ijk = np.clip(ijk, [0, 0, 0], [nx - 1, ny - 1, nz - 1])
    flat = ijk[:, 0] * (ny * nz) + ijk[:, 1] * nz + ijk[:, 2]
    cell_min = bounds_min + ijk * dx
    xi_local = np.clip(2.0 * (ip_xyz - cell_min) / dx - 1.0, -1.0, 1.0)
    return flat, xi_local
