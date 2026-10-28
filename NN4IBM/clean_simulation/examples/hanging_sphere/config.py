"""
Configuration and helper functions for hanging sphere simulations with QuESo.
Centralizes hardcoded constants and sphere geometry operations.
"""

from pathlib import Path
import numpy as np
from typing import Dict, Any, Tuple


# Default hanging sphere configuration
DEFAULT_HANGING_SPHERE_CONFIG = {
    "sphere": {
        "radius": 7.5,
        "center": (0.0, 0.0, 0.0),
    },
    "material": {
        "E": 1.0,
        "nu": 0.30,
        "density": 7850.0,
    },
    "grid": {
        "bounds": [[-15, -15, -15], [15, 15, 15]],
        "resolution": [30, 30, 30],
    },
    "integration": {
        "poly_order": 1,
        "gauss_order": 2,
        "moment_fitting_residual": 1e-8,
    },
    "loads": {
        "gravity": [0.0, 0.0, -9.81e6],
        "direction": [0.0, 0.0, -1.0],
    },
    "bc": {
        "nitsche_penalty": 100.0,
        "top_cap_fraction": 0.06,
    },
}


# System paths (hardcoded for local environment)
PYTHON_312_EXECUTABLE = Path(r"C:/Users/z0058p1u/AppData/Local/Programs/Python/Python312/python.exe")
BOTTOM_FACE_Z_THRESHOLD = -0.2  # Threshold for identifying bottom faces


def get_hanging_sphere_config(custom_config: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Get hanging sphere configuration, optionally overriding defaults.
    
    Parameters
    ----------
    custom_config : dict, optional
        Dictionary with keys matching DEFAULT_HANGING_SPHERE_CONFIG structure.
        Will be merged with defaults (custom values override).
    
    Returns
    -------
    dict
        Complete configuration dictionary.
    """
    config = dict(DEFAULT_HANGING_SPHERE_CONFIG)
    if custom_config:
        for key, value in custom_config.items():
            if isinstance(value, dict) and key in config:
                config[key].update(value)
            else:
                config[key] = value
    return config


def extract_top_cap_faces(sphere_mesh, config: Dict[str, Any]):
    """
    Extract top cap triangular faces from sphere mesh based on z-height and normal.
    
    Parameters
    ----------
    sphere_mesh : trimesh.Trimesh
        The sphere surface mesh.
    config : dict
        Configuration with bc.top_cap_fraction.
    
    Returns
    -------
    faces : ndarray
        Indices of top cap faces.
    attach_z_min : float
        Minimum z-coordinate of attachment region.
    """
    face_normals = sphere_mesh.face_normals
    face_centers = sphere_mesh.triangles_center
    
    z_max = float(sphere_mesh.vertices[:, 2].max())
    attach_cap_h = max(
        1e-6,
        float(config["bc"]["top_cap_fraction"]) * float(config["sphere"]["radius"])
    )
    attach_z_min = z_max - attach_cap_h
    
    top_mask = (face_centers[:, 2] >= attach_z_min) & (face_normals[:, 2] > 0.0)
    top_face_indices = np.where(top_mask)[0]
    
    return sphere_mesh.faces[top_face_indices], attach_z_min


def extract_bottom_cap_faces(sphere_mesh):
    """
    Extract bottom cap triangular faces from sphere mesh.
    
    Parameters
    ----------
    sphere_mesh : trimesh.Trimesh
        The sphere surface mesh.
    
    Returns
    -------
    faces : ndarray
        Indices of bottom cap faces.
    """
    face_normals = sphere_mesh.face_normals
    bottom_mask = face_normals[:, 2] < BOTTOM_FACE_Z_THRESHOLD
    bottom_face_indices = np.where(bottom_mask)[0]
    
    return sphere_mesh.faces[bottom_face_indices]
