import numpy as np
import torch
import trimesh

from core.grid_indexing import structured_node_shape, structured_total_nodes


class Volume:
    def __init__(self, mesh, device="cpu"):
        self.device = "cuda" if (device == "cuda" and torch.cuda.is_available()) else "cpu"
        self.mesh = mesh
        self._cached_proximity_query = None

    def signed_distance(self, points):
        """Accept numpy or tensor, always return numpy."""
        if isinstance(points, torch.Tensor):
            points = points.detach().cpu().numpy()
        if self._cached_proximity_query is None:
            self._cached_proximity_query = trimesh.proximity.ProximityQuery(self.mesh)
        return self._cached_proximity_query.signed_distance(points)

    def signed_distance_torch(self, x):
        """Accept numpy or tensor, always return float64 tensor on self.device."""
        if isinstance(x, torch.Tensor):
            x_np = x.detach().cpu().numpy()
        else:
            x_np = x
        if self._cached_proximity_query is None:
            self._cached_proximity_query = trimesh.proximity.ProximityQuery(self.mesh)
        phi_np = self._cached_proximity_query.signed_distance(x_np)
        return torch.from_numpy(phi_np).to(dtype=torch.float64, device=self.device)


class HexGrid:
    def __init__(self, bounds, resolution, device="cuda"):
        self.bounds  = bounds
        self.resolution = resolution
        self.device  = device

        self.centers, self.shape = self._build_centers()
        self.dx, self.dy, self.dz = self._cell_size()

    def _build_centers(self):
        (xmin, ymin, zmin), (xmax, ymax, zmax) = self.bounds
        nx, ny, nz = self.resolution

        # torch.linspace is always inclusive of both endpoints,
        # so replicate np.linspace(..., endpoint=False) + half-step manually
        def _cell_axis(lo, hi, n):
            step = (hi - lo) / n
            return torch.arange(n, dtype=torch.float64, device=self.device) * step + lo + step / 2

        x = _cell_axis(xmin, xmax, nx)
        y = _cell_axis(ymin, ymax, ny)
        z = _cell_axis(zmin, zmax, nz)

        X, Y, Z = torch.meshgrid(x, y, z, indexing="ij")
        centers  = torch.stack([X, Y, Z], dim=-1)   # (nx, ny, nz, 3)

        return centers, tuple(centers.shape[:3])

    def _cell_size(self):
        (xmin, ymin, zmin), (xmax, ymax, zmax) = self.bounds
        nx, ny, nz = self.resolution
        return (xmax - xmin) / nx, (ymax - ymin) / ny, (zmax - zmin) / nz

    def node_axes(self, poly_order=1):
        sx, sy, sz = structured_node_shape(self.resolution, poly_order=poly_order)
        (xmin, ymin, zmin), (xmax, ymax, zmax) = self.bounds
        return (
            torch.linspace(xmin, xmax, sx, dtype=torch.float64, device=self.device),
            torch.linspace(ymin, ymax, sy, dtype=torch.float64, device=self.device),
            torch.linspace(zmin, zmax, sz, dtype=torch.float64, device=self.device),
        )

    def node_coordinates(self, poly_order=1):
        x, y, z = self.node_axes(poly_order=poly_order)
        X, Y, Z = torch.meshgrid(x, y, z, indexing="ij")
        return torch.stack([X, Y, Z], dim=-1).reshape(-1, 3)   # (Nn, 3) on device

    def total_nodes(self, poly_order=1):
        return structured_total_nodes(self.resolution, poly_order=poly_order)

    @staticmethod
    def _local_node_offsets(dx, dy, dz, poly_order=1, device="cpu"):
        p = int(poly_order)
        if p < 1:
            raise ValueError("poly_order must be >= 1")

        if p == 1:
            hx, hy, hz = dx / 2, dy / 2, dz / 2
            return torch.tensor([
                [-hx, -hy, -hz],
                [ hx, -hy, -hz],
                [ hx,  hy, -hz],
                [-hx,  hy, -hz],
                [-hx, -hy,  hz],
                [ hx, -hy,  hz],
                [ hx,  hy,  hz],
                [-hx,  hy,  hz],
            ], dtype=torch.float64, device=device)

        x = torch.linspace(-dx / 2, dx / 2, p + 1, dtype=torch.float64, device=device)
        y = torch.linspace(-dy / 2, dy / 2, p + 1, dtype=torch.float64, device=device)
        z = torch.linspace(-dz / 2, dz / 2, p + 1, dtype=torch.float64, device=device)
        gx, gy, gz = torch.meshgrid(x, y, z, indexing="ij")
        return torch.stack([gx, gy, gz], dim=-1).reshape(-1, 3)

    @staticmethod
    def cell_nodes_from_centers(centers, dx, dy, dz, poly_order=1):
        centers = torch.as_tensor(centers, dtype=torch.float64)
        if centers.ndim != 1 or centers.shape[0] != 3:
            raise ValueError(f"centers must have shape (3,); got {centers.shape}.")
        offsets = HexGrid._local_node_offsets(dx, dy, dz, poly_order=poly_order,
                                              device=centers.device)
        return centers[None, :] + offsets

    @staticmethod
    def build_node_index(grid, poly_order=1):
        """
        Returns elem_to_node : (N, K) int64 tensor on grid.device
        Fully vectorised — no Python loops over cells.
        """
        nx, ny, nz = grid.resolution
        p          = int(poly_order)
        sx, sy, sz = structured_node_shape(grid.resolution, poly_order=poly_order)
        dev        = grid.device

        # node_id[i, j, k] = flat index into node array
        node_id = torch.arange(sx * sy * sz, dtype=torch.int64, device=dev).reshape(sx, sy, sz)

        if p == 1:
            # Corner offsets for a single hex element: (8, 3)
            di = torch.tensor([0, 1, 1, 0, 0, 1, 1, 0], dtype=torch.int64, device=dev)
            dj = torch.tensor([0, 0, 1, 1, 0, 0, 1, 1], dtype=torch.int64, device=dev)
            dk = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.int64, device=dev)
        else:
            # Higher-order: (p+1)^3 nodes per element
            r  = torch.arange(p + 1, dtype=torch.int64, device=dev)
            gz, gy, gx = torch.meshgrid(r, r, r, indexing="ij")
            di = gx.reshape(-1)   # fast (x) index varies fastest
            dj = gy.reshape(-1)
            dk = gz.reshape(-1)

        # Cell base indices: (nx,) (ny,) (nz,)
        ci = torch.arange(nx, dtype=torch.int64, device=dev) * p
        cj = torch.arange(ny, dtype=torch.int64, device=dev) * p
        ck = torch.arange(nz, dtype=torch.int64, device=dev) * p

        # Broadcast to (nx, ny, nz, K)
        I = (ci[:, None, None, None] + di[None, None, None, :])  # (nx, ny, nz, K)
        J = (cj[None, :, None, None] + dj[None, None, None, :])
        K = (ck[None, None, :, None] + dk[None, None, None, :])

        elem_to_node = node_id[I, J, K]                          # (nx, ny, nz, K)
        return elem_to_node.reshape(nx * ny * nz, -1)            # (N, K)