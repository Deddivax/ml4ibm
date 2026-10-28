import pyvista as pv
import numpy as np
import torch
from geometry import HexGrid

class Visualizer:

    def __init__(self, window_size=(900, 700)):
        self.plotter = pv.Plotter(window_size=window_size)

    # ==========================================================
    # ✅ CORRECT SDF SURFACE
    # ==========================================================
    def add_volume_sdf(self, volume, grid, iso=0.0, opacity=1):
        """
        Plot iso-surface of SDF (correctly aligned with grid)
        """

        pts = grid.centers.reshape(-1, 3)
        sdf = volume.signed_distance(pts).reshape(grid.shape)

        nx, ny, nz = grid.shape

        # ✅ FIX: build UNIFORM GRID using spacing (not linspace)
        origin = (
            grid.bounds[0][0] + 0.5 * grid.dx,
            grid.bounds[0][1] + 0.5 * grid.dy,
            grid.bounds[0][2] + 0.5 * grid.dz,
        )

        spacing = (grid.dx, grid.dy, grid.dz)

        grid_pv = pv.ImageData()
        grid_pv.dimensions = (nx, ny, nz)
        grid_pv.origin = origin
        grid_pv.spacing = spacing

        grid_pv["sdf"] = sdf.flatten(order="F")

        contour = grid_pv.contour([iso])

        self.plotter.add_mesh(
            contour,
            color="lightblue",
            opacity=0.3
        )

        return contour
    

    def add_displaced_surface(self, surface, coords_nodes, U_nodes, scale=1.0):

        # surface points
        pts = surface.points.copy()

        # map each surface point to nearest node
        # (cheap but works for structured grids)
        from scipy.spatial import cKDTree
        tree = cKDTree(coords_nodes)

        _, idx = tree.query(pts)

        U_surface = U_nodes[idx]

        pts_def = pts + scale * U_surface

        surface_def = pv.PolyData(pts_def, surface.faces)

        self.plotter.add_mesh(
            surface_def,
            color="red",
            opacity=0.3
        )

    # ==========================================================
    # ✅ ADAPTIVE CELLS (THIN BAND)
    # ==========================================================
    def add_adaptive_cells(self, centers, h_values,
                           color="yellow",
                           opacity=0.6,
                           max_size=None):
        """
        Plot octree cells.

        max_size:
            if set → filters OUT large cells (keeps only thin boundary)
        """

        if centers is None or len(centers) == 0:
            return

        centers_np = centers.cpu().numpy()
        sizes_np   = (2.0 * h_values.cpu().numpy())

        # ✅ FIX: keep only small cells (thin band)
        if max_size is not None:
            mask = sizes_np < max_size
            centers_np = centers_np[mask]
            sizes_np   = sizes_np[mask]

            if len(centers_np) == 0:
                return

        points = pv.PolyData(centers_np)
        points["size"] = sizes_np

        cubes = pv.Cube()
        glyphs = points.glyph(scale="size", geom=cubes)

        self.plotter.add_mesh(
            glyphs,
            color=color,
            opacity=opacity
        )

    # ==========================================================
    # ✅ DEBUG GRID CELLS (OPTIONAL)
    # ==========================================================
    def add_partial_cells(self, grid, partial_mask,
                          color="red", opacity=0.2):
        """
        Visualize coarse partial cells (for debugging classification)
        """

        centers = grid.centers.reshape(-1, 3)
        mask = partial_mask.ravel()

        pts = centers[mask]

        if len(pts) == 0:
            return

        sizes = np.full(len(pts), grid.dx)

        pdata = pv.PolyData(pts)
        pdata["size"] = sizes

        glyphs = pdata.glyph(scale="size", geom=pv.Cube())

        self.plotter.add_mesh(
            glyphs,
            color=color,
            opacity=opacity
        )

    def add_cells_wireframe(self, centers, h_values,
                            color="yellow",
                            line_width=1):

        if centers is None or len(centers) == 0:
            return

        centers_np = centers.cpu().numpy()
        sizes_np   = (2.0 * h_values.cpu().numpy())

        points = pv.PolyData(centers_np)
        points["size"] = sizes_np

        cube = pv.Cube()

        glyphs = points.glyph(scale="size", geom=cube)

        self.plotter.add_mesh(
            glyphs,
            color=color,
            style="wireframe",
            line_width=line_width
        )

    def add_cut_cells(self, partial_mask, grid,
                    color="red",
                    line_width=2):

        # Flatten grid centers
        centers = grid.centers.reshape(-1, 3)

        # Flatten mask
        mask = partial_mask.reshape(-1)

        # Select only boundary cells
        pts = centers[mask]

        if pts.shape[0] == 0:
            return

        # Convert to torch
        pts = torch.tensor(pts, dtype=torch.float32)

        # Cell size = grid spacing
        h_value = grid.dx / 2.0
        h = torch.full((pts.shape[0],), h_value)

        self.add_cells_wireframe(pts, h, color=color, line_width=line_width)

    def add_displacement_active(self, coords_nodes, U_nodes, node_mask, scale=1.0):
        """
        Visualize displacement on active nodes only.

        Parameters
        ----------
        coords_nodes : (N,3) numpy array
        U_nodes      : (N,3) numpy array
        node_mask    : (N,)  boolean numpy array
        scale        : float
        """

        # safety checks
        assert coords_nodes.shape == U_nodes.shape
        assert coords_nodes.shape[0] == node_mask.shape[0]

        # select active nodes
        X = coords_nodes[node_mask]
        U = U_nodes[node_mask]

        # deformed coordinates
        X_def = X + scale * U

        # build point cloud
        cloud = pv.PolyData(X_def)

        # displacement magnitude
        cloud["|u|"] = np.linalg.norm(U, axis=1)

        self.plotter.add_mesh(
            cloud,
            scalars="|u|",
            render_points_as_spheres=True,
            point_size=6,
            cmap="jet"
        )


    def add_nodes(self, coords_nodes, node_mask, color="white", size=3):
        X = coords_nodes[node_mask]

        cloud = pv.PolyData(X)

        self.plotter.add_mesh(
            cloud,
            color=color,
            render_points_as_spheres=True,
            point_size=size
        )

    # ==========================================================
    def show(self, dynamic=True):
        self.plotter.show_grid()
        if dynamic:
            try:
                pv.set_jupyter_backend("trame")
                self.plotter.show(jupyter_backend="trame")
                return
            except Exception:
                pass

        try:
            pv.set_jupyter_backend("static")
            self.plotter.show(jupyter_backend="static")
        except Exception:
            self.plotter.show()