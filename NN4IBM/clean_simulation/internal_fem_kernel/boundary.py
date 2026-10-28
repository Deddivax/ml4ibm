import torch

class Boundary:

    @staticmethod
    def _axis_to_index(axis):
        if isinstance(axis, int):
            if axis in (0, 1, 2):
                return axis
            raise ValueError("Axis index must be 0, 1, or 2.")

        axis_map = {"x": 0, "y": 1, "z": 2}
        axis_key = str(axis).strip().lower()
        if axis_key not in axis_map:
            raise ValueError("Axis must be one of: 'x', 'y', 'z', 0, 1, 2.")
        return axis_map[axis_key]

    @staticmethod
    def get_fixed_nodes(node_coords, constraints=None, tol=1e-4):
        """Return a boolean mask of nodes fixed by plane constraints.

        Args:
            node_coords: (N, 3) tensor/array of node coordinates.
            constraints: list of tuples (axis, value), e.g. [("z", 1.0)].
            tol: absolute tolerance for coordinate matching (default 1e-4 to catch
                 all nodes on boundary planes reliably with floating-point arithmetic).
        """
        coords = torch.as_tensor(node_coords)
        if coords.ndim != 2 or coords.shape[1] != 3:
            raise ValueError("node_coords must have shape (N, 3).")

        if constraints is None:
            constraints = [("z", -1.0)]

        if isinstance(constraints, tuple) and len(constraints) == 2:
            constraints = [constraints]

        fixed_mask = torch.zeros(coords.shape[0], dtype=torch.bool, device=coords.device)
        for axis, value in constraints:
            axis_idx = Boundary._axis_to_index(axis)
            fixed_mask |= torch.abs(coords[:, axis_idx] - float(value)) <= tol

        return fixed_mask

    @staticmethod
    def get_fixed_dofs(node_coords, constraints=None, tol=1e-6, dofs_per_node=3):
        fixed_nodes = Boundary.get_fixed_nodes(node_coords, constraints=constraints, tol=tol)
        fixed_node_idx = torch.where(fixed_nodes)[0]

        dof_offsets = torch.arange(dofs_per_node, device=fixed_node_idx.device)
        fixed_dofs = (fixed_node_idx.unsqueeze(1) * dofs_per_node + dof_offsets).reshape(-1)

        return fixed_nodes, fixed_dofs

    @staticmethod
    def apply_dirichlet(K, F, node_coords, constraints=None, tol=1e-6, return_indices=False):
        raise RuntimeError(
            "Boundary.apply_dirichlet is disabled in basic_simulation. "
            "Use NitscheAssembly.compute_terms(...) and GlobalAssembly.assemble_system(...)."
        )