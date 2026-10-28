import numpy as np

from integration.nitsche_bc import NitscheBC


class NitscheAssembly:

    @staticmethod
    def compute_terms(
        grid,
        material,
        active_indices,
        nitsche_bcs,
        device="cpu",
        poly_order=1,
    ):
        """Compute per-BC Nitsche sets; global assembly is handled by GlobalAssembly."""
        if not nitsche_bcs:
            return []

        bc_specs = nitsche_bcs if isinstance(nitsche_bcs, (list, tuple)) else [nitsche_bcs]
        nitsche_sets = []

        for bc in bc_specs:
            if not isinstance(bc, dict):
                raise TypeError("Each Nitsche BC specification must be a dict.")

            bc_args = dict(bc)
            if "triangles" in bc_args:
                k_i, f_i, boundary_elems, gamma_v = NitscheBC.compute_surface_terms(
                    grid=grid,
                    material=material,
                    active_indices=active_indices,
                    device=device,
                    poly_order=poly_order,
                    **bc_args,
                )
            else:
                k_i, f_i, boundary_elems, gamma_v = NitscheBC.compute_plane_terms(
                    grid=grid,
                    material=material,
                    active_indices=active_indices,
                    device=device,
                    poly_order=poly_order,
                    **bc_args,
                )

            nitsche_sets.append(
                {
                    "k_add": k_i,
                    "f_add": f_i,
                    "boundary_elems": np.asarray(boundary_elems, dtype=np.int64),
                    "gamma_v": gamma_v,
                }
            )

        return nitsche_sets
