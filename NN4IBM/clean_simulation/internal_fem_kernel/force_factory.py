import torch


class ForceFactory:
    """
    Factory for body forces and surface tractions.
    All functions are vectorized and torch-compatible.
    """

    # ==========================================================
    # BASIC FORCES
    # ==========================================================

    @staticmethod
    def constant(force_vector):
        """
        Constant force everywhere.

        f(x) = c
        """
        c = torch.tensor(force_vector, dtype=torch.float64)

        def f(x):
            return c.expand_as(x)

        return f

    @staticmethod
    def gravity(g=(0, 0, -9.81), density=1.0):
        """
        Gravity force.

        f(x) = rho * g
        """
        g = torch.tensor(g, dtype=torch.float64) * density

        def f(x):
            return g.expand_as(x)

        return f

    # ==========================================================
    # POSITION-DEPENDENT FORCES
    # ==========================================================

    @staticmethod
    def linear(A, b):
        """
        Linear force field:

        f(x) = A x + b

        A: (3,3)
        b: (3,)
        """
        A = torch.tensor(A, dtype=torch.float64)
        b = torch.tensor(b, dtype=torch.float64)

        def f(x):
            return torch.einsum('ij,...j->...i', A, x) + b

        return f

    @staticmethod
    def radial(center=(0, 0, 0), strength=1.0):
        """
        Radial force:

        f(x) = strength * (x - c)
        """
        c = torch.tensor(center, dtype=torch.float64)

        def f(x):
            return strength * (x - c)

        return f

    @staticmethod
    def inverse_radial(center=(0, 0, 0), strength=1.0, eps=1e-8):
        """
        Inverse radial force:

        f(x) = strength * (x - c) / ||x - c||^3
        """
        c = torch.tensor(center, dtype=torch.float64)

        def f(x):
            r = x - c
            norm = torch.linalg.norm(r, dim=-1, keepdim=True) + eps
            return strength * r / norm**3

        return f

    # ==========================================================
    # SURFACE TRACTIONS
    # ==========================================================

    @staticmethod
    def constant_traction(t_vec):
        """
        Constant traction on surface.
        """
        t_vec = torch.tensor(t_vec, dtype=torch.float64)

        def t(x, n):
            return t_vec.expand_as(x)

        return t

    @staticmethod
    def normal_pressure(p):
        """
        Pressure applied along normal:

        t = p * n
        """
        def t(x, n):
            return p * n

        return t

    @staticmethod
    def directional_pressure(direction, magnitude):
        """
        Constant direction traction:

        t = magnitude * d
        """
        d = torch.tensor(direction, dtype=torch.float64)

        def t(x, n):
            return magnitude * d.expand_as(x)

        return t

    @staticmethod
    def radial_pressure(center=(0, 0, 0), magnitude=1.0):
        """
        Radial outward traction.
        """
        c = torch.tensor(center, dtype=torch.float64)

        def t(x, n):
            r = x - c
            r = r / (torch.linalg.norm(r, dim=-1, keepdim=True) + 1e-8)
            return magnitude * r

        return t

    # ==========================================================
    # MASKED / CONDITIONAL FORCES
    # ==========================================================

    @staticmethod
    def masked(force_func, mask_func):
        """
        Apply force only where mask is True.

        mask_func(x) -> (...,) boolean
        """
        def f(x):
            mask = mask_func(x).unsqueeze(-1)
            return torch.where(mask, force_func(x), torch.zeros_like(x))

        return f

    @staticmethod
    def box_mask(bounds_min, bounds_max):
        """
        Returns a mask function for a box region.
        """
        bmin = torch.tensor(bounds_min, dtype=torch.float64)
        bmax = torch.tensor(bounds_max, dtype=torch.float64)

        def mask(x):
            return ((x >= bmin) & (x <= bmax)).all(dim=-1)

        return mask

    # ==========================================================
    # CUSTOM FORCE WRAPPER
    # ==========================================================

    @staticmethod
    def from_callable(func):
        """
        Wrap a user-defined function safely.

        func(x) must return (...,3)
        """
        def f(x):
            out = func(x)
            assert out.shape[-1] == 3
            return out

        return f