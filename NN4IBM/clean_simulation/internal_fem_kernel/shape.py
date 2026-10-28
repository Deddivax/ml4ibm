import torch


class Shape:
    @staticmethod
    def _lagrange_1d(x, poly_order):
        if poly_order < 1:
            raise ValueError("poly_order must be >= 1")

        nodes = torch.linspace(-1.0, 1.0, poly_order + 1, dtype=x.dtype, device=x.device)
        n = poly_order + 1
        L = torch.ones((x.shape[0], n), dtype=x.dtype, device=x.device)
        dL = torch.zeros((x.shape[0], n), dtype=x.dtype, device=x.device)

        for i in range(n):
            for j in range(n):
                if j != i:
                    L[:, i] *= (x - nodes[j]) / (nodes[i] - nodes[j])

            acc = torch.zeros(x.shape[0], dtype=x.dtype, device=x.device)
            for m in range(n):
                if m == i:
                    continue
                term = torch.full((x.shape[0],), 1.0 / (nodes[i] - nodes[m]), dtype=x.dtype, device=x.device)
                for j in range(n):
                    if j != i and j != m:
                        term *= (x - nodes[j]) / (nodes[i] - nodes[j])
                acc += term
            dL[:, i] = acc

        return L, dL

    @staticmethod
    def infer_poly_order_from_nnodes(num_nodes):
        if num_nodes < 8:
            raise ValueError("num_nodes must be >= 8")
        p1 = int(round(num_nodes ** (1.0 / 3.0)))
        if p1 ** 3 != num_nodes:
            raise ValueError(f"num_nodes={num_nodes} is not a tensor-product hex node count (expected (p+1)^3).")
        return p1 - 1

    @staticmethod
    def N3(xi, eta, zeta, poly_order=1):
        if poly_order == 1:
            return 0.125 * torch.stack(
                [
                    (1 - xi) * (1 - eta) * (1 - zeta),
                    (1 + xi) * (1 - eta) * (1 - zeta),
                    (1 + xi) * (1 + eta) * (1 - zeta),
                    (1 - xi) * (1 + eta) * (1 - zeta),
                    (1 - xi) * (1 - eta) * (1 + zeta),
                    (1 + xi) * (1 - eta) * (1 + zeta),
                    (1 + xi) * (1 + eta) * (1 + zeta),
                    (1 - xi) * (1 + eta) * (1 + zeta),
                ],
                dim=-1,
            )

        Lx, _ = Shape._lagrange_1d(xi, poly_order)
        Ly, _ = Shape._lagrange_1d(eta, poly_order)
        Lz, _ = Shape._lagrange_1d(zeta, poly_order)

        rows = []
        n = poly_order + 1
        for k in range(n):
            for j in range(n):
                for i in range(n):
                    rows.append(Lx[:, i] * Ly[:, j] * Lz[:, k])
        return torch.stack(rows, dim=-1)

    @staticmethod
    def N2(u, v):
        return 0.25 * torch.stack(
            [
                (1 - u) * (1 - v),
                (1 + u) * (1 - v),
                (1 - u) * (1 + v),
                (1 + u) * (1 + v),
            ],
            dim=-1,
        )

    @staticmethod
    def dN2(u, v):
        return 0.25 * torch.stack(
            [
                torch.stack([-(1 - v), (1 - v), -(1 + v), (1 + v)], dim=-1),
                torch.stack([-(1 - u), -(1 + u), (1 - u), (1 + u)], dim=-1),
            ],
            dim=0,
        )

    @staticmethod
    def Qf(u, v, X):
        return torch.dot(Shape.N2(u, v), X)

    @staticmethod
    def cross_det(u, v, X):
        dN = Shape.dN2(u, v)
        xu = dN[0] @ X
        xv = dN[1] @ X
        return torch.cross(xu, xv)

    @staticmethod
    def dN3_dxi(xi, eta, zeta, poly_order=1):
        if poly_order == 1:
            return 0.125 * torch.stack(
                [
                    torch.stack([-(1 - eta) * (1 - zeta), -(1 - xi) * (1 - zeta), -(1 - xi) * (1 - eta)], dim=-1),
                    torch.stack([(1 - eta) * (1 - zeta), -(1 + xi) * (1 - zeta), -(1 + xi) * (1 - eta)], dim=-1),
                    torch.stack([(1 + eta) * (1 - zeta), (1 + xi) * (1 - zeta), -(1 + xi) * (1 + eta)], dim=-1),
                    torch.stack([-(1 + eta) * (1 - zeta), (1 - xi) * (1 - zeta), -(1 - xi) * (1 + eta)], dim=-1),
                    torch.stack([-(1 - eta) * (1 + zeta), -(1 - xi) * (1 + zeta), (1 - xi) * (1 - eta)], dim=-1),
                    torch.stack([(1 - eta) * (1 + zeta), -(1 + xi) * (1 + zeta), (1 + xi) * (1 - eta)], dim=-1),
                    torch.stack([(1 + eta) * (1 + zeta), (1 + xi) * (1 + zeta), (1 + xi) * (1 + eta)], dim=-1),
                    torch.stack([-(1 + eta) * (1 + zeta), (1 - xi) * (1 + zeta), (1 - xi) * (1 + eta)], dim=-1),
                ],
                dim=1,
            )

        Lx, dLx = Shape._lagrange_1d(xi, poly_order)
        Ly, dLy = Shape._lagrange_1d(eta, poly_order)
        Lz, dLz = Shape._lagrange_1d(zeta, poly_order)

        rows = []
        n = poly_order + 1
        for k in range(n):
            for j in range(n):
                for i in range(n):
                    rows.append(
                        torch.stack(
                            [
                                dLx[:, i] * Ly[:, j] * Lz[:, k],
                                Lx[:, i] * dLy[:, j] * Lz[:, k],
                                Lx[:, i] * Ly[:, j] * dLz[:, k],
                            ],
                            dim=-1,
                        )
                    )
        return torch.stack(rows, dim=1)

    @staticmethod
    def _canonical_coords(coords):
        if coords.ndim == 3 and coords.shape[0] == 1:
            return coords[0]
        return coords

    @staticmethod
    def _canonical_dN(dN3_dxi):
        if dN3_dxi.ndim == 4 and dN3_dxi.shape[-2:] == (3, 8):
            return dN3_dxi.transpose(-1, -2)
        if dN3_dxi.ndim == 3 and dN3_dxi.shape[-2:] == (3, 8):
            return dN3_dxi.transpose(-1, -2)
        return dN3_dxi

    @staticmethod
    def jacobian(dN3_dxi, coords):
        coords = Shape._canonical_coords(coords)
        dN3_dxi = Shape._canonical_dN(dN3_dxi)

        if dN3_dxi.ndim == 4:
            if coords.ndim != 3 or coords.shape[-2:] != (dN3_dxi.shape[-2], 3):
                raise ValueError(f"Unsupported shapes: dN3_dxi={dN3_dxi.shape}, coords={coords.shape}")
            return torch.einsum("egnk,enj->egkj", dN3_dxi, coords)

        if dN3_dxi.ndim == 3:
            if coords.ndim != 2 or coords.shape[-2:] != (dN3_dxi.shape[-2], 3):
                raise ValueError(f"Unsupported shapes: dN3_dxi={dN3_dxi.shape}, coords={coords.shape}")
            return torch.einsum("gnk,nj->gkj", dN3_dxi, coords)

        raise ValueError(f"Unsupported shapes: dN3_dxi={dN3_dxi.shape}, coords={coords.shape}")

    @staticmethod
    def dN3_dx(dN3_dxi, coords):
        J = Shape.jacobian(dN3_dxi, coords)
        invJ = torch.linalg.inv(J)
        dN3_dxi = Shape._canonical_dN(dN3_dxi)
        dN3_dx = torch.einsum("...nk,...ki->...ni", dN3_dxi, invJ)
        return dN3_dx, J

    @staticmethod
    def inverse_map(x, coords, tol=1e-10, max_iter=20, poly_order=1):
        xi = torch.zeros(3, dtype=torch.float64, device=x.device)
        coords = Shape._canonical_coords(coords)

        for _ in range(max_iter):
            N3 = Shape.N3(xi[0:1], xi[1:2], xi[2:3], poly_order=poly_order).squeeze(0)
            x_map = torch.matmul(N3, coords)
            r = x_map - x
            if torch.norm(r) < tol:
                return xi

            dN3 = Shape.dN3_dxi(xi[0:1], xi[1:2], xi[2:3], poly_order=poly_order).squeeze(0)
            J = torch.matmul(dN3.transpose(0, 1), coords)
            xi = xi - torch.linalg.solve(J, r)

        return xi