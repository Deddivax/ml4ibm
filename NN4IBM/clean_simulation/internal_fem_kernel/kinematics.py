import torch


class Kinematics:
    @staticmethod
    def B_matrix(dN3_dx):
        dN3_dx = torch.as_tensor(dN3_dx)
        if dN3_dx.ndim not in (3, 4):
            raise ValueError(f"dN3_dx must have 3 or 4 dimensions, got {dN3_dx.shape}")

        if dN3_dx.ndim == 3:
            dN3_dx = dN3_dx.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False

        # Preserve all batch dimensions that precede (n_nodes, 3), including
        # the Gauss-point dimension for inputs shaped like (g, n_nodes, 3).
        batch_shape = dN3_dx.shape[:-2]
        n_nodes = dN3_dx.shape[-2]
        flat = dN3_dx.reshape(-1, n_nodes, 3)
        B = torch.zeros((flat.shape[0], 6, 3 * n_nodes), dtype=dN3_dx.dtype, device=dN3_dx.device)

        dN3x = flat[..., 0]
        dN3y = flat[..., 1]
        dN3z = flat[..., 2]

        B[:, 0, 0::3] = dN3x
        B[:, 1, 1::3] = dN3y
        B[:, 2, 2::3] = dN3z
        B[:, 3, 0::3] = dN3y
        B[:, 3, 1::3] = dN3x
        B[:, 4, 1::3] = dN3z
        B[:, 4, 2::3] = dN3y
        B[:, 5, 0::3] = dN3z
        B[:, 5, 2::3] = dN3x

        B = B.reshape(*batch_shape, 6, 3 * n_nodes)
        return B[0] if squeeze else B