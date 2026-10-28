import numpy as np

class Material:
    def __init__(self, E, nu):
        self.E = E
        self.nu = nu

        self.C = self._compute_elasticity_matrix()

    def _compute_elasticity_matrix(self):
        E = self.E
        nu = self.nu

        coef = E / ((1 + nu) * (1 - 2 * nu))

        C = coef * np.array([
            [1-nu,   nu,   nu,   0,     0,     0],
            [nu,   1-nu,   nu,   0,     0,     0],
            [nu,     nu, 1-nu,   0,     0,     0],
            [0,       0,    0, (1-2*nu)/2, 0, 0],
            [0,       0,    0, 0, (1-2*nu)/2, 0],
            [0,       0,    0, 0, 0, (1-2*nu)/2]
        ])

        return C