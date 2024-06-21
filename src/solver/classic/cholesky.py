import numpy as np
from src.utils.utils import timing
from src.solver.solver import Solver
from src.linalg.decomposition.chol import cholesky_decompose


class Cholesky(Solver):

    def __init__(self):
        super().__init__()
        self.L = None

    @timing
    def solve(self):
        """
        Cholesky Solver
        """
        if self.A is None or self.b is None:
            raise ValueError("Matrix A and vector b must be set before solving.")

        self.L = cholesky_decompose(self.A)     # decompose
        self.x = self.cholesky_solve()          # solve

    def cholesky_solve(self):
        """
        Solve Linear System with Lower Triangular Matrix
        quadratic complexity O(n^2)
        """

        # initialize vectors
        z = np.zeros(self.N)
        self.x = np.zeros(self.N)

        # forward substitution
        z[0] = self.b[0] / self.L[0, 0]
        for i in range(1, self.N):
            _sum = 0.0
            for j in range(i + 1):
                _sum += self.L[i, j] * z[j]
            z[i] = (self.b[i] - _sum) / self.L[i, i]

        # backward substitution
        self.L = np.transpose(self.L)
        for i in range(self.N - 1, -1, -1):
            _sum = 0.0
            for j in range(i + 1, self.N):
                _sum += self.L[i, j] * self.x[j]
            self.x[i] = (z[i] - _sum) / self.L[i, i]

        return self.x