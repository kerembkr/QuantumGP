import numpy as np
from src.utils.utils import timing
from src.solver.solver import Solver
from src.linalg.decomposition.cholesky import cholesky


class Cholesky(Solver):

    def __init__(self):
        super().__init__()
        self.L = None

    @timing
    def solve(self):
        """
        Cholesky Solver
        """

        # self.L = cholesky(self.A)     # decompose
        self.L = partial_cholesky(self.A, p=3)  # decompose
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
        for i in range(self.N - 1, -1, -1):
            _sum = 0.0
            for j in range(i + 1, self.N):
                _sum += self.L[j, i] * self.x[j]
            self.x[i] = (z[i] - _sum) / self.L[i, i]

        return self.x


if __name__ == "__main__":

    np.random.seed(42)
    N = 5
    A = np.random.rand(N, N)
    A = A @ A.T
    b = np.random.rand(N)

    solver = Cholesky()
    solver.set_lse(A=A, b=b)
    solver.solve()

    print(solver.x)
    print(np.linalg.solve(A, b))
