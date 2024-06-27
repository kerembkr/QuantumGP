import numpy as np
from src.utils.utils import spd
from src.solver.solver import Solver
from src.linalg.cholesky import cholesky


class Cholesky(Solver):

    def __init__(self, rank=None, rnd_idx=False):
        super().__init__()
        self.L = None
        self.rank = rank
        self.rnd_idx = rnd_idx
        self.invM = None

    def solve(self):
        """
        Cholesky Solver
        """

        self.L, self.invM = cholesky(self.A, p=self.rank, rnd_idx=self.rnd_idx)    # decompose
        # self.x = self.cholesky_solve()  # solve
        self.x = self.invM @ self.b

        return self.x

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

    # fix random seed
    np.random.seed(42)

    # linear system
    N = 300
    A = spd(N)
    b = np.random.rand(N)

    # solve system
    solver = Cholesky(rank=N)
    solver.set_lse(A=A, b=b)
    solver.solve()

    # accuracy
    print("sol ", np.linalg.norm(solver.x - np.linalg.solve(A, b)))
    print("inv ", np.linalg.norm(solver.invM - np.linalg.inv(A)))
