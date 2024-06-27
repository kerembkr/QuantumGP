import numpy as np
from src.utils.utils import spd
from src.solver.solver import Solver
from src.linalg.preconditioned_conjugate_gradient import pcg
from src.linalg.cholesky import cholesky


class PCG(Solver):

    def __init__(self, rank=None, pre_iters=None, tol=1e-8):
        super().__init__()
        self.iters: int = 0
        self.rank = rank
        self.tol = tol
        self.invM = None
        self.P = None
        self.pre_iters = pre_iters

    def solve(self):
        """
        Solve the linear system of equations Ax=b using the conjugate gradient method
        """

        if self.pre_iters is None:
            self.P = np.eye(self.N)
        else:
            L, _ = cholesky(self.A, p=self.pre_iters, rnd_idx=True)
            self.P = L @ L.T

        self.x, self.invM = pcg(A=self.A,
                                b=self.b,
                                maxiter=self.rank,
                                atol=self.tol,
                                rtol=self.tol,
                                P=self.P)

        return self.x


if __name__ == "__main__":
    # fix random seed
    np.random.seed(42)

    # linear system
    N = 200
    K = spd(N)
    y = np.random.rand(N)

    # solve system
    solver = PCG()
    solver.set_lse(A=K, b=y)
    x, invK = solver.solve()

    print(np.linalg.norm(x - np.linalg.solve(K, y)))
