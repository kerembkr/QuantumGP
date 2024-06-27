import numpy as np
from src.utils.utils import spd
from src.solver.solver import Solver
from src.linalg.preconditioned_conjugate_gradient import pcg, pcg_winv
from src.linalg.cholesky import cholesky


class PCG(Solver):

    def __init__(self, rank=None, tol=1e-8):
        super().__init__()
        self.iters: int = 0
        self.rank = rank
        self.tol = tol
        self.invM = None
        self.P = None

    def solve(self):
        """
        Solve the linear system of equations Ax=b using the conjugate gradient method
        """

        L, _ = cholesky(self.A, p=5, rnd_idx=True)

        self.P = L @ L.T

        self.x, self.invM = pcg_winv(A=self.A,
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
