import numpy as np
from src.utils.utils import spd
from src.solver.solver import Solver
from src.linalg.conjugate_gradient import cg


class CG(Solver):

    def __init__(self, rank=None, tol=1e-8):
        super().__init__()
        self.iters: int = 0
        self.rank = rank
        self.tol = tol
        self.invM = None

    def solve(self):
        """
        Solve the linear system of equations Ax=b using the conjugate gradient method
        """

        self.x, self.invM = cg(A=self.A,
                               b=self.b,
                               maxiter=self.rank,
                               atol=self.tol,
                               rtol=self.tol)

        return self.x


if __name__ == "__main__":
    # fix random seed
    np.random.seed(42)

    # linear system
    N = 200
    K = spd(N)
    y = np.random.rand(N)

    # solve system
    solver = CG()
    solver.set_lse(A=K, b=y)
    x, invK = solver.solve()

    print(np.linalg.norm(x - np.linalg.solve(K, y)))
