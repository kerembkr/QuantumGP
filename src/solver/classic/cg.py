import numpy as np
from src.utils.utils import spd
from src.solver.solver import Solver


class CG(Solver):

    def __init__(self, maxiter=None, tol=1e-8):
        super().__init__()
        self.iters: int = 0
        self.maxiter = maxiter
        self.tol: float = tol

    def solve(self):
        """
        Solve the linear system of equations Ax=b using the conjugate gradient method
        """

        self.x = self.cg()

    def cg(self):
        """
        Conjugate gradient method

        Returns
        -------

        """
        if self.maxiter is None:
            self.maxiter = 10 * self.N

        self.x = np.zeros(self.N)  # initial solution guess
        r = self.b - self.A @ self.x  # initial residual
        d = r.copy()  # initial search direction
        i = 0  # iteration counter
        while np.linalg.norm(r) > self.tol:  # start CG method

            self.x = self.x + (r.T @ r) / (d.T @ (self.A @ d)) * d  # update solution
            delta_old = r.T @ r  # save old squared residual
            r = self.b - self.A @ self.x  # update residual
            d = r + (r.T @ r / delta_old) * d  # update search direction

            i += 1  # update iteration counter
            if i == self.maxiter:  # convergence criteria
                raise RuntimeError("No convergence.")  # no convergence

        return self.x

    def cg_winv(self):
        """
        Conjugate gradient method with inverse approximation

        Returns
        -------

        """

        if self.maxiter is None:
            maxiter = 10 * self.N

        x = np.zeros(self.N)  # initial solution guess
        C = np.zeros_like(self.A)  # inverse approximation
        r = self.b - self.A @ x  # initial residual
        i = 0  # iteration counter
        while np.linalg.norm(r) > self.tol:  # CG loop
            r = self.b - self.A @ x  # residual
            s = r  # action
            alpha = s.T @ r  # observation
            d = (np.eye(self.N) - C @ self.A) @ s  # search direction
            eta = s.T @ (self.A @ d)  # normalization constant
            C += 1.0 / eta * np.outer(d, d)  # inverse estimate
            x += alpha / eta * d  # solution estimate
            i += 1  # update iteration counter
            if i == maxiter:  # convergence criteria
                raise RuntimeError("No convergence.")  # no convergence

        return x, C


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
    solver.solve()
