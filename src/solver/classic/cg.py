import numpy as np
from solver import Solver


class CG(Solver):

    def __init__(self, A, b, maxiter=None, tol=1e-8):
        super().__init__(A, b)
        self.maxiter = maxiter
        self.tol = tol
        if maxiter is None:
            self.maxiter = len(b)

    def solve(self):
        """
        Conjugate Gradient Method

        """
        self.x = np.zeros(len(self.A))

        # initialization
        r = self.b - self.A @ self.x
        d = np.zeros(len(self.b))
        i = 0

        while (np.linalg.norm(r) > self.tol) and (i <= self.maxiter):

            # residual
            r = self.b - self.A @ self.x

            # search direction
            if i == 0:
                dp = r
            else:
                dp = r - (r.T @ (self.A @ d)) / (d.T @ (self.A @ d)) * d

            # solution estimate
            self.x = self.x + (r.T @ r) / (dp.T @ (self.A @ dp)) * dp

            # update iteration counter
            i += 1
            d = dp

            # convergence criteria
            if i == self.maxiter:
                raise BaseException("no convergence")
