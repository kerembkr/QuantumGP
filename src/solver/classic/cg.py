import numpy as np
from src.solver.classic.solver import Solver


class CG(Solver):

    def __init__(self, A, b, maxiter=20, tol=1e-8):
        super().__init__(A, b)
        self.maxiter = maxiter
        self.tol = tol

    def solve(self):
        """
        Conjugate Gradient Method
        """

        self.x = np.zeros(len(self.A))                  # initial solution guess
        r = self.b - self.A @ self.x                    # initial residual
        d = r.copy()                                    # initial search direction
        delta_new: float = r.T @ r                      # initial squared residual
        i: int = 0                                      # iteration counter

        while (np.sqrt(delta_new) > self.tol) and (i < self.maxiter):

            q = self.A @ d                              # matrix-vector product Ad
            alpha: float = delta_new / (d.T @ q)        # step size
            self.x = self.x + alpha * d                 # update solution
            r = r - alpha * q                           # update residual
            delta_old: float = delta_new                # save old squared residual
            delta_new = r.T @ r                         # new squared residual
            print(np.sqrt(delta_new))
            beta: float = delta_new / delta_old         # calculate beta
            d = r + beta * d                            # update search direction
            i += 1                                      # update iteration counter
            if i == self.maxiter:                       # convergence criteria
                raise BaseException("no convergence")   # no convergence

