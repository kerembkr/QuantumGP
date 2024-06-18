import numpy as np
from src.solver.classic.solver import Solver
from src.utils_gpr.utils import timing


class PCG(Solver):

    def __init__(self, A, b, M=None, maxiter=20, tol=1e-8):
        super().__init__(A, b)
        self.iters = None
        self.M = M if M is not None else np.eye(A.shape[0])
        self.maxiter = maxiter
        self.tol = tol

    @timing
    def solve(self):
        """
        Conjugate Gradient Method with Preconditioning
        """

        self.x = np.zeros(self.N)                                       # initial solution guess
        r = self.b - self.A @ self.x                                    # initial residual
        z = np.linalg.solve(self.M, r)                                  # apply preconditioner
        d = z.copy()                                                    # initial search direction
        delta_new: float = r.T @ z                                      # initial squared residual
        i: int = 0                                                      # iteration counter
        while (np.sqrt(delta_new) > self.tol) and (i < self.maxiter):   # start CG method
            q = self.A @ d                                              # matrix-vector product Ad
            alpha: float = delta_new / (d.T @ q)                        # step size
            self.x = self.x + alpha * d                                 # update solution
            r = r - alpha * q                                           # update residual
            z = np.linalg.solve(self.M, r)                              # apply preconditioner
            delta_old: float = delta_new                                # save old squared residual
            delta_new = r.T @ z                                         # new squared residual
            beta: float = delta_new / delta_old                         # calculate beta
            d = z + beta * d                                            # update search direction
            i += 1                                                      # update iteration counter
            if i == self.maxiter:                                       # convergence criteria
                self.iters = self.maxiter                               # maximum number of iterations needed
                raise BaseException("no convergence")                   # no convergence
        self.iters = i                                                  # save number of iterations needed
