import numpy as np
from src.utils.utils import timing
from src.solver.solver import Solver
from src.utils.assertions import (assert_not_none, assert_symmetric, assert_square, assert_not_singular,
                                  assert_positive_definite)

class CG(Solver):

    def __init__(self, maxiter=None, tol=1e-8):
        super().__init__()
        self.iters: int = 0
        self.maxiter = maxiter
        self.tol: float = tol

    @timing
    def solve(self):
        """
        Conjugate Gradient Method
        """

        assert_not_none(self.A, "Matrix A must be set before solving.")
        assert_not_none(self.b, "Vector b must be set before solving.")
        # assert_symmetric(self.A)
        assert_square(self.A)
        assert_not_singular(self.A)
        # assert_positive_definite(self.A)

        if self.maxiter is None:
            self.maxiter = 10 * self.N

        self.x = np.zeros(self.N)  # initial solution guess
        r = self.b - self.A @ self.x  # initial residual
        d = r.copy()  # initial search direction
        delta_new: float = r.T @ r  # initial squared residual
        i: int = 0  # iteration counter
        while (np.sqrt(delta_new) > self.tol) and (i < self.maxiter):  # start CG method
            q = self.A @ d  # matrix-vector product Ad
            alpha: float = delta_new / (d.T @ q)  # step size
            self.x = self.x + alpha * d  # update solution
            r = r - alpha * q  # update residual
            delta_old: float = delta_new  # save old squared residual
            delta_new = r.T @ r  # new squared residual
            beta: float = delta_new / delta_old  # calculate beta
            d = r + beta * d  # update search direction
            i += 1  # update iteration counter
            if i == self.maxiter:  # convergence criteria
                self.iters = self.maxiter  # maximum number of iterations needed
                raise RuntimeError("No convergence.")  # no convergence
        self.iters = i  # save number of iterations needed

