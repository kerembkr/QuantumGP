import unittest
import numpy as np
from parameterized import parameterized
from src.solver.classic.cholesky import Cholesky
from src.solver.classic.cg import CG
from src.solver.classic.pcg import PCG
from src.solver.quantum.vqls.vqls import VQLS


class TestSolvers(unittest.TestCase):

    @parameterized.expand([
        ("Cholesky", Cholesky),
        ("CG", CG),
        ("PCG", PCG),
    ])
    def test_solver(self, name, Solver):
        np.random.seed(42)          # fix random seed
        A = np.random.rand(20, 20)  # random matrix
        A = A @ A.T                 # create an spd matrix
        b = np.random.rand(20)      # random vector

        solver = Solver()           # initialize solver
        solver.set_lse(A, b)        # set linear system
        solver.solve()              # solve system with solver

        # check accuracy
        np.allclose(np.linalg.solve(A, b), solver.x, atol=1e-8)

    @parameterized.expand([
        ("Cholesky", Cholesky),
        ("CG", CG),
        ("PCG", PCG),
    ])
    def test_singular_matrix(self, name, Solver):
        A = np.array([[1, 2], [2, 4]])  # singular matrix
        b = np.array([1, 2])

        solver = Solver()
        with self.assertRaises(np.linalg.LinAlgError):
            solver.set_lse(A, b)
            solver.solve()

    @parameterized.expand([
        ("VQLS", VQLS),
    ])
    def test_probabilities(self, name, Solver):
        np.random.seed(42)          # fix random seed
        A = np.random.rand(2, 2)    # random matrix
        A = A @ A.T                 # create an spd matrix
        b = np.random.rand(2)       # random vector
        solver = Solver()           # initialize solver
        solver.set_lse(A, b)        # set linear system
        solver.solve()              # solve system with solver

        # compute normed solution
        x = np.linalg.solve(A, b)
        x = x / np.linalg.norm(x)
        x = x**2

        # check accuracy
        np.allclose(x, solver.xprobs, atol=1e-8)




if __name__ == '__main__':
    unittest.main()
