import unittest
import numpy as np
from src.solver.classic.cholesky import Cholesky
from src.solver.classic.cg import CG
from src.solver.classic.pcg import PCG


class TestCholesky(unittest.TestCase):

    def test_sol(self):

        np.random.seed(42)          # fix random seed
        A = np.random.rand(20, 20)  # random matrix
        A = A @ A.T                 # create an spd matrix
        b = np.random.rand(20)      # random vector
        chol = Cholesky()           # init Cholesky solver
        chol.set_lse(A, b)          # set linear system
        chol.solve()                # solve system with Cholesky

        # check accuracy
        np.allclose(np.linalg.solve(A, b), chol.x, atol=1e-8)

    def test_sym_but_not_posdef(self):
        A = np.array([[0, 1], [1, 0]])  # symmetric but not positive definite
        b = np.array([1, 2])
        chol = Cholesky()
        with self.assertRaises(ValueError):
            chol.set_lse(A, b)
            chol.solve()

    def test_singular_matrix(self):
        A = np.array([[1, 2], [2, 4]])  # singular matrix
        b = np.array([1, 2])
        chol = Cholesky()
        with self.assertRaises(np.linalg.LinAlgError):
            chol.set_lse(A, b)
            chol.solve()


class TestCG(unittest.TestCase):

    def test_sol(self):

        np.random.seed(42)          # fix random seed
        A = np.random.rand(20, 20)  # random matrix
        A = A @ A.T                 # create an spd matrix
        b = np.random.rand(20)      # random vector
        cg = CG()           # init Cholesky solver
        cg.set_lse(A, b)          # set linear system
        cg.solve()                # solve system with Cholesky

        # check accuracy
        np.allclose(np.linalg.solve(A, b), cg.x, atol=1e-8)

    def test_singular_matrix(self):
        A = np.array([[1, 2], [2, 4]])  # singular matrix
        b = np.array([1, 2])
        cg = CG()
        with self.assertRaises(np.linalg.LinAlgError):
            cg.set_lse(A, b)
            cg.solve()


class TestPCG(unittest.TestCase):

    def test_sol(self):

        np.random.seed(42)          # fix random seed
        A = np.random.rand(20, 20)  # random matrix
        A = A @ A.T                 # create an spd matrix
        b = np.random.rand(20)      # random vector
        pcg = PCG()           # init Cholesky solver
        pcg.set_lse(A, b)          # set linear system
        pcg.solve()                # solve system with Cholesky

        # check accuracy
        np.allclose(np.linalg.solve(A, b), pcg.x, atol=1e-8)

    def test_singular_matrix(self):
        A = np.array([[1, 2], [2, 4]])  # singular matrix
        b = np.array([1, 2])
        pcg = PCG()
        with self.assertRaises(np.linalg.LinAlgError):
            pcg.set_lse(A, b)
            pcg.solve()


if __name__ == '__main__':
    unittest.main()
