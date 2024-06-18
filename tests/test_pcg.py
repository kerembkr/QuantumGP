import numpy as np
from src.solver.classic.cg import CG
from src.solver.classic.pcg import PCG
from src.solver.classic.solver import Solver
from scipy.sparse import csc_matrix

# Example usage
np.random.seed(42)
N = 500
A = np.random.rand(N, N)
A = A @ A.T
b = np.random.rand(N)

solver_basic = Solver(A, b)
solver_basic.solve()

solver_cg = CG(A, b, maxiter=10*N)
solver_cg.solve()
print(solver_cg.iters)

A_sparse = csc_matrix(A)
solver_pcg = PCG(A_sparse, b, maxiter=10*N, M=np.diag(np.diag(A)))
solver_pcg.solve()
print(solver_pcg.iters)
