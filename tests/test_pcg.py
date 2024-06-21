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

solver_basic = Solver()
solver_basic.N = N
solver_basic.b = b
solver_basic.A = A
solver_basic.solve()

# test