import numpy as np
from src.solver.solver import Solver

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