import numpy as np
from src.solver.classic.cg import CG
from src.solver.classic.solver import Solver

# Example usage
A = np.array([[3, 1], [1, 2]])
b = np.array([9, 8])

solver_basic = Solver(A, b)
solver_basic.solve()
print("Solution x:", solver_basic.x)

solver_cg = CG(A, b, maxiter=20)
solver_cg.solve()
print("Solution x:", solver_cg.x)
