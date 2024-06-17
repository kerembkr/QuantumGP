import numpy as np
from abc import abstractmethod

class Solver:

    def __init__(self, A, b):
        self.A = A
        self.b = b
        self.x = None

    def solve(self):
        self.x = np.linalg.solve(self.A, self.b)

    @abstractmethod
    def precon(self):

    @property
    def x(self):
        return self._x

    @x.setter
    def x(self, value):
        self._x = value

    @property
    def b(self):
        return self._b

    @b.setter
    def b(self, value):
        self._b = value

    @property
    def A(self):
        return self._A

    @A.setter
    def A(self, value):
        self._A = value

# Example usage
A = np.array([[3, 1], [1, 2]])
b = np.array([9, 8])

solver = Solver(A, b)
solver.solve()
print("Solution x:", solver.x)
