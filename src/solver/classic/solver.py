import numpy as np
from abc import abstractmethod
from src.utils_gpr.utils import timing


class Solver:

    def __init__(self, A, b):
        self.A = A
        self.b = b
        self.x = None
        self.N = len(b)

    @timing
    def solve(self):
        self.x = np.linalg.solve(self.A, self.b)

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

    @property
    def condA(self):
        return np.linalg.cond(self._A)
