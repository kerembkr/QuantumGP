import numpy as np
from src.utils.utils import timing


class Solver:

    def __init__(self):
        self._A = None
        self._b = None
        self._x = None
        self._N = None

    @timing
    def solve(self):
        self._x = np.linalg.solve(self._A, self._b)

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
    def N(self):
        return self._N

    @N.setter
    def N(self, value):
        self._N = value

    @property
    def A(self):
        return self._A

    @A.setter
    def A(self, value):
        self._A = value

    def set_lse(self, A, b):
        self.A = A
        self.b = b
        self.N = len(b)

    @property
    def condA(self):
        return np.linalg.cond(self._A)
