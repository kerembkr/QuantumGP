

class Solver:

    def __init__(self, A, b):
        self.A = A
        self.b = b
        self.x = None

    def solve(self):
        pass

    @property
    def vec_x(self):
        return self.x

    @property
    def vec_b(self):
        return self.b

    @property
    def mat_A(self):
        return self.A
