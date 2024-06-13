import numpy as np


def quadratic(x, y):
    # 2D quadratic function
    return x ** 2 + y ** 2


def sinc_2d(x, y):
    # 2D sinc function
    return np.sinc(np.sqrt(x ** 2 + y ** 2))


def gaussian_2d(x, y):
    # 2D Gaussian function
    return np.exp(-(x ** 2 + y ** 2))


def sinusoidal_2d(x, y):
    # 2D sinusoidal function
    return np.sin(x) * np.cos(y)


def rastrigin_2d(x, y):
    # 2D Rastrigin function (commonly used in optimization)
    return 10 * 2 + (x ** 2 - 10 * np.cos(2 * np.pi * x)) + (y ** 2 - 10 * np.cos(2 * np.pi * y))


def rosenbrock_2d(x, y):
    # 2D Rosenbrock function (commonly used in optimization)
    return (1 - x) ** 2 + 100 * (y - x ** 2) ** 2
