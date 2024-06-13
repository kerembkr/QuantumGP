import numpy as np


def sphere(x):
    # N-dimensional Sphere function
    return np.sum(x ** 2, axis=-1)


def ackley(x):
    # N-dimensional Ackley function
    a = 20
    b = 0.2
    c = 2 * np.pi
    d = x.shape[-1]
    sum1 = np.sum(x ** 2, axis=-1)
    sum2 = np.sum(np.cos(c * x), axis=-1)
    term1 = -a * np.exp(-b * np.sqrt(sum1 / d))
    term2 = -np.exp(sum2 / d)
    return term1 + term2 + a + np.exp(1)


def rosenbrock(x):
    # N-dimensional Rosenbrock function
    return np.sum(100.0 * (x[..., 1:] - x[..., :-1] ** 2.0) ** 2.0 + (1 - x[..., :-1]) ** 2.0, axis=-1)


def rastrigin(x):
    # N-dimensional Rastrigin function
    d = x.shape[-1]
    return 10 * d + np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x), axis=-1)


def griewank(x):
    # N-dimensional Griewank function
    d = x.shape[-1]
    sum_term = np.sum(x ** 2 / 4000.0, axis=-1)
    prod_term = np.prod(np.cos(x / np.sqrt(np.arange(1, d + 1))), axis=-1)
    return sum_term - prod_term + 1


def levy(x):
    # N-dimensional Levy function
    w = 1 + (x - 1) / 4
    term1 = np.sin(np.pi * w[..., 0]) ** 2
    term3 = (w[..., -1] - 1) ** 2 * (1 + np.sin(2 * np.pi * w[..., -1]) ** 2)
    term2 = np.sum((w[..., :-1] - 1) ** 2 * (1 + 10 * np.sin(np.pi * w[..., :-1] + 1) ** 2), axis=-1)
    return term1 + term2 + term3
