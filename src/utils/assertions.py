import numpy as np


def assert_not_none(value, message):
    if value is None:
        raise ValueError(message)


def assert_symmetric(A):
    if not (A == A.T).all():
        raise ValueError("Matrix A must be symmetric.")


def assert_square(A):
    if A.shape[0] != A.shape[1]:
        raise ValueError("Matrix A must be square.")


def assert_positive_definite(A):
    if not np.all(np.linalg.eigvals(A) > 0):
        raise ValueError("Matrix A must be positive definite.")


def assert_not_singular(A):
    if np.any(np.linalg.eigvals(A) == 0):
        raise np.linalg.LinAlgError("Matrix A is singular.")
