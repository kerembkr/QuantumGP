import numpy as np


def partial_cholesky(A):
    """
    Partial Cholesky Decomposition

    Parameters
    ----------
    A

    Returns
    -------

    """

    n = len(A)

    L = np.zeros_like(A)

    for k in range(n):
        L[k, k] = np.sqrt(L[k, k])
        for i in range(k + 1, n):
            if L[i, k] != 0:
                L[i, k] = L[i, k] / L[k, k]

        for j in range(k + 1, n):
            for i in range(j, n):
                if L[i, j] != 0:
                    L[i, j] = L[i, j] - L[i, k] * L[j, k]

        for i in range(n):
            for j in range(i + 1, n):
                L[i, j] = 0

    return L

