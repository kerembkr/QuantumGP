import numpy as np




def cholesky_solve(L, b):
    """
  quadratic complexity O(n^2)
  """

    # dimension
    n = len(b)

    # initialize vectors
    z = np.zeros(n)
    x = np.zeros(n)

    # forward substitution
    z[0] = b[0] / L[0, 0]
    for i in range(1, n):
        sum = 0.0
        for j in range(i + 1):
            sum += L[i, j] * z[j]
        z[i] = (b[i] - sum) / L[i, i]

    # backward substitution
    L = np.transpose(L)
    for i in range(n - 1, -1, -1):
        sum = 0.0
        for j in range(i + 1, n):
            sum += L[i, j] * x[j]
        x[i] = (z[i] - sum) / L[i, i]

    return x
