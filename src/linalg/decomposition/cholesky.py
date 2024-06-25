import numpy as np
from scipy.linalg import cholesky as cholesky_scipy


def cholesky(A):
    """
    Performs the Cholesky decomposition of a positive definite matrix A.
    The decomposition is such that A = L * L.T where L is a lower triangular matrix.

    Parameters
    ----------
    A : numpy.ndarray
        The matrix to be decomposed. Must be a square, symmetric, positive definite matrix.

    Returns
    -------
    L : numpy.ndarray
        The lower triangular matrix resulting from the decomposition.

    Raises
    ------
    ValueError
        If the matrix A is not square or not positive definite.
    """
    # Ensure A is a numpy array
    A = np.array(A, dtype=float)

    # Check if the matrix is square
    n, m = A.shape
    if n != m:
        raise ValueError("Matrix must be square")

    # Check if the matrix is symmetric
    if not np.allclose(A, A.T):
        raise ValueError("Matrix must be symmetric")

    n = A.shape[0]
    L = np.zeros_like(A, dtype=float)
    A_ = A.copy()

    for j in range(n):

        # compute j-th row of L
        I = np.zeros(n)
        I[j] = np.sqrt(A[j, j] - np.sum(L[j, :j] ** 2))
        for i in range(j + 1, n):
            I[i] = (A[i, j] - np.sum(L[i, :j] * L[j, :j])) / I[j]

        # save j-th column in L
        L[:, j] = I

        # residual
        A_ = A_ - np.outer(I, I.T)

    return L


if __name__ == "__main__":

    # Example usage
    M = np.array([[4, 12, -16],
                  [12, 37, -43],
                  [-16, -43, 98]], dtype=float)
    M = M @ M.T

    lower = cholesky(M)
    lower_scipy = cholesky_scipy(M, lower=True)
    print(lower)
    print(lower_scipy)
