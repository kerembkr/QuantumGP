import numpy as np


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

    for j in range(n):
        # Compute L[j, j]
        L[j, j] = np.sqrt(A[j, j] - np.sum(L[j, :j] ** 2))

        # Compute L[j+1:n, j]
        for i in range(j + 1, n):
            L[i, j] = (A[i, j] - np.sum(L[i, :j] * L[j, :j])) / L[j, j]

    return L


if __name__ == "__main__":

    # Example usage
    M = np.array([[4, 12, -16],
                  [12, 37, -43],
                  [-16, -43, 98]], dtype=float)
    M = M @ M.T

    lower = cholesky(M)
    print(lower)
