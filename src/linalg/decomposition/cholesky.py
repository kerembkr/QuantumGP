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

    # Initialize the lower triangular matrix L
    L = np.zeros_like(A)

    # Perform the Cholesky decomposition
    for i in range(n):
        for j in range(i + 1):
            _sum = sum(L[i, k] * L[j, k] for k in range(j))
            if i == j:  # Diagonal elements
                if A[i, i] - _sum <= 0:
                    raise ValueError("Matrix is not positive definite")
                L[i, j] = np.sqrt(A[i, i] - _sum)
            else:
                L[i, j] = (A[i, j] - _sum) / L[j, j]

    return L


if __name__ == "__main__":

    # Example usage
    A = np.array([[4, 12, -16],
                  [12, 37, -43],
                  [-16, -43, 98]], dtype=float)

    try:u
        L = cholesky(A)
        print("Matrix A:")
        print(A)
        print("\nCholesky Decomposition (L):")
        print(L)
    except ValueError as e:
        print(e)
