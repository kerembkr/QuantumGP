import numpy as np
from scipy.linalg import cholesky as cholesky_scipy
import random


def cholesky(A, p=None, rnd_idx=False):
    """
    Performs the Cholesky decomposition of a positive definite matrix A and
    provides an approximation to the inverse of A.

    The decomposition results in a lower triangular matrix L such that A ≈ L * L.T,
    and a low-rank approximation C ≈ A^(-1).

    Parameters
    ----------
    rnd_idx : boolean
        If set to true, the indices are chosen randomly, else they are chosen in a lexicographic order.
    A : numpy.ndarray
        The matrix to be decomposed. Must be a square, symmetric, positive definite matrix.
    p : int, optional
        The number of steps to perform in the decomposition. Defaults to the dimension
        of A if not provided.

    Returns
    -------
    L : numpy.ndarray
        The lower triangular matrix resulting from the decomposition.
    C : numpy.ndarray
        The low-rank approximation of the inverse of A.

    Raises
    ------
    ValueError
        If the matrix A is not square or not symmetric.


    Notes
    -----
    The residual can be computed using  A_ = A - L @ L.T. It is defined as the current Cholesky approximation subtracted
    from the matrix A.
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

    # dimension
    n = A.shape[0]

    # reduction
    if p is None:
        p = n

    # memory allocation
    L = np.zeros_like(A, dtype=float)  # lower triangular Cholesky factor
    C = np.zeros_like(A, dtype=float)  # inverse approximation

    # choice of index order (lexicographic or random)
    indices = list(range(n))

    for i in range(p):                              # Cholesky decomposition with Inverse Approximation
        index = i                                   # lexicographic order
        if rnd_idx:                                 # if random order was chosen
            index = random.choice(indices)          # choose random index
            indices.remove(index)                   # remove index from list
        e_i = np.eye(n)[:, index]                   # unit vector
        s_i = e_i                                   # action
        d_i = (np.eye(n) - C @ A) @ s_i             # search direction
        eta_i = s_i.T @ (A @ d_i)                   # normalization constant
        l_i = (1.0 / np.sqrt(eta_i)) * (A @ d_i)    # matrix observation
        C += (1.0 / eta_i) * np.outer(d_i, d_i)     # inverse estimate
        L[:, i] = l_i                               # lower Cholesky factor

    return L, C


if __name__ == "__main__":

    # random spd matrix
    N = 500
    M = np.random.rand(N, N)
    M = M @ M.T

    lower_, invM_ = cholesky(M, p=N)
    lower = cholesky_scipy(M, lower=True)
    invM = np.linalg.inv(M)

    # approximation quality
    print("res_L    : {:.4e}".format(np.linalg.norm(lower_ - lower)))
    print("res_invA : {:.4e}".format(np.linalg.norm(invM_ - invM)))
