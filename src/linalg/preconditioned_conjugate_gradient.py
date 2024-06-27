import numpy as np


def pcg(A, b, maxiter=None, tol=1e-8):
    """
    Conjugate gradient method

    Parameters
    ----------
    A
    b
    maxiter
    tol

    Returns
    -------

    """

    n = len(b)

    if maxiter is None:
        maxiter = 10 * n

    x = np.zeros(n)  # initial solution guess
    r = b - A @ x  # initial residual
    d = r.copy()  # initial search direction
    i = 0  # iteration counter
    while np.linalg.norm(r) > tol:  # start CG method

        x = x + (r.T @ r) / (d.T @ (A @ d)) * d  # update solution
        delta_old = r.T @ r  # save old squared residual
        r = b - A @ x  # update residual
        d = r + (r.T @ r / delta_old) * d  # update search direction

        i += 1  # update iteration counter
        if i == maxiter:  # convergence criteria
            # raise RuntimeError("No convergence.")           # no convergence
            return x

    return x


def pcg_winv(A, b, maxiter=None, rtol=1e-6, atol=1e-6, P=None):
    """
    Conjugate gradient method

    Parameters
    ----------
    P
    atol
    rtol
    A
    b
    maxiter

    Returns
    -------

    """

    n = len(b)
    if maxiter is None:
        maxiter = 10 * n

    if P is None:
        P = np.eye(n)

    invP = mat_inv_lemma(np.eye(n), P, np.eye(n), np.eye(n))

    x = np.zeros(n)  # initial solution guess
    C = np.zeros_like(A)  # inverse approximation
    r = b - A @ x  # initial residual
    i = 0  # iteration counter
    tol = max(rtol * np.linalg.norm(b), atol)  # threshold
    while np.linalg.norm(r) > tol:  # CG loop
        r = b - A @ x  # residual
        s = invP @ r  # action (NEW)
        alpha = s.T @ r  # observation
        d = (np.eye(n) - C @ A) @ s  # search direction
        eta = s.T @ (A @ d)  # normalization constant
        C += 1.0 / eta * np.outer(d, d)  # inverse estimate
        x += alpha / eta * d  # solution estimate
        i += 1  # update iteration counter
        if i == maxiter:  # convergence criteria
            # raise RuntimeError("No convergence.")   # no convergence
            return x, C

    return x, C


def mat_inv_lemma(A, U, C, V):

    invA = np.linalg.inv(A)
    invC = np.linalg.inv(C)

    a = invA @ U
    b = np.linalg.inv(invC + V.T @ (invA @ U))
    c = V.T @ invA

    inv_mat = invA - a @ (b @ c)

    return inv_mat


if __name__ == "__main__":
    np.random.seed(42)
    N = 3
    A_ = np.random.rand(N, N)
    A_ = A_ @ A_.T
    b_ = np.random.rand(N)

    xsol1 = pcg(A_, b_)
    xsol2, invA_ = pcg_winv(A_, b_)

    print(np.linalg.norm(xsol1 - np.linalg.solve(A_, b_)))
