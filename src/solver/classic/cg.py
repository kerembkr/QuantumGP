
def cg(_A, _b, maxiter=100, tol=1e-8):
    """
    Conjugate Gradient Method

    :param _A: matrix
    :param _b: vector
    :param maxiter: maximum number of iterations
    :param tol: tolerance
    :return: solution vector x
    """
    x = np.zeros(len(_A))

    # initialization
    r = _b - _A @ x
    d = np.zeros(len(_b))
    i = 0

    while (np.linalg.norm(r) > tol) and (i <= maxiter):

        # residual
        r = _b - _A @ x

        # search direction
        if i == 0:
            dp = r
        else:
            dp = r - (r.T @ (_A @ d)) / (d.T @ (_A @ d)) * d

        # solution estimate
        x = x + (r.T @ r) / (dp.T @ (_A @ dp)) * dp

        # update iteration counter
        i += 1
        d = dp

        # convergence criteria
        if i == maxiter:
            raise BaseException("no convergence")

    return x
