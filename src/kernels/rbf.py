import numpy as np
from src.kernels.kernel import Kernel


class RBFKernel(Kernel):

    def __init__(self, theta, bounds=None, hyperparams=None):
        super().__init__(theta, bounds, hyperparams)
        self.hyperparams = ["sigma", "length_scale"]

        np.testing.assert_equal(len(self.theta), len(self.hyperparams),
                                err_msg="theta and hyperparams must have the same length")

    def __call__(self, X1, X2=None, eval_gradient=False):

        def kernelval(x1, x2, eval_gradient=False):

            # kernel
            k_ = self.theta[0] ** 2 * np.exp(-(np.linalg.norm(x1 - x2) ** 2) / 2 / self.theta[1] ** 2)

            if eval_gradient:
                # kernel gradient
                dk0 = 2.0 * self.theta[0] * np.exp(-(np.linalg.norm(x1 - x2) ** 2) / 2 / self.theta[1] ** 2)
                dk1 = self.theta[0] ** 2 * np.exp(-(np.linalg.norm(x1 - x2) ** 2) / 2 / self.theta[1] ** 2) * (
                        np.linalg.norm(x1 - x2) ** 2) / self.theta[1] ** 3
                dk_ = np.array([dk0, dk1])
                return k_, dk_
            else:
                return k_

        if X2 is None:
            X2 = X1.copy()

        # Covariance matrix
        K_ = np.array([[kernelval(x1, x2) for x2 in X2] for x1 in X1])

        if eval_gradient:
            dK = np.zeros((len(self.theta), len(X1), len(X2)))
            for i, x1 in enumerate(X1):
                for j, x2 in enumerate(X2):
                    _, dk_ = kernelval(x1, x2, eval_gradient=True)
                    for k in range(len(self.theta)):
                        dK[k, i, j] = dk_[k]

            return K_, dK  # return covariance matrix and its gradient
        else:
            return K_  # only return covariance matrix
