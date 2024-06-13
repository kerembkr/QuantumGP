import numpy as np
from abc import ABC, abstractmethod


class Kernel(ABC):

    def __init__(self, theta, bounds=None, hyperparams=None):

        self.theta = np.array(theta)
        self.hyperparams = hyperparams

        if bounds is None:
            self.bounds = [(1e-05, 100000.0)] * len(self.theta)
        else:
            self.bounds = bounds

    # def __call__(self, X1, X2=None, eval_gradient=False):
    #
    #     if X2 is None:
    #         X2 = X1.copy()
    #
    #     # Covariance matrix
    #     K_ = np.array([[self.k(x1, x2) for x2 in X2] for x1 in X1])
    #
    #     if eval_gradient:
    #         dK = np.zeros((len(self.theta), len(X1), len(X2)))
    #         for i, x1 in enumerate(X1):
    #             for j, x2 in enumerate(X2):
    #                 _, dk_ = self.k(x1, x2, eval_gradient=True)
    #                 for k in range(len(self.theta)):
    #                     dK[k, i, j] = dk_[k]
    #
    #         return K_, dK  # return covariance matrix and its gradient
    #     else:
    #         return K_  # only return covariance matrix

    @abstractmethod
    def __call__(self, X1, X2=None, eval_gradient=False):
        """abstract method for __call__ """

    def __add__(self, other):
        if not isinstance(other, Kernel):
            raise TypeError("Can only add Kernel objects.")
        # Create a new instance of the Sum class with both kernels
        return Sum([self, other])

    def __mul__(self, other):
        if not isinstance(other, Kernel):
            raise TypeError("Can only add Kernel objects.")
        # Create a new instance of the Sum class with both kernels
        return Product([self, other])

    @property
    def n_dims(self):
        """Returns the number of non-fixed hyperparameters of the kernel."""
        return self.theta.shape[0]

    @property
    def hyperparameters(self):
        """Returns a list of all hyperparameter specifications."""
        return [getattr(self, attr) for attr in dir(self) if attr.startswith("hyperparams")]

    def __repr__(self):
        params_repr = ', '.join(f"{name}={value!r}" for name, value in zip(self.hyperparams, self.theta))
        return f"{self.__class__.__name__}({params_repr})"


class Sum(Kernel):
    def __init__(self, kernels):
        # Initialize the hyperparameters and bounds
        theta = np.hstack([kernel.theta for kernel in kernels])
        hyperparams = [param for kernel in kernels for param in kernel.hyperparams]
        bounds = [bound for kernel in kernels for bound in kernel.bounds]
        super().__init__(theta, bounds, hyperparams)

        self.kernel1 = kernels[0]
        self.kernel2 = kernels[1]

    def __call__(self, X1, X2=None, eval_gradient=False):

        if X2 is None:
            X2 = X1.copy()

        if eval_gradient:
            K1, K1_gradient = self.kernel1(X1, X2, eval_gradient=True)
            K2, K2_gradient = self.kernel2(X1, X2, eval_gradient=True)
            return K1 + K2, [*K1_gradient, *K2_gradient]
        else:
            return self.kernel1(X1, X2) + self.kernel2(X1, X2).squeeze()

    def __repr__(self):
        return "{0} + {1}".format(self.kernel1, self.kernel2)


class Product(Kernel):
    def __init__(self, kernels):
        # Initialize the hyperparameters and bounds
        theta = np.hstack([kernel.theta for kernel in kernels])
        hyperparams = [param for kernel in kernels for param in kernel.hyperparams]
        bounds = [bound for kernel in kernels for bound in kernel.bounds]
        super().__init__(theta, bounds, hyperparams)

        self.kernel1 = kernels[0]
        self.kernel2 = kernels[1]

    def __call__(self, X1, X2=None, eval_gradient=False):

        if X2 is None:
            X2 = X1.copy()

        if eval_gradient:
            K1, K1_gradient = self.kernel1(X1, X2, eval_gradient=True)
            K2, K2_gradient = self.kernel2(X1, X2, eval_gradient=True)
            grad_term1 = K1_gradient * K2[np.newaxis, :, :]
            grad_term2 = K2_gradient * K1[np.newaxis, :, :]
            return K1 * K2, [*grad_term1, *grad_term2]
        else:
            return self.kernel1(X1, X2) * self.kernel2(X1, X2)

    def __repr__(self):
        return "{0} * {1}".format(self.kernel1, self.kernel2)
