import numpy as np
from scipy.stats import norm
from abc import ABC, abstractmethod


class Acquisition(ABC):

    def __init__(self, model, bounds):
        self.model = model
        self.bounds = bounds

    @abstractmethod
    def __call__(self):
        pass


class ExpectedImprovement(Acquisition):
    
    def __init__(self, model, bounds, xi=0.0):
        super().__init__(model, bounds)
        self.xi = xi

    def __call__(self, x, f_star):
        """
        Compute the Expected Improvement (EI) for a given point.

        Parameters
        ----------
        x : array_like
            The point(s) where the acquisition function is evaluated.
        f_star : float
            The current best known function value.

        Returns
        -------
        float
            The expected improvement (EI) for the point(s) x.

        """

        # get mean and standard deviation of point x
        mu, sig = self.model.predict(x)

        sig = np.diagonal(sig)

        # helping value for simplicity
        arg = (mu-f_star-self.xi)/sig

        # expected improvement
        EI = arg * sig * norm.cdf(arg) + sig * norm.cdf(arg)

        return -EI
