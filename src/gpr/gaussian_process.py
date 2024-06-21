import numpy as np
import scipy.optimize
from numpy.random import randn
import matplotlib.pyplot as plt
from operator import itemgetter
from src.utils.utils import save_fig
from matplotlib.ticker import MaxNLocator
from scipy.linalg import cholesky, solve_triangular
from src.utils.acquisition import ExpectedImprovement

class GP:
    def __init__(self, kernel, optimizer=None, alpha_=1e-10, n_restarts_optimizer=0, solver=None, precon=None,
                 acq_func=None):
        self.f_star = None
        self.optimizer = optimizer
        self.solver = solver
        self.alpha_ = alpha_
        self.precon = precon
        self.n_targets = None
        self.alpha = None
        self.L = None
        self.y_train = None
        self.X_train = None
        self.X_test = None
        self.kernel = kernel
        self.n = None
        self.n_restarts_optimizer = n_restarts_optimizer
        self.acq_func = acq_func

    def fit(self, X, y):
        """Fit Gaussian process regression model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features) or list of object
            Feature vectors or other representations of training data.

        y : array-like of shape (n_samples,) or (n_samples, n_targets)
            Target values.

        Returns
        -------
        self : object
            GaussianProcessRegressor class instance.
        """

        self.X_train = X
        self.y_train = y
        self.n = len(y)

        # Choose hyperparameters based on maximizing the log-marginal likelihood
        if self.optimizer is not None:
            self.hyperopt()

        # K_ = K + sigma^2 I
        K_ = self.kernel(self.X_train)

        K_[np.diag_indices_from(K_)] += self.alpha_

        # NEW
        self.solver.set_lse(A=K_, b=self.y_train)
        self.solver.solve()
        self.alpha = self.solver.x

        # OLD (CHOLESKY)
        # K_ = L*L^T --> L
        self.L = cholesky(K_, lower=True, check_finite=False)

        #  alpha = L^T \ (L \ y)
        #self.alpha = cho_solve((self.L, True), self.y_train, check_finite=False)

        return self

    def hyperopt(self):

        def obj_func(theta, eval_gradient=True):
            if eval_gradient:
                lml, grad = self.log_marginal_likelihood(theta, eval_gradient=True)
                return lml, grad  # why not working for -lml, -grad??
            else:
                lml = self.log_marginal_likelihood(theta, eval_gradient=False)
                return lml

        # First optimize starting from theta specified in kernel
        optima = [(self._constrained_optimization(obj_func, self.kernel.theta, self.kernel.bounds))]

        # Additional runs
        if self.n_restarts_optimizer > 0:
            if not np.isfinite(self.kernel.bounds).all():
                raise ValueError("Multiple optimizer restarts requires that all bounds are finite.")
            bounds = self.kernel.bounds
            for iteration in range(self.n_restarts_optimizer):
                # theta_initial = np.random.uniform(bounds[:, 0], bounds[:, 1])
                theta_initial = np.random.rand(len(self.kernel.theta))
                optima.append(self._constrained_optimization(obj_func, theta_initial, bounds))
        # Select result from run with minimal (negative) log-marginal likelihood
        lml_values = list(map(itemgetter(1), optima))
        self.kernel.theta = optima[np.argmin(lml_values)][0]  # optimal hyperparameters

    def _constrained_optimization(self, obj_func, initial_theta, bounds):
        if self.optimizer == "fmin_l_bfgs_b":
            opt_res = scipy.optimize.minimize(obj_func, initial_theta, method="L-BFGS-B", jac=True, bounds=bounds,
                                              options={'disp': False})
            theta_opt, func_min = opt_res.x, opt_res.fun
        else:
            try:
                opt_res = scipy.optimize.minimize(obj_func, initial_theta, method=self.optimizer, jac=True,
                                                  bounds=bounds, options={'disp': True})
                theta_opt, func_min = opt_res.x, opt_res.fun
            except ValueError:
                raise ValueError(f"Unknown optimizer {self.optimizer}.")

        return theta_opt, func_min

    def predict(self, X):
        """
        Predict using the Gaussian process regression model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features) or list of object
            Query points where the GP is evaluated.

        Returns
        -------
        y_mean : ndarray of shape (n_samples,) or (n_samples, n_targets)
            Mean of predictive distribution a query points.

        y_cov : ndarray of shape (n_samples, n_samples) or \
                (n_samples, n_samples, n_targets), optional
            Covariance of joint predictive distribution a query points.
            Only returned when `return_cov` is True.
        """

        self.X_test = X

        if not hasattr(self, "X_train"):  # Unfitted;predict based on GP prior

            n_targets = self.n_targets if self.n_targets is not None else 1
            y_mean_ = np.zeros(shape=(X.shape[0], n_targets)).squeeze()

            # covariance matrix
            y_cov_ = self.kernel(X)

            if n_targets > 1:
                y_cov_ = np.repeat(
                    np.expand_dims(y_cov_, -1), repeats=n_targets, axis=-1
                )
            return y_mean_, y_cov_

        else:  # Predict based on GP posterior

            # K(X_test, X_train)
            K_trans = self.kernel(X, self.X_train)

            # MEAN
            y_mean_ = K_trans @ self.alpha

            # STDDEV
            V = solve_triangular(self.L, K_trans.T, lower=True, check_finite=False)
            y_cov_ = self.kernel(X) - V.T @ V

            return y_mean_, y_cov_

    def log_marginal_likelihood(self, hypers, eval_gradient=False):
        """

        Parameters
        ----------
        hypers
        eval_gradient

        Returns
        -------

        """

        self.kernel.theta = hypers

        # prerequisites
        K, dK = self.kernel(self.X_train, eval_gradient=True)  # build Gram matrix, with derivatives

        G = K + self.alpha_ * np.eye(self.n)  # add noise

        (s, ld) = np.linalg.slogdet(G)  # compute log determinant of symmetric pos.def. matrix
        a = np.linalg.solve(G, self.y_train)  # G \\ Y

        # log likelihood
        loglik = np.inner(self.y_train, a) + ld  # (Y / G) * Y + log |G|

        # gradient
        dloglik = np.zeros(len(hypers))
        for i in range(len(hypers)):
            dloglik[i] = -np.inner(a, dK[i] @ a) + np.trace(np.linalg.solve(G, dK[i]))

        if eval_gradient:
            return loglik, dloglik
        else:
            return loglik

    def select_next_point(self):
        """
        Select the next point to evaluate the objective function using the acquisition function.

        Returns
        -------
        array_like
            The next point to evaluate.
        """
        self.f_star = np.min(self.y_train)  # Current best known function value
        self.acq_func = ExpectedImprovement(model=self, bounds=[(min(self.X_train), max(self.X_train))])

        opt_res = scipy.optimize.minimize(
            self.acq_func,
            # np.zeros(self.X_train.shape[1]),
            np.zeros(1),
            args=(self.f_star,),
            method="L-BFGS-B",
            bounds=self.acq_func.bounds,
            options={'disp': False}
        )

        return opt_res.x

    def plot_acquisition(self, X_test):
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        ax.set_xlabel("$X$", fontsize=15)
        ax.set_ylabel("$EI$", fontsize=15)
        ax.set_xlim([min(X_test), max(X_test)])
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.tick_params(direction="in", labelsize=15, length=10, width=0.8, colors='k')
        for edge in ["top", "bottom", "left", "right"]:
            ax.spines[edge].set_linewidth(2.0)
        plt.plot(X_test, -self.acq_func(X_test, self.f_star))
        save_fig("acquisition" + "_" + str(len(self.X_train)))

    def plot_samples(self, nsamples, save_png=False):
        """
        Plot samples with GP Model

        :param save_png:
        :param nsamples: number of samples
        :return: None
        """

        noise = 0.1

        K = self.kernel(self.X_train)

        # plot
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        ax.set_xlabel("$X$", fontsize=15)
        ax.set_ylabel("$y$", fontsize=15)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.tick_params(direction="in", labelsize=15, length=10, width=0.8, colors='k')
        for edge in ["top", "bottom", "left", "right"]:
            ax.spines[edge].set_linewidth(2.0)

        # Sort the data by the x-axis
        sorted_indices = np.argsort(self.X_train)
        X_train_sorted = self.X_train[sorted_indices]
        y_train_sorted = np.array(self.y_train)[sorted_indices]
        prior_samples = cholesky(K + self.alpha_ * np.eye(len(self.X_train))) @ randn(len(self.X_train), nsamples)
        n = X_train_sorted.shape[0]
        plt.plot(X_train_sorted, y_train_sorted, ".-", label="Training Data")
        plt.plot(X_train_sorted, prior_samples + noise * randn(n, nsamples), ".-", alpha=0.3)
        # plt.plot(X_train_sorted, prior_samples + self.alpha_ * randn(n, nsamples), ".-")
        delta = (max(self.y_train)-min(self.y_train))/5.0
        ax.set_ylim([min(self.y_train)-delta, max(self.y_train)+delta])

        plt.legend()
        # save sample plots
        if save_png:
            save_fig("samples")

    def plot_gp(self, X, mu, cov, post=False):
        delta = 1.96
        if post is True:
            delta = (max(mu) - min(mu)) / 10
        xmin = min(X)
        xmax = max(X)
        ymin = min(mu) - delta
        ymax = max(mu) + delta
        X = X.ravel()
        mu = mu.ravel()
        samples = np.random.multivariate_normal(mu, cov, 10)
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        ax.set_xlabel("$X$", fontsize=15)
        ax.set_ylabel("$y$", fontsize=15)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.tick_params(direction="in", labelsize=15, length=10, width=0.8, colors='k')
        for edge in ["top", "bottom", "left", "right"]:
            ax.spines[edge].set_linewidth(2.0)
        plt.plot(X, mu, color="purple", lw=2)
        for i, sample in enumerate(samples):
            plt.plot(X, sample, lw=0.5, ls='-', color="purple")
        if post:
            plt.scatter(self.X_train, self.y_train, color='k', linestyle='None', linewidth=1.0)
        stdpi = np.sqrt(np.diag(cov))[:, np.newaxis]
        yy = np.linspace(ymin, ymax, len(X)).reshape([len(X), 1])
        P = np.exp(-0.5 * (yy - mu.T) ** 2 / (stdpi ** 2).T)
        ax.imshow(P, extent=[xmin, xmax, ymin, ymax], aspect="auto", origin="lower", cmap="Purples", alpha=0.6)

        if post:
            # save_fig("posterior")
            save_fig("posterior" + "_" + str(len(self.X_train)))
        else:
            save_fig("prior")
