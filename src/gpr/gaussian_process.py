import numpy as np
import scipy.optimize
from numpy.random import randn
import matplotlib.pyplot as plt
from operator import itemgetter
from src.utils.utils import save_fig
from matplotlib.ticker import MaxNLocator
from scipy.linalg import cholesky, cho_solve, solve_triangular
from src.utils.acquisition import ExpectedImprovement


class GP:
    def __init__(self, kernel, func, optimizer=None, alpha_=1e-10, n_restarts_optimizer=0, solver=None, precon=None,
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
        self.func = func
        self.invK = None

        plt.rcParams['text.usetex'] = True

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

        if self.solver is not None:  # custom solver
            self.solver.set_lse(A=K_, b=self.y_train)
            print("cond(A) =", self.solver.condA)
            self.alpha = self.solver.solve()
        else:  # standard Cholesky
            self.L = cholesky(K_, lower=True, check_finite=False)  # K_ = L*L^T --> L
            self.alpha = cho_solve((self.L, True), self.y_train, check_finite=False)  # alpha = L^T \ (L \ y)

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

            # inference
            if self.solver is not None:  # custom solver
                y_cov_ = self.kernel(X) - K_trans @ (self.solver.invM @ K_trans.T)  # std dev
            else:  # basic Cholesky
                V = solve_triangular(self.L, K_trans.T, lower=True, check_finite=False)  # std dev
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

        if self.solver is not None:
            self.solver.set_lse(G, self.y_train)
            a = self.solver.solve()
        else:
            a = np.linalg.solve(G, self.y_train)  # G \\ Y

        # log likelihood
        loglik = np.inner(self.y_train, a) + ld  # (Y / G) * Y + log |G|

        # gradient
        if eval_gradient:
            dloglik = np.zeros(len(hypers))
            for i in range(len(hypers)):
                if self.solver is not None:
                    dloglik[i] = -np.inner(a, dK[i] @ a) + np.trace(self.solver.invM @ dK[i])
                else:
                    dloglik[i] = -np.inner(a, dK[i] @ a) + np.trace(np.linalg.solve(G, dK[i]))
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

        self.acq_func = ExpectedImprovement(model=self,
                                            xi=0.1,  # exploration exploitation trade-off
                                            bounds=[(min(self.X_train), max(self.X_train))])

        # maximize acquisition function = minimize negative acquisition function
        opt_res = scipy.optimize.minimize(
            self.acq_func,
            np.zeros(1),
            args=(self.f_star,),
            method="L-BFGS-B",
            bounds=self.acq_func.bounds,
            options={'disp': False}
        )

        return opt_res.x

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
        delta = (max(self.y_train) - min(self.y_train)) / 5.0
        ax.set_ylim([min(self.y_train) - delta, max(self.y_train) + delta])

        plt.legend()
        # save sample plots
        if save_png:
            save_fig("samples")

    def plot_gp(self, X, mu, cov, post=False, plot_acq=False):

        # Create a figure
        if plot_acq:
            fig = plt.figure(figsize=(14, 4))
            ax1 = fig.add_subplot(121)
            ax2 = fig.add_subplot(122)
        else:
            fig = plt.figure(figsize=(7, 4))
            ax1 = fig.add_subplot(111)

        # Always plot the posterior
        X = X.ravel()
        mu = mu.ravel()
        ax1.set_xlabel("$\mathcal{X}$", fontsize=15)
        ax1.set_ylabel("$\mathcal{Y}$", fontsize=15)
        for edge in ["top", "bottom", "left", "right"]:
            ax1.spines[edge].set_linewidth(2.0)
        ax1.plot(X, mu, color="purple", lw=3.0)
        ax1.plot(X, self.func(X), "--", color="grey", lw=3.0)
        if post:
            ax1.scatter(self.X_train, self.y_train, color='k', linestyle='None', linewidth=3.0)

        # Calculate the first standard deviation
        std = np.sqrt(np.diag(cov))
        # Plot the standard deviation and fill the area between
        ax1.fill_between(X, mu - std, mu + std, color="purple", alpha=0.3)
        ax1.plot(X, mu - std, color="grey", alpha=0.3, lw=3.0)
        ax1.plot(X, mu + std, color="grey", alpha=0.3, lw=3.0)
        ax1.set_xticks([])
        ax1.set_yticks([])

        # Plot acquisition function if plot_acq is True
        if plot_acq:
            self.f_star = np.min(self.y_train)  # Current best known function value
            self.acq_func = ExpectedImprovement(model=self, xi=0.01, bounds=[(min(self.X_train), max(self.X_train))])
            ax2.set_xlabel("$\mathcal{X}$", fontsize=15)
            ax2.set_ylabel("$EI$", fontsize=15)
            ax2.set_xticks([])
            ax2.set_yticks([])
            ax2.set_xlim([min(X), max(X)])
            for edge in ["top", "bottom", "left", "right"]:
                ax2.spines[edge].set_linewidth(2.0)
            ax2.plot(X, -self.acq_func(X, self.f_star), lw=3.0, color="blue")
            ax2.fill_between(X, np.zeros_like(mu), -self.acq_func(X, self.f_star), color="blue", alpha=0.3)

        plt.tight_layout()

        if post:
            save_fig("gp" + "_" + str(len(self.X_train)))
        else:
            save_fig("gp_0")
