import numpy as np
from src.kernels.rbf import RBFKernel
from src.gpr.gaussian_process import GP
from src.utils.utils import data_from_func
from input.testfuncs_1d import oscillatory_increasing_amplitude
from src.solver.classic.cholesky import Cholesky

# choose function
func = oscillatory_increasing_amplitude
X_train, X_test, y_train = data_from_func(f=func, N=4, M=500, xx=[-2.0, 6.0, -2.0, 6.0], noise=0.1)

# choose kernel
kernel = RBFKernel(theta=[1.0, 1.0])

# noise
eps = 0.1

# choose solver
solver = Cholesky()

# choose preconditioner
precon = None

# choose acquisition function
acq_func = "EI"

# create GP model
model = GP(kernel=kernel,
           optimizer="fmin_l_bfgs_b",
           alpha_=eps ** 2,
           n_restarts_optimizer=5,
           solver=solver,
           precon=precon,
           acq_func=acq_func)

# fit
model.fit(X_train, y_train)

# predict
y_mean, y_cov = model.predict(X_test)

# plot posterior
model.plot_gp(X=X_test, mu=y_mean, cov=y_cov, post=True)

# Bayesian Optimization
for i in range(2):
    x_next = model.select_next_point()                          # minimize acquisition function
    y_next = func(x_next)                                       # compute new y
    X_train = np.append(X_train, x_next, axis=0)                # add new x to X_train
    y_train = np.append(y_train, y_next, axis=0)                # add new y to y_train
    model.fit(X_train, y_train)                                 # fit
    y_mean, y_cov = model.predict(X_test)                       # predict
    model.plot_gp(X=X_test, mu=y_mean, cov=y_cov, post=True)    # plot posterior
    model.plot_acquisition(X_test)                              # plot acquisition
