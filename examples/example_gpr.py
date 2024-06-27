import numpy as np
from src.kernels.rbf import RBFKernel
from src.gpr.gaussian_process import GP
from src.utils.utils import data_from_func
from input.testfuncs_1d import oscillatory_increasing_amplitude
from src.solver.solver import Solver
from src.solver.classic.cg import CG
from src.solver.classic.pcg import PCG
from src.solver.classic.chol import Cholesky

# choose function
func = oscillatory_increasing_amplitude
X_train, X_test, y_train = data_from_func(f=func, N=30, M=500, xx=[0.0, 4.0, -2.0, 6.0], noise=0.1)

# choose kernel
kernel = RBFKernel(theta=[1.0, 1.0])

# noise
eps = 0.1

# choose solver
solver = Cholesky()
# solver = CG()
solver = None

# choose preconditioner
precon = None

# create GP model
model = GP(kernel=kernel,
           optimizer="fmin_l_bfgs_b",
           alpha_=eps ** 2,
           n_restarts_optimizer=5,
           solver=solver,
           precon=precon,
           func=func)

# fit
model.fit(X_train, y_train)

# predict
y_mean, y_cov = model.predict(X_test)

# plot posterior
model.plot_gp(X=X_test, mu=y_mean, cov=y_cov, post=True)
