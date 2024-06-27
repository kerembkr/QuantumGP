import numpy as np
from time import time
from src.kernels.rbf import RBFKernel
from src.gpr.gaussian_process import GP
from src.utils.utils import data_from_func
from src.solver.classic.pcg import PCG
from input.testfuncs_1d import oscillatory_increasing_amplitude

np.random.seed(42)

# choose function
func = oscillatory_increasing_amplitude
X_train, X_test, y_train = data_from_func(f=func, N=30, M=500, xx=[0.0, 4.0, -2.0, 6.0], noise=0.1)

# choose kernel
kernel = RBFKernel(theta=[1.0, 1.0])

# noise
eps = 0.1

# choose solver
solver = PCG(rank=10, pre_iters=5)

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
t0 = time()
model.fit(X_train, y_train)
print("fit     : {:.4f} sec".format(time()-t0))

# predict
t0 = time()
y_mean, y_cov = model.predict(X_test)
print("predict : {:.4f} sec".format(time()-t0))

# plot posterior
model.plot_gp(X=X_test, mu=y_mean, cov=y_cov, post=True)
