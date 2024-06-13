from src.gpr.gaussian_process import GP
from src.kernels.rbf import RBFKernel
from src.kernels.linear import LinearKernel
from src.kernels.periodic import PeriodicKernel
from src.utils_gpr.utils import data_from_func
from input.testfuncs_1d import (oscillatory_increasing_amplitude, sine_plus_linear, cubic_quadratic_polynomial,
                                absolute_value, exponential_growth, high_frequency_sine)
import numpy as np

# choose function
func = oscillatory_increasing_amplitude
X_train, X_test, y_train = data_from_func(f=func, N=30, M=500, xx=[0.0, 4.0, -2.0, 6.0], noise=0.1)

# choose kernel
kernel = RBFKernel(theta=[1.0, 1.0])

# noise
eps = 0.1

# create GP model
model = GP(kernel=kernel, optimizer="fmin_l_bfgs_b", alpha_=eps ** 2, n_restarts_optimizer=5)

# fit
model.fit(X_train, y_train)

# predict
y_mean, y_cov = model.predict(X_test)

# plot prior
model.plot_gp(X=X_test, mu=np.zeros(len(X_test)), cov=model.kernel(X_test))
# plot posterior
model.plot_gp(X=X_test, mu=y_mean, cov=y_cov, post=True)
# plot samples
model.plot_samples(5, save_png=True)
