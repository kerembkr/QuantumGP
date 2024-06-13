import numpy as np
from BoostedGPR.src.input.testfuncs_1d import f1
from BoostedGPR.src.gaussian_process import GP
from BoostedGPR.src.utils.kernel import RBFKernel
from BoostedGPR.src.utils.utils import data_from_func
import sys
print(sys.path)
sys.path.append('/Users/kerembuekrue/Documents/code/VQBayesOpt/NoPlateauVQLS/src')
from NoPlateauVQLS.src.vqls_fast_and_slow import FastSlowVQLS


# choose function
X_train, X_test, y_train = data_from_func(f=f1, N=20, M=500, xx=[-2.0, 2.0, -4.0, 4.0], noise=0.1)

# choose kernel
kernel = RBFKernel(theta=[1.0, 1.0])

# noise
eps = 0.1

# create GP model
model = GP(kernel=kernel, optimizer="fmin_l_bfgs_b", alpha_=eps ** 2, n_restarts_optimizer=2)

# fit
model.fit(X_train, y_train)

# predict
y_mean, y_cov = model.predict(X_test)

# plot prior
model.plot_gp(X=X_test, mu=np.zeros(len(X_test)), cov=model.kernel(X_test))
# plot posterior
model.plot_gp(X=X_test, mu=y_mean, cov=y_cov, post=True)
