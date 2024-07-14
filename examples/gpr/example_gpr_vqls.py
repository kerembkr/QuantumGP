import numpy as np
from time import time
from src.kernels.rbf import RBFKernel
from src.gpr.gaussian_process import GP
from src.utils.utils import data_from_func
from src.solver.classic.cg import CG
from src.solver.quantum.vqls.vqls import VQLS
from input.testfuncs_1d import oscillatory_increasing_amplitude
from src.utils.backend import DefaultQubit
from src.utils.ansatz import HardwareEfficient
from src.optimizers.optim_qml import AdamQML
from src.utils.embedding import MottonenStatePrep

# fix random seed
np.random.seed(42)

n_train = 8  # training points
n_test = 64  # testing points

# choose function
func = oscillatory_increasing_amplitude
X_train, X_test, y_train = data_from_func(f=func, N=n_train, M=n_test, xx=[0.0, 4.0, -2.0, 6.0], noise=0.1)

# choose kernel
kernel = RBFKernel(theta=[1.0, 1.0])

# noise
eps = 0.1

# quantum solver
solver = VQLS()
maxiter = 50
nqubits = 3
nlayers = 1
ansatz_ = HardwareEfficient(nqubits=nqubits, nlayers=nlayers)
prep_ = MottonenStatePrep(wires=range(nqubits))
solver.setup(optimizer=AdamQML(), ansatz=ansatz_, stateprep=prep_, backend=DefaultQubit(wires=nqubits + 1),
             epochs=maxiter)

# classic solver
# solver = CG()

# create GP model
model = GP(kernel=kernel,
           optimizer="fmin_l_bfgs_b",
           alpha_=eps ** 2,
           n_restarts_optimizer=0,
           solver=solver,
           precon=None,
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
model.plot_gp(X=X_test, mu=y_mean, cov=y_cov, post=True, title="n8_-n64_vqls")
