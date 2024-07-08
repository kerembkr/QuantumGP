from src.utils.ansatz import HardwareEfficient
from src.utils.embedding import *
from src.optimizers.optim_qml import *
from src.utils.backend import DefaultQubit, DefaultQubitTorch
from src.utils.utils import get_random_ls
from src.solver.quantum.vqls.vqls import VQLS
from src.solver.quantum.vqls.vqls_fast_and_slow import FastSlowVQLS
from src.utils.plotting import plot_costs
from src.solver.quantum.vqls.vqls_dnn import DeepVQLS
from src.optimizers.optim_torch import SGDTorch
from src.solver.quantum.vqls.vqls_escape import EscapeVQLS

# reproducibility
np.random.seed(42)

# number of qubits & layers
nqubits = 1
nlayers = 1

# maximum number of iterations
maxiter = 100

# random symmetric positive definite matrix
A0, b0 = get_random_ls(nqubits, easy_example=True)

# init solvers
solver1 = VQLS()
solver2 = FastSlowVQLS()
solver3 = DeepVQLS()
solver4 = EscapeVQLS()

# set linear system
solver1.set_lse(A=A0, b=b0)
solver2.set_lse(A=A0, b=b0)
solver3.set_lse(A=A0, b=b0)
solver4.set_lse(A=A0, b=b0)

# choose optimizer, ansatz, state preparation, backend
ansatz_ = HardwareEfficient(nqubits=nqubits, nlayers=nlayers)
prep_ = MottonenStatePrep(wires=range(nqubits))

# setup VQLS solvers
solver1.setup(optimizer=AdamQML(), ansatz=ansatz_, stateprep=prep_, backend=DefaultQubit(wires=nqubits + 1),
              epochs=maxiter, tol=1e-7)
solver2.setup(optimizer=AdamQML(), ansatz=ansatz_, stateprep=prep_, backend=DefaultQubit(wires=nqubits + 1),
              epochs=maxiter, epochs_bo=None, tol=1e-7)
solver3.setup(optimizer=SGDTorch(), ansatz=ansatz_, stateprep=prep_, backend=DefaultQubitTorch(wires=nqubits + 1),
              epochs=maxiter, tol=1e-7, nhidden=16, ninputs=8)
solver4.setup(optimizer=SGDTorch(), ansatz=ansatz_, stateprep=prep_, backend=DefaultQubitTorch(wires=nqubits + 1),
              epochs=maxiter, tol=1e-7, nhidden=16, ninputs=8)

# solve linear systems
xopt1 = solver1.solve()
print("")
xopt2 = solver2.solve()
print("")
xopt3 = solver3.solve()
print("")
xopt4 = solver4.solve()

# solution of linear system
print("")
print("xsol =", np.linalg.solve(A0, b0))
print("vqls =", np.round(xopt1, 2))
print("fast =", np.round(xopt2, 2))
print("deep =", np.round(xopt3, 2))
print("escp =", np.round(xopt4, 2))


# plot loss functions
losses = {"VQLS": solver1.loss, "FastSlowVQLS": solver2.loss, "DeepVQLS": solver3.loss, "EscapeVQLS": solver4.loss}
title = "qubits = {:d}    layers = {:d}".format(nqubits, nlayers)
plot_costs(data=losses, save_png=True, title=title, fname="vqls_variations")
