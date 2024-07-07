from src.utils.ansatz import HardwareEfficient
from src.utils.embedding import MottonenStatePrep
from src.optimizers.optim_torch import AdamTorch, AdagradTorch, RMSPropTorch, SGDTorch
from src.utils.backend import DefaultQubitTorch
from src.utils.utils import get_random_ls
from src.solver.quantum.vqls.vqls_dnn import DeepVQLS
from src.utils.plotting import plot_costs
import numpy as np

# reproducibility
np.random.seed(42)

# number of qubits & layers
nqubits = 1
nlayers = 1

maxiter = 100

# init
solver1 = DeepVQLS()
solver2 = DeepVQLS()
solver3 = DeepVQLS()
solver4 = DeepVQLS()

# set linear system
A0, b0 = get_random_ls(nqubits, easy_example=True)  # random spd
solver1.set_lse(A=A0, b=b0)
solver2.set_lse(A=A0, b=b0)
solver3.set_lse(A=A0, b=b0)
solver4.set_lse(A=A0, b=b0)

# choose ansatz, state preparation, backend
ansatz_ = HardwareEfficient(nqubits=nqubits, nlayers=nlayers)
prep_ = MottonenStatePrep(wires=range(nqubits))
backend_ = DefaultQubitTorch(wires=nqubits + 1)

# setup
solver1.setup(optimizer=AdamTorch(), ansatz=ansatz_, stateprep=prep_, backend=backend_, epochs=maxiter, tol=1e-5)
solver2.setup(optimizer=AdagradTorch(), ansatz=ansatz_, stateprep=prep_, backend=backend_, epochs=maxiter, tol=1e-5)
solver3.setup(optimizer=RMSPropTorch(), ansatz=ansatz_, stateprep=prep_, backend=backend_, epochs=maxiter, tol=1e-5)
solver4.setup(optimizer=SGDTorch(), ansatz=ansatz_, stateprep=prep_, backend=backend_, epochs=maxiter, tol=1e-5)

# solve linear system
xopt1 = solver1.solve()
xopt2 = solver2.solve()
xopt3 = solver3.solve()
xopt4 = solver4.solve()

# loss curves
cost_hists = {"Adam": solver1.loss,
              "Adagrad": solver2.loss,
              # "RMSProp": solver3.loss,
              "SGD": solver4.loss}

plot_costs(data=cost_hists, save_png=True, title=None, fname="deep_vqls_optimizer_comparison_hea_nq{:d}_nl{:d}".format(
    nqubits, nlayers))
