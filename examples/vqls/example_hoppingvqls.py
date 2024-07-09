from src.utils.ansatz import HardwareEfficient
from src.utils.embedding import MottonenStatePrep
from src.optimizers.optim_qml import NesterovMomentumQML, AdagradQML
from src.utils.backend import DefaultQubit
from src.utils.utils import get_random_ls
from src.solver.quantum.vqls.vqls_hopping import HoppingVQLS
from src.utils.plotting import plot_costs
import numpy as np
from src.solver.quantum.vqls.vqls import VQLS

# reproducibility
np.random.seed(42)


# number of qubits & layers
nqubits = 2
nlayers = 2
maxiter = 100

# choose ansatz, state preparation, backend
ansatz_ = HardwareEfficient(nqubits=nqubits, nlayers=nlayers)
prep_ = MottonenStatePrep(wires=range(nqubits))
backend_ = DefaultQubit(wires=nqubits + 1)

# solve linear system
solver1 = VQLS()
solver2 = HoppingVQLS()
A0, b0 = get_random_ls(nqubits, easy_example=True)  # random spd
solver1.set_lse(A=A0, b=b0)
solver2.set_lse(A=A0, b=b0)

# setup
solver1.setup(optimizer=NesterovMomentumQML(eta=0.3), ansatz=ansatz_, stateprep=prep_, backend=backend_, epochs=maxiter, tol=1e-5)
solver2.setup(optimizer=NesterovMomentumQML(eta=0.3), ansatz=ansatz_, stateprep=prep_, backend=backend_, epochs=maxiter, tol=1e-5)

# solve linear system
solver1.solve()
solver2.solve()

# loss curves
cost_hists = {"VQLS": solver1.loss, "HoppingVQLS": solver2.loss}

plot_costs(data=cost_hists, save_png=True, title=None, fname="hopping_vqls".format(nqubits, nlayers))
