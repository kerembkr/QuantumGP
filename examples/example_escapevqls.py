from src.utils.ansatz import HardwareEfficient
from src.utils.embedding import MottonenStatePrep
from src.optimizers.optim_torch import SGDTorch
from src.utils.backend import DefaultQubitTorch
from src.utils.utils import get_random_ls
from src.solver.quantum.vqls.vqls_escape import EscapeVQLS
from src.utils.plotting import plot_costs
import numpy as np

# reproducibility
np.random.seed(42)

# number of qubits & layers
nqubits = 1
nlayers = 1
ninputs = 4  # number of inputs for the neural network
nhidden = 8  # number of nodes per hidden layer
maxiter = 100

# choose ansatz, state preparation, backend
ansatz_ = HardwareEfficient(nqubits=nqubits, nlayers=nlayers)
prep_ = MottonenStatePrep(wires=range(nqubits))
backend_ = DefaultQubitTorch(wires=nqubits + 1)

# solve linear system
solver = EscapeVQLS()
A0, b0 = get_random_ls(nqubits, easy_example=True)  # random spd
solver.set_lse(A=A0, b=b0)

# setup
solver.setup(optimizer=SGDTorch(eta=0.01), ansatz=ansatz_, stateprep=prep_, backend=backend_, epochs=maxiter, tol=1e-5,
             ninputs=ninputs, nhidden=nhidden)

# solve linear system
solver.solve()

# loss curves
cost_hists = {"EscapeVQLS": solver.loss}

plot_costs(data=cost_hists, save_png=True, title=None, fname="escape_vqls".format(nqubits, nlayers))
