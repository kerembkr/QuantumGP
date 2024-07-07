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

nsolvers = 10

# choose ansatz, state preparation, backend
ansatz_ = HardwareEfficient(nqubits=nqubits, nlayers=nlayers)
prep_ = MottonenStatePrep(wires=range(nqubits))
backend_ = DefaultQubitTorch(wires=nqubits + 1)

cost_hists = {}

for i in range(nsolvers):

    ninputs = 4*(i+1)
    nhidden = 8

    # init
    solver = DeepVQLS()

    # set linear system
    A0, b0 = get_random_ls(nqubits, easy_example=True)  # random spd
    solver.set_lse(A=A0, b=b0)

    # setup
    solver.setup(optimizer=SGDTorch(), ansatz=ansatz_, stateprep=prep_, backend=backend_, epochs=maxiter, tol=1e-5,
                 ninputs=ninputs, nhidden=nhidden)

    # solve linear system
    solver.solve()

    # loss curves
    cost_hists["ninputs_{:d}".format(ninputs)] = solver.loss

plot_costs(data=cost_hists, save_png=True, title=None, fname="deep_vqls_nn_architecture_hea_nq{:d}_nl{:d}".format(
    nqubits, nlayers))
