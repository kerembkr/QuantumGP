from src.utils.ansatz import BasicEntangling, StrongEntangling, RotY, HardwareEfficient
from src.utils.embedding import AmplitudeEmbedding, MottonenStatePrep
from src.optimizers.optim_qml import (GradientDescentQML, AdamQML, AdagradQML, MomentumQML, NesterovMomentumQML,
                                      RMSPropQML)
from src.utils.backend import DefaultQubit, LightningQubit
from src.utils.utils import get_random_ls
from src.solver.quantum.vqls.vqls import VQLS
from src.utils.plotting import plot_costs
import numpy as np

# reproducibility
np.random.seed(42)

# number of qubits & layers
nqubits = 1
nlayers = 1

maxiter = 50

# init
solver = VQLS()

# set linear system
A0, b0 = get_random_ls(nqubits, easy_example=True)  # random spd
solver.set_lse(A=A0, b=b0)

# choose optimizer
optims = [GradientDescentQML(),
          AdamQML(),
          AdagradQML(),
          MomentumQML(),
          NesterovMomentumQML(),
          RMSPropQML()]

# ansatz_ = StrongEntangling(nqubits=nqubits, nlayers=nlayers)
ansatz_ = HardwareEfficient(nqubits=nqubits, nlayers=nlayers)
prep_ = MottonenStatePrep(wires=range(nqubits))
backend_ = DefaultQubit(wires=nqubits + 1)

cost_hists = {}
wopts = {}

for optim in optims:

    # setup
    solver.setup(optimizer=optim, ansatz=ansatz_, stateprep=prep_, backend=backend_, epochs=maxiter, tol=1e-5)

    # solve linear system
    xopt = solver.solve()

    # save loss data
    cost_hists[optim.name] = solver.loss

    # print solution
    print("xopt :", xopt)

title = "VQLS   {:s}    qubits = {:d}    layers = {:d}".format(ansatz_.__class__.__name__, nqubits, nlayers)
plot_costs(data=cost_hists, save_png=True, title=title, fname="vqls_optimizer_comparison")
