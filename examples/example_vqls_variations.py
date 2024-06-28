from src.utils.ansatz import *
from src.utils.embedding import *
from src.optimizers.optim_qml import *
from src.utils.backend import DefaultQubit
from src.utils.utils import get_random_ls
from src.solver.quantum.vqls.vqls import VQLS
from src.solver.quantum.vqls.vqls_fast_and_slow import FastSlowVQLS
from src.utils.plotting import plot_costs

# reproducibility
np.random.seed(42)

# number of qubits & layers
nqubits = 1
nlayers = 1

# maximum number of iterations
maxiter = 10

# random symmetric positive definite matrix
A0, b0 = get_random_ls(nqubits, easy_example=False)

# init
solver1 = VQLS()
solver1.set_lse(A=A0, b=b0)
solver2 = FastSlowVQLS()
solver2.set_lse(A=A0, b=b0)

# choose optimizer, ansatz, state preparation, backend
optim_ = AdamQML()
ansatz_ = StrongEntangling(nqubits=nqubits, nlayers=nlayers)
prep_ = MottonenStatePrep(wires=range(nqubits))
backend_ = DefaultQubit(wires=nqubits + 1)

wopt1, loss1 = solver1.opt(optimizer=optim_, ansatz=ansatz_, stateprep=prep_, backend=backend_, epochs=maxiter,
                           tol=1e-6)
wopt2, loss2 = solver2.opt(optimizer=optim_, ansatz=ansatz_, stateprep=prep_, backend=backend_, epochs=maxiter,
                           epochs_bo=10, tol=1e-6)

losses = {"VQLS": loss1, "FastSlowVQLS": loss2}

title = "qubits = {:d}    layers = {:d}".format(nqubits, nlayers)
plot_costs(data=losses, save_png=True, title=title, fname="test")
