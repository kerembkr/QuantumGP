from src.utils.ansatz import *
from src.utils.embedding import *
from src.optimizers.optim_qml import *
from src.utils.backend import DefaultQubit
from src.utils.utils import get_random_ls
from src.solver.quantum.vqls.vqls import VQLS
from src.solver.quantum.vqls.vqls_fast_and_slow import FastSlowVQLS
from src.utils.plotting import plot_costs
from src.solver.quantum.vqls.vqls_dnn import DeepVQLS

# reproducibility
np.random.seed(42)

# number of qubits & layers
nqubits = 1
nlayers = 1

# maximum number of iterations
maxiter = 200

# random symmetric positive definite matrix
A0, b0 = get_random_ls(nqubits, easy_example=False)

# init solvers
solver1 = VQLS()
solver2 = FastSlowVQLS()
solver3 = DeepVQLS()

# set linear system
solver1.set_lse(A=A0, b=b0)
solver2.set_lse(A=A0, b=b0)
solver3.set_lse(A=A0, b=b0)

# choose optimizer, ansatz, state preparation, backend
optim_ = AdamQML()
ansatz_ = StrongEntangling(nqubits=nqubits, nlayers=nlayers)
prep_ = MottonenStatePrep(wires=range(nqubits))
backend_ = DefaultQubit(wires=nqubits + 1)

solver1.setup(optimizer=optim_, ansatz=ansatz_, stateprep=prep_, backend=backend_, epochs=maxiter, tol=1e-5)
solver2.setup(optimizer=optim_, ansatz=ansatz_, stateprep=prep_, backend=backend_, epochs=maxiter, epochs_bo=10, tol=1e-5)
solver3.setup(optimizer=optim_, ansatz=ansatz_, stateprep=prep_, backend=backend_, epochs=maxiter, tol=1e-5)

xopt1 = solver1.solve()
xopt2 = solver2.solve()
xopt3 = solver3.solve()

print("xopt1", xopt1)
print("xopt2", xopt2)
print("xopt3", xopt3)

losses = {"VQLS": solver1.loss, "FastSlowVQLS": solver2.loss, "DeepVQLS": solver3.loss}

title = "qubits = {:d}    layers = {:d}".format(nqubits, nlayers)
plot_costs(data=losses, save_png=True, title=title, fname="test")
