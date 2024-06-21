from src.utils.ansatz import *
from src.utils.embedding import *
from src.optimizers.optim_qml import *
from src.utils.backend import DefaultQubit, LightningQubit
from src.utils.qutils import get_random_ls
from src.solver.quantum.vqls.vqls_fast_and_slow import FastSlowVQLS
from src.utils.qutils import plot_costs

# reproducibility
# np.random.seed(42)

# number of qubits & layers
nqubits = 1
nlayers = 2

maxiter = 200

# random symmetric positive definite matrix
A0, b0 = get_random_ls(nqubits, easy_example=False)

# init
solver = FastSlowVQLS(A=A0, b=b0)

# choose optimizer
optims = [GradientDescentQML(),
          AdamQML(),
          AdagradQML(),
          MomentumQML(),
          NesterovMomentumQML(),
          RMSPropQML()]

ansatz_ = StrongEntangling(nqubits=nqubits, nlayers=nlayers)

prep_ = MottonenStatePrep(wires=range(nqubits))

backend_ = DefaultQubit(wires=nqubits + 1)

cost_hists = {}
wopts = {}

for optim in optims:
    wopt, cost_hist = solver.opt(optimizer=optim,
                                 ansatz=ansatz_,
                                 stateprep=prep_,
                                 backend=backend_,
                                 epochs=maxiter,
                                 epochs_bo=None,
                                 tol=1e-6)

    cost_hists[optim.name] = cost_hist

    wopts[optim.name] = wopt

title = "{:s}    qubits = {:d}    layers = {:d}".format(ansatz_.__class__.__name__, nqubits, nlayers)
plot_costs(data=cost_hists, save_png=True, title=title)

device_probs = LightningQubit(wires=nqubits, shots=10000)

solver.plot_probs(device_probs, wopts)
