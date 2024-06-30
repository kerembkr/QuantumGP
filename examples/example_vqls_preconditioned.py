from src.utils.ansatz import *
from src.utils.embedding import *
from src.optimizers.optim_qml import *
from src.utils.backend import DefaultQubit
from src.utils.utils import get_random_ls
from src.solver.quantum.vqls.vqls import VQLS
from src.utils.plotting import plot_costs
from src.linalg.cholesky import cholesky
from src.linalg.preconditioned_conjugate_gradient import mat_inv_lemma

# reproducibility
np.random.seed(42)

# number of qubits & layers
nqubits = 2
nlayers = 2

# maximum number of iterations
maxiter = 100

# random symmetric positive definite matrix
A0, b0 = get_random_ls(nqubits, easy_example=True)

L, _ = cholesky(A=A0, p=2**nqubits, rnd_idx=True)  # preconditioning matrix
P = L[:, :2**nqubits]

# compute inverse using matrix inversion lemma
invP = mat_inv_lemma(A=np.eye(2**nqubits) * 0.1**2,
                     U=P,
                     C=np.eye(np.shape(P)[1]),
                     V=P)

A0p = invP @ A0
b0p = invP @ b0

print(np.linalg.cond(A0))
print(np.linalg.cond(A0p))

# init
solver1 = VQLS()
solver2 = VQLS()

# set linear system
solver1.set_lse(A=A0, b=b0)
solver2.set_lse(A=A0p, b=b0p)

# choose optimizer, ansatz, state preparation, backend
optim_ = NesterovMomentumQML()
ansatz_ = HardwareEfficient(nqubits=nqubits, nlayers=nlayers)
prep_ = MottonenStatePrep(wires=range(nqubits))
backend_ = DefaultQubit(wires=nqubits + 1)

# setup
solver1.setup(optimizer=optim_, ansatz=ansatz_, stateprep=prep_, backend=backend_, epochs=maxiter, tol=1e-5)
solver2.setup(optimizer=optim_, ansatz=ansatz_, stateprep=prep_, backend=backend_, epochs=maxiter, tol=1e-5)

# solve
xopt1 = solver1.solve()
xopt2 = solver2.solve()

losses = {"VQLS": solver1.loss,
          "VQLS with PC": solver2.loss}

title = "qubits = {:d}    layers = {:d}".format(nqubits, nlayers)
plot_costs(data=losses, save_png=True, title=title, fname="vqls_precon")
