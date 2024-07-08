from src.utils.ansatz import *
from src.utils.embedding import *
from src.optimizers.optim_qml import *
from src.utils.backend import DefaultQubit
from src.solver.quantum.vqls.vqls import VQLS


class HoppingVQLS(VQLS):
    def __init__(self):
        super().__init__()

    def setup(self, optimizer=None, ansatz=None, stateprep=None, backend=None, epochs=100, tol=1e-4):

        self.epochs = epochs
        self.tol = tol

        if optimizer is None:
            self.optimizer = GradientDescentQML()
        else:
            self.optimizer = optimizer

        if ansatz is None:
            self.ansatz = StrongEntangling(nqubits=self.nqubits, nlayers=1)
        else:
            self.ansatz = ansatz

        if stateprep is None:
            self.stateprep = AmplitudeEmbedding(wires=range(self.nqubits))
        else:
            self.stateprep = stateprep

        if backend is None:
            self.backend = DefaultQubit(wires=self.nqubits + 1)
        else:
            self.backend = backend

    def opt(self, optimizer=None, ansatz=None, stateprep=None, backend=None, epochs=100, tol=1e-4):
        """
        Parameters
        ----------
        optimizer : Optimizer object, optional
            The local optimizer to use.
        ansatz : Ansatz object, optional
            The ansatz to use for the optimization.
        stateprep : StatePreparation object, optional
            The state preparation routine to use.
        backend : Backend object, optional
            The backend to use for running the circuits.
        epochs : int, optional
            The number of epochs to run the local optimization for. Default is 100.
        tol : float, optional
            The tolerance for convergence. Default is 1e-4.

        Returns
        -------
        w : ndarray
            The optimized weights.
        cost_vals : list
            The cost function values during the optimization.
        """

        # STANDARD VQLS
        # -------------
        # initial weights
        w = ansatz.init_weights()

        # solve linear system
        w, cost_vals, iters = optimizer.optimize(func=self.cost, w=w, epochs=epochs, tol=tol)

        # global minimum reached in first try
        if np.linalg.norm(cost_vals[-1]) < tol:
            return w, cost_vals

        # BASIN HOPPING
        # -------------
        # Initialize variables
        nhops = 5
        step_size = np.pi * 0.7
        cost_vals = [self.cost(w)]  # Initial cost value
        best_x = np.copy(w)
        best_f = cost_vals[-1]
        x_current = np.copy(w)
        f_current = cost_vals[-1]
        temp = 1.0

        for i in range(nhops):
            if np.linalg.norm(f_current) > tol:
                w = x_current + np.random.uniform(-step_size, step_size, size=w.shape)
            else:
                return w, cost_vals

            # Local optimization
            w, cost_vals, iters = optimizer.optimize(func=self.cost, w=w, epochs=epochs, tol=tol)

            # Metropolis acceptance criterion
            # if cost_vals[-1] < f_current or np.exp((f_current - cost_vals[-1]) / temp) > np.random.rand():
            if cost_vals[-1] < f_current or np.exp((f_current - cost_vals[-1]) / temp) > np.random.rand():

                x_current = w
                f_current = cost_vals[-1]

            # Update the best solution found
            if f_current < best_f:
                best_x = x_current
                best_f = f_current

        return best_x, cost_vals
