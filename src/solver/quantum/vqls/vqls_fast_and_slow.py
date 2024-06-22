from skopt import gp_minimize
from src.utils.ansatz import *
from src.utils.embedding import *
from src.optimizers.optim_qml import *
from src.utils.backend import DefaultQubit
from src.solver.quantum.vqls.vqls import VQLS


class FastSlowVQLS(VQLS):
    def __init__(self, A, b):
        super().__init__(A, b)

    def opt(self, optimizer=None, ansatz=None, stateprep=None, backend=None, epochs=100, epochs_bo=None, tol=1e-4):
        """

        Parameters
        ----------
        optimizer
        ansatz
        stateprep
        backend
        epochs
        epochs_bo
        tol

        Returns
        -------

        """

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

        # initial weights
        w = self.ansatz.init_weights()

        # global optimization
        if epochs_bo is not None:
            w, _ = self.bayesopt_init(epochs_bo=epochs_bo)

        # local optimization
        w, cost_vals, iters = self.optimizer.optimize(func=self.cost, w=w, epochs=epochs, tol=tol)

        return w, cost_vals

    def bayesopt_init(self, epochs_bo=10):
        """
        Perform Bayesian Optimization to initialize the weights of the variational quantum circuit.

        :param epochs_bo: Number of Bayesian optimization steps (default: 10).

        :return: Tuple (w, cost_hist_bo):
                 - w: Initial weights for the gradient-based optimizer.
                 - cost_hist_bo: List of cost function values during Bayesian optimization.

        Example:
        --------
        w, cost_hist_bo = bayesopt_init(epochs_bo=20)

        Notes:
        ------
        - This method uses Bayesian optimization to find an initial set of weights that minimize the cost function.
        - The optimization progress is printed at each step.

        """

        def print_progress(res_):
            print("{:20s}    Step {:3d}    obj = {:9.7f} ".format(
                "Bayesian Optimization", len(res_.func_vals), res_.func_vals[-1]))

        # set parameter space
        dimensions = [(-np.pi, +np.pi) for _ in range(self.ansatz.nweights)]

        # bayesian optimization
        res = gp_minimize(func=self.cost,
                          dimensions=dimensions,
                          callback=[print_progress],
                          acq_func="EI",
                          n_calls=epochs_bo)

        # save cost function values
        cost_hist_bo = res.func_vals.tolist()

        # initial guess for gradient optimizer
        w = np.tensor(res.x, requires_grad=True)

        # reshape weights
        w = self.ansatz.prep_weights(w)

        return w, cost_hist_bo
