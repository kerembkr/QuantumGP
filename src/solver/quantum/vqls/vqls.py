import src.utils.utils as utils
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from src.utils.ansatz import *
from src.utils.embedding import *
from src.optimizers.optim_qml import *
from src.utils.backend import DefaultQubit
from src.utils.utils import get_paulis
from src.solver.solver import Solver


class VQLS(Solver):
    def __init__(self):
        super().__init__()
        self.tol = None
        self.epochs = None
        self.c = None
        self.wires = None
        self.mats = None
        self.nqubits = None
        self.b = None
        self.A = None
        self.optimizer = None
        self.stateprep = None
        self.ansatz = None
        self.backend = None
        self.loss = None
        self.wopt = None

    def solve(self):

        wopt, loss_hist = self.opt(optimizer=self.optimizer,
                                   ansatz=self.ansatz,
                                   stateprep=self.stateprep,
                                   backend=self.backend,
                                   epochs=self.epochs,
                                   tol=self.tol)

        self.loss = loss_hist
        self.wopt = wopt

        @qml.qnode(qml.device("default.qubit", wires=self.nqubits))
        def prepare_and_sample(weights):
            self.V(weights)
            return qml.state()

        self.x = prepare_and_sample(self.wopt)

        self.x = self.correct_scaling_err(self.x)  # correct the scaling error

        self.x = np.real(self.x)  # only regard the real part

        return self.x

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

    def set_lse(self, A, b):
        self.A = A
        self.b = b

        self.bnorm = np.linalg.norm(self.b)

        # number of qubits
        self.nqubits = int(np.log(len(b)) / np.log(2))

        # Pauli decomposition
        self.mats, self.wires, self.c = get_paulis(self.A)

    def opt(self, optimizer=None, ansatz=None, stateprep=None, backend=None, epochs=100, tol=1e-4):
        """

        Parameters
        ----------
        optimizer
        ansatz
        stateprep
        backend
        epochs
        tol

        Returns
        -------

        """

        # initial weights
        w = self.ansatz.init_weights()

        # local optimization
        w, cost_vals, iters = self.optimizer.optimize(func=self.cost, w=w, epochs=self.epochs, tol=self.tol)

        return w, cost_vals

    def qlayer(self, l=None, lp=None, j=None, part=None):
        """
        Construct a quantum node representing a layer of the quantum circuit for estimating the expectation value
        of the ancillary qubit in the Pauli-Z basis.

        Parameters
        ----------
        l : int, optional
            Index of the unitary component A_l in the problem matrix A.
        lp : int, optional
            Index of the unitary component A_lp in the problem matrix A.
        j : int, optional
            Index of the qubit to apply the controlled Z operator. If -1, apply the identity.
        part : {'Re', 'Im'}, optional
            Specifies whether to estimate the real ('Re') or imaginary ('Im') part of the coefficient 'mu'.

        Returns
        -------
        callable
            Quantum circuit representing the layer of the quantum circuit.

        """

        @qml.qnode(self.backend.qdevice)
        def qcircuit(weights):

            # First Hadamard gate applied to the ancillary qubit.
            qml.Hadamard(wires=self.nqubits)

            # For estimating the imaginary part of the coefficient "mu", we must add a "-i"
            # phase gate.
            if part == "Im" or part == "im":
                qml.PhaseShift(-np.pi / 2, wires=self.nqubits)

            # Variational circuit generating a guess for the solution vector |x>
            self.V(weights)

            # Controlled application of the unitary component A_l of the problem matrix A.
            self.CA(l, self.mats, self.wires)

            # Adjoint of the unitary U_b associated to the problem vector |b>.
            qml.adjoint(self.U_b)(self.b)

            # Controlled Z operator at position j. If j = -1, apply the identity.
            if j != -1:
                qml.CZ(wires=[self.nqubits, j])

            # Unitary U_b associated to the problem vector |b>.
            self.U_b(self.b)

            # Controlled application of Adjoint(A_lp).
            qml.adjoint(self.CA)(lp, self.mats, self.wires)

            # Second Hadamard gate applied to the ancillary qubit.
            qml.Hadamard(wires=self.nqubits)

            # Expectation value of Z for the ancillary qubit.
            return qml.expval(qml.PauliZ(wires=self.nqubits))

        return qcircuit

    def U_b(self, vec):
        """
        Apply a unitary operation to embed the problem vector |b> into the quantum state.

        This method creates a unitary matrix that rotates the ground state |0> to the problem vector |b>.
        The operation uses amplitude embedding to encode the vector into the quantum state.

        :param vec: A list or array representing the problem vector |b> to be embedded.
                    The length of vec should be 2^n, where n is the number of qubits.

        Example:
        --------
        vec = [0.5, 0.5, 0.5, 0.5]  # Example vector for 2 qubits
        U_b(vec)

        Notes:
        ------
        - This method uses the AmplitudeEmbedding function from the PennyLane library to normalize and embed the vector.
        - Ensure that the length of vec is compatible with the number of qubits (2^n).
        """
        # qml.AmplitudeEmbedding(features=vec, wires=range(self.nqubits), normalize=True)  # O(n^2)
        self.stateprep.prep(vec=vec)

    def CA(self, idx, matrices, qubits):
        """
        Apply a controlled unitary operation for the specified component of the problem matrix A.

        This method applies the controlled version of a unitary matrix A_l, where the control qubit is the last qubit
        and the target qubits are specified by the qubits parameter.

        :param idx: Index of the unitary component in the matrices list.
        :param matrices: List of unitary matrices representing components of the problem matrix A.
        :param qubits: List of qubit indices on which each unitary matrix should act.

        Example:
        --------
        matrices = [U1, U2, U3]  # List of unitary matrices
        qubits = [0, 1, 2]  # Corresponding target qubits for each unitary matrix
        CA(idx=1, matrices=matrices, qubits=qubits)

        Notes:
        ------
        - The control qubit is assumed to be the last qubit (self.nqubits).
        - Ensure that idx is within the range of the matrices and qubits lists.
        - The qml.ControlledQubitUnitary function from PennyLane is used to apply the controlled unitary operation.
        """
        qml.ControlledQubitUnitary(matrices[idx], control_wires=[self.nqubits], wires=qubits[idx])

    def V(self, weights):
        """
        Apply a variational circuit to map the ground state |0> to the ansatz state |x>.

        This method uses the provided weights to prepare and apply a variational quantum circuit (VQC),
        transforming the initial ground state |0> into the ansatz state |x>.

        :param weights: A list or array of weights used for the variational circuit.
                        These weights parameterize the quantum gates in the ansatz.

        Example:
        --------
        weights = [0.1, 0.2, 0.3, 0.4]  # Example weights for the variational circuit
        V(weights)

        Notes:
        ------
        - The weights are first prepared and possibly reshaped using the ansatz's prep_weights method.
        - The variational quantum circuit is then applied using the ansatz's vqc method.
        """

        weights = self.ansatz.prep_weights(weights)

        # apply unitary ansatz
        self.ansatz.vqc(weights=weights)

    def cost(self, weights):
        """
        Calculate the cost function value for the current set of trainable parameters.

        The cost function measures the proximity between the encoded state A|x> and the target state |b>.
        It tends to zero when A|x> is proportional to |b>.

        :param weights: Trainable parameters for the variational circuit.

        :return: Cost function value (float).

        Example:
        --------
        weights = [0.1, 0.2, 0.3, 0.4]  # Example weights for the variational circuit
        cost_value = cost(weights)

        Notes:
        ------
        - This method calculates the cost function by evaluating the quantum nodes for both the real and imaginary parts.
        - It iterates over the quantum layers and qubits to compute the norm of the encoded state (psi_norm) and the
          expectation value of the problem matrix (mu_sum).
        - The cost function is then calculated as 0.5 - 0.5 * abs(mu_sum) / (nqubits * abs(psi_norm)).
        """

        mu_sum = 0.0
        psi_norm = 0.0
        for l in range(0, len(self.c)):
            for lp in range(0, len(self.c)):
                psi_real_qnode = self.qlayer(l=l, lp=lp, j=-1, part="Re")
                psi_imag_qnode = self.qlayer(l=l, lp=lp, j=-1, part="Im")
                psi_real = psi_real_qnode(weights)
                psi_imag = psi_imag_qnode(weights)
                psi_norm += self.c[l] * np.conj(self.c[lp]) * (psi_real + 1.0j * psi_imag)
                for j in range(0, self.nqubits):
                    mu_real_qnode = self.qlayer(l=l, lp=lp, j=j, part="Re")
                    mu_imag_qnode = self.qlayer(l=l, lp=lp, j=j, part="Im")
                    mu_real = mu_real_qnode(weights)
                    mu_imag = mu_imag_qnode(weights)
                    mu_sum += self.c[l] * np.conj(self.c[lp]) * (mu_real + 1.0j * mu_imag)

        # Cost function
        try:
            return float(0.5 - 0.5 * abs(mu_sum) / (self.nqubits * abs(psi_norm)))
        except TypeError:
            return 0.5 - 0.5 * abs(mu_sum) / (self.nqubits * abs(psi_norm))

    def solve_classic(self):
        return np.linalg.solve(self.A, self.b)

    def plot_probs(self, device, params_opt):

        # classical probabilities
        x = self.solve_classic()
        c_probs = (x / np.linalg.norm(x)) ** 2

        @qml.qnode(device.qdevice, interface=device.interface)
        def prepare_and_sample(weights):
            self.V(weights)
            return qml.sample()

        # Adjust the figure size to fit the optimizer names
        fig, axs = plt.subplots(len(params_opt.items()), 2, figsize=(12, 7),
                                constrained_layout=True)

        for i, (label, params) in enumerate(params_opt.items()):

            raw_samples = prepare_and_sample(params)
            if self.nqubits == 1:
                raw_samples = [[_] for _ in raw_samples]
            samples = []
            for sam in raw_samples:
                samples.append(int("".join(str(bs) for bs in sam), base=2))
            q_probs = np.round(np.bincount(samples) / device.shots, 2)

            axs[i, 0].xaxis.set_major_locator(MaxNLocator(integer=True))
            axs[i, 1].xaxis.set_major_locator(MaxNLocator(integer=True))

            # plot probabilities
            axs[i, 0].bar(np.arange(0, 2 ** self.nqubits), c_probs, color="skyblue")
            axs[i, 1].bar(np.arange(0, 2 ** self.nqubits), q_probs, color="plum")

            if i == len(params_opt.items()) - 1:
                axs[i, 0].set_xlabel("Vector space basis", fontsize=18, fontname='serif')
                axs[i, 1].set_xlabel("Hilbert space basis", fontsize=18, fontname='serif')

            if i == 0:
                axs[i, 1].set_title("Quantum probabilities", fontsize=18, fontname='serif')
                axs[i, 0].set_title("Classical probabilities", fontsize=18, fontname='serif')

            # Add optimizer name next to the right plot
            axs[i, 1].annotate(label, xy=(1.05, 0.5), xycoords='axes fraction', va='center', fontsize=18,
                               fontname='serif')

            # Remove x axis ticks except for the lowest plot
            if i < len(params_opt.items()) - 1:
                axs[i, 0].set_xticklabels([])
                axs[i, 1].set_xticklabels([])

        utils.save_fig(["vqls/", "probs"])

    def correct_scaling_err(self, x):

        try:
            x = x.detach().numpy()
        except:
            pass

        # Step 4: Calculate the optimal scaling factor mu
        Ax_normed = self.A @ x  # O(n^2)
        mu = self.bnorm / np.linalg.norm(Ax_normed)

        # Step 5: Scale x_normed to get the original solution x
        x = mu * x  # O(n)

        return x
