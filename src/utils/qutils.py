import pennylane as qml
import pennylane.numpy as np


def get_paulis(mat):
    """
      Decompose the input matrix into its Pauli components in O(4^n) time

      Args:
          mat (np.array): Matrix to decompose.

      Returns:
          mats (list): Pauli matrices
          wires(list): wire indices, where the Pauli matrices are applied

      """

    # decompose
    pauli_matrix = qml.pauli_decompose(mat, check_hermitian=True, pauli=False)

    # get coefficients and operators
    coeffs = pauli_matrix.coeffs
    ops = pauli_matrix.ops

    # create Pauli word
    pw = qml.pauli.PauliWord({i: pauli for i, pauli in enumerate(ops)})

    # get wires
    qubits = [pw[i].wires for i in range(len(pw))]

    # convert Pauli operator to matrix
    matrices = [qml.pauli.pauli_word_to_matrix(pw[i]) for i in range(len(pw))]

    return matrices, qubits, coeffs


def get_random_ls(nqubits, easy_example=False):
    if easy_example:
        A_ = np.eye(2 ** nqubits)
        A_[0, 0] = 2.0
        b_ = np.ones(2 ** nqubits)
        return A_, b_

    M = np.random.rand(2 ** nqubits, 2 ** nqubits)
    A_ = M @ M.T
    # vector
    b_ = np.random.rand(2 ** nqubits)
    b_ = b_ / np.linalg.norm(b_)

    return A_, b_


def combine_lists(cost_history):
    # make one single list
    cost_history = [item for sublist in cost_history for item in
                    (sublist if isinstance(sublist, list) else [sublist])]

    return cost_history
