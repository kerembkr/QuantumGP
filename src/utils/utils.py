import os
import numpy as np
from time import time
from functools import wraps
import matplotlib.pyplot as plt
import pennylane as qml




def spd(n):

    rnd_mat = np.random.rand(n, n)

    rnd_spd = rnd_mat @ rnd_mat.T

    return rnd_spd

def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        print('func:%r args:[%r, %r] took: %2.4f sec' % \
              (f.__name__, args, kw, te - ts))
        return result

    return wrap


def data_from_func(f, N, M, xx, noise=0.1, rand=False):
    """

    Parameters
    ----------
    f
    N
    M
    xx
    noise
    rand

    Returns
    -------

    """

    # set sample space
    xmin_tr, xmax_tr, xmin_te, xmax_te = xx

    # training data
    if rand:
        X_train = np.random.rand(N)*(xmax_tr-xmin_tr) + xmin_tr
    else:
        X_train = np.linspace(xmin_tr, xmax_tr, N)
    y_train = [f(X_) + np.random.rand() * 2 * noise - noise for X_ in X_train]

    # testing data
    X_test = np.linspace(xmin_te, xmax_te, M).reshape(-1, 1)

    return X_train, X_test, y_train


def save_fig(name):
    output_dir = '/Users/kerembuekrue/Documents/code/QuantumGP/output/'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    plt.savefig(os.path.join(output_dir, name + ".png"), dpi=300)


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
