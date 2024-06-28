from src.solver.quantum.vqls.vqls import VQLS
import torch.nn as nn
import functools
import torch
import pennylane.numpy as np


class DeepVQLS(VQLS):
    def __init__(self):
        super().__init__()


class HybridNeuralNetwork(nn.Module):

    def __init__(self, qnode, nqubits, nlayers, ninputs, npaulis):
        super().__init__()

        # number of qubits & layers & inputs & paulis
        self.nqubits = nqubits
        self.nlayers = nlayers
        self.ninputs = ninputs
        self.npaulis = npaulis

        # classical layers
        self.lin1 = nn.Linear(self.ninputs, 8)
        self.lin2 = nn.Linear(8, 8)
        self.lin3 = nn.Linear(8, self.nqubits)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

        # new
        self.qnode = qnode
        self.qlayers = {}
        it = 0
        for l in range(self.npaulis):
            for lp in range(self.npaulis):
                for part in ["Re", "Im"]:
                    # setattr(self, f"xx_{it}", qml.qnn.TorchLayer(self.qnode[(l, lp, -1, part)], {"weights": (self.nqubits)}))
                    setattr(self, f"xx_{it}", self.qnode[(l, lp, -1, part)])
                    self.qlayers[(l, lp, -1, part)] = getattr(self, f"xx_{it}")
                    it += 1
                    for j in range(self.nqubits):
                        # setattr(self, f"xx_{it}", qml.qnn.TorchLayer(self.qnode[(l, lp, j, part)], {"weights": (self.nqubits)}))
                        setattr(self, f"xx_{it}", self.qnode[(l, lp, j, part)])
                        self.qlayers[(l, lp, j, part)] = getattr(self, f"xx_{it}")
                        it += 1

        # something new
        init_method = functools.partial(torch.nn.init.uniform_, b=2 * np.pi)
        self.alpha = torch.nn.Parameter(init_method(torch.Tensor(1, self.ninputs)), requires_grad=True)

    def forward(self, x):

        # neural network architecture
        y = self.lin1(self.alpha)
        y = self.tanh(y)
        y = self.lin2(y)
        y = self.tanh(y)
        y = self.lin3(y)
        # y = self.tanh(y)
        # y = torch.reshape(y, (self.nqubits,))
        y = torch.reshape(y, (1, self.nqubits))

        # use output of the DNN for every VQC
        outputs = {}
        for l in range(self.npaulis):
            for lp in range(self.npaulis):
                for part in ["Re", "Im"]:
                    outputs[(l, lp, -1, part)] = self.qlayers[(l, lp, -1, part)](y)
                    for j in range(self.nqubits):
                        outputs[(l, lp, j, part)] = self.qlayers[(l, lp, j, part)](y)

        return outputs, y
