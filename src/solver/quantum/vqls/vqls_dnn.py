from src.solver.quantum.vqls.vqls import VQLS
import torch.nn as nn
import functools
import torch
import pennylane.numpy as np


class DeepVQLS(VQLS):
    def __init__(self):
        super().__init__()

    def create_qnodes(self):
        qnode_dict = {}
        for l in range(len(self.c)):
            for lp in range(len(self.c)):
                # psi
                qn_re_psi = self.qlayer(l=l, lp=lp, j=-1, part="Re")
                qn_im_psi = self.qlayer(l=l, lp=lp, j=-1, part="Im")
                qnode_dict[(l, lp, -1, "Re")] = qn_re_psi
                qnode_dict[(l, lp, -1, "Im")] = qn_im_psi
                for j in range(self.nqubits):
                    # mu
                    qn_re_mu = self.qlayer(l=l, lp=lp, j=j, part="Re")
                    qn_im_mu = self.qlayer(l=l, lp=lp, j=j, part="Im")
                    qnode_dict[(l, lp, j, "Re")] = qn_re_mu
                    qnode_dict[(l, lp, j, "Im")] = qn_im_mu

        return qnode_dict

    def opt(self, optimizer=None, ansatz=None, stateprep=None, backend=None, epochs=100, tol=1e-4):

        # init hybrid model
        qlayers = self.create_qnodes()

        ni = 4

        # numpy to torch
        self.c = torch.from_numpy(self.c)

        model = HybridNeuralNetwork(qnode=qlayers, nqubits=self.nqubits, nlayers=self.ansatz.nlayers, ninputs=ni,
                                    npaulis=len(self.c))

        # initial weights
        w = self.ansatz.init_weights()

        # local optimization
        w, cost_vals, iters = self.optimizer.optimize(model=model, w=w, epochs=self.epochs, tol=self.tol)

        return w, cost_vals


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
        self.lin3 = nn.Linear(8, self.nqubits + self.nqubits * self.nlayers)  # number of ansatz weights
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

        # new
        self.qnode = qnode
        self.qlayers = {}
        it = 0
        for l in range(self.npaulis):
            for lp in range(self.npaulis):
                for part in ["Re", "Im"]:
                    setattr(self, f"xx_{it}", self.qnode[(l, lp, -1, part)])
                    self.qlayers[(l, lp, -1, part)] = getattr(self, f"xx_{it}")
                    it += 1
                    for j in range(self.nqubits):
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
        # y = torch.reshape(y, (1, self.nqubits))
        y = torch.reshape(y, (1, self.nqubits + self.nqubits * self.nlayers))  # number of ansatz weights

        # use output of the DNN for every VQC
        outputs = {}
        for l in range(self.npaulis):
            for lp in range(self.npaulis):
                for part in ["Re", "Im"]:
                    outputs[(l, lp, -1, part)] = self.qlayers[(l, lp, -1, part)](y)
                    for j in range(self.nqubits):
                        outputs[(l, lp, j, part)] = self.qlayers[(l, lp, j, part)](y)

        return outputs, y
