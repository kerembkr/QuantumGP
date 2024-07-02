from src.solver.quantum.vqls.vqls import VQLS
import torch.nn as nn
import functools
import torch
import pennylane.numpy as np


class DeepVQLS(VQLS):
    def __init__(self):
        super().__init__()

    def create_qnodes(self, n, lenc, qlayer):
        qnode_dict = {}
        for l in range(lenc):
            for lp in range(lenc):
                # psi
                qn_re_psi = self.qlayer(n=n, l=l, lp=lp, j=-1, part="Re")
                qn_im_psi = self.qlayer(n=n, l=l, lp=lp, j=-1, part="Im")
                qnode_dict[(l, lp, -1, "Re")] = qn_re_psi
                qnode_dict[(l, lp, -1, "Im")] = qn_im_psi
                for j in range(n):
                    # mu
                    qn_re_mu = self.qlayer(n=n, l=l, lp=lp, j=j, part="Re")
                    qn_im_mu = self.qlayer(n=n, l=l, lp=lp, j=j, part="Im")
                    qnode_dict[(l, lp, j, "Re")] = qn_re_mu
                    qnode_dict[(l, lp, j, "Im")] = qn_im_mu

        return qnode_dict

    def opt(self, optimizer=None, ansatz=None, stateprep=None, backend=None, epochs=100, tol=1e-4):

        # init hybrid model
        qlayers = self.create_qnodes(n, len(c), qlayer)

        model = HybridNeuralNetwork(qnode=qlayers, nqubits=n, nlayers=L, ninputs=ni, npaulis=len(c))

        maxiter = 200  # max iterations
        tol = 0.001  # threshold
        eta = 0.1  # learning rate

        # List to save data
        cost_hist = []

        # features
        x = torch.ones(ni) * (np.pi / 4)

        # Optimizer
        opt = torch.optim.SGD(model.parameters(), lr=eta)

        for i in range(maxiter):

            # track time
            t0 = time()

            opt.zero_grad()  # init gradient
            out, _ = model(x)  # forward pass
            loss = cost(out)  # compute loss
            loss.backward()  # backpropagation
            opt.step()  # update weights

            # save cost function value
            cost_hist.append(loss.item())

            # print information
            if i % 1 == 0:
                print("iter {:4d}    cost  {:.5f}    time  {:.4f}".format(i, loss, time() - t0))

            if loss.item() < tol:  # breaking condition
                print("\nOptimum found after {:3d} Steps!".format(i))
                _, opti_mopti = model(x)
                return cost_hist, opti_mopti, 0
        _, opti_mopti = model(x)
        return cost_hist, opti_mopti, -1

        # initial weights
        w = self.ansatz.init_weights()

        # local optimization
        w, cost_vals, iters = self.optimizer.optimize(func=self.cost, w=w, epochs=self.epochs, tol=self.tol)

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
