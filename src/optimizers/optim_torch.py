import pennylane.numpy as np
from abc import ABC, abstractmethod
import torch
from time import time


class OptimizerTorch(ABC):
    def __init__(self):
        self.name = None

    def optimize(self, func, model, w, epochs, tol):

        # features
        x = torch.ones(model.ninputs) * (np.pi / 4)

        # Optimizer
        opt = self.get_optimizer(model)

        # Optimization loop
        cost_vals = []
        for it in range(epochs):
            ta = time()

            # neural network maths
            opt.zero_grad()  # init gradient
            out, _ = model(x)  # forward pass
            # loss = self.cost(out)  # compute loss
            loss = func(out)  # compute loss

            loss.backward()  # backpropagation
            opt.step()  # update weights

            print("{:20s}     Step {:3d}    obj = {:9.7f}    time = {:9.7f} sec".format(self.name, it, loss.item(), time() - ta))
            cost_vals.append(loss.item())  # save cost function value
            if np.abs(loss.item()) < tol:  # breaking condition
                _, w = model(x)
                return w, cost_vals, it+1

        _, w = model(x)

        return w, cost_vals, epochs

    @abstractmethod
    def get_optimizer(self, model):
        pass


class SGDTorch(OptimizerTorch):
    def __init__(self, eta=0.1):
        super().__init__()
        self.eta = eta
        self.name = "SGD"

    def get_optimizer(self, model):
        return torch.optim.SGD(params=model.parameters(), lr=self.eta)
