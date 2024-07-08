import pennylane.numpy as np
from abc import ABC, abstractmethod
import torch
from time import time
from tqdm import tqdm


class OptimizerTorch(ABC):
    def __init__(self):
        self.name = None

    # def optimize(self, func, model, w, epochs, tol):
    #
    #     # features
    #     x = torch.ones(model.ninputs) * (np.pi / 4)
    #
    #     # Optimizer
    #     opt = self.get_optimizer(model)
    #
    #     # Optimization loop
    #     cost_vals = []
    #     for it in range(epochs):
    #         ta = time()
    #
    #         # neural network maths
    #         opt.zero_grad()  # init gradient
    #         out, _ = model(x)  # forward pass
    #         loss = func(out)  # compute loss
    #         loss.backward()  # backpropagation
    #         opt.step()  # update weights
    #
    #         print("{:s}     Step {:3d}    obj = {:9.7f}    time = {:9.7f} sec".format(self.name, it, loss.item(),
    #                                                                                   time() - ta))
    #         cost_vals.append(loss.item())  # save cost function value
    #         if np.abs(loss.item()) < tol:  # breaking condition
    #             _, w = model(x)
    #             return w, cost_vals, it+1
    #
    #     _, w = model(x)
    #
    #     return w, cost_vals, epochs

    def optimize(self, func, model, w, epochs, tol):
        # Features
        x = torch.ones(model.ninputs) * (np.pi / 4)

        # Optimizer
        opt = self.get_optimizer(model)

        # Optimization loop
        cost_vals = []
        with tqdm(total=epochs, desc=self.name) as pbar:
            for it in range(epochs):
                ta = time()

                # Neural network maths
                opt.zero_grad()  # init gradient
                out, _ = model(x)  # forward pass
                loss = func(out)  # compute loss
                loss.backward()  # backpropagation
                opt.step()  # update weights

                elapsed_time = time() - ta
                pbar.set_postfix(obj="{:9.7f}".format(loss.item()), time="{:9.7f} sec".format(elapsed_time))
                pbar.update(1)
                #
                # print("{:s}     Step {:3d}    obj = {:9.7f}    time = {:9.7f} sec".format(self.name, it, loss.item(),
                #                                                                           elapsed_time))
                cost_vals.append(loss.item())  # save cost function value

                if np.abs(loss.item()) < tol:  # breaking condition
                    _, w = model(x)
                    return w, cost_vals, it + 1

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


class AdamTorch(OptimizerTorch):
    def __init__(self, eta=0.1, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        super().__init__()
        self.eta = eta
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.name = "Adam"

    def get_optimizer(self, model):
        return torch.optim.Adam(params=model.parameters(), lr=self.eta, betas=self.betas, eps=self.eps,
                                weight_decay=self.weight_decay)


class AdagradTorch(OptimizerTorch):
    def __init__(self, eta=0.1, lr_decay=0, weight_decay=0, ini_acc_val=0, eps=1e-10):
        super().__init__()
        self.eta = eta
        self.lr_decay = lr_decay
        self.weight_decay = weight_decay
        self.ini_acc_val = ini_acc_val
        self.eps = eps
        self.name = "Adagrad"

    def get_optimizer(self, model):
        return torch.optim.Adagrad(params=model.parameters(), lr=self.eta, lr_decay=self.lr_decay,
                                   weight_decay=self.weight_decay, initial_accumulator_value=self.ini_acc_val,
                                   eps=self.eps)


class RMSPropTorch(OptimizerTorch):
    def __init__(self, eta=0.1, alpha=0.99, eps=1e-8, weight_decay=0, momentum=0):
        super().__init__()
        self.eta = eta
        self.alpha = alpha
        self.eps = eps
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.name = "RMSProp"

    def get_optimizer(self, model):
        return torch.optim.RMSprop(params=model.parameters(), lr=self.eta, alpha=self.alpha, eps=self.eps,
                                   weight_decay=self.weight_decay, momentum=self.momentum)
