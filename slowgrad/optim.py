import numpy as np
from slowgrad import Tensor

class Optimizer(object):
    def __init__(self, parameters):
        self.parameters = [x for x in parameters if x.requires_grad]

    def get_params(self):
        return self.parameters

    def zero_grad(self):
        for p in self.parameters:
            p.grad = None


class SGD(Optimizer):
    def __init__(self, parameters, lr=0.001):

        super().__init__(parameters)
        self.lr = lr

    def step(self):

        for p in self.parameters:
            p -= p.grad * self.lr


class Adam(Optimizer):
    def __init__(self, parameters, lr=0.001, b1=0.9, b2=0.999, e=10e-8):

        super().__init__(parameters)
        self.lr = lr
        self.b1 = b1
        self.b2 = b2
        self.e = e

        self.t = 0

        self.m = [Tensor(np.zeros_like(p)) for p in self.parameters]
        self.v = [Tensor(np.zeros_like(p)) for p in self.parameters]

    def step(self):

        self.t += 1
        a = self.lr * ((1.0 - self.b2**self.t)**0.5) / (1.0-self.b1**self.t)
        for i, p in enumerate(self.parameters):

            self.m[i] = self.b1*self.m[i] + (1.0 - self.b1)*p.grad
            self.v[i] = self.b2*self.v[i] + (1.0 -self.b2)*p.grad*p.grad


            p -= a*self.m[i].div((self.v[i].sqrt() + self.e))


class RMSProp(Optimizer):
    def __init__(self, parameters, lr=0.001, decay=0.9,  e=10e-8):

        super().__init__(parameters)
        self.lr = lr
        self.decay = decay
        self.e = e
        
        self.v = [Tensor(np.zeros_like(p)) for p in parameters]
    def step(self):
        
        for i,p in enumerate(self.parameters):

            self.v[i] = self.decay*self.v[i] + (1.0-self.decay)*p.grad*p.grad
            p -= (p.grad * self.lr).div(self.v[i].sqrt() + self.e)


