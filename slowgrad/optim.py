class Optimizer(object):
    def __init__(self, parameters):
        self.parameters = parameters

    def get_params(self):
        return self.parameters

    def zero_grad(self):
        for p in self.parameters:
            p.grad = None


class SGD(Optimizer):
    def __init__(self, parameters, lr=0.1):

        super().__init__(parameters)
        self.lr = lr

    def step(self):

        for p in self.parameters:
            p.data -= p.grad.data * self.lr


class Adam(Optimizer):
    pass


class RMSProp(Optimizer):
    pass
