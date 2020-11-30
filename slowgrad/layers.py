import numpy as np
from slowgrad.tensor import Tensor


class Layer(object):
    def __init__(self, init_fn=None):
        self.parameters = list()
        self.init_fn = init_fn

    def get_parameters(self):
        return self.parameters


class Linear(Layer):
    def __init__(self, n_inputs, n_outputs, init_fn=None):
        super().__init__(init_fn=init_fn)

        if self.init_fn is None:
            W = np.random.randn(n_inputs, n_outputs) * np.sqrt(2.0 /
                                                               (n_inputs))
        else:
            W = init_fn(n_inputs, n_outputs)

        self.weight = Tensor(W, requires_grad=True)

        self.parameters.append(self.weight)

    def forward(self, input):
        return input.dot(self.weight)

    def __call__(self, input):
        return input.dot(self.weight)


class Conv2d(Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=(2, 2),
                 stride=1,
                 init_fn=None):
        super().__init__(init_fn=init_fn)

        shape = (out_channels, in_channels, *kernel_size)

        if self.init_fn is None:
            W = np.random.randn(*shape) * np.sqrt(2.0 / (n_inputs))
        else:
            W = init_fn(*shape)

        self.weight = Tensor(W, requires_grad=True)
        self.parameters.append(self.weight)

    def forward(self, input):
        return input.conv2d(self.weight)

    def __call__(self, input):
        return input.conv2d(self.weight)
