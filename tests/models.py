
import numpy as np

from slowgrad.tensor import Tensor
from slowgrad.utils import layer_init_uniform
import slowgrad.optim as optim
from slowgrad.layers import Linear, Conv2d


# create a model
class TinyBobNet:
    def __init__(self):
        self.l1 = Tensor(layer_init_uniform(784, 128), requires_grad=True)
        self.l2 = Tensor(layer_init_uniform(128, 10), requires_grad=True)

    def parameters(self):
        return [self.l1, self.l2]

    def forward(self, x):
        return x.dot(self.l1).relu().dot(self.l2).logsoftmax()


# create a model
class TinyConvNet:
    def __init__(self):
        # https://keras.io/examples/vision/mnist_convnet/
        conv = 3
        #inter_chan, out_chan = 32, 64
        inter_chan, out_chan = 8, 16  # for speed
        self.c1 = Tensor(layer_init_uniform(inter_chan, 1, conv, conv))
        self.c2 = Tensor(layer_init_uniform(out_chan, inter_chan, conv, conv))
        self.l1 = Tensor(layer_init_uniform(out_chan * 5 * 5, 10))

    def parameters(self):
        return [self.l1, self.c1, self.c2]

    def forward(self, x):
        x = x.reshape(shape=(-1, 1, 28, 28))  # hacks
        x = x.conv2d(self.c1).relu().max_pool2d()
        x = x.conv2d(self.c2).relu().max_pool2d()
        x = x.reshape(shape=[x.shape[0], -1])
        return x.dot(self.l1).logsoftmax()


# create a model
class TinyConvNetLayer:
    def __init__(self):
        # https://keras.io/examples/vision/mnist_convnet/
        conv = 3
        #inter_chan, out_chan = 32, 64
        inter_chan, out_chan = 8, 16  # for speed
        self.c1 = Conv2d(1, 8, kernel_size=(3, 3), init_fn=layer_init_uniform)
        self.c2 = Conv2d(8, 16, kernel_size=(3, 3), init_fn=layer_init_uniform)
        self.l1 = Linear(784, 10, init_fn=layer_init_uniform)

    def parameters(self):
        return [
            *self.l1.get_parameters(), *self.c1.get_parameters(),
            *self.c2.get_parameters()
        ]

    def forward(self, x):
        x = x.reshape(shape=(-1, 1, 28, 28))  # hacks
        x = self.c1(x).relu().max_pool2d()
        x = self.c2(x).relu().max_pool2d()
        x = x.reshape(shape=[x.shape[0], -1])
        return self.l1(x).logsoftmax()


# create a model
class TinyBobNetLayer:
    def __init__(self):
        self.fc1 = Linear(784, 128, init_fn=layer_init_uniform)
        self.fc2 = Linear(128, 10, init_fn=layer_init_uniform)

    def parameters(self):
        return [*self.fc1.get_parameters(), *self.fc1.get_parameters()]

    def forward(self, x):
        x = self.fc1.forward(x)
        x = x.relu()
        x = self.fc2.forward(x)
        return x.logsoftmax()
