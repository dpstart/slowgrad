#!/usr/bin/env python
import os
import unittest
import numpy as np
from slowdl.tensor import Tensor
from slowdl.utils import fetch, layer_init_uniform
import slowdl.optim as optim
from tqdm import trange

from slowdl.layers import Linear, Conv2d

# mnist loader
def fetch_mnist():
  import gzip
  parse = lambda dat: np.frombuffer(gzip.decompress(dat), dtype=np.uint8).copy()
  X_train = parse(fetch("http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz"))[0x10:].reshape((-1, 28, 28))
  Y_train = parse(fetch("http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz"))[8:]
  X_test = parse(fetch("http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz"))[0x10:].reshape((-1, 28, 28))
  Y_test = parse(fetch("http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz"))[8:]
  return X_train, Y_train, X_test, Y_test

# load the mnist dataset
X_train, Y_train, X_test, Y_test = fetch_mnist()

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
    inter_chan, out_chan = 8, 16   # for speed
    self.c1 = Tensor(layer_init_uniform(inter_chan,1,conv,conv))
    self.c2 = Tensor(layer_init_uniform(out_chan,inter_chan,conv,conv))
    self.l1 = Tensor(layer_init_uniform(out_chan*5*5, 10))

  def parameters(self):
    return [self.l1, self.c1, self.c2]

  def forward(self, x):
    x = x.reshape(shape=(-1, 1, 28, 28)) # hacks
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
    inter_chan, out_chan = 8, 16   # for speed
    self.c1 = Conv2d(1,8,kernel_size=(3,3), init_fn=layer_init_uniform)
    self.c2 = Conv2d(8,16,kernel_size=(3,3), init_fn=layer_init_uniform)
    self.l1 = Linear(16*5*5,10,init_fn=layer_init_uniform)

  def parameters(self):
    return [*self.l1.get_parameters(), *self.c1.get_parameters(), *self.c2.get_parameters()]

  def forward(self, x):
    x = x.reshape(shape=(-1, 1, 28, 28)) # hacks
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


def train(model, optim, steps, BS=128):
  losses, accuracies = [], []

  t = trange(steps)
  for i in t:
    optim.zero_grad()
    samp = np.random.randint(0, X_train.shape[0], size=(BS))

    x = Tensor(X_train[samp].reshape((-1, 28*28)).astype(np.float32))
    Y = Y_train[samp]
    y = np.zeros((len(samp),10), np.float32)
    # correct loss for NLL, torch NLL loss returns one per row
    y[range(y.shape[0]),Y] = -10.0
    y = Tensor(y)

    # network
    out = model.forward(x)

    # NLL loss function
    loss = out.mul(y).mean()
    loss.backward()
    optim.step()

    cat = np.argmax(out.data, axis=1)
    accuracy = (cat == Y).mean()

    # printing
    loss = loss.data
    losses.append(loss)
    accuracies.append(accuracy)
    t.set_description("loss %.2f accuracy %.2f" % (loss, accuracy))

def evaluate(model):
  def numpy_eval():
    Y_test_preds_out = model.forward(Tensor(X_test.reshape((-1, 28*28)).astype(np.float32)))
    Y_test_preds = np.argmax(Y_test_preds_out.data, axis=1)
    return (Y_test == Y_test_preds).mean()

  accuracy = numpy_eval()
  print("test set accuracy is %f" % accuracy)
  assert accuracy > 0.95


class TestMNIST(unittest.TestCase):
    def test_sgd(self):
        np.random.seed(1337)
        model = TinyBobNet()
        optimizer = optim.SGD(model.parameters(), lr=0.001)
        train(model, optimizer, steps=1000)
        evaluate(model)
    def test_sgd_layer(self):
        np.random.seed(1337)
        model = TinyBobNetLayer()
        optimizer = optim.SGD(model.parameters(), lr=0.001)
        train(model, optimizer, steps=1000)
        evaluate(model)
    # def test_convnet(self):
    #   np.random.seed(1337)
    #   model = TinyConvNet()
    #   optimizer = optim.SGD(model.parameters(), lr=0.001)
    #   train(model, optimizer, steps=1000)
    #   evaluate(model)

    # def test_convnet_layer(self):
    #   np.random.seed(1337)
    #   model = TinyConvNetLayer()
    #   optimizer = optim.SGD(model.parameters(), lr=0.001)
    #   train(model, optimizer, steps=1000)
    #   evaluate(model)

if __name__ == "__main__":

  np.random.seed(1337)
  model = TinyConvNetLayer()
  optimizer = optim.SGD(model.parameters(), lr=0.001)
  train(model, optimizer, steps=1000)
  evaluate(model)