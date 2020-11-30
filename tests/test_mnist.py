#!/usr/bin/env python
import os
import unittest
import numpy as np

from slowgrad.tensor import Tensor
from slowgrad.utils import fetch
import slowgrad.optim as optim
from tqdm import trange

from models import *


# mnist loader
def fetch_mnist():
    import gzip
    parse = lambda dat: np.frombuffer(gzip.decompress(dat), dtype=np.uint8
                                      ).copy()
    X_train = parse(
        fetch("http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz")
    )[0x10:].reshape((-1, 28, 28))
    Y_train = parse(
        fetch(
            "http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz"))[8:]
    X_test = parse(
        fetch("http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz")
    )[0x10:].reshape((-1, 28, 28))
    Y_test = parse(
        fetch(
            "http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz"))[8:]
    return X_train, Y_train, X_test, Y_test


# load the mnist dataset
X_train, Y_train, X_test, Y_test = fetch_mnist()


def train(model, optim, steps, BS=128):
    losses, accuracies = [], []

    t = trange(steps)
    for i in t:
        optim.zero_grad()
        samp = np.random.randint(0, X_train.shape[0], size=(BS))

        x = Tensor(X_train[samp].reshape((-1, 28 * 28)).astype(np.float32))
        Y = Y_train[samp]
        y = np.zeros((len(samp), 10), np.float32)
        # correct loss for NLL, torch NLL loss returns one per row
        y[range(y.shape[0]), Y] = -10.0
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
        Y_test_preds_out = model.forward(
            Tensor(X_test.reshape((-1, 28 * 28)).astype(np.float32)))
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
