import numpy as np
import os

from slowgrad.tensor import Tensor
from slowgrad.utils import layer_init_uniform, fetch
import slowgrad.optim as optim
from slowgrad.layers import Linear, Conv2d

from tqdm import trange
import gzip



def numpy_eval():
    Y_test_preds_out = model.forward(
        Tensor(X_test.reshape((-1, 28 * 28)).astype(np.float32)))
    Y_test_preds = np.argmax(Y_test_preds_out.data, axis=1)
    return (Y_test == Y_test_preds).mean()

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

class TinyConvNet:
    def __init__(self):

        self.c1 = Conv2d(1, 8, kernel_size=(3, 3), init_fn=layer_init_uniform)
        self.c2 = Conv2d(8, 16, kernel_size=(3, 3), init_fn=layer_init_uniform)
        self.l1 = Linear(16 * 5 * 5, 10, init_fn=layer_init_uniform)

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
    


# load the mnist dataset
X_train, Y_train, X_test, Y_test = fetch_mnist()

model = TinyConvNet()
optim = optim.SGD(model.parameters(), lr=0.001)

BS = 128

losses, accuracies = [], []

steps = 1000
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




accuracy = numpy_eval()
print("test set accuracy is %f" % accuracy)
assert accuracy > 0.95