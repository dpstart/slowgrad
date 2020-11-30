<h1 align="center">
  slowgrad
 </h1>
<p align="center">
    <img src="https://github.com/dpstart/slowgrad/workflows/Unit%20tests/badge.svg" alt="Unit tests" />
</p>

A small neural network library optimized for learning.

Inspired by PyTorch, micrograd, and tinygrad.

## Build an MNIST Convnet

```python
from slowgrad.layers import Linear, Conv2d

class TinyConvNetLayer:
  def __init__(self):

    self.c1 = Conv2d(1,8,kernel_size=(3,3))
    self.c2 = Conv2d(8,16,kernel_size=(3,3))
    self.l1 = Linear(16*5*5,10)

  def parameters(self):
    return [*self.l1.get_parameters(), *self.c1.get_parameters(), *self.c2.get_parameters()]

  def forward(self, x):
    x = x.reshape(shape=(-1, 1, 28, 28))
    x = self.c1(x).relu().max_pool2d()
    x = self.c2(x).relu().max_pool2d()
    x = x.reshape(shape=[x.shape[0], -1])
    return self.l1(x).logsoftmax()
```

--------------------------------------------------------------------
