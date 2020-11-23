import numpy as np
from typing import Union, List

import torch


class Tensor(object):
    def __init__(self,
                 data: Union[List, np.array],
                 genesis_op=None,
                 requires_grad=False,
                 children=()):
        self.data = np.array(data)
        self.genesis_op = genesis_op
        self.requires_grad = requires_grad

        self.backward_fn = None

        self.grad = None
        self.children = set(children)

    def backward(self, grad=None):

        if self.requires_grad:

            if grad is None:
                grad = Tensor(np.ones_like(self.data))

            if self.grad is None:
                self.grad = grad
            else:
                self.grad += grad

            topo = []
            visited = set()

            def build_topo(v):
                if v not in visited:
                    visited.add(v)
                    for child in v.children:
                        build_topo(child)
                    topo.append(v)

            build_topo(self)

            for v in reversed(topo):
                v.backward_fn()

    @property
    def shape(self):
        return self.data.shape

    def __add__(self, other):

        if self.requires_grad and other.requires_grad:
            out = Tensor(self.data + other.data,
                         genesis_op="add",
                         requires_grad=True)
        else:
            out = Tensor(self.data + other.data)

        def _backward():

            if self.grad is None:
                self.grad = out.grad
            else:
                self.grad += out.grad

            if other.grad is None:
                other.grad = out.grad
            else:
                other.grad += out.grad

        out.backward_fn = _backward
        return out

    def __neg__(self):

        if (self.requires_grad):
            out = Tensor(self.data * -1, genesis_op="neg", requires_grad=True)
        else:
            out = Tensor(self.data * -1)

        def _backward():

            if self.grad is None:
                self.grad = Tensor(-1 * out.grad.data)
            else:
                self.grad += Tensor(-1 * out.grad.data)

        out.backward_fn = _backward
        return out

    def __sub__(self, other):
        if self.requires_grad and other.requires_grad:
            out = Tensor(self.data - other.data,
                         genesis_op="sub",
                         requires_grad=True)
        else:
            out = Tensor(self.data - other.data)

        def _backward():

            if self.grad is None:
                self.grad = out.grad
            else:
                self.grad += out.grad

            if other.grad is None:
                other.grad = Tensor(-1 * out.grad.data)
            else:
                other.grad += Tensor(-1 * out.grad.data)

        out.backward_fn = _backward
        return out

    def dot(self, other):
        if (self.requires_grad):
            out = Tensor(self.data.dot(other.data),
                         requires_grad=True,
                         genesis_op="dot")

        else:
            out = Tensor(self.data.dot(x.data))

        def _backward():

            if self.grad is None:
                self.grad = Tensor(out.grad.data.dot(other.data.T))
            else:
                self.grad += Tensor(out.grad.data.dot(other.data.T))

            if other.grad is None:
                other.grad = Tensor(out.grad.data.T.dot(self.data).T)
            else:
                other.grad += Tensor(out.grad.data.T.dot(self.data).T)

        out.backward_fn = _backward
        return out

    def __repr__(self):
        return str(self.data.__repr__())

    def __str__(self):
        return f"Tensor: {str(self.data.__str__())}, requires_grad={self.requires_grad}"


if __name__ == "__main__":

    data = Tensor(np.array([[0, 0], [0, 1], [1, 0], [1, 1]]),
                  requires_grad=True)
    w0 = Tensor(np.array([[1, 2, 3], [1, 2, 3]]), requires_grad=True)

    pred = data.dot(w0)
    pred.backward()

    print(pred)

    data = torch.tensor(np.array([[0, 0], [0, 1], [1, 0], [1, 1]],
                                 dtype=np.float32),
                        requires_grad=True).float()
    w0 = torch.tensor(np.array([[1, 2, 3], [1, 2, 3]], dtype=np.float32),
                      requires_grad=True).float()

    pred = data.mm(w0)
    print(pred.grad_fn)
