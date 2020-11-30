import numpy as np
from typing import Union, List


class Tensor(object):
    def __init__(self,
                 data: Union[List, np.array],
                 parents=None,
                 genesis_op=None,
                 requires_grad=False,
                 id=None):
        self.data = np.array(data)
        self.parents = parents
        self.genesis_op = genesis_op
        self.requires_grad = requires_grad

        if (id is None):
            self.id = np.random.randint(0, 100000)
        else:
            self.id = id

        self.grad = None
        self.children = {}

        if parents is not None:
            for p in parents:
                if self.id not in p.children:
                    p.children[self.id] = 1
                else:
                    p.children[self.id] += 1

    def all_children_grads_accounted_for(self):
        for id, cnt in self.children.items():
            if (cnt != 0):
                return False
        return True

    def backward(self, grad=None, grad_origin=None):

        if self.requires_grad:

            if grad is None:
                grad = Tensor(np.ones_like(self.data))

            if (grad_origin is not None):
                if (self.children[grad_origin.id] == 0):
                    raise Exception("cannot backprop more than once")
                else:
                    self.children[grad_origin.id] -= 1

            if self.grad is None:
                self.grad = grad
            else:
                self.grad += grad

            if (self.parents is not None
                    and (self.all_children_grads_accounted_for()
                         or grad_origin is None)):

                if self.genesis_op == "add":
                    self.parents[0].backward(self.grad, self)
                    self.parents[1].backward(self.grad, self)
                elif self.genesis_op == "neg":
                    self.parents[0].backward(self.grad.__neg__(), self)
                elif self.genesis_op == "sub":
                    self.parents[0].backward(Tensor(self.grad.data), self)
                    self.parents[1].backward(self.grad.__neg__().data, self)

    @property
    def shape(self):
        return self.data.shape

    def __add__(self, other):
        if self.requires_grad and other.requires_grad:
            out = Tensor(self.data + other.data,
                         parents=[self, other],
                         genesis_op="add",
                         requires_grad=True)

        out = Tensor(self.data + other.data)
        return out

    def __neg__(self):

        if (self.requires_grad):
            return Tensor(self.data * -1,
                          parents=[self],
                          genesis_op="neg",
                          requires_grad=True)
        return Tensor(self.data * -1)

    def __sub__(self, other):
        if self.requires_grad and other.requires_grad:
            return Tensor(self.data - other.data,
                          parents=[self, other],
                          genesis_op="sub",
                          requires_grad=True)

        return Tensor(self.data - other.data)

    def __repr__(self):
        return str(self.data.__repr__())

    def __str__(self):
        return str(self.data.__str__())


if __name__ == "__main__":

    a = Tensor([1, 2, 3, 4, 5], requires_grad=True)
    b = Tensor([2, 2, 2, 2, 2], requires_grad=True)
    c = Tensor([5, 4, 3, 2, 1], requires_grad=True)

    d = a + b
    e = b + c
    f = d + e
    f.backward()
    f = f + b
    f.backward()

    print(b.grad.data)
