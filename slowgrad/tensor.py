import numpy as np
from typing import Union, List, Any
from inspect import signature

import torch


class Tensor(object):
    def __init__(self, data: Union[List, np.array], requires_grad=False):
        self.data = np.array(data, dtype=np.float32)
        self.requires_grad = requires_grad

        self.grad = None

        # A context object is an instance of a Function.
        # Its purpose is to save information used when backpropagating.
        self._ctx = None

    # A dictionary that assigns the operations to the class.
    ops = {}

    def assign(self, x):
        self.data = x.data

    def _requires_grad_(self):
        self.requires_grad = True

    @staticmethod
    def zeros(*shape):
        return Tensor(np.zeros(shape, dtype=np.float32))

    @staticmethod
    def ones(*shape):
        return Tensor(np.ones(shape, dtype=np.float32))

    @staticmethod
    def randn(*shape):
        return Tensor(np.random.randn(*shape).astype(np.float32))

    @staticmethod
    def eye(dim):
        return Tensor(np.eye(dim).astype(np.float32))

    def backward(self):

        # A Tensor that is created as a result of an operation will have a Context set.
        if self._ctx is None:
            return

        if self.grad is None:
            # fill in the first grad with one
            # this is "implicit gradient creation"
            self.grad = Tensor(np.ones(self.data.shape, dtype=self.data.dtype))

        # Visited will containes Tensors in topological order starting from the "first children"
        visited, nodes = set(), []

        def deepwalk(node):
            visited.add(node)
            if node._ctx:
                for i in node._ctx.parents:
                    if i not in visited:
                        deepwalk(i)
                nodes.append(node)

        deepwalk(self)

        # Go through nodes starting from the current one.
        # For each node, compute the gradient by calling the `backward` function of the Context object.
        for t0 in reversed(nodes):

            assert (t0.grad is not None)

            grads = t0._ctx.backward(t0._ctx, t0.grad.data)
            if len(t0._ctx.parents) == 1:
                grads = [grads]
            for t, g in zip(t0._ctx.parents, grads):
                if g is None:
                    continue
                assert g.shape == t.data.shape, \
                "grad shape must match tensor shape in %r, %r != %r" % (self._ctx, g.shape, t.data.shape)

                # Assign gradient to parent node (or accumulate if something's there already)
                t.grad = Tensor(g) if t.grad is None else (t.grad + Tensor(g))

    def mean(self):
        div = Tensor(np.array([1 / np.prod(self.shape)],
                              dtype=self.data.dtype),
                     requires_grad=True)
        return self.sum().mul(div)

    @property
    def shape(self):
        return self.data.shape

    def __repr__(self):
        return str(self.data.__repr__())

    def __str__(self):
        return f"Tensor: {str(self.data.__str__())}, requires_grad={self.requires_grad}"


class Function():
    r"""
    """
    def __init__(self, *tensors):
        self.parents = tensors
        self.to_saves = []

    def apply(self, *x, **kwargs):
        op = self
        ctx = op(*x)
        # use default params
        params = signature(op.forward).parameters
        for p in params.values():
            if p.default is not p.empty:
                setattr(ctx, p.name, p.default)
        # overwrite with passed params
        for k, v in kwargs.items():
            setattr(ctx, k, v)
        ret = Tensor(op.forward(ctx, *[t.data for t in x], **kwargs))
        ret._ctx = ctx
        return ret

    def save_for_backward(self, *tensors):
        r"""Saves given tensors for a future call to :func:`~Function.backward`.

        **This should be called at most once, and only from inside the**
        :func:`forward` **method.**

        Later, saved tensors can be accessed through the :attr:`saved_tensors`
        attribute. Before returning them to the user, a check is made to ensure
        they weren't used in any in-place operation that modified their content.

        Arguments can also be ``None``.
        """
        self.to_save = tensors

    @staticmethod
    def forward(ctx: Any, *args: Any, **kwargs: Any) -> Any:
        r"""
        """
        raise NotImplementedError(
            "You must implement the forward function for custom"
            " autograd.Function.")

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        r"""
        """
        raise NotImplementedError(
            "You must implement the backward function for custom"
            " autograd.Function.")


def register(name, fxn):
    """Register an op in the Tensor class. 
    Override operators when possible.

    Args:
        name (str): Name of the op
        fxn (Function subclass): A subclass of Function that defines the operation
    """

    Tensor.ops[name] = fxn

    def dispatch(*x, **kwargs):
        f = Tensor.ops[name]
        return f.apply(f, *x, **kwargs)

    setattr(Tensor, name, dispatch)
    if name in ['add', 'sub', 'mul', 'div']:
        setattr(Tensor, "__%s__" % name, dispatch)
        setattr(Tensor, "__i%s__" % name,
                lambda self, x: self.assign(dispatch(self, x)))


# Inspired from tinygrad. This import is used for registering the operation in the Tensor class.
# TODO: change
import slowgrad.ops
