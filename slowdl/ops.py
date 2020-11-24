import numpy as np
from .tensor3 import Function, register


class Add(Function):
    @staticmethod
    def forward(ctx, x, y):
        ctx.save_for_backward(x.shape, y.shape)
        return x + y

    @staticmethod
    def backward(ctx, grad_output):
        shape_x, shape_y = ctx.to_save
        return grad_output, grad_output


register('add', Add)


class Sub(Function):
    @staticmethod
    def forward(ctx, x, y):
        ctx.save_for_backward(x.shape, y.shape)
        return x - y

    @staticmethod
    def backward(ctx, grad_output):
        shape_x, shape_y = ctx.to_save
        return grad_output, -grad_output


register('sub', Sub)


class MatMul(Function):
    @staticmethod
    def forward(ctx, x, y):
        ctx.save_for_backward(x, y)
        return x.dot(y)

    @staticmethod
    def backward(ctx, grad_output):
        x, y = ctx.to_save
        grad_x = grad_output.dot(y.T)
        grad_y = x.T.dot(grad_output)
        return grad_x, grad_y


register('mm', MatMul)
register('dot', MatMul)


class Sum(Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return np.array([input.sum()])

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.to_save
        return grad_output * np.ones_like(input)


register('sum', Sum)
