import numpy as np
from .tensor import Function, register


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

class Mul(Function):
  @staticmethod
  def forward(ctx, x, y):
    ctx.save_for_backward(x, y)
    return x*y

  @staticmethod
  def backward(ctx, grad_output):
    x,y = ctx.to_save
    return y*grad_output, x*grad_output
register('mul', Mul)

class Div(Function):
  @staticmethod
  def forward(ctx, x, y):
    ctx.save_for_backward(x, y)
    return x/y

  @staticmethod
  def backward(ctx, grad_output):
    x,y = ctx.to_save
    return (1/y)*grad_output, -(x/np.square(y))*grad_output
register('div', Div)

class Pow(Function):
    @staticmethod
    def forward(ctx, input, power):
        ctx.save_for_backward(input, power)
        return input**power

    @staticmethod
    def backward(ctx, grad_output):
        input, power = ctx.to_save
        return grad_output * power*(input**(power-1)), grad_output* input**power*np.log(input)


register('pow', Pow)

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


class Relu(Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return np.array(np.maximum(input,0))

    @staticmethod
    def backward(ctx, grad_output):
        input,  = ctx.to_save
        return grad_output * (input>=0)


register('relu', Relu)

class LogSoftmax(Function):
  @staticmethod
  def forward(ctx, input):
    def logsumexp(x):
      #return np.log(np.exp(x).sum(axis=1))
      c = x.max(axis=1)
      return c + np.log(np.exp(x-c.reshape((-1, 1))).sum(axis=1))
    output = input - logsumexp(input).reshape((-1, 1))
    ctx.save_for_backward(output)
    return output

  @staticmethod
  def backward(ctx, grad_output):
    output, = ctx.to_save
    return grad_output - np.exp(output)*(grad_output.sum(axis=1).reshape((-1, 1)))
register('logsoftmax', LogSoftmax)


