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
        return x * y

    @staticmethod
    def backward(ctx, grad_output):
        x, y = ctx.to_save
        return y * grad_output, x * grad_output


register('mul', Mul)


class Div(Function):
    @staticmethod
    def forward(ctx, x, y):
        ctx.save_for_backward(x, y)
        return x / y

    @staticmethod
    def backward(ctx, grad_output):
        x, y = ctx.to_save
        return (1 / y) * grad_output, -(x / np.square(y)) * grad_output


register('div', Div)


class Pow(Function):
    @staticmethod
    def forward(ctx, input, power):
        ctx.save_for_backward(input, power)
        return input**power

    @staticmethod
    def backward(ctx, grad_output):
        input, power = ctx.to_save
        return grad_output * power * (input**(
            power - 1)), grad_output * input**power * np.log(input)


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
        return np.array(np.maximum(input, 0))

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.to_save
        return grad_output * (input >= 0)


register('relu', Relu)


class LogSoftmax(Function):
    @staticmethod
    def forward(ctx, input):
        def logsumexp(x):
            #return np.log(np.exp(x).sum(axis=1))
            c = x.max(axis=1)
            return c + np.log(np.exp(x - c.reshape((-1, 1))).sum(axis=1))

        output = input - logsumexp(input).reshape((-1, 1))
        ctx.save_for_backward(output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        output, = ctx.to_save
        return grad_output - np.exp(output) * (grad_output.sum(axis=1).reshape(
            (-1, 1)))


register('logsoftmax', LogSoftmax)


class Reshape(Function):
    @staticmethod
    def forward(ctx, x, shape):
        ctx.save_for_backward(x.shape)
        return x.reshape(shape)

    @staticmethod
    def backward(ctx, grad_output):
        in_shape, = ctx.to_save
        return grad_output.reshape(in_shape)


register('reshape', Reshape)


class Conv2d(Function):
    @staticmethod
    def forward(ctx, x, w, stride=1, groups=1):
        if type(ctx.stride) == int:
            ctx.stride = (ctx.stride, ctx.stride)

        cout, cin, H, W = w.shape
        ys, xs = ctx.stride
        # HxW of input
        bs, cin_ = x.shape[0], x.shape[1]

        #output H and W
        oy, ox = (x.shape[2] - (H - ys)) // ys, (x.shape[3] - (W - xs)) // xs
        assert cin * ctx.groups == cin_
        assert cout % ctx.groups == 0
        rcout = cout // ctx.groups

        #batch size X groups X dim3 X H x W
        gx = x.reshape(bs, ctx.groups, cin, x.shape[2], x.shape[3])

        tx = np.lib.stride_tricks.as_strided(
            gx,
            shape=(bs, ctx.groups, cin, oy, ox, H, W),
            strides=(gx.strides[0], gx.strides[1], gx.strides[2],
                     gx.strides[3] * ys, gx.strides[4] * xs, gx.strides[3],
                     gx.strides[4]),
            writeable=False,
        )
        tw = w.reshape(ctx.groups, rcout, cin, H, W)
        ctx.save_for_backward(tx, tw, x.shape)

        ret = np.zeros((bs, ctx.groups, oy, ox, rcout), dtype=x.dtype)
        for g in range(ctx.groups):
            #ijYXyx,kjyx -> iYXk ->ikYX
            ret[:, g] += np.tensordot(tx[:, g], tw[g], ((1, 4, 5), (1, 2, 3)))
        return np.moveaxis(ret, 4, 2).reshape(bs, cout, oy, ox)

    @staticmethod
    def backward(ctx, grad_output):
        bs, _, oy, ox = grad_output.shape
        tx, tw, x_shape = ctx.to_save
        _, rcout, cin, H, W = tw.shape
        ys, xs = ctx.stride
        OY, OX = x_shape[2:4]

        ggg = grad_output.reshape(bs, ctx.groups, rcout, oy, ox)

        gdw = np.zeros((ctx.groups, rcout, cin, H, W), dtype=tx.dtype)
        for g in range(ctx.groups):
            #'ikYX,ijYXyx -> kjyx'
            gdw[g] += np.tensordot(ggg[:, g], tx[:, g], ((0, 2, 3), (0, 2, 3)))

        # needs to be optimized
        gdx = np.zeros((bs, ctx.groups, cin, OY, OX), dtype=tx.dtype)
        for Y in range(grad_output.shape[2]):
            for X in range(grad_output.shape[3]):
                iY, iX = Y * ys, X * xs
                #gdx[:,:,: , iY:iY+H, iX:iX+W] += np.einsum('igk,gkjyx->igjyx', ggg[:,:,:,Y,X], tw)
                for g in range(ctx.groups):
                    tg = np.dot(ggg[:, g, :, Y, X].reshape(bs, -1),
                                tw[g].reshape(rcout, -1))
                    gdx[:, g, :, iY:iY + H, iX:iX + W] += tg.reshape(
                        (bs, cin, H, W))

        return gdx.reshape((bs, ctx.groups * cin, OY, OX)), gdw.reshape(
            (ctx.groups * rcout, cin, H, W))


register('conv2d', Conv2d)


def stack_for_pool(x, py, px):
    my, mx = (x.shape[2] // py) * py, (x.shape[3] // px) * px
    stack = []
    xup = x[:, :, :my, :mx]
    for Y in range(py):
        for X in range(px):
            stack.append(xup[:, :, Y::py, X::px][None])
    return np.concatenate(stack, axis=0)


def unstack_for_pool(fxn, s, py, px):
    my, mx = (s[2] // py) * py, (s[3] // px) * px
    for Y in range(py):
        for X in range(px):
            ll = fxn(Y * px + X)
            if X == 0 and Y == 0:
                ret = np.zeros(s, dtype=ll.dtype)
            ret[:, :, Y:my:py, X:mx:px] = ll
    return ret


class MaxPool2D(Function):
    @staticmethod
    def forward(ctx, x, kernel_size=(2, 2)):
        stack = stack_for_pool(x, *kernel_size)
        idxs = np.argmax(stack, axis=0)
        ctx.save_for_backward(idxs, x.shape)
        return np.max(stack, axis=0)

    @staticmethod
    def backward(ctx, grad_output):
        idxs, s = ctx.to_save
        return unstack_for_pool(lambda idx: grad_output * (idxs == idx), s,
                                *ctx.kernel_size)


register('max_pool2d', MaxPool2D)
