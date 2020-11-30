import torch
import numpy as np

import unittest
import pytest
import timeit
import functools

from slowgrad.tensor import Tensor


def helper_test_op(shps,
                   torch_fxn,
                   slowgrad_fxn,
                   atol=0,
                   rtol=1e-6,
                   grad_atol=0,
                   grad_rtol=1e-6,
                   forward_only=False):
    torch.manual_seed(0)
    ts = [torch.rand(x, requires_grad=True) for x in shps]
    tst = [Tensor(x.detach().numpy(), requires_grad=True) for x in ts]

    out = torch_fxn(*ts)
    ret = slowgrad_fxn(*tst)

    np.testing.assert_allclose(ret.data,
                               out.detach().numpy(),
                               atol=atol,
                               rtol=rtol)

    if not forward_only:
        out.mean().backward()
        ret.mean().backward()

        for t, tt in zip(ts, tst):
            np.testing.assert_allclose(t.grad,
                                       tt.grad.data,
                                       atol=grad_atol,
                                       rtol=grad_rtol)

    # speed
    torch_fp = timeit.Timer(functools.partial(torch_fxn, *
                                              ts)).timeit(5) * 1000 / 5
    slowgrad_fp = timeit.Timer(functools.partial(slowgrad_fxn, *
                                                 tst)).timeit(5) * 1000 / 5

    if not forward_only:
        torch_fbp = timeit.Timer(
            functools.partial(lambda f, x: f(*x).mean().backward(), torch_fxn,
                              ts)).timeit(5) * 1000 / 5
        slowgrad_fbp = timeit.Timer(
            functools.partial(lambda f, x: f(*x).mean().backward(),
                              slowgrad_fxn, tst)).timeit(5) * 1000 / 5
    else:
        torch_fbp, slowgrad_fbp = np.nan, np.nan

    print(
        "testing %30r   torch/slowgrad fp: %.2f / %.2f ms  bp: %.2f / %.2f ms"
        % (shps, torch_fp, slowgrad_fp, torch_fbp - torch_fp,
           slowgrad_fbp - slowgrad_fp))


class TestOps(unittest.TestCase):
    def test_add(self):
        helper_test_op([(45, 65), (45, 65)],
                       lambda x, y: x + y,
                       Tensor.add,
                       forward_only=False)

    def test_sub(self):
        helper_test_op([(45, 65), (45, 65)],
                       lambda x, y: x - y,
                       Tensor.sub,
                       forward_only=True)

    def test_mul(self):
        helper_test_op([(45, 65), (45, 65)], lambda x, y: x * y, Tensor.mul)

    def test_div(self):
        helper_test_op([(45, 65), (45, 65)], lambda x, y: x / y, Tensor.div)

    def test_pow(self):
        helper_test_op([(45, 65), (45, 65)], lambda x, y: x**y, Tensor.pow)
