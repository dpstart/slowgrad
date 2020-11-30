import pytest
import numpy as np
from slowgrad.tensor import Tensor


@pytest.mark.unittest
def test_create_tensor():
    data = np.zeros([5, 5])
    tensor = Tensor(data)

    assert type(tensor) == Tensor
    assert (tensor.data == data).all()


@pytest.mark.unittest
def test_add_gradient():
    a = Tensor([1, 2, 3, 4, 5], requires_grad=True)
    b = Tensor([2, 2, 2, 2, 2], requires_grad=True)

    c = a + b
    c.backward()

    assert (a.grad.data == np.array([
        1,
        1,
        1,
        1,
        1,
    ])).all()
