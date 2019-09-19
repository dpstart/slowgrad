import pytest
import numpy as np
from slowdl.tensor import Tensor


@pytest.mark.unittest
def test_create_tensor():
    data = np.zeros([5, 5])
    tensor = Tensor(data)

    assert type(tensor) == Tensor
    assert (tensor.data == data).all()
