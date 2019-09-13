import pytest
from slowdl.tensor import Tensor


@pytest.mark.unittest
def test_create_tensor():

    tensor = Tensor([])
    assert type(tensor) == Tensor
