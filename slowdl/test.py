from slowdl.tensor3 import Tensor
import numpy as np

a = Tensor([1, 2, 3, 4, 5], requires_grad=True)
b = Tensor([2, 2, 2, 2, 2], requires_grad=True)

c = a + b
c.backward()

print(a.grad)
