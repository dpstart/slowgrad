from slowgrad.tensor import Tensor
import numpy as np

x = Tensor(np.eye(3), requires_grad=True)
y = Tensor([[2.0, 0, -2.0]], requires_grad=True)
z = y.mm(x).sum()
z.backward()

print(x.grad)  # dz/dx
print(y.grad)  # dz/dy
