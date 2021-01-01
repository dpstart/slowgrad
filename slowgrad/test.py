from slowgrad.tensor import Tensor
import numpy as np


x = Tensor(np.random.rand(1,3,32,32), requires_grad=True)
w = Tensor(np.random.rand(1,3,5,5), requires_grad=True)
out = x.conv2d(w)


print(out.shape)
