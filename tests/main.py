from slowgrad.tensor import Tensor
import numpy as np
img = Tensor(np.random.rand(1, 1, 10, 10))
ker = Tensor(np.random.rand(1, 1, 3, 3))

print(img.conv2d(ker))
