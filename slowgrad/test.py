from slowgrad import Tensor
import numpy as np
import torch.nn.functional as F
import torch

inn = np.random.rand(1,3,32,32)
w_in = np.random.rand(5,3,5,5)
x = Tensor(inn, requires_grad=True)
w = Tensor(w_in, requires_grad=True)


out = x.conv2d(w)
out_torch = F.conv2d(torch.tensor(inn),torch.tensor(w_in), stride=1,padding=1)



print(out.data.shape)
print(out_torch.data.shape)
np.testing.assert_allclose(out.data, out_torch.data)

