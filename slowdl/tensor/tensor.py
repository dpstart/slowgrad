import numpy as np


class Tensor(object):
    def __init__(self, data, grad=True):
        self.data = data
        self.grad = grad
