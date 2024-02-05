import numpy as np


def identity(x, der=0):
    if der == 0:
        return x
    else:
        return 1


def numpy_tanh(x, der=0):
    tanh_result = np.tanh(x)
    if der == 0:
        return tanh_result
    else:
        tanh_derivative = 1 - tanh_result ** 2
        return tanh_derivative
