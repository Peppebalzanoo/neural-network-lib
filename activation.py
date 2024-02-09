import numpy as np


def identity(A_layer, der=0):
    if der == 0:
        return A_layer
    else:
        return 1


def numpy_tanh(A_layer, der=0):
    tanh_result = np.tanh(A_layer)
    if der == 0:
        return tanh_result
    else:
        tanh_derivative = 1 - tanh_result ** 2
        return tanh_derivative
