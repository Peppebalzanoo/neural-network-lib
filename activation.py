import numpy as np


def identity(A_layer, der=0):
    if der == 0:
        return A_layer
    else:
        return 1


def tanh(A_layer, der=0):
    tanh_result = np.tanh(A_layer)
    if der == 0:
        return tanh_result
    else:
        tanh_derivative = 1 - tanh_result ** 2
        return tanh_derivative


def ReLU(A_layer, der=0):
    if der == 0:
        return np.where(A_layer > 0, A_layer, 0)
    else:
        return np.where(A_layer > 0, 1, 0)


def leaky_ReLU(A_layer, der=0, alpha=0.1):
    if der == 0:
        return np.where(A_layer >= 0, alpha * A_layer, 0)
    else:
        return np.where(A_layer >= 0, alpha * 1, 0)
