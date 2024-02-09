import numpy as np


def soft_max(Z):
    # Z - Z.max(0) for avoid overflow
    Y_exp = np.exp(Z - Z.max(0))
    # Y_exp is a matrix of e^z
    z = Y_exp / sum(Y_exp, 0)  # soft-max formula
    return z


def cross_entropy(Z_out, Y_gold, der=0):
    # Z is a matrix with probability
    Z = soft_max(Z_out) + 1e-10  # smoothing
    if der == 0:
        return -(Y_gold * np.log(Z)).sum()
    else:
        # Derivative of error function respect weights
        return Z - Y_gold
