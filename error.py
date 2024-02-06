import numpy as np


def soft_max(y):
    y_exp = np.exp(y - y.max(0))
    z = y_exp / sum(y_exp, 0)
    return z


def cross_entropy(y, t, der=0):
    z = soft_max(y) + 1e-10  # smoothing
    if der == 0:
        return -(t * np.log(z)).sum()
    else:
        return z - t
