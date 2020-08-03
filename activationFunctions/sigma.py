import numpy as np


def sigma(z):
    return 1.0 / (1 + np.exp(-z))


def derivative_sigma(z):
    """Derivative of the sigmoid function."""
    return sigma(z) * (1 - sigma(z))
