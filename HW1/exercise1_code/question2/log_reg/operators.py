import numpy as np 


def l1_prox(y, weight):
    """Projection onto the l1-ball.
    """
    return np.maximum(np.abs(y) - weight, 0) * np.sign(y)


def l2_prox(y, weight):
    """Projection onto the l2-ball.
    """
    return (1.0 / (weight + 1)) * y


def norm1(x):
    """Returns the l1 norm `x`.
    """
    return np.linalg.norm(x, 1)


def norm2sq(x):
    """Returns the l2 norm squared of `x`.
    """
    return (1.0 / 2) * np.linalg.norm(x) ** 2
