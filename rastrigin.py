import numpy as np

def rastrigin(X):
    """
    Rastrigin function (2D version) for optimization.
    Global minimum at (0, 0) with value 0.
    """
    x, y = X
    return 20 + x**2 + y**2 - 10 * (np.cos(2 * np.pi * x) + np.cos(2 * np.pi * y))
