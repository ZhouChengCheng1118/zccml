# __Author__:Zcc
import numpy as np


def euclidean_distance(a, b):
    if isinstance(a, list) and isinstance(b, list):
        a = np.array(a)
        b = np.array(b)

    return np.sqrt(sum((a - b) ** 2))