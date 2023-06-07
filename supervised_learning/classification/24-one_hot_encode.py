#!/usr/bin/env python3
"""hot encoding!"""
import numpy as np


def one_hot_encode(Y, classes):
    """
    Converts a numeric label vector into a one-hot matrix
    Y: a numpy.ndarray with shape (m,) containing numeric class labels
    m: number of examples
    classes: the maximum number of classes found in Y
    Returns a one-hot encoding of Y with shape (classes, m), or None on failure
    """
    if not isinstance(Y, np.ndarray) or len(Y) == 0:
        return None
    if not isinstance(classes, int) or classes <= np.max(Y):
        return None

    # Create a new array of zeros with shape (classes, m)
    one_hot = np.zeros((classes, Y.shape[0]))

    # For each class, set the column corresponding to the indices in Y to 1
    for i, value in enumerate(Y):
        one_hot[value][i] = 1

    return one_hot
