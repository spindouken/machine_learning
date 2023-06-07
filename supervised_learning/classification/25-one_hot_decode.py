#!/usr/bin/env python3
"""hot decoding!"""
import numpy as np


def one_hot_decode(one_hot):
    """
    Converts a one-hot matrix into a vector of labels
    one_hot: a one-hot encoded numpy.ndarray with shape (classes, m)
    Returns a numpy.ndarray with shape (m,)
        containing the numeric labels for each example, or None on failure
    """
    # Check the validity of the parameter
    if not isinstance(one_hot, np.ndarray) or len(one_hot.shape) != 2:
        return None

    # Use the argmax function to find the index of the 1 in each column
    labels = np.argmax(one_hot, axis=0)

    return labels
