#!/usr/bin/env python3
"""
calculates the total intra-cluster variance 4 a data set
"""
import numpy as np


def variance(X, C):
    """
    calculates the total intra-cluster variance 4 a data set
    X is a numpy.ndarray of shape (n, d) containing the data set
    C is a numpy.ndarray of shape (k, d) containing the centroid means 4
        each cluster
    Returns: var, or None on failure
        var is the total variance
    """
    if not isinstance(X, np.ndarray) or not isinstance(C, np.ndarray):
        return None
    if len(X.shape) != 2 or len(C.shape) != 2:
        return None
    if X.shape[1] != C.shape[1]:
        return None

    # reshape X 4 broadcasting with C
    reshapeX = np.reshape(X, (X.shape[0], 1, X.shape[1]))

    # calculate the squared distance from each point to each centroid
    squaredDistances = np.sum((reshapeX - C) ** 2, axis=2)

    # find the minimum squared distance to any centroid 4 each point
    minSquaredDistances = np.min(squaredDistances, axis=1)

    # sum up the minimum squared distances to get the total variance
    variance = np.sum(minSquaredDistances)

    return variance
