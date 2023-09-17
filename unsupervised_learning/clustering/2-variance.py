#!/usr/bin/env python3
"""
calculates the total intra-cluster variance for a data set
"""
import numpy as np


def variance(X, C):
    """
    calculates the total intra-cluster variance for a data set
    X is a numpy.ndarray of shape (n, d) containing the data set
    C is a numpy.ndarray of shape (k, d) containing the centroid means for
        each cluster
    Returns: var, or None on failure
        var is the total variance
    """
    if not isinstance(X, np.ndarray) or not isinstance(C, np.ndarray):
        return None
    if len(X.shape) != 2 or len(C.shape) != 2:
        return None
