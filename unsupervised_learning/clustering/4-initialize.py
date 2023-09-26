#!/usr/bin/env python3
"""
initializes variables for a Gaussian Mixture Model
"""
import numpy as np
kmeans = __import__('1-kmeans').kmeans


def initialize(X, k):
    """
    X is a numpy.ndarray of shape (n, d) containing the data set
    k is a positive integer containing the number of clusters
    You are not allowed to use any loops
    Returns: pi, m, S, or None, None, None on failure
        pi is a numpy.ndarray of shape (k,) containing the priors for each
            cluster, initialized evenly
        m is a numpy.ndarray of shape (k, d) containing the centroid means for
            each cluster, initialized with K-means
        S is a numpy.ndarray of shape (k, d, d) containing the covariance
            matrices for each cluster, initialized as identity matrices
    """
    if not isinstance(X, np.ndarray) or not isinstance(k, int):
        return None, None, None
    if k < 1 or len(X.shape) != 2:
        return None, None, None
