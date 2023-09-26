#!/usr/bin/env python3
"""
initializes cluster centroids for K-means
"""
import numpy as np


def initialize(X, k):
    """
    X is a numpy.ndarray of shape (n, d) containing the dataset that will be
        used for K-means clustering
            n is the number of data points
            d is the number of dimensions for each data point
    k is a positive integer containing the number of clusters
    Returns: a numpy.ndarray of shape (k, d) containing the initialized
        centroids for each cluster, or None on failure
    """
    if not isinstance(X, np.ndarray) or not isinstance(k, int):
        return None
    if k < 1:
        return None

    n, d = X.shape

    minValues = np.min(X, axis=0)
    maxValues = np.max(X, axis=0)

    # multivariate uniform distribution
    centroids = np.random.uniform(minValues, maxValues, (k, d))

    # return initialized centroids
    return centroids
