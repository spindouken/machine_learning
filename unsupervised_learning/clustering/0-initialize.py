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
    if len(X.shape) != 2 or k < 1:
        return None
