#!/usr/bin/env python3
"""
performs K-means on a dataset
"""
import numpy as np


def kmeans(X, k, iterations=1000):
    """
    X is a numpy.ndarray of shape (n, d) containing the dataset
        n is the number of data points
        d is the number of dimensions for each data point
    k is a positive integer containing the number of clusters
    iterations is a positive integer containing the maximum number of
        iterations that should be performed
    If no change in the cluster centroids occurs between iterations,
        function should return
    Initialize the cluster centroids using a multivariate uniform distribution
        (based on 0-initialize.py)
    If a cluster contains no data points during the update step, reinitialize
        its centroid
    Returns: C, clss, or None, None on failure
    C is a numpy.ndarray of shape (k, d) containing the centroid means for
            each cluster
        clss is a numpy.ndarray of shape (n,) containing
            the index of the cluster
                in C that each data point belongs to
    """
    if not isinstance(X, np.ndarray) or not isinstance(k, int):
        return None
    if len(X.shape) != 2 or k < 1:
        return None
