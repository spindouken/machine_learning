#!/usr/bin/env python3
"""
performs K-means on a dataset
"""
import sklearn.cluster
import numpy as np


def kmeans(X, k):
    """
    X is a numpy.ndarray of shape (n, d) containing the dataset
    k is the number of clusters
    Returns: C, clss
        C is a numpy.ndarray of shape (k, d) containing the centroid means
            for each cluster
        clss is a numpy.ndarray of shape (n,) containing the index of the
            cluster in C that each data point belongs to
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None
    if not isinstance(k, int) or k <= 0:
        return None

    cluster = sklearn.cluster
    C, clss, _ = cluster.k_means(X, k)
    return C, clss
