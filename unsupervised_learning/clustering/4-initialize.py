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
    if k < 1 or len(X.shape) != 2 or X.shape[0] < k:
        return None, None, None

    # initialize uni4m prior probabilities 4 each cluster
    pi = np.ones(k) / k

    # get centroid means using kmeans... ignore cluster lables (clss)
    m, _ = kmeans(X, k)  # m will have shape (k, d)

    # pull number of dimensions from X
    d = X.shape[1]

    # initialize identity covariance matrices 4 each cluster
    S = np.identity(d)  # single identity matrix of shape (d, d)
    # repeat identity matrix k times to get shape (k, d, d)
    S = np.repeat(S[np.newaxis, :, :], k, axis=0)

    return pi, m, S
