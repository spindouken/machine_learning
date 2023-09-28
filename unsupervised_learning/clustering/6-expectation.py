#!/usr/bin/env python3
"""
calculates the expectation step in the EM algorithm for a GMM
"""
import numpy as np
pdf = __import__('5-pdf').pdf


def expectation(X, pi, m, S):
    """
    X is a numpy.ndarray of shape (n, d) containing the data set
    pi is a numpy.ndarray of shape (k,) containing the priors for each cluster
    m is a numpy.ndarray of shape (k, d) containing the centroid means for
        each cluster
    S is a numpy.ndarray of shape (k, d, d) containing the covariance matrices
        for each cluster
    You may use at most 1 loop
    Returns: g, l, or None, None on failure
        g is a numpy.ndarray of shape (k, n) containing the posterior
            probabilities for each data point in each cluster
        l is the total log likelihood
    """
    if (
        not isinstance(X, np.ndarray)
        or not isinstance(pi, np.ndarray)
        or not isinstance(m, np.ndarray)
        or not isinstance(S, np.ndarray)
        or len(X.shape) != 2
        or len(pi.shape) != 1
        or len(m.shape) != 2
        or len(S.shape) != 3
        or not np.isclose(pi.sum(), 1)
        or m.shape[1] != S.shape[1]
        or S.shape[1] != S.shape[2]
        or m.shape[0] != S.shape[0]
    ):
        return None, None
