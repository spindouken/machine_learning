#!/usr/bin/env python3
"""
calculates the maximization step in the EM algorithm for a GMM
"""
import numpy as np


def maximization(X, g):
    """
    X is a numpy.ndarray of shape (n, d) containing the data set
    g is a numpy.ndarray of shape (k, n) containing the posterior probabilities
        for each data point in each cluster
    You may use at most 1 loop
    Returns: pi, m, S, or None, None, None on failure
        pi is a numpy.ndarray of shape (k,) containing the updated priors for
            each cluster
        m is a numpy.ndarray of shape (k, d) containing the updated centroid
            means for each cluster
        S is a numpy.ndarray of shape (k, d, d) containing the updated
            covariance matrices for each cluster
    """
    if (
        not isinstance(X, np.ndarray)
        or not isinstance(g, np.ndarray)
        or not isinstance(pi, np.ndarray)
        or not isinstance(m, np.ndarray)
        or not isinstance(S, np.ndarray)
    ):
        return None, None, None

    if (
        len(X.shape) != 2
        or len(g.shape) != 2
        or len(pi.shape) != 1
        or len(m.shape) != 2
        or len(S.shape) != 3
    ):
        return None, None, None
