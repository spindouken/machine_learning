#!/usr/bin/env python3
"""
calculates the probability density function of a Gaussian distribution
"""
import numpy as np


def pdf(X, m, S):
    """
    X is a numpy.ndarray of shape (n, d) containing the data points whose PDF
        should be evaluated
    m is a numpy.ndarray of shape (d,) containing the mean of the distribution
    S is a numpy.ndarray of shape (d, d) containing the covariance of the
        distribution
    Returns: P, or None on failure
        P is a numpy.ndarray of shape (n,) containing the PDF values for each
            data point
    """
    if (
        not isinstance(X, np.ndarray)
        or not isinstance(m, np.ndarray)
        or not isinstance(S, np.ndarray)
    ):
        return None
    if len(X.shape) != 2 or len(m.shape) != 1 or len(S.shape) != 2:
        return None
